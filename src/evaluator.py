"""
evaluator.py -- Benchmark Evaluation Engine
============================================
Ground-truth comparison engine for the dual-tier BGP anomaly monitor.

Intercepts every detection result from Tier 1 (RuleEngineResult) and
Tier 2 (Tier2Result), strips the internal hidden ground-truth tag, and
computes classification and latency metrics across five test suites.

Test Suites
-----------
S1  Pure Detection       -- All 5 event classes end-to-end.
S2  MOAS Stress Test     -- Focus on LEGIT_MOAS / LEGIT_TE false-positive
                            suppression.
S3  Temporal Precision   -- Origin-stability & burst-entropy sensitivity
                            across window boundaries.
S4  Latency Profiling    -- Per-tier and end-to-end latency budgets.
S5  Ablation Study       -- Tier 1-only vs Tier 1 + Tier 2 comparison.

Metrics (per suite)
-------------------
* Precision   = TP / (TP + FP)
* Recall      = TP / (TP + FN)
* F1-Score    = 2 * Precision * Recall / (Precision + Recall)
* FPR         = FP / (FP + TN)
* Alert Latency (mean, P50, P95, P99, max) in milliseconds
"""

from __future__ import annotations

import collections
import json
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from data_engine import BGPUpdate, EventClass
from rule_engine import RuleEngineResult, ROVState
from ml_engine import Tier2Result


# ===================================================================
# Constants
# ===================================================================

#: Ground-truth event classes that are TRUE anomalies.
ANOMALY_CLASSES: Set[EventClass] = {
    EventClass.HIJACK_ORIGIN,
    EventClass.HIJACK_SUBPREFIX,
    EventClass.HIJACK_MAXLEN,
}

#: Ground-truth event classes that are LEGITIMATE (should NOT alert).
LEGIT_CLASSES: Set[EventClass] = {
    EventClass.LEGIT_MOAS,
    EventClass.LEGIT_TE,
}

#: Benign baseline.
BENIGN_CLASSES: Set[EventClass] = {
    EventClass.BENIGN,
}


class TestSuite(Enum):
    """Identifiers for the five evaluation test suites."""

    S1_PURE_DETECTION = "S1"
    S2_MOAS_STRESS = "S2"
    S3_TEMPORAL_PRECISION = "S3"
    S4_LATENCY_PROFILING = "S4"
    S5_ABLATION_STUDY = "S5"


# ===================================================================
# Data Models
# ===================================================================

@dataclass
class ConfusionMatrix:
    """Standard binary confusion matrix."""

    tp: int = 0  # True Positive: anomaly correctly flagged
    fp: int = 0  # False Positive: benign/legit incorrectly flagged
    tn: int = 0  # True Negative: benign/legit correctly passed
    fn: int = 0  # False Negative: anomaly missed

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate = FP / (FP + TN)."""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1_score": round(self.f1_score, 6),
            "fpr": round(self.fpr, 6),
            "total": self.total,
        }


@dataclass
class LatencyProfile:
    """Aggregated latency statistics in milliseconds."""

    samples: List[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = min(int(p / 100.0 * len(s)), len(s) - 1)
        return s[idx]

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def under_5ms_pct(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s < 5.0) / len(self.samples) * 100

    @property
    def under_50ms_pct(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s < 50.0) / len(self.samples) * 100

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 4),
            "median_ms": round(self.median, 4),
            "p95_ms": round(self.p95, 4),
            "p99_ms": round(self.p99, 4),
            "max_ms": round(self.max_ms, 4),
            "under_5ms_pct": round(self.under_5ms_pct, 2),
            "under_50ms_pct": round(self.under_50ms_pct, 2),
        }


@dataclass
class PerClassMetrics:
    """Metrics broken down by ground-truth event class."""

    class_name: str
    total: int = 0
    detected_tier1: int = 0
    detected_tier2: int = 0
    detected_combined: int = 0
    missed: int = 0

    @property
    def detection_rate(self) -> float:
        return self.detected_combined / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "total": self.total,
            "detected_tier1": self.detected_tier1,
            "detected_tier2": self.detected_tier2,
            "detected_combined": self.detected_combined,
            "missed": self.missed,
            "detection_rate": round(self.detection_rate, 6),
        }


@dataclass
class SuiteReport:
    """Complete report for one test suite."""

    suite: TestSuite
    description: str
    confusion: ConfusionMatrix
    per_class: Dict[str, PerClassMetrics]
    latency_tier1: LatencyProfile
    latency_tier2: LatencyProfile
    latency_e2e: LatencyProfile
    n_updates: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite.value,
            "description": self.description,
            "n_updates": self.n_updates,
            "confusion_matrix": self.confusion.to_dict(),
            "per_class_metrics": {k: v.to_dict()
                                  for k, v in self.per_class.items()},
            "latency_tier1": self.latency_tier1.to_dict(),
            "latency_tier2": self.latency_tier2.to_dict(),
            "latency_e2e": self.latency_e2e.to_dict(),
            "extra": self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ===================================================================
# Unified Detection Record
# ===================================================================

@dataclass
class DetectionRecord:
    """Unified record linking a BGP UPDATE to its detection outcome
    across both tiers.

    This is the fundamental unit consumed by the evaluator: one record
    per update flowing through the pipeline.
    """

    update: BGPUpdate
    ground_truth: EventClass

    # Tier 1 outcome
    tier1_result: Optional[RuleEngineResult] = None
    tier1_flagged: bool = False          # short-circuited = alert emitted
    tier1_latency_ms: float = 0.0

    # Tier 2 outcome
    tier2_result: Optional[Tier2Result] = None
    tier2_flagged: bool = False          # anomaly_score >= theta
    tier2_latency_ms: float = 0.0

    # Combined outcome
    flagged: bool = False                # flagged by either tier
    e2e_latency_ms: float = 0.0         # total pipeline latency


# ===================================================================
# Record Builder
# ===================================================================

def build_detection_records(
    tier1_results: Sequence[RuleEngineResult],
    tier2_results: Optional[Sequence[Tier2Result]] = None,
) -> List[DetectionRecord]:
    """Build DetectionRecord objects from pipeline outputs.

    Joins Tier 1 results with optional Tier 2 results by matching on
    the underlying BGPUpdate object identity.

    Parameters
    ----------
    tier1_results : sequence of RuleEngineResult
        Complete Tier 1 output (one per update).
    tier2_results : sequence of Tier2Result or None
        Tier 2 output (one per non-short-circuited update).
        If None, only Tier 1 is evaluated (ablation mode).
    """
    # Index Tier 2 results by update id for fast lookup
    t2_by_id: Dict[int, Tier2Result] = {}
    if tier2_results:
        for t2r in tier2_results:
            t2_by_id[id(t2r.feature_vector.update)] = t2r

    records: List[DetectionRecord] = []
    for t1r in tier1_results:
        update = t1r.update
        gt = update._ground_truth.event_class

        rec = DetectionRecord(
            update=update,
            ground_truth=gt,
            tier1_result=t1r,
            tier1_flagged=t1r.short_circuit,
            tier1_latency_ms=t1r.processing_time_ms,
        )

        # Check Tier 2
        t2r = t2_by_id.get(id(update))
        if t2r is not None:
            rec.tier2_result = t2r
            rec.tier2_flagged = t2r.is_anomaly
            rec.tier2_latency_ms = t2r.processing_time_ms

        # Combined flag: alerted by either tier
        rec.flagged = rec.tier1_flagged or rec.tier2_flagged

        # End-to-end latency
        rec.e2e_latency_ms = rec.tier1_latency_ms + rec.tier2_latency_ms

        records.append(rec)

    return records


# ===================================================================
# Core Evaluation Engine
# ===================================================================

class Evaluator:
    """Benchmark Evaluation Engine.

    Consumes DetectionRecord objects and produces SuiteReport results
    across five standardised test suites.

    Usage
    -----
    >>> evaluator = Evaluator()
    >>> records = build_detection_records(tier1_results, tier2_results)
    >>> full_report = evaluator.run_all_suites(records)
    """

    def __init__(self) -> None:
        self._reports: Dict[TestSuite, SuiteReport] = {}

    # ------------------------------------------------------------------
    # Suite runners
    # ------------------------------------------------------------------

    def run_all_suites(
        self,
        records: Sequence[DetectionRecord],
        tier1_only_records: Optional[Sequence[DetectionRecord]] = None,
    ) -> Dict[TestSuite, SuiteReport]:
        """Run all five test suites and return their reports.

        Parameters
        ----------
        records : sequence of DetectionRecord
            Full pipeline records (Tier 1 + Tier 2).
        tier1_only_records : sequence of DetectionRecord or None
            Records processed by Tier 1 only (for S5 ablation).
            If None, S5 uses *records* in Tier 1-only mode.
        """
        self._reports[TestSuite.S1_PURE_DETECTION] = self._run_s1(records)
        self._reports[TestSuite.S2_MOAS_STRESS] = self._run_s2(records)
        self._reports[TestSuite.S3_TEMPORAL_PRECISION] = self._run_s3(records)
        self._reports[TestSuite.S4_LATENCY_PROFILING] = self._run_s4(records)
        self._reports[TestSuite.S5_ABLATION_STUDY] = self._run_s5(
            records, tier1_only_records
        )
        return dict(self._reports)

    # ------------------------------------------------------------------
    # S1: Pure Detection
    # ------------------------------------------------------------------

    def _run_s1(self, records: Sequence[DetectionRecord]) -> SuiteReport:
        """S1 -- Pure Detection: evaluate all 5 event classes end-to-end.

        Classification rule:
        - Anomaly classes flagged by either tier = TP
        - Anomaly classes NOT flagged = FN
        - Benign / Legit classes flagged = FP
        - Benign / Legit classes NOT flagged = TN
        """
        cm = ConfusionMatrix()
        per_class: Dict[str, PerClassMetrics] = {}
        lat_t1 = LatencyProfile()
        lat_t2 = LatencyProfile()
        lat_e2e = LatencyProfile()

        for rec in records:
            gt = rec.ground_truth
            cls_name = gt.name
            if cls_name not in per_class:
                per_class[cls_name] = PerClassMetrics(class_name=cls_name)
            pc = per_class[cls_name]
            pc.total += 1

            is_true_anomaly = gt in ANOMALY_CLASSES

            if rec.flagged:
                if is_true_anomaly:
                    cm.tp += 1
                    pc.detected_combined += 1
                    if rec.tier1_flagged:
                        pc.detected_tier1 += 1
                    if rec.tier2_flagged:
                        pc.detected_tier2 += 1
                else:
                    cm.fp += 1
            else:
                if is_true_anomaly:
                    cm.fn += 1
                    pc.missed += 1
                else:
                    cm.tn += 1

            lat_t1.record(rec.tier1_latency_ms)
            if rec.tier2_latency_ms > 0:
                lat_t2.record(rec.tier2_latency_ms)
            lat_e2e.record(rec.e2e_latency_ms)

        return SuiteReport(
            suite=TestSuite.S1_PURE_DETECTION,
            description="Pure Detection: all 5 event classes, full pipeline",
            confusion=cm,
            per_class=per_class,
            latency_tier1=lat_t1,
            latency_tier2=lat_t2,
            latency_e2e=lat_e2e,
            n_updates=len(records),
        )

    # ------------------------------------------------------------------
    # S2: MOAS Stress Test
    # ------------------------------------------------------------------

    def _run_s2(self, records: Sequence[DetectionRecord]) -> SuiteReport:
        """S2 -- MOAS Stress Test: focus on false-positive suppression
        for LEGIT_MOAS and LEGIT_TE.

        Filters to only LEGIT_MOAS, LEGIT_TE, and a sample of BENIGN
        records.  Measures FPR, precision, and per-class false-alarm
        rates.
        """
        # Include: LEGIT_MOAS, LEGIT_TE, BENIGN, and anomalies for context
        relevant = [r for r in records if r.ground_truth in (
            LEGIT_CLASSES | BENIGN_CLASSES | ANOMALY_CLASSES
        )]

        cm = ConfusionMatrix()
        per_class: Dict[str, PerClassMetrics] = {}
        lat_t1 = LatencyProfile()
        lat_t2 = LatencyProfile()
        lat_e2e = LatencyProfile()

        moas_false_alarms = 0
        te_false_alarms = 0
        moas_total = 0
        te_total = 0

        for rec in relevant:
            gt = rec.ground_truth
            cls_name = gt.name
            if cls_name not in per_class:
                per_class[cls_name] = PerClassMetrics(class_name=cls_name)
            pc = per_class[cls_name]
            pc.total += 1

            is_true_anomaly = gt in ANOMALY_CLASSES

            if rec.flagged:
                if is_true_anomaly:
                    cm.tp += 1
                    pc.detected_combined += 1
                else:
                    cm.fp += 1
                    if gt == EventClass.LEGIT_MOAS:
                        moas_false_alarms += 1
                    elif gt == EventClass.LEGIT_TE:
                        te_false_alarms += 1
            else:
                if is_true_anomaly:
                    cm.fn += 1
                    pc.missed += 1
                else:
                    cm.tn += 1

            if gt == EventClass.LEGIT_MOAS:
                moas_total += 1
            elif gt == EventClass.LEGIT_TE:
                te_total += 1

            lat_t1.record(rec.tier1_latency_ms)
            if rec.tier2_latency_ms > 0:
                lat_t2.record(rec.tier2_latency_ms)
            lat_e2e.record(rec.e2e_latency_ms)

        return SuiteReport(
            suite=TestSuite.S2_MOAS_STRESS,
            description="MOAS Stress Test: LEGIT_MOAS/TE false-positive suppression",
            confusion=cm,
            per_class=per_class,
            latency_tier1=lat_t1,
            latency_tier2=lat_t2,
            latency_e2e=lat_e2e,
            n_updates=len(relevant),
            extra={
                "moas_total": moas_total,
                "moas_false_alarms": moas_false_alarms,
                "moas_false_alarm_rate": round(
                    moas_false_alarms / moas_total, 6) if moas_total else 0.0,
                "te_total": te_total,
                "te_false_alarms": te_false_alarms,
                "te_false_alarm_rate": round(
                    te_false_alarms / te_total, 6) if te_total else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # S3: Temporal Precision
    # ------------------------------------------------------------------

    def _run_s3(self, records: Sequence[DetectionRecord]) -> SuiteReport:
        """S3 -- Temporal Precision: evaluate detection sensitivity
        across tumbling window boundaries.

        Groups records by window_id (from Tier 2 FeatureVector) and
        computes per-window detection rates to assess whether anomalies
        near window edges are missed.
        """
        cm = ConfusionMatrix()
        per_class: Dict[str, PerClassMetrics] = {}
        lat_t1 = LatencyProfile()
        lat_t2 = LatencyProfile()
        lat_e2e = LatencyProfile()

        # Per-window breakdown
        window_stats: Dict[int, Dict[str, int]] = collections.defaultdict(
            lambda: {"total": 0, "anomaly": 0, "detected": 0, "missed": 0}
        )

        for rec in records:
            gt = rec.ground_truth
            cls_name = gt.name
            if cls_name not in per_class:
                per_class[cls_name] = PerClassMetrics(class_name=cls_name)
            pc = per_class[cls_name]
            pc.total += 1

            is_true_anomaly = gt in ANOMALY_CLASSES

            # Determine window_id
            window_id = -1
            if rec.tier2_result and rec.tier2_result.feature_vector:
                window_id = rec.tier2_result.feature_vector.window_id

            ws = window_stats[window_id]
            ws["total"] += 1

            if rec.flagged:
                if is_true_anomaly:
                    cm.tp += 1
                    pc.detected_combined += 1
                    ws["detected"] += 1
                else:
                    cm.fp += 1
            else:
                if is_true_anomaly:
                    cm.fn += 1
                    pc.missed += 1
                    ws["missed"] += 1
                else:
                    cm.tn += 1

            if is_true_anomaly:
                ws["anomaly"] += 1

            lat_t1.record(rec.tier1_latency_ms)
            if rec.tier2_latency_ms > 0:
                lat_t2.record(rec.tier2_latency_ms)
            lat_e2e.record(rec.e2e_latency_ms)

        # Compute per-window detection rates
        window_detection_rates = {}
        for wid, ws in sorted(window_stats.items()):
            if ws["anomaly"] > 0:
                rate = ws["detected"] / ws["anomaly"]
            else:
                rate = None  # no anomalies in this window
            window_detection_rates[f"window_{wid}"] = {
                "total_updates": ws["total"],
                "anomalies": ws["anomaly"],
                "detected": ws["detected"],
                "missed": ws["missed"],
                "detection_rate": round(rate, 4) if rate is not None else "N/A",
            }

        return SuiteReport(
            suite=TestSuite.S3_TEMPORAL_PRECISION,
            description="Temporal Precision: detection across window boundaries",
            confusion=cm,
            per_class=per_class,
            latency_tier1=lat_t1,
            latency_tier2=lat_t2,
            latency_e2e=lat_e2e,
            n_updates=len(records),
            extra={"per_window": window_detection_rates},
        )

    # ------------------------------------------------------------------
    # S4: Latency Profiling
    # ------------------------------------------------------------------

    def _run_s4(self, records: Sequence[DetectionRecord]) -> SuiteReport:
        """S4 -- Latency Profiling: per-tier and end-to-end latency
        budgets with SLA compliance checks.

        SLA targets (from presentation):
        - Tier 1: <5 ms per update
        - Tier 2 (ML + SHAP): <50 ms per update
        """
        cm = ConfusionMatrix()  # minimal; focus is latency
        per_class: Dict[str, PerClassMetrics] = {}
        lat_t1 = LatencyProfile()
        lat_t2 = LatencyProfile()
        lat_e2e = LatencyProfile()

        # Separate latency profiles for flagged vs non-flagged
        lat_t1_flagged = LatencyProfile()
        lat_t1_passed = LatencyProfile()
        lat_t2_flagged = LatencyProfile()
        lat_t2_passed = LatencyProfile()

        for rec in records:
            gt = rec.ground_truth
            cls_name = gt.name
            if cls_name not in per_class:
                per_class[cls_name] = PerClassMetrics(class_name=cls_name)
            per_class[cls_name].total += 1

            is_true_anomaly = gt in ANOMALY_CLASSES
            if rec.flagged:
                if is_true_anomaly:
                    cm.tp += 1
                else:
                    cm.fp += 1
            else:
                if is_true_anomaly:
                    cm.fn += 1
                else:
                    cm.tn += 1

            lat_t1.record(rec.tier1_latency_ms)
            lat_e2e.record(rec.e2e_latency_ms)

            if rec.tier1_flagged:
                lat_t1_flagged.record(rec.tier1_latency_ms)
            else:
                lat_t1_passed.record(rec.tier1_latency_ms)

            if rec.tier2_latency_ms > 0:
                lat_t2.record(rec.tier2_latency_ms)
                if rec.tier2_flagged:
                    lat_t2_flagged.record(rec.tier2_latency_ms)
                else:
                    lat_t2_passed.record(rec.tier2_latency_ms)

        return SuiteReport(
            suite=TestSuite.S4_LATENCY_PROFILING,
            description="Latency Profiling: per-tier SLA compliance",
            confusion=cm,
            per_class=per_class,
            latency_tier1=lat_t1,
            latency_tier2=lat_t2,
            latency_e2e=lat_e2e,
            n_updates=len(records),
            extra={
                "sla_tier1_target_ms": 5.0,
                "sla_tier2_target_ms": 50.0,
                "tier1_sla_compliance_pct": round(lat_t1.under_5ms_pct, 2),
                "tier2_sla_compliance_pct": round(lat_t2.under_50ms_pct, 2),
                "tier1_flagged_latency": lat_t1_flagged.to_dict(),
                "tier1_passed_latency": lat_t1_passed.to_dict(),
                "tier2_flagged_latency": lat_t2_flagged.to_dict(),
                "tier2_passed_latency": lat_t2_passed.to_dict(),
            },
        )

    # ------------------------------------------------------------------
    # S5: Ablation Study
    # ------------------------------------------------------------------

    def _run_s5(
        self,
        full_records: Sequence[DetectionRecord],
        tier1_only_records: Optional[Sequence[DetectionRecord]] = None,
    ) -> SuiteReport:
        """S5 -- Ablation Study: compare Tier 1-only vs Tier 1 + Tier 2.

        If *tier1_only_records* is None, the ablation is computed by
        ignoring Tier 2 flags in *full_records*.
        """
        # Tier 1+2 metrics
        cm_full = self._compute_cm(full_records, use_combined=True)

        # Tier 1-only metrics
        if tier1_only_records is not None:
            cm_t1only = self._compute_cm(tier1_only_records, use_combined=True)
        else:
            # Simulate T1-only by considering only tier1_flagged
            cm_t1only = self._compute_cm(full_records, use_combined=False)

        per_class: Dict[str, PerClassMetrics] = {}
        lat_t1 = LatencyProfile()
        lat_t2 = LatencyProfile()
        lat_e2e = LatencyProfile()

        for rec in full_records:
            gt = rec.ground_truth
            cls_name = gt.name
            if cls_name not in per_class:
                per_class[cls_name] = PerClassMetrics(class_name=cls_name)
            pc = per_class[cls_name]
            pc.total += 1
            if rec.tier1_flagged:
                pc.detected_tier1 += 1
            if rec.tier2_flagged:
                pc.detected_tier2 += 1
            if rec.flagged:
                pc.detected_combined += 1
            elif gt in ANOMALY_CLASSES:
                pc.missed += 1

            lat_t1.record(rec.tier1_latency_ms)
            if rec.tier2_latency_ms > 0:
                lat_t2.record(rec.tier2_latency_ms)
            lat_e2e.record(rec.e2e_latency_ms)

        # Compute lift from Tier 2
        recall_lift = cm_full.recall - cm_t1only.recall
        precision_delta = cm_full.precision - cm_t1only.precision
        f1_lift = cm_full.f1_score - cm_t1only.f1_score
        fpr_delta = cm_full.fpr - cm_t1only.fpr

        return SuiteReport(
            suite=TestSuite.S5_ABLATION_STUDY,
            description="Ablation Study: Tier 1-only vs Tier 1 + Tier 2",
            confusion=cm_full,
            per_class=per_class,
            latency_tier1=lat_t1,
            latency_tier2=lat_t2,
            latency_e2e=lat_e2e,
            n_updates=len(full_records),
            extra={
                "tier1_only": cm_t1only.to_dict(),
                "tier1_plus_tier2": cm_full.to_dict(),
                "recall_lift": round(recall_lift, 6),
                "precision_delta": round(precision_delta, 6),
                "f1_lift": round(f1_lift, 6),
                "fpr_delta": round(fpr_delta, 6),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cm(
        records: Sequence[DetectionRecord],
        use_combined: bool = True,
    ) -> ConfusionMatrix:
        """Compute a confusion matrix from detection records.

        If *use_combined* is True, ``rec.flagged`` (Tier 1 OR Tier 2)
        is used.  If False, only ``rec.tier1_flagged`` is used
        (simulating Tier 1-only ablation).
        """
        cm = ConfusionMatrix()
        for rec in records:
            flagged = rec.flagged if use_combined else rec.tier1_flagged
            is_true_anomaly = rec.ground_truth in ANOMALY_CLASSES

            if flagged:
                if is_true_anomaly:
                    cm.tp += 1
                else:
                    cm.fp += 1
            else:
                if is_true_anomaly:
                    cm.fn += 1
                else:
                    cm.tn += 1
        return cm

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def reports(self) -> Dict[TestSuite, SuiteReport]:
        return dict(self._reports)

    def summary_table(self) -> str:
        """Return a formatted summary table of all suite results."""
        lines = []
        header = (f"  {'Suite':<6s} {'Prec':>8s} {'Recall':>8s} "
                  f"{'F1':>8s} {'FPR':>8s} {'TP':>6s} {'FP':>6s} "
                  f"{'TN':>6s} {'FN':>6s} {'N':>6s}")
        sep = "  " + "-" * (len(header) - 2)
        lines.append(header)
        lines.append(sep)

        for suite in TestSuite:
            report = self._reports.get(suite)
            if report is None:
                continue
            cm = report.confusion
            lines.append(
                f"  {suite.value:<6s} {cm.precision:>8.4f} {cm.recall:>8.4f} "
                f"{cm.f1_score:>8.4f} {cm.fpr:>8.4f} {cm.tp:>6d} {cm.fp:>6d} "
                f"{cm.tn:>6d} {cm.fn:>6d} {cm.total:>6d}"
            )
        return "\n".join(lines)


# ===================================================================
# CLI Smoke Test -- Full Pipeline
# ===================================================================

def main() -> None:
    """Full pipeline: data -> rules -> features -> ML -> evaluation."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from data_engine import SyntheticBGPFeedGenerator
    from features import FeatureExtractor
    from ml_engine import MLEngine, DEFAULT_THETA_ANOMALY
    from rule_engine import (
        RuleEngine,
        build_default_inventory,
        build_default_vrp_trie,
    )

    print("=" * 74)
    print("  Benchmark Evaluation Engine -- Full Pipeline Smoke Test")
    print("=" * 74)

    # ---- Stage 1: Generate synthetic feed ----
    gen = SyntheticBGPFeedGenerator(
        injection_rate=0.12,
        anomaly_weight=2.0,
        seed=42,
        start_ts=1_700_000_000.0,
        update_interval=0.5,
    )
    updates = gen.generate_batch(count=3000)
    print(f"\n  [1] Generated {len(updates)} synthetic updates")

    # ---- Stage 2: Tier 1 ----
    vrp_trie = build_default_vrp_trie()
    inventory = build_default_inventory()
    rule_engine = RuleEngine(vrp_trie=vrp_trie, inventory=inventory)
    tier1_results = rule_engine.evaluate_batch(updates)
    t1s = rule_engine.stats
    print(f"  [2] Tier 1: {t1s['short_circuited']} short-circuited, "
          f"{t1s['total'] - t1s['short_circuited']} forwarded to Tier 2")

    # ---- Stage 3: Feature extraction ----
    extractor = FeatureExtractor()
    vectors = extractor.process_batch(tier1_results)
    print(f"  [3] Feature vectors: {len(vectors)} emitted")

    # ---- Stage 4: Train & score with Isolation Forest ----
    benign_vectors = [
        v for v in vectors
        if v.update._ground_truth.event_class == EventClass.BENIGN
    ]
    print(f"  [4] Training IsolationForest on {len(benign_vectors)} "
          f"benign vectors ...")

    # Use a lower threshold for the smoke test to exercise the
    # full SHAP -> Evidence Card pathway despite the small dataset.
    ml = MLEngine(
        n_estimators=100,
        contamination=0.05,
        theta_anomaly=0.55,
        random_state=42,
    )
    ml.train(benign_vectors)
    tier2_results = ml.score_batch(vectors)
    mls = ml.stats
    print(f"      Scored: {mls['total_scored']}, "
          f"Flagged: {mls['anomalies_flagged']}, "
          f"Evidence cards: {mls['evidence_cards_emitted']}")

    # ---- Stage 5: Build detection records ----
    records = build_detection_records(tier1_results, tier2_results)
    print(f"  [5] Detection records: {len(records)}")

    # ---- Stage 6: Run all evaluation suites ----
    print(f"  [6] Running evaluation suites S1-S5 ...\n")
    evaluator = Evaluator()
    reports = evaluator.run_all_suites(records)

    # ---- Print summary table ----
    print("  Evaluation Summary")
    print("  " + "=" * 70)
    print(evaluator.summary_table())

    # ---- Detailed per-suite output ----
    for suite in TestSuite:
        report = reports.get(suite)
        if report is None:
            continue
        print(f"\n  {'=' * 70}")
        print(f"  {suite.value}: {report.description}")
        print(f"  {'=' * 70}")

        cm = report.confusion
        print(f"    Confusion Matrix:")
        print(f"      TP={cm.tp}  FP={cm.fp}  TN={cm.tn}  FN={cm.fn}")
        print(f"      Precision : {cm.precision:.4f}")
        print(f"      Recall    : {cm.recall:.4f}")
        print(f"      F1-Score  : {cm.f1_score:.4f}")
        print(f"      FPR       : {cm.fpr:.4f}")

        if report.per_class:
            print(f"    Per-Class Detection Rates:")
            print(f"      {'Class':<22s} {'Total':>6s} {'T1':>5s} "
                  f"{'T2':>5s} {'Both':>5s} {'Miss':>5s} {'Rate':>8s}")
            print(f"      {'-'*22} {'-'*6} {'-'*5} {'-'*5} {'-'*5} "
                  f"{'-'*5} {'-'*8}")
            for cls_name in sorted(report.per_class.keys()):
                pc = report.per_class[cls_name]
                print(f"      {pc.class_name:<22s} {pc.total:>6d} "
                      f"{pc.detected_tier1:>5d} {pc.detected_tier2:>5d} "
                      f"{pc.detected_combined:>5d} {pc.missed:>5d} "
                      f"{pc.detection_rate:>8.4f}")

        if report.latency_tier1.count > 0:
            print(f"    Latency (Tier 1): mean={report.latency_tier1.mean:.4f} ms, "
                  f"P99={report.latency_tier1.p99:.4f} ms, "
                  f"<5ms={report.latency_tier1.under_5ms_pct:.1f}%")
        if report.latency_tier2.count > 0:
            print(f"    Latency (Tier 2): mean={report.latency_tier2.mean:.4f} ms, "
                  f"P99={report.latency_tier2.p99:.4f} ms, "
                  f"<50ms={report.latency_tier2.under_50ms_pct:.1f}%")
        if report.latency_e2e.count > 0:
            print(f"    Latency (E2E)   : mean={report.latency_e2e.mean:.4f} ms, "
                  f"P99={report.latency_e2e.p99:.4f} ms")

        # Suite-specific extras
        if suite == TestSuite.S2_MOAS_STRESS and report.extra:
            ex = report.extra
            print(f"    MOAS specifics:")
            print(f"      LEGIT_MOAS: {ex['moas_total']} total, "
                  f"{ex['moas_false_alarms']} false alarms "
                  f"(rate={ex['moas_false_alarm_rate']:.4f})")
            print(f"      LEGIT_TE  : {ex['te_total']} total, "
                  f"{ex['te_false_alarms']} false alarms "
                  f"(rate={ex['te_false_alarm_rate']:.4f})")

        if suite == TestSuite.S4_LATENCY_PROFILING and report.extra:
            ex = report.extra
            print(f"    SLA Compliance:")
            print(f"      Tier 1 (<5ms) : {ex['tier1_sla_compliance_pct']:.1f}%")
            print(f"      Tier 2 (<50ms): {ex['tier2_sla_compliance_pct']:.1f}%")

        if suite == TestSuite.S5_ABLATION_STUDY and report.extra:
            ex = report.extra
            t1o = ex["tier1_only"]
            t12 = ex["tier1_plus_tier2"]
            print(f"    Ablation Comparison:")
            print(f"      {'Metric':<14s} {'T1-only':>10s} {'T1+T2':>10s} "
                  f"{'Delta':>10s}")
            print(f"      {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
            for metric in ["precision", "recall", "f1_score", "fpr"]:
                v1 = t1o[metric]
                v2 = t12[metric]
                delta = v2 - v1
                print(f"      {metric:<14s} {v1:>10.4f} {v2:>10.4f} "
                      f"{delta:>+10.4f}")

    print("\n" + "=" * 74)
    print("  Evaluation complete.")
    print("=" * 74)


if __name__ == "__main__":
    main()

