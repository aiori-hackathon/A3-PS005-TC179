"""
ml_engine.py -- Tier 2: Isolation Forest Detector & TreeSHAP Explainer
======================================================================
Unsupervised behavioural anomaly detection for BGP updates that pass
Tier 1 without a deterministic violation (i.e., NotFound or
whitelisted-Valid routes forwarded from the rule engine).

Pipeline
--------
1. **Train** a single ``IsolationForest`` on baseline (benign) feature
   vectors produced by ``features.py``.
2. **Score** incoming feature vectors.  The anomaly score *s* is
   normalised to [0, 1] where 1.0 = most anomalous.
3. When *s >= theta_anomaly* (default 0.80), pass the vector to
   **TreeSHAP** (``shap.TreeExplainer``) to extract exact marginal
   feature contributions (Shapley values) in polynomial time.
4. Convert positive SHAP attributions into a **Comparative Evidence
   Card** -- a structured JSON object designed for NOC operators.

References
----------
[Liu 2008]     F.T. Liu et al., "Isolation Forest", IEEE ICDM 2008.
[Lundberg 2020] S.M. Lundberg et al., "From local explanations to
                global understanding with explainable AI for trees",
                Nature Machine Intelligence, vol. 2, 2020.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

# Lazy-import shap to keep module load fast when only training
_shap = None

def _get_shap():
    """Lazy-load the shap library on first use."""
    global _shap
    if _shap is None:
        import shap
        _shap = shap
    return _shap


from data_engine import BGPUpdate, EventClass
from features import FEATURE_NAMES, FeatureVector
from rule_engine import ROVState


# ===================================================================
# Constants
# ===================================================================

#: Default anomaly score threshold.  Vectors scoring at or above this
#: value are flagged and sent to TreeSHAP for explanation.
DEFAULT_THETA_ANOMALY: float = 0.80

#: Default IsolationForest hyper-parameters.
DEFAULT_N_ESTIMATORS: int = 200
DEFAULT_MAX_SAMPLES: int | str = "auto"
DEFAULT_CONTAMINATION: float = 0.01  # expected anomaly fraction
DEFAULT_RANDOM_STATE: int = 42


# ===================================================================
# Data Models
# ===================================================================

@dataclass
class SHAPAttribution:
    """One feature's SHAP contribution to an anomaly score.

    Attributes
    ----------
    feature_name : str
        Human-readable feature name from ``FEATURE_NAMES``.
    feature_index : int
        Positional index in the 8D vector.
    feature_value : float
        The raw feature value for this update.
    shap_value : float
        Marginal Shapley contribution (positive = drives toward anomaly,
        negative = suppresses anomaly).
    direction : str
        "anomaly" if shap_value > 0, "suppressing" otherwise.
    """

    feature_name: str
    feature_index: int
    feature_value: float
    shap_value: float
    direction: str


@dataclass
class EvidenceCard:
    """Comparative Evidence Card for a Tier 2 flagged anomaly.

    Produced when a feature vector's anomaly score exceeds theta_anomaly
    AND TreeSHAP has been run.  Designed to give NOC operators immediate,
    explainable forensic context.

    Attributes
    ----------
    card_id : str
        Unique UUID for this evidence card.
    timestamp : float
        Unix epoch when the card was generated.
    announced_prefix : str
        The NLRI prefix from the BGP UPDATE.
    announced_length : int
        Bit-length of the announced prefix.
    origin_asn : int
        The claimed origin ASN.
    rov_state : str
        The RFC 6811 validation state (from Tier 1).
    anomaly_score : float
        Normalised isolation score in [0, 1].
    theta_anomaly : float
        The threshold that was exceeded.
    attributions : list[SHAPAttribution]
        All 8 SHAP attributions, sorted by absolute value descending.
    top_drivers : list[dict]
        Positive-only attributions (anomaly drivers), serialised.
    top_suppressors : list[dict]
        Negative-only attributions (suppressing features), serialised.
    feature_vector : dict
        The full 8D vector as ``{name: value}``.
    peer_asn : int
        BGP peer that propagated the update.
    collector : str
        RouteViews / RIPE RIS collector name.
    as_path : list[int]
        Full AS_PATH attribute.
    detection_latency_ms : float
        Wall-clock time for scoring + SHAP explanation.
    """

    card_id: str
    timestamp: float
    announced_prefix: str
    announced_length: int
    origin_asn: int
    rov_state: str
    anomaly_score: float
    theta_anomaly: float
    attributions: List[SHAPAttribution]
    top_drivers: List[Dict[str, Any]]
    top_suppressors: List[Dict[str, Any]]
    feature_vector: Dict[str, float]
    peer_asn: int
    collector: str
    as_path: List[int]
    detection_latency_ms: float

    def to_json(self) -> str:
        """Serialise to JSON for SIEM / console ingestion."""
        d = {
            "card_id": self.card_id,
            "timestamp": self.timestamp,
            "announced_prefix": self.announced_prefix,
            "announced_length": self.announced_length,
            "origin_asn": self.origin_asn,
            "rov_state": self.rov_state,
            "anomaly_score": round(self.anomaly_score, 6),
            "theta_anomaly": self.theta_anomaly,
            "top_drivers": self.top_drivers,
            "top_suppressors": self.top_suppressors,
            "feature_vector": {k: round(v, 4) for k, v in
                               self.feature_vector.items()},
            "peer_asn": self.peer_asn,
            "collector": self.collector,
            "as_path": self.as_path,
            "detection_latency_ms": round(self.detection_latency_ms, 4),
        }
        return json.dumps(d, indent=2)


@dataclass
class Tier2Result:
    """Verdict returned by the Tier 2 ML engine for a single vector.

    Attributes
    ----------
    feature_vector : FeatureVector
        The input feature vector.
    anomaly_score : float
        Normalised isolation score [0, 1].
    is_anomaly : bool
        True if anomaly_score >= theta_anomaly.
    evidence_card : EvidenceCard | None
        Comparative Evidence Card (only when is_anomaly is True and
        SHAP has been run).
    processing_time_ms : float
        Wall-clock time for the full Tier 2 evaluation.
    """

    feature_vector: FeatureVector
    anomaly_score: float
    is_anomaly: bool
    evidence_card: Optional[EvidenceCard]
    processing_time_ms: float


# ===================================================================
# Score Normalisation
# ===================================================================

def _normalise_score(raw_score: float, offset: float = 0.0) -> float:
    """Convert sklearn's IsolationForest ``score_samples`` output
    to a [0, 1] anomaly score where 1.0 = most anomalous.

    sklearn's ``score_samples`` returns the raw anomaly score
    (negative of the mean path length normalised by average path
    length in a random tree):
    - More negative -> more anomalous (shorter isolation paths)
    - Less negative / closer to 0 -> more normal

    We apply a shifted sigmoid:
        ``s = 1 / (1 + exp((raw_score - offset) * k))``
    where *offset* is the training median (centring the sigmoid on
    "typical" scores) and k controls steepness.
    """
    k = 10.0  # steepness
    x = np.clip((raw_score - offset) * k, -500, 500)
    return float(1.0 / (1.0 + np.exp(x)))


# ===================================================================
# Tier 2 ML Engine
# ===================================================================

class MLEngine:
    """Tier 2: Isolation Forest anomaly detector with TreeSHAP explainer.

    Lifecycle
    ---------
    1. **``train()``** -- Fit the IsolationForest on baseline feature
       vectors (typically the benign-majority output from the feature
       extractor's first pass).
    2. **``score()``** -- Score a single feature vector, returning a
       ``Tier2Result``.  If the anomaly score exceeds ``theta_anomaly``,
       TreeSHAP is invoked automatically.
    3. **``score_batch()``** -- Score a batch of vectors.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    max_samples : int | str
        Number of samples to draw for each tree.
    contamination : float
        Expected fraction of anomalies in the training data.
    theta_anomaly : float
        Anomaly score threshold in [0, 1].
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_samples: int | str = DEFAULT_MAX_SAMPLES,
        contamination: float = DEFAULT_CONTAMINATION,
        theta_anomaly: float = DEFAULT_THETA_ANOMALY,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.theta_anomaly = theta_anomaly
        self._iforest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._explainer: Any = None  # shap.TreeExplainer (lazy)
        self._is_trained: bool = False
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None

        # Statistics
        self._stats: Dict[str, int] = {
            "total_scored": 0,
            "anomalies_flagged": 0,
            "shap_explanations": 0,
            "evidence_cards_emitted": 0,
        }
        self._evidence_cards: List[EvidenceCard] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, vectors: Sequence[FeatureVector]) -> Dict[str, Any]:
        """Fit the IsolationForest on baseline feature vectors.

        Parameters
        ----------
        vectors : sequence of FeatureVector
            Training data.  Should be predominantly benign baseline
            traffic so the forest learns the normal manifold.

        Returns
        -------
        dict
            Training summary with sample count, feature statistics,
            and forest parameters.
        """
        if not vectors:
            raise ValueError("Cannot train on empty vector set")

        X = np.array([v.to_list() for v in vectors], dtype=np.float64)

        # Store training statistics for optional normalisation
        self._training_mean = X.mean(axis=0)
        self._training_std = X.std(axis=0)
        # Avoid division by zero for constant features
        self._training_std[self._training_std == 0] = 1.0

        self._iforest.fit(X)
        self._is_trained = True

        # Build TreeSHAP explainer
        shap = _get_shap()
        self._explainer = shap.TreeExplainer(
            self._iforest,
            feature_names=FEATURE_NAMES,
        )

        return {
            "n_samples": len(vectors),
            "n_features": X.shape[1],
            "feature_means": {n: round(float(m), 4) for n, m in
                              zip(FEATURE_NAMES, self._training_mean)},
            "feature_stds": {n: round(float(s), 4) for n, s in
                             zip(FEATURE_NAMES, self._training_std)},
            "n_estimators": self._iforest.n_estimators,
            "contamination": self._iforest.contamination,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, vector: FeatureVector) -> Tier2Result:
        """Score a single feature vector and optionally explain.

        1. Compute the raw ``decision_function`` score.
        2. Normalise to [0, 1] via sigmoid.
        3. If score >= theta_anomaly, invoke TreeSHAP.
        4. Build an EvidenceCard from positive SHAP attributions.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError("MLEngine.train() must be called before scoring")

        t_start = time.perf_counter_ns()
        self._stats["total_scored"] += 1

        x = np.array([vector.to_list()], dtype=np.float64)

        # -- Step 1-2: Score --
        raw = float(self._iforest.decision_function(x)[0])
        anomaly_score = _normalise_score(raw)

        is_anomaly = anomaly_score >= self.theta_anomaly
        evidence_card: Optional[EvidenceCard] = None

        if is_anomaly:
            self._stats["anomalies_flagged"] += 1

            # -- Step 3: TreeSHAP --
            shap_values = self._explain(x)
            self._stats["shap_explanations"] += 1

            # -- Step 4: Build Evidence Card --
            processing_ms = (time.perf_counter_ns() - t_start) / 1_000_000
            evidence_card = self._build_evidence_card(
                vector, anomaly_score, shap_values, processing_ms
            )
            self._evidence_cards.append(evidence_card)
            self._stats["evidence_cards_emitted"] += 1

        processing_ms = (time.perf_counter_ns() - t_start) / 1_000_000

        return Tier2Result(
            feature_vector=vector,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            evidence_card=evidence_card,
            processing_time_ms=processing_ms,
        )

    def score_batch(
        self, vectors: Sequence[FeatureVector]
    ) -> List[Tier2Result]:
        """Score a batch of feature vectors.  Returns results in order."""
        return [self.score(v) for v in vectors]

    # ------------------------------------------------------------------
    # SHAP explanation
    # ------------------------------------------------------------------

    def _explain(self, x: np.ndarray) -> np.ndarray:
        """Run TreeSHAP on a single sample and return SHAP values.

        Returns a 1D array of shape (n_features,).
        """
        sv = self._explainer.shap_values(x)
        # shap.TreeExplainer on IsolationForest returns shape (1, n_features)
        if sv.ndim == 2:
            return sv[0]
        return sv

    # ------------------------------------------------------------------
    # Evidence Card construction
    # ------------------------------------------------------------------

    def _build_evidence_card(
        self,
        vector: FeatureVector,
        anomaly_score: float,
        shap_values: np.ndarray,
        latency_ms: float,
    ) -> EvidenceCard:
        """Construct a Comparative Evidence Card from SHAP attributions."""
        update = vector.update

        # Build attribution objects
        attributions: List[SHAPAttribution] = []
        for i, (name, sv) in enumerate(zip(FEATURE_NAMES, shap_values)):
            attr = SHAPAttribution(
                feature_name=name,
                feature_index=i,
                feature_value=float(vector.values[i]),
                shap_value=float(sv),
                direction="anomaly" if sv > 0 else "suppressing",
            )
            attributions.append(attr)

        # Sort by absolute SHAP value descending
        attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)

        # Split drivers vs suppressors
        def _attr_dict(a: SHAPAttribution) -> Dict[str, Any]:
            return {
                "feature": a.feature_name,
                "value": round(a.feature_value, 4),
                "shap_contribution": round(a.shap_value, 6),
                "direction": a.direction,
            }

        top_drivers = [_attr_dict(a) for a in attributions if a.shap_value > 0]
        top_suppressors = [_attr_dict(a) for a in attributions if a.shap_value <= 0]

        return EvidenceCard(
            card_id=uuid.uuid4().hex,
            timestamp=time.time(),
            announced_prefix=update.prefix,
            announced_length=update.prefix_length,
            origin_asn=update.origin_asn,
            rov_state=vector.rov_state.value,
            anomaly_score=anomaly_score,
            theta_anomaly=self.theta_anomaly,
            attributions=attributions,
            top_drivers=top_drivers,
            top_suppressors=top_suppressors,
            feature_vector=vector.to_dict(),
            peer_asn=update.peer_asn,
            collector=update.collector,
            as_path=list(update.as_path),
            detection_latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def evidence_cards(self) -> List[EvidenceCard]:
        return list(self._evidence_cards)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0


# ===================================================================
# CLI Smoke Test
# ===================================================================

def main() -> None:
    """Full pipeline smoke test: data -> rules -> features -> ML."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from data_engine import SyntheticBGPFeedGenerator
    from features import FeatureExtractor
    from rule_engine import (
        RuleEngine,
        build_default_inventory,
        build_default_vrp_trie,
    )

    print("=" * 72)
    print("  Tier 2 ML Engine -- Smoke Test")
    print("=" * 72)

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

    # ---- Stage 2: Tier 1 rule engine ----
    vrp_trie = build_default_vrp_trie()
    inventory = build_default_inventory()
    rule_engine = RuleEngine(vrp_trie=vrp_trie, inventory=inventory)
    tier1_results = rule_engine.evaluate_batch(updates)
    t1_stats = rule_engine.stats
    print(f"  [2] Tier 1: {t1_stats['short_circuited']} short-circuited, "
          f"{t1_stats['total'] - t1_stats['short_circuited']} forwarded")

    # ---- Stage 3: Feature extraction ----
    extractor = FeatureExtractor()
    vectors = extractor.process_batch(tier1_results)
    print(f"  [3] Feature vectors: {len(vectors)} emitted")

    # ---- Stage 4: Train Isolation Forest on baseline vectors ----
    # Use benign vectors for training
    benign_vectors = [
        v for v in vectors
        if v.update._ground_truth.event_class == EventClass.BENIGN
    ]
    print(f"  [4] Training on {len(benign_vectors)} benign vectors ...")

    ml = MLEngine(
        n_estimators=200,
        contamination=0.01,
        theta_anomaly=DEFAULT_THETA_ANOMALY,
        random_state=42,
    )
    train_summary = ml.train(benign_vectors)
    print(f"      Forest: {train_summary['n_estimators']} trees, "
          f"{train_summary['n_samples']} samples, "
          f"{train_summary['n_features']} features")

    # ---- Stage 5: Score ALL vectors ----
    print(f"  [5] Scoring {len(vectors)} vectors ...")
    t_score_start = time.perf_counter_ns()
    tier2_results = ml.score_batch(vectors)
    t_score_ms = (time.perf_counter_ns() - t_score_start) / 1_000_000

    ml_stats = ml.stats
    print(f"      Total scored       : {ml_stats['total_scored']}")
    print(f"      Anomalies flagged  : {ml_stats['anomalies_flagged']}")
    print(f"      Evidence cards     : {ml_stats['evidence_cards_emitted']}")
    print(f"      Scoring latency    : {t_score_ms:.1f} ms total, "
          f"{t_score_ms / len(vectors):.3f} ms/vector")

    # ---- Score distribution ----
    scores = [r.anomaly_score for r in tier2_results]
    import statistics
    print(f"\n  Anomaly Score Distribution")
    print("  " + "-" * 50)
    print(f"    Min    : {min(scores):.4f}")
    print(f"    Mean   : {statistics.mean(scores):.4f}")
    print(f"    Median : {statistics.median(scores):.4f}")
    print(f"    P95    : {sorted(scores)[int(0.95*len(scores))]:.4f}")
    print(f"    P99    : {sorted(scores)[int(0.99*len(scores))]:.4f}")
    print(f"    Max    : {max(scores):.4f}")

    # ---- Ground truth vs detection ----
    print(f"\n  Detection vs Ground Truth")
    print("  " + "-" * 50)
    gt_counts: Dict[str, Dict[str, int]] = {}
    for r in tier2_results:
        gt = r.feature_vector.update._ground_truth.event_class.name
        if gt not in gt_counts:
            gt_counts[gt] = {"total": 0, "flagged": 0}
        gt_counts[gt]["total"] += 1
        if r.is_anomaly:
            gt_counts[gt]["flagged"] += 1

    print(f"    {'GT Class':<22s} {'Total':>6s} {'Flagged':>8s} {'Rate':>8s}")
    print(f"    {'-'*22} {'-'*6} {'-'*8} {'-'*8}")
    for cls in sorted(gt_counts.keys()):
        c = gt_counts[cls]
        rate = c["flagged"] / c["total"] * 100 if c["total"] else 0
        print(f"    {cls:<22s} {c['total']:>6d} {c['flagged']:>8d} "
              f"{rate:>7.1f}%")

    # ---- Sample Evidence Card ----
    cards = ml.evidence_cards
    if cards:
        print(f"\n  Sample Comparative Evidence Card:")
        print("  " + "-" * 68)
        card = cards[0]
        for line in card.to_json().split("\n"):
            print(f"    {line}")

        # Show SHAP attribution breakdown for the first card
        print(f"\n  SHAP Attribution Breakdown (card {card.card_id[:12]}...):")
        print("  " + "-" * 68)
        print(f"    {'Feature':<24s} {'Value':>8s} {'SHAP':>10s} "
              f"{'Direction':<12s}")
        print(f"    {'-'*24} {'-'*8} {'-'*10} {'-'*12}")
        for attr in card.attributions:
            print(f"    {attr.feature_name:<24s} {attr.feature_value:>8.3f} "
                  f"{attr.shap_value:>+10.6f} {attr.direction:<12s}")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
