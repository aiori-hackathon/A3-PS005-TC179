"""
features.py -- Stateful Feature Extraction with Simulated Loc-RIB
=================================================================
Tier 2 pre-processing: maintains a simulated Loc-RIB and a 5-minute
tumbling window over the BGP UPDATE stream.  For every update that
passes Tier 1 without a short-circuit (i.e., Valid or NotFound ROV
states, plus whitelisted Invalid states), this module computes an
8-dimensional numerical feature vector consumed by the Isolation
Forest detector.

Feature Vector (8D)
-------------------
 Index  Name                Description
 -----  ------------------  -------------------------------------------
   0    prefix_length       NLRI bit-length of the announced prefix.
   1    subprefix_degree    How many existing Loc-RIB routes this prefix
                            is a strict subset (more-specific) of, with
                            a *different* origin ASN.
   2    update_frequency    Number of UPDATEs (A+W) for this prefix
                            within the current 5-min window.
   3    origin_stability    Fraction of trailing 30-day history where the
                            *current* origin ASN was the observed origin.
                            1.0 = perfectly stable; 0.0 = never seen.
   4    moas_conflict_count Number of distinct origin ASNs simultaneously
                            announced for this prefix in the Loc-RIB.
   5    rov_state_encoded   Numeric encoding of the RFC 6811 state:
                            Valid = 1, NotFound = 0, Invalid = -1.
   6    prefix_visibility   Count of distinct collector peers that
                            observed this prefix in the current window.
   7    burst_entropy       Shannon entropy of inter-arrival intervals
                            for this prefix within the window.  High
                            entropy = irregular bursts; low = steady.
"""

from __future__ import annotations

import collections
import ipaddress
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from data_engine import BGPUpdate, UpdateType
from rule_engine import ROVState, RuleEngineResult


# ===================================================================
# Constants
# ===================================================================

#: Duration of each tumbling window in seconds.
WINDOW_DURATION: float = 300.0  # 5 minutes

#: Trailing history horizon for origin_stability (seconds).
STABILITY_HORIZON: float = 30 * 24 * 3600.0  # 30 days

#: Number of discrete bins for inter-arrival histogram (burst_entropy).
ENTROPY_BINS: int = 16

#: Ordered names of the 8 features (used for labelling / SHAP).
FEATURE_NAMES: List[str] = [
    "prefix_length",
    "subprefix_degree",
    "update_frequency",
    "origin_stability",
    "moas_conflict_count",
    "rov_state_encoded",
    "prefix_visibility",
    "burst_entropy",
]

#: Mapping from ROVState to the numeric encoding.
_ROV_ENCODING: Dict[ROVState, float] = {
    ROVState.VALID: 1.0,
    ROVState.NOT_FOUND: 0.0,
    ROVState.INVALID_AS: -1.0,
    ROVState.INVALID_ML: -1.0,
    ROVState.INVALID_ML_AS: -1.0,
}


# ===================================================================
# Data Models
# ===================================================================

@dataclass
class FeatureVector:
    """8-dimensional feature vector attached to a single BGP UPDATE.

    Attributes
    ----------
    values : list[float]
        The 8 feature values in the order defined by ``FEATURE_NAMES``.
    update : BGPUpdate
        The originating BGP UPDATE for traceability.
    rov_state : ROVState
        The RFC 6811 state from Tier 1 evaluation.
    window_id : int
        Monotonic identifier of the tumbling window that produced
        this vector.
    """

    values: List[float]
    update: BGPUpdate
    rov_state: ROVState
    window_id: int

    def __repr__(self) -> str:
        abbrev = ", ".join(f"{n}={v:.3f}" for n, v in
                           zip(FEATURE_NAMES, self.values))
        return (f"FeatureVector(prefix={self.update.prefix}, "
                f"origin=AS{self.update.origin_asn}, [{abbrev}])")

    def to_dict(self) -> Dict[str, float]:
        """Return ``{feature_name: value}`` mapping."""
        return dict(zip(FEATURE_NAMES, self.values))

    def to_list(self) -> List[float]:
        """Return a plain list (for NumPy / sklearn input)."""
        return list(self.values)


# ===================================================================
# Simulated Loc-RIB
# ===================================================================

@dataclass
class _LocRIBEntry:
    """One route in the simulated Loc-RIB.

    Keyed by (prefix, peer_asn) to model per-peer Adj-RIB-In before
    the best-path selection.
    """

    prefix: str
    prefix_length: int
    origin_asn: int
    as_path: List[int]
    peer_asn: int
    collector: str
    timestamp: float


class LocRIB:
    """Simulated Local Routing Information Base.

    Maintains the set of currently-active routes as observed in the
    replayed BGP stream.  Routes are keyed by ``(prefix, peer_asn)``
    -- one entry per peer -- mirroring the Adj-RIB-In -> Loc-RIB
    pipeline of a real router.

    The Loc-RIB exposes analytical queries used by the feature
    extractor:

    * **subprefix_degree** -- count of covering (less-specific) routes
      with a *different* origin ASN.
    * **moas_conflict_count** -- count of distinct origin ASNs
      simultaneously announced for a given prefix across all peers.
    """

    def __init__(self) -> None:
        # (prefix_str, peer_asn) -> _LocRIBEntry
        self._routes: Dict[Tuple[str, int], _LocRIBEntry] = {}

        # prefix_str -> set of peer_asns that have an active route
        self._prefix_peers: Dict[str, Set[int]] = collections.defaultdict(set)

        # Reverse index: prefix_length -> set of prefix_str
        # for efficient covering-prefix search.
        self._by_length: Dict[int, Set[str]] = collections.defaultdict(set)

    # ------------------------------------------------------------------
    # Route mutations
    # ------------------------------------------------------------------

    def process_update(self, update: BGPUpdate) -> None:
        """Apply a BGP UPDATE to the Loc-RIB.

        ANNOUNCE: insert or replace the route for (prefix, peer_asn).
        WITHDRAW: remove the route for (prefix, peer_asn).
        """
        net = ipaddress.IPv4Network(update.prefix, strict=False)
        prefix_str = str(net)
        key = (prefix_str, update.peer_asn)

        if update.update_type == UpdateType.ANNOUNCE:
            entry = _LocRIBEntry(
                prefix=prefix_str,
                prefix_length=update.prefix_length,
                origin_asn=update.origin_asn,
                as_path=list(update.as_path),
                peer_asn=update.peer_asn,
                collector=update.collector,
                timestamp=update.timestamp,
            )
            self._routes[key] = entry
            self._prefix_peers[prefix_str].add(update.peer_asn)
            self._by_length[update.prefix_length].add(prefix_str)
        elif update.update_type == UpdateType.WITHDRAW:
            if key in self._routes:
                del self._routes[key]
                self._prefix_peers[prefix_str].discard(update.peer_asn)
                if not self._prefix_peers[prefix_str]:
                    del self._prefix_peers[prefix_str]
                    self._by_length[update.prefix_length].discard(prefix_str)

    # ------------------------------------------------------------------
    # Analytical queries
    # ------------------------------------------------------------------

    def subprefix_degree(self, prefix: str, origin_asn: int) -> int:
        """Count covering (less-specific) Loc-RIB routes with a
        *different* origin ASN.

        A non-zero value means this prefix is a strict subset of an
        existing route originated by another AS -- the hallmark of a
        subprefix hijack.
        """
        query_net = ipaddress.IPv4Network(prefix, strict=False)
        query_len = query_net.prefixlen
        count = 0

        for plen in sorted(self._by_length.keys()):
            if plen >= query_len:
                break  # only want strictly less-specific
            for candidate_prefix in self._by_length[plen]:
                candidate_net = ipaddress.IPv4Network(
                    candidate_prefix, strict=False
                )
                if query_net.subnet_of(candidate_net):
                    # Check if any peer has a different origin
                    for peer in self._prefix_peers.get(candidate_prefix, set()):
                        entry = self._routes.get((candidate_prefix, peer))
                        if entry and entry.origin_asn != origin_asn:
                            count += 1
                            break  # count each covering prefix once

        return count

    def moas_conflict_count(self, prefix: str) -> int:
        """Count distinct origin ASNs for *prefix* across all peers."""
        net = ipaddress.IPv4Network(prefix, strict=False)
        prefix_str = str(net)
        origins: Set[int] = set()
        for peer in self._prefix_peers.get(prefix_str, set()):
            entry = self._routes.get((prefix_str, peer))
            if entry:
                origins.add(entry.origin_asn)
        return len(origins)

    @property
    def route_count(self) -> int:
        return len(self._routes)

    @property
    def prefix_count(self) -> int:
        return len(self._prefix_peers)


# ===================================================================
# Origin Stability Tracker
# ===================================================================

class OriginStabilityTracker:
    """Tracks the historical origin ASN for each prefix over a
    trailing window (default 30 days) to compute stability scores.

    For each prefix, maintains a bounded deque of
    ``(timestamp, origin_asn)`` observations.  The stability score
    for a (prefix, origin_asn) pair is the fraction of observations
    where that ASN was the origin.
    """

    def __init__(self, horizon: float = STABILITY_HORIZON, max_samples: int = 10_000) -> None:
        self._horizon = horizon
        self._max_samples = max_samples
        # prefix_str -> deque of (timestamp, origin_asn)
        self._history: Dict[str, Deque[Tuple[float, int]]] = (
            collections.defaultdict(lambda: collections.deque(maxlen=max_samples))
        )

    def record(self, prefix: str, origin_asn: int, timestamp: float) -> None:
        """Record an observation of (prefix, origin_asn) at *timestamp*."""
        net = ipaddress.IPv4Network(prefix, strict=False)
        self._history[str(net)].append((timestamp, origin_asn))

    def stability(self, prefix: str, origin_asn: int, now: float) -> float:
        """Return the stability score in [0.0, 1.0].

        Computed as the fraction of observations within the trailing
        ``_horizon`` where ``origin_asn`` was the recorded origin.
        Returns 0.0 if there is no history at all (brand-new prefix).
        """
        net = ipaddress.IPv4Network(prefix, strict=False)
        prefix_str = str(net)
        hist = self._history.get(prefix_str)
        if not hist:
            return 0.0

        cutoff = now - self._horizon
        total = 0
        match = 0
        for ts, asn in hist:
            if ts >= cutoff:
                total += 1
                if asn == origin_asn:
                    match += 1

        return match / total if total > 0 else 0.0


# ===================================================================
# Tumbling Window Buffer
# ===================================================================

@dataclass
class _WindowState:
    """Accumulated state for one 5-minute tumbling window."""

    window_id: int
    window_start: float
    window_end: float

    # Per-prefix counters: prefix_str -> list of timestamps
    update_timestamps: Dict[str, List[float]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )

    # Per-prefix peer visibility: prefix_str -> set of (collector, peer_asn)
    visibility: Dict[str, Set[Tuple[str, int]]] = field(
        default_factory=lambda: collections.defaultdict(set)
    )


# ===================================================================
# Shannon Entropy Helper
# ===================================================================

def _shannon_entropy(intervals: List[float], n_bins: int = ENTROPY_BINS) -> float:
    """Compute Shannon entropy of inter-arrival intervals.

    The intervals are binned into ``n_bins`` equal-width buckets
    spanning [0, max_interval].  The entropy is normalised to [0, 1]
    by dividing by log2(n_bins).

    Returns 0.0 if there are fewer than 2 intervals (no meaningful
    distribution).
    """
    if len(intervals) < 2:
        return 0.0

    min_val = 0.0
    max_val = max(intervals)
    if max_val <= min_val:
        return 0.0  # all intervals identical

    bin_width = (max_val - min_val) / n_bins
    counts = [0] * n_bins
    for v in intervals:
        idx = int((v - min_val) / bin_width)
        idx = min(idx, n_bins - 1)  # clamp last edge
        counts[idx] += 1

    total = len(intervals)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(n_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _inter_arrival_intervals(timestamps: List[float]) -> List[float]:
    """Compute pairwise inter-arrival intervals from sorted timestamps."""
    if len(timestamps) < 2:
        return []
    ts_sorted = sorted(timestamps)
    return [ts_sorted[i+1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]


# ===================================================================
# Stateful Feature Extractor
# ===================================================================

class FeatureExtractor:
    """5-minute tumbling window feature extractor for Tier 2.

    Maintains a simulated Loc-RIB, an origin-stability tracker, and
    a per-window accumulator.  For each ``RuleEngineResult`` that was
    NOT short-circuited by Tier 1, produces an 8D ``FeatureVector``.

    Usage
    -----
    >>> extractor = FeatureExtractor()
    >>> for result in tier1_results:
    ...     vec = extractor.process(result)
    ...     if vec is not None:
    ...         tier2_input.append(vec)

    Parameters
    ----------
    window_duration : float
        Width of each tumbling window in seconds.  Default 300 (5 min).
    stability_horizon : float
        Trailing observation horizon for origin_stability.  Default 30 days.
    """

    def __init__(
        self,
        window_duration: float = WINDOW_DURATION,
        stability_horizon: float = STABILITY_HORIZON,
    ) -> None:
        self.window_duration = window_duration

        # -- Persistent state (survives window resets) --
        self.loc_rib = LocRIB()
        self.stability_tracker = OriginStabilityTracker(
            horizon=stability_horizon
        )

        # -- Per-window state --
        self._window: Optional[_WindowState] = None
        self._window_counter: int = 0

        # -- Statistics --
        self._stats: Dict[str, int] = {
            "total_processed": 0,
            "short_circuited_skipped": 0,
            "vectors_emitted": 0,
            "windows_completed": 0,
        }

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _ensure_window(self, timestamp: float) -> _WindowState:
        """Return the current window, advancing if necessary.

        If *timestamp* falls outside the current window, the window is
        closed and a new one is opened.  This implements the tumbling
        (non-overlapping) window semantics.
        """
        if self._window is None:
            # First update -- initialise the first window
            w_start = timestamp
            w_end = w_start + self.window_duration
            self._window = _WindowState(
                window_id=self._window_counter,
                window_start=w_start,
                window_end=w_end,
            )
            return self._window

        if timestamp >= self._window.window_end:
            # Current window expired -- advance
            self._stats["windows_completed"] += 1
            self._window_counter += 1
            # Align new window to grid (handles gaps in the stream)
            elapsed = timestamp - self._window.window_start
            windows_skipped = int(elapsed / self.window_duration)
            new_start = self._window.window_start + windows_skipped * self.window_duration
            self._window = _WindowState(
                window_id=self._window_counter,
                window_start=new_start,
                window_end=new_start + self.window_duration,
            )

        return self._window

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, result: RuleEngineResult) -> Optional[FeatureVector]:
        """Process a Tier 1 result and return a FeatureVector if applicable.

        Returns ``None`` for short-circuited updates (Tier 1 already
        handled them with a Forensic Ledger Alert).

        For all other updates:
        1. Update the Loc-RIB with the announcement/withdrawal.
        2. Record the observation in the stability tracker.
        3. Accumulate window-level counters.
        4. Compute and return the 8D feature vector.
        """
        self._stats["total_processed"] += 1
        update = result.update

        if result.short_circuit:
            # Tier 1 already emitted an alert -- skip Tier 2
            self._stats["short_circuited_skipped"] += 1
            # Still update Loc-RIB so it stays consistent
            self.loc_rib.process_update(update)
            return None

        # -- 1. Update Loc-RIB --
        self.loc_rib.process_update(update)

        # -- 2. Record origin observation --
        self.stability_tracker.record(
            update.prefix, update.origin_asn, update.timestamp
        )

        # -- 3. Advance / initialise window --
        window = self._ensure_window(update.timestamp)

        # Accumulate per-prefix timestamps
        net = ipaddress.IPv4Network(update.prefix, strict=False)
        prefix_str = str(net)
        window.update_timestamps[prefix_str].append(update.timestamp)
        window.visibility[prefix_str].add(
            (update.collector, update.peer_asn)
        )

        # -- 4. Compute 8D feature vector --
        vec = self._compute_vector(update, result.rov_state, window, prefix_str)

        self._stats["vectors_emitted"] += 1
        return vec

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_vector(
        self,
        update: BGPUpdate,
        rov_state: ROVState,
        window: _WindowState,
        prefix_str: str,
    ) -> FeatureVector:
        """Compute the 8D feature vector for a single update."""

        # F0: prefix_length -- direct from the update NLRI
        f_prefix_length = float(update.prefix_length)

        # F1: subprefix_degree -- covering routes with different origin
        f_subprefix_degree = float(
            self.loc_rib.subprefix_degree(prefix_str, update.origin_asn)
        )

        # F2: update_frequency -- count of updates for this prefix in window
        f_update_frequency = float(
            len(window.update_timestamps.get(prefix_str, []))
        )

        # F3: origin_stability -- trailing 30-day fraction
        f_origin_stability = self.stability_tracker.stability(
            prefix_str, update.origin_asn, update.timestamp
        )

        # F4: moas_conflict_count -- distinct origins in Loc-RIB
        f_moas_conflict = float(
            self.loc_rib.moas_conflict_count(prefix_str)
        )

        # F5: rov_state_encoded -- Valid=1, NotFound=0, Invalid=-1
        f_rov_encoded = _ROV_ENCODING.get(rov_state, 0.0)

        # F6: prefix_visibility -- distinct (collector, peer) pairs
        f_visibility = float(
            len(window.visibility.get(prefix_str, set()))
        )

        # F7: burst_entropy -- Shannon entropy of inter-arrival intervals
        timestamps = window.update_timestamps.get(prefix_str, [])
        intervals = _inter_arrival_intervals(timestamps)
        f_burst_entropy = _shannon_entropy(intervals)

        values = [
            f_prefix_length,       # 0
            f_subprefix_degree,    # 1
            f_update_frequency,    # 2
            f_origin_stability,    # 3
            f_moas_conflict,       # 4
            f_rov_encoded,         # 5
            f_visibility,          # 6
            f_burst_entropy,       # 7
        ]

        return FeatureVector(
            values=values,
            update=update,
            rov_state=rov_state,
            window_id=window.window_id,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self, results: Sequence[RuleEngineResult]
    ) -> List[FeatureVector]:
        """Process a batch of Tier 1 results, returning feature vectors
        for non-short-circuited updates."""
        vectors = []
        for r in results:
            v = self.process(r)
            if v is not None:
                vectors.append(v)
        return vectors

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def current_window_id(self) -> int:
        return self._window.window_id if self._window else -1


# ===================================================================
# CLI Smoke Test
# ===================================================================

def main() -> None:
    """End-to-end smoke test: data_engine -> rule_engine -> features."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from data_engine import EventClass, SyntheticBGPFeedGenerator
    from rule_engine import (
        RuleEngine,
        build_default_inventory,
        build_default_vrp_trie,
    )

    print("=" * 72)
    print("  Feature Extractor -- Smoke Test")
    print("=" * 72)

    # -- Build Tier 1 --
    vrp_trie = build_default_vrp_trie()
    inventory = build_default_inventory()
    engine = RuleEngine(vrp_trie=vrp_trie, inventory=inventory)

    # -- Generate synthetic feed --
    gen = SyntheticBGPFeedGenerator(
        injection_rate=0.15,
        anomaly_weight=2.0,
        seed=42,
        start_ts=1_700_000_000.0,
        update_interval=0.5,
    )
    updates = gen.generate_batch(count=2000)
    print(f"\n  Generated {len(updates)} updates")

    # -- Run Tier 1 --
    tier1_results = engine.evaluate_batch(updates)
    print(f"  Tier 1 processed: {engine.stats['total']} updates")
    print(f"  Tier 1 short-circuited: {engine.stats['short_circuited']}")

    # -- Run Feature Extractor --
    extractor = FeatureExtractor(
        window_duration=WINDOW_DURATION,
    )
    vectors = extractor.process_batch(tier1_results)

    print(f"\n  Feature vectors emitted : {len(vectors)}")
    print(f"  Loc-RIB routes          : {extractor.loc_rib.route_count}")
    print(f"  Loc-RIB prefixes        : {extractor.loc_rib.prefix_count}")
    print(f"  Windows completed       : {extractor.stats['windows_completed']}")

    # -- Feature statistics --
    if vectors:
        import statistics

        print(f"\n  Feature Statistics (across {len(vectors)} vectors)")
        print("  " + "-" * 68)
        print(f"    {'Feature':<24s} {'Min':>8s} {'Mean':>8s} "
              f"{'Max':>8s} {'Stdev':>8s}")
        print(f"    {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for i, name in enumerate(FEATURE_NAMES):
            vals = [v.values[i] for v in vectors]
            mn = min(vals)
            mx = max(vals)
            avg = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"    {name:<24s} {mn:>8.3f} {avg:>8.3f} "
                  f"{mx:>8.3f} {sd:>8.3f}")

    # -- Show sample vectors by ground-truth class --
    print(f"\n  Sample Feature Vectors by Ground-Truth Class")
    print("  " + "-" * 68)

    by_class: Dict[str, List[FeatureVector]] = collections.defaultdict(list)
    for v in vectors:
        gt = v.update._ground_truth.event_class.name
        by_class[gt].append(v)

    for cls_name in sorted(by_class.keys()):
        sample = by_class[cls_name][0]
        fdict = sample.to_dict()
        print(f"\n    [{cls_name}]  prefix={sample.update.prefix}  "
              f"origin=AS{sample.update.origin_asn}  "
              f"ROV={sample.rov_state.value}")
        for fname, fval in fdict.items():
            bar_len = int(min(fval, 30))
            bar = "#" * max(bar_len, 0)
            print(f"      {fname:<24s} {fval:>8.3f}  {bar}")

    # -- Verify NotFound updates get vectors --
    notfound_vectors = [v for v in vectors
                        if v.rov_state == ROVState.NOT_FOUND]
    print(f"\n  NotFound vectors (Tier 2 candidates): {len(notfound_vectors)}")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()

