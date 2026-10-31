"""
data_engine.py – Synthetic BGP Feed Generator & Event Injector
==============================================================
Generates a realistic stream of BGPUpdate objects with programmatic
injection of 3 anomaly classes and 2 legitimate scenarios.  Every
update carries an internal, detector-blind ground-truth tag used
exclusively by the downstream Benchmark Evaluation Engine.

Event Classes
-------------
Anomaly:
    HIJACK_ORIGIN   – exact-prefix origin hijack by a rogue ASN.
    HIJACK_SUBPREFIX – de-aggregated /24 subset from a hostile ASN.
    HIJACK_MAXLEN   – legitimate ASN exceeds ROA maxLength.

Legitimate:
    LEGIT_MOAS – whitelisted multi-origin Anycast / CDN route.
    LEGIT_TE   – traffic-engineering split within ROA bounds.
"""

from __future__ import annotations

import hashlib
import ipaddress
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HMAC key used to sign the hidden ground-truth tag so that downstream
#: detectors cannot forge or reverse-engineer the label from the update
#: payload alone.  In production this would be read from a secure vault.
_GT_HMAC_KEY: bytes = b"aiori-v3-hackathon-gt-key-tc179"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventClass(Enum):
    """Enumeration of the five synthetic event classes."""

    # Anomaly classes
    HIJACK_ORIGIN = auto()
    HIJACK_SUBPREFIX = auto()
    HIJACK_MAXLEN = auto()

    # Legitimate classes
    LEGIT_MOAS = auto()
    LEGIT_TE = auto()

    # Baseline (unmodified) updates from the seed topology
    BENIGN = auto()


class UpdateType(Enum):
    """BGP UPDATE message type."""

    ANNOUNCE = "A"
    WITHDRAW = "W"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class GroundTruthTag:
    """Hidden, detector-blind metadata attached to every BGPUpdate.

    Attributes
    ----------
    event_class : EventClass
        The true label of the update (e.g., HIJACK_ORIGIN).
    injection_id : str
        A unique identifier tying an injected event to its injection
        campaign, enabling multi-update correlation.
    hmac_digest : str
        HMAC-SHA256 digest of ``event_class.name + injection_id`` keyed
        with ``_GT_HMAC_KEY``.  The Benchmark Evaluation Engine uses
        this to verify tag integrity.
    metadata : dict
        Arbitrary key-value pairs describing injection parameters
        (e.g., the rogue ASN, the legitimate owner, the parent prefix).
    """

    event_class: EventClass
    injection_id: str
    hmac_digest: str
    metadata: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        event_class: EventClass,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "GroundTruthTag":
        injection_id = uuid.uuid4().hex
        digest = _compute_hmac(event_class.name, injection_id)
        return cls(
            event_class=event_class,
            injection_id=injection_id,
            hmac_digest=digest,
            metadata=metadata or {},
        )

    def verify(self) -> bool:
        """Return True if the HMAC digest is consistent."""
        return self.hmac_digest == _compute_hmac(
            self.event_class.name, self.injection_id
        )


@dataclass
class BGPUpdate:
    """Representation of a single BGP UPDATE message.

    Attributes
    ----------
    timestamp : float
        Unix epoch seconds when the update was observed / generated.
    update_type : UpdateType
        ``ANNOUNCE`` or ``WITHDRAW``.
    prefix : str
        NLRI prefix in CIDR notation (e.g., ``"203.0.113.0/24"``).
    prefix_length : int
        Bit-length of the announced prefix.
    origin_asn : int
        Rightmost ASN in the AS_PATH (the origin).
    as_path : List[int]
        Full AS_PATH attribute (left = nearest, right = origin).
    peer_asn : int
        ASN of the BGP peer / collector that received the update.
    collector : str
        Identifier of the RouteViews / RIPE RIS collector.
    communities : List[str]
        BGP community strings attached to the update.
    nexthop : str
        NEXT_HOP IP address.
    _ground_truth : GroundTruthTag
        **Internal, hidden** ground-truth label.  This attribute is
        name-mangled (leading underscore) and must *never* be read by
        any detection component.  Only the Benchmark Evaluation Engine
        is permitted to access it.
    """

    timestamp: float
    update_type: UpdateType
    prefix: str
    prefix_length: int
    origin_asn: int
    as_path: List[int]
    peer_asn: int
    collector: str
    communities: List[str]
    nexthop: str
    _ground_truth: GroundTruthTag

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(self.prefix, strict=False)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BGPUpdate(ts={self.timestamp:.0f}, "
            f"type={self.update_type.value}, "
            f"prefix={self.prefix}, "
            f"origin=AS{self.origin_asn}, "
            f"path={self.as_path})"
        )


# ---------------------------------------------------------------------------
# VRP / ROA stub for the generator's internal bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class VRPRecord:
    """Validated ROA Payload record (simplified).

    Used by the generator to know which prefixes are RPKI-covered so
    that injections can be crafted realistically.
    """

    prefix: str
    max_length: int
    authorized_asn: int


# ---------------------------------------------------------------------------
# Seed Topology & Prefix Catalogue
# ---------------------------------------------------------------------------

@dataclass
class PrefixEntry:
    """One entry in the simulated routing table seed."""

    prefix: str
    prefix_length: int
    owner_asn: int
    vrp: Optional[VRPRecord]
    # Additional ASNs that may legitimately originate this prefix
    # (Anycast / CDN peers, TE partners).
    secondary_asns: List[int] = field(default_factory=list)
    # Maximum allowed de-aggregation depth for TE splits.
    max_deagg: int = 0


def _build_default_topology() -> List[PrefixEntry]:
    """Return a realistic seed topology of 20 prefix entries.

    The topology is designed to exercise every code-path in the
    dual-tier detection pipeline:

    * Entries **with** VRP coverage (Valid / Invalid test targets).
    * Entries **without** VRP coverage (NotFound – Tier 2 targets).
    * Entries with secondary ASNs (MOAS whitelist targets).
    * Entries with non-zero ``max_deagg`` (TE split targets).
    """

    return [
        # ---- RPKI-covered prefixes (Valid under normal operation) ----
        PrefixEntry(
            prefix="203.0.113.0/22",
            prefix_length=22,
            owner_asn=64500,
            vrp=VRPRecord("203.0.113.0/22", max_length=22, authorized_asn=64500),
        ),
        PrefixEntry(
            prefix="198.51.100.0/24",
            prefix_length=24,
            owner_asn=64501,
            vrp=VRPRecord("198.51.100.0/24", max_length=24, authorized_asn=64501),
        ),
        PrefixEntry(
            prefix="192.0.2.0/24",
            prefix_length=24,
            owner_asn=64502,
            vrp=VRPRecord("192.0.2.0/24", max_length=24, authorized_asn=64502),
        ),
        PrefixEntry(
            prefix="100.64.0.0/16",
            prefix_length=16,
            owner_asn=64503,
            vrp=VRPRecord("100.64.0.0/16", max_length=20, authorized_asn=64503),
            secondary_asns=[64550, 64551],
            max_deagg=4,
        ),
        PrefixEntry(
            prefix="10.0.0.0/8",
            prefix_length=8,
            owner_asn=64504,
            vrp=VRPRecord("10.0.0.0/8", max_length=16, authorized_asn=64504),
            max_deagg=8,
        ),
        PrefixEntry(
            prefix="172.16.0.0/12",
            prefix_length=12,
            owner_asn=64505,
            vrp=VRPRecord("172.16.0.0/12", max_length=16, authorized_asn=64505),
        ),
        PrefixEntry(
            prefix="44.0.0.0/16",
            prefix_length=16,
            owner_asn=64520,
            vrp=VRPRecord("44.0.0.0/16", max_length=18, authorized_asn=64520),
            secondary_asns=[64521],
        ),
        # ---- Anycast / CDN entries with MOAS whitelist ----
        PrefixEntry(
            prefix="185.0.0.0/22",
            prefix_length=22,
            owner_asn=13335,
            vrp=VRPRecord("185.0.0.0/22", max_length=24, authorized_asn=13335),
            secondary_asns=[13336, 13337],
        ),
        PrefixEntry(
            prefix="104.16.0.0/20",
            prefix_length=20,
            owner_asn=13335,
            vrp=VRPRecord("104.16.0.0/20", max_length=24, authorized_asn=13335),
            secondary_asns=[13336],
        ),
        # ---- Prefixes WITHOUT RPKI coverage (NotFound) ----
        PrefixEntry(
            prefix="45.33.32.0/24",
            prefix_length=24,
            owner_asn=63949,
            vrp=None,
        ),
        PrefixEntry(
            prefix="91.189.88.0/21",
            prefix_length=21,
            owner_asn=41231,
            vrp=None,
        ),
        PrefixEntry(
            prefix="151.101.0.0/16",
            prefix_length=16,
            owner_asn=54113,
            vrp=None,
            secondary_asns=[54114],
        ),
        PrefixEntry(
            prefix="140.82.112.0/20",
            prefix_length=20,
            owner_asn=36459,
            vrp=None,
        ),
        PrefixEntry(
            prefix="31.13.64.0/18",
            prefix_length=18,
            owner_asn=32934,
            vrp=None,
            secondary_asns=[32935],
        ),
        PrefixEntry(
            prefix="52.84.0.0/15",
            prefix_length=15,
            owner_asn=16509,
            vrp=None,
        ),
        # ---- Extra covered entries for variety ----
        PrefixEntry(
            prefix="23.0.0.0/12",
            prefix_length=12,
            owner_asn=15169,
            vrp=VRPRecord("23.0.0.0/12", max_length=16, authorized_asn=15169),
            secondary_asns=[15170],
            max_deagg=4,
        ),
        PrefixEntry(
            prefix="35.192.0.0/14",
            prefix_length=14,
            owner_asn=15169,
            vrp=VRPRecord("35.192.0.0/14", max_length=18, authorized_asn=15169),
        ),
        PrefixEntry(
            prefix="64.233.160.0/19",
            prefix_length=19,
            owner_asn=15169,
            vrp=VRPRecord("64.233.160.0/19", max_length=24, authorized_asn=15169),
            max_deagg=5,
        ),
        PrefixEntry(
            prefix="208.80.152.0/22",
            prefix_length=22,
            owner_asn=14907,
            vrp=VRPRecord("208.80.152.0/22", max_length=24, authorized_asn=14907),
        ),
        PrefixEntry(
            prefix="199.232.0.0/16",
            prefix_length=16,
            owner_asn=54113,
            vrp=None,
            secondary_asns=[54114, 54115],
            max_deagg=8,
        ),
    ]


# ---------------------------------------------------------------------------
# Rogue ASN Pool
# ---------------------------------------------------------------------------

_ROGUE_ASNS: List[int] = [
    64512, 64513, 64514, 64515, 64516,
    64600, 64601, 64602, 64603, 64604,
    65000, 65001, 65002, 65003, 65004,
]


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _compute_hmac(event_name: str, injection_id: str) -> str:
    """Compute HMAC-SHA256 for ground-truth tag integrity."""
    msg = f"{event_name}:{injection_id}".encode()
    return hashlib.sha256(_GT_HMAC_KEY + msg).hexdigest()


def _random_as_path(origin: int, depth_range: Tuple[int, int] = (2, 6)) -> List[int]:
    """Synthesize a plausible AS_PATH terminating at *origin*.

    The path is built right-to-left: ``[transit_n, ..., transit_1, origin]``.
    """
    depth = random.randint(*depth_range)
    transit_asns = random.sample(range(1000, 60000), k=depth - 1)
    return transit_asns + [origin]


def _random_collector() -> str:
    collectors = [
        "route-views2", "route-views.linx", "route-views.wide",
        "rrc00", "rrc01", "rrc03", "rrc04", "rrc10", "rrc11",
        "rrc12", "rrc13", "rrc14", "rrc15", "rrc21", "rrc23",
    ]
    return random.choice(collectors)


def _random_peer_asn() -> int:
    return random.choice([
        3356, 174, 1299, 6939, 2914, 6461, 3257, 7018, 701, 6762,
    ])


def _random_nexthop() -> str:
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _deaggregate(prefix_str: str, new_length: int) -> str:
    """Return the first subnet of *prefix_str* with *new_length*.

    Example: ``_deaggregate("203.0.113.0/22", 24)`` → ``"203.0.113.0/24"``
    """
    net = ipaddress.IPv4Network(prefix_str, strict=False)
    subnets = list(net.subnets(new_prefix=new_length))
    return str(random.choice(subnets))


# ---------------------------------------------------------------------------
# Individual Injection Factories
# ---------------------------------------------------------------------------

def _make_benign(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """Create a baseline (unmodified) announcement."""
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=entry.prefix,
        prefix_length=entry.prefix_length,
        origin_asn=entry.owner_asn,
        as_path=_random_as_path(entry.owner_asn),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=[],
        nexthop=_random_nexthop(),
        _ground_truth=GroundTruthTag.create(EventClass.BENIGN),
    )


def _inject_hijack_origin(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """HIJACK_ORIGIN – Rogue ASN claims the exact same prefix."""
    rogue = random.choice(_ROGUE_ASNS)
    tag = GroundTruthTag.create(
        EventClass.HIJACK_ORIGIN,
        metadata={
            "legitimate_owner": entry.owner_asn,
            "rogue_asn": rogue,
            "target_prefix": entry.prefix,
        },
    )
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=entry.prefix,
        prefix_length=entry.prefix_length,
        origin_asn=rogue,
        as_path=_random_as_path(rogue, depth_range=(2, 4)),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=[],
        nexthop=_random_nexthop(),
        _ground_truth=tag,
    )


def _inject_hijack_subprefix(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """HIJACK_SUBPREFIX – Rogue ASN announces a more-specific subset."""
    rogue = random.choice(_ROGUE_ASNS)
    parent_net = ipaddress.IPv4Network(entry.prefix, strict=False)
    # Pick a subprefix that is 2-4 bits more specific than the parent,
    # clamped to /28 to keep it realistic.
    new_len = min(entry.prefix_length + random.randint(2, 4), 28)
    if new_len <= entry.prefix_length:
        new_len = entry.prefix_length + 1
    sub_prefix = _deaggregate(entry.prefix, new_len)
    tag = GroundTruthTag.create(
        EventClass.HIJACK_SUBPREFIX,
        metadata={
            "legitimate_owner": entry.owner_asn,
            "rogue_asn": rogue,
            "parent_prefix": entry.prefix,
            "hijacked_subprefix": sub_prefix,
        },
    )
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=sub_prefix,
        prefix_length=new_len,
        origin_asn=rogue,
        as_path=_random_as_path(rogue, depth_range=(2, 3)),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=[],
        nexthop=_random_nexthop(),
        _ground_truth=tag,
    )


def _inject_hijack_maxlen(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """HIJACK_MAXLEN – Legitimate ASN exceeds the ROA maxLength.

    Only applicable when the entry has a VRP with a finite maxLength
    that leaves room for a violation (prefix_length < 32 and
    vrp.max_length < 32).
    """
    assert entry.vrp is not None, "HIJACK_MAXLEN requires VRP coverage"
    # Announce a prefix 1–3 bits longer than the allowed maxLength
    violation_len = entry.vrp.max_length + random.randint(1, 3)
    violation_len = min(violation_len, 28)
    if violation_len <= entry.vrp.max_length:
        violation_len = entry.vrp.max_length + 1
    sub_prefix = _deaggregate(entry.prefix, violation_len)
    tag = GroundTruthTag.create(
        EventClass.HIJACK_MAXLEN,
        metadata={
            "authorized_asn": entry.vrp.authorized_asn,
            "parent_prefix": entry.prefix,
            "violated_prefix": sub_prefix,
            "roa_max_length": entry.vrp.max_length,
            "announced_length": violation_len,
        },
    )
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=sub_prefix,
        prefix_length=violation_len,
        origin_asn=entry.vrp.authorized_asn,  # legit ASN, bad length
        as_path=_random_as_path(entry.vrp.authorized_asn),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=[],
        nexthop=_random_nexthop(),
        _ground_truth=tag,
    )


def _inject_legit_moas(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """LEGIT_MOAS – Whitelisted secondary ASN originates the prefix.

    Simulates Anycast DNS / CDN multi-origin announcements.
    """
    secondary = random.choice(entry.secondary_asns)
    tag = GroundTruthTag.create(
        EventClass.LEGIT_MOAS,
        metadata={
            "primary_asn": entry.owner_asn,
            "secondary_asn": secondary,
            "prefix": entry.prefix,
            "reason": "anycast_cdn",
        },
    )
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=entry.prefix,
        prefix_length=entry.prefix_length,
        origin_asn=secondary,
        as_path=_random_as_path(secondary),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=["13335:10000"],  # Anycast community tag
        nexthop=_random_nexthop(),
        _ground_truth=tag,
    )


def _inject_legit_te(entry: PrefixEntry, ts: float) -> BGPUpdate:
    """LEGIT_TE – Traffic-engineering split within ROA bounds.

    The owner de-aggregates the prefix but stays within
    ``max_deagg`` and the ROA ``maxLength`` (if present).
    """
    max_allowed = entry.prefix_length + entry.max_deagg
    if entry.vrp:
        max_allowed = min(max_allowed, entry.vrp.max_length)
    if max_allowed <= entry.prefix_length:
        # Cannot de-aggregate further; just re-announce the same prefix
        te_prefix = entry.prefix
        te_len = entry.prefix_length
    else:
        te_len = random.randint(entry.prefix_length + 1, max_allowed)
        te_prefix = _deaggregate(entry.prefix, te_len)
    tag = GroundTruthTag.create(
        EventClass.LEGIT_TE,
        metadata={
            "owner_asn": entry.owner_asn,
            "parent_prefix": entry.prefix,
            "te_prefix": te_prefix,
            "max_deagg": entry.max_deagg,
        },
    )
    return BGPUpdate(
        timestamp=ts,
        update_type=UpdateType.ANNOUNCE,
        prefix=te_prefix,
        prefix_length=te_len,
        origin_asn=entry.owner_asn,
        as_path=_random_as_path(entry.owner_asn),
        peer_asn=_random_peer_asn(),
        collector=_random_collector(),
        communities=["65000:100"],  # TE community tag
        nexthop=_random_nexthop(),
        _ground_truth=tag,
    )


# ---------------------------------------------------------------------------
# Eligibility Filters
# ---------------------------------------------------------------------------

def _eligible_for(entry: PrefixEntry, event_class: EventClass) -> bool:
    """Return True if *entry* can host an injection of *event_class*."""

    if event_class == EventClass.HIJACK_ORIGIN:
        return True  # any prefix can be origin-hijacked

    if event_class == EventClass.HIJACK_SUBPREFIX:
        # Must have room for a more-specific (prefix_length < 28)
        return entry.prefix_length < 28

    if event_class == EventClass.HIJACK_MAXLEN:
        # Needs VRP with maxLength < 28 so we can exceed it
        return (
            entry.vrp is not None
            and entry.vrp.max_length < 28
        )

    if event_class == EventClass.LEGIT_MOAS:
        return len(entry.secondary_asns) > 0

    if event_class == EventClass.LEGIT_TE:
        return entry.max_deagg > 0

    return True


# ---------------------------------------------------------------------------
# Injection Dispatch Table
# ---------------------------------------------------------------------------

_INJECTORS = {
    EventClass.HIJACK_ORIGIN: _inject_hijack_origin,
    EventClass.HIJACK_SUBPREFIX: _inject_hijack_subprefix,
    EventClass.HIJACK_MAXLEN: _inject_hijack_maxlen,
    EventClass.LEGIT_MOAS: _inject_legit_moas,
    EventClass.LEGIT_TE: _inject_legit_te,
}


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class SyntheticBGPFeedGenerator:
    """Configurable synthetic BGP feed with fault injection.

    Parameters
    ----------
    topology : list[PrefixEntry] | None
        Seed routing topology.  If ``None``, the built-in 20-entry
        default topology is used.
    injection_rate : float
        Probability that any given update slot is replaced by a
        synthetic injection (0.0–1.0).  Default is 0.05 (5%).
    anomaly_weight : float
        Relative weight of anomaly vs. legitimate injections.
        ``1.0`` means equal; ``3.0`` means anomaly classes are
        three times more likely than legitimate classes.
    seed : int | None
        Random seed for reproducibility.
    start_ts : float | None
        Unix epoch start timestamp.  Defaults to ``time.time()``.
    update_interval : float
        Seconds between successive updates in the generated stream.
    """

    def __init__(
        self,
        topology: Optional[List[PrefixEntry]] = None,
        injection_rate: float = 0.05,
        anomaly_weight: float = 1.0,
        seed: Optional[int] = None,
        start_ts: Optional[float] = None,
        update_interval: float = 0.1,
    ) -> None:
        self.topology = topology or _build_default_topology()
        self.injection_rate = injection_rate
        self.anomaly_weight = anomaly_weight
        self.update_interval = update_interval
        self._rng = random.Random(seed)
        self._ts = start_ts or time.time()

        # Pre-compute eligibility indices for fast injection selection
        self._eligible: Dict[EventClass, List[PrefixEntry]] = {}
        for ec in EventClass:
            if ec is EventClass.BENIGN:
                continue
            self._eligible[ec] = [
                e for e in self.topology if _eligible_for(e, ec)
            ]

    # ------------------------------------------------------------------
    # Internal selection helpers
    # ------------------------------------------------------------------

    def _pick_event_class(self) -> EventClass:
        """Weighted random selection of an injection class."""
        anomaly_classes = [
            EventClass.HIJACK_ORIGIN,
            EventClass.HIJACK_SUBPREFIX,
            EventClass.HIJACK_MAXLEN,
        ]
        legit_classes = [
            EventClass.LEGIT_MOAS,
            EventClass.LEGIT_TE,
        ]

        # Filter to classes that have at least one eligible entry
        available_anomaly = [c for c in anomaly_classes if self._eligible[c]]
        available_legit = [c for c in legit_classes if self._eligible[c]]

        if not available_anomaly and not available_legit:
            return EventClass.BENIGN  # fallback

        # Build weighted pool
        pool: List[EventClass] = []
        weights: List[float] = []
        for c in available_anomaly:
            pool.append(c)
            weights.append(self.anomaly_weight)
        for c in available_legit:
            pool.append(c)
            weights.append(1.0)

        chosen = self._rng.choices(pool, weights=weights, k=1)[0]
        return chosen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, count: int = 1000
    ) -> Generator[BGPUpdate, None, None]:
        """Yield *count* BGPUpdate objects.

        Each update is either a benign baseline announcement or a
        synthetically injected event, determined by ``injection_rate``.

        Yields
        ------
        BGPUpdate
            A fully-formed update with a hidden ground-truth tag.
        """
        for _ in range(count):
            ts = self._ts
            self._ts += self.update_interval

            if self._rng.random() < self.injection_rate:
                ec = self._pick_event_class()
                if ec is EventClass.BENIGN:
                    entry = self._rng.choice(self.topology)
                    yield _make_benign(entry, ts)
                else:
                    entry = self._rng.choice(self._eligible[ec])
                    yield _INJECTORS[ec](entry, ts)
            else:
                entry = self._rng.choice(self.topology)
                yield _make_benign(entry, ts)

    def generate_batch(self, count: int = 1000) -> List[BGPUpdate]:
        """Return a list of *count* BGPUpdate objects (non-lazy)."""
        return list(self.generate(count))


# ---------------------------------------------------------------------------
# Convenience: quick summary of a generated batch
# ---------------------------------------------------------------------------

def summarize_feed(updates: Sequence[BGPUpdate]) -> Dict[str, int]:
    """Return a ``{event_class_name: count}`` dict for *updates*."""
    counts: Dict[str, int] = {}
    for u in updates:
        key = u._ground_truth.event_class.name
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# CLI entry-point for smoke testing
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate a sample feed and print summary statistics."""
    gen = SyntheticBGPFeedGenerator(
        injection_rate=0.10,
        anomaly_weight=1.5,
        seed=42,
        start_ts=1_700_000_000.0,
        update_interval=0.5,
    )

    updates = gen.generate_batch(count=1000)

    print("=" * 60)
    print("  Synthetic BGP Feed - Sample Generation Report")
    print("=" * 60)

    summary = summarize_feed(updates)
    total = len(updates)
    print(f"\n  Total updates generated : {total}\n")
    print(f"  {'Event Class':<22s} {'Count':>6s} {'Share':>8s}")
    print(f"  {'-'*22} {'-'*6} {'-'*8}")
    for cls_name, cnt in summary.items():
        pct = 100.0 * cnt / total
        print(f"  {cls_name:<22s} {cnt:>6d} {pct:>7.2f}%")

    # Verify HMAC integrity of every tag
    ok = all(u._ground_truth.verify() for u in updates)
    print(f"\n  HMAC integrity check : {'PASS' if ok else 'FAIL'}")

    # Print a few sample updates
    print("\n  Sample injected updates:")
    injected = [u for u in updates if u._ground_truth.event_class is not EventClass.BENIGN]
    for u in injected[:5]:
        gt = u._ground_truth
        print(f"    {u}")
        print(f"      -> GT: {gt.event_class.name}  meta={gt.metadata}")
    print()


if __name__ == "__main__":
    main()
