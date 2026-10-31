"""
api.py -- FastAPI Server for Web Dashboard & REST APIs
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_engine import SyntheticBGPFeedGenerator
from rule_engine import (
    RuleEngine,
    build_default_vrp_trie,
    build_default_inventory,
    InventoryDisposition,
)
from features import FeatureExtractor
from ml_engine import MLEngine
from evaluator import Evaluator, build_detection_records, TestSuite

# Create the FastAPI application instance
app = FastAPI(title="Explainable BGP Anomaly Monitor API", version="1.1.0")

# Enable CORS so index.html can communicate with this server from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Simple check to verify the API server is online."""
    return {"status": "online", "system": "Explainable BGP Monitor"}


@app.get("/api/run-benchmark")
def trigger_benchmark(count: int = 1000):
    """
    Triggers an end-to-end benchmark run across Tier 1 and Tier 2,
    returning live KPIs, telemetry counts, and alert cards to index.html.
    """
    # 1. Initialize Subsystems
    feed_generator = SyntheticBGPFeedGenerator(
        seed=42,
        injection_rate=0.10,
        anomaly_weight=1.5,
    )
    rule_engine = RuleEngine(
        vrp_trie=build_default_vrp_trie(),
        inventory=build_default_inventory(),
    )
    feature_extractor = FeatureExtractor()
    ml_engine = MLEngine(theta_anomaly=0.80)

    # 2. Train Tier 2 Isolation Forest on baseline seed traffic
    seed_updates = feed_generator.generate_batch(count=1000)
    train_vectors = []
    for u in seed_updates:
        r1 = rule_engine.evaluate(u)
        if not r1.short_circuit:
            fv = feature_extractor.process(r1)
            if fv is not None:
                train_vectors.append(fv)
    ml_engine.train(train_vectors)

    # Reset counters before scoring the live batch
    rule_engine.reset_stats()
    ml_engine.reset_stats()

    # 3. Process and score the requested number of updates
    live_updates = feed_generator.generate_batch(count=count)
    rule_results = []
    tier2_results = []

    for u in live_updates:
        r1 = rule_engine.evaluate(u)
        rule_results.append(r1)
        if not r1.short_circuit:
            fv = feature_extractor.process(r1)
            if fv is not None:
                r2 = ml_engine.score(fv)
                tier2_results.append(r2)

    # 4. Evaluate Benchmark Performance (Suites S1-S5)
    records = build_detection_records(rule_results, tier2_results)
    evaluator = Evaluator()
    reports = evaluator.run_all_suites(records)
    s1 = reports[TestSuite.S1_PURE_DETECTION]

    # 5. Format notable events for the index.html table
    events_payload = []
    t2_by_id = {id(r.feature_vector.update): r for r in tier2_results}

    for idx, r1 in enumerate(rule_results):
        u = r1.update
        t2 = t2_by_id.get(id(u))

        if r1.short_circuit and r1.alert is not None:
            events_payload.append({
                "id": f"t1-{idx}",
                "timestamp": u.timestamp,
                "prefix": u.prefix,
                "prefix_length": u.prefix_length,
                "origin_asn": u.origin_asn,
                "as_path": u.as_path,
                "peer_asn": u.peer_asn,
                "collector": u.collector,
                "rov_state": r1.rov_state.value,
                "category": "TIER1_ALERT",
                "tier_label": "Tier 1 (Fast Path)",
                "latency_ms": r1.processing_time_ms,
                "short_circuit": True,
                "inventory_disposition": r1.inventory_disposition.value,
                "authorised_asns": r1.alert.authorised_asns,
                "max_lengths": r1.alert.max_lengths,
                "parent_prefix": (
                    r1.alert.covering_vrps[0]["prefix"]
                    if r1.alert.covering_vrps
                    else u.prefix
                ),
                "explanation": r1.alert.explanation,
            })
        elif t2 is not None and t2.is_anomaly and t2.evidence_card is not None:
            card = t2.evidence_card
            events_payload.append({
                "id": f"t2-{idx}",
                "timestamp": u.timestamp,
                "prefix": u.prefix,
                "prefix_length": u.prefix_length,
                "origin_asn": u.origin_asn,
                "as_path": u.as_path,
                "peer_asn": u.peer_asn,
                "collector": u.collector,
                "rov_state": r1.rov_state.value,
                "category": "TIER2_ALERT",
                "tier_label": f"Tier 2 (iForest s={t2.anomaly_score:.2f})",
                "latency_ms": t2.processing_time_ms,
                "short_circuit": False,
                "anomaly_score": t2.anomaly_score,
                "theta_anomaly": card.theta_anomaly,
                "feature_vector": card.feature_vector,
                "attributions": [
                    {
                        "feature": a.feature_name,
                        "desc": f"Raw value: {a.feature_value:.2f}",
                        "value": a.feature_value,
                        "shap": a.shap_value,
                    }
                    for a in card.attributions[:5]
                ],
            })
        elif r1.inventory_disposition == InventoryDisposition.WHITELISTED:
            events_payload.append({
                "id": f"wl-{idx}",
                "timestamp": u.timestamp,
                "prefix": u.prefix,
                "prefix_length": u.prefix_length,
                "origin_asn": u.origin_asn,
                "as_path": u.as_path,
                "peer_asn": u.peer_asn,
                "collector": u.collector,
                "rov_state": r1.rov_state.value,
                "category": "WHITELISTED",
                "tier_label": "Inventory Suppressed",
                "latency_ms": r1.processing_time_ms,
                "short_circuit": False,
                "inventory_disposition": r1.inventory_disposition.value,
                "authorised_asns": [u.origin_asn],
                "max_lengths": [u.prefix_length],
                "parent_prefix": u.prefix,
                "explanation": (
                    "Matched Relational Operator Inventory whitelist "
                    "(MOAS/Anycast/TE). False-positive alert suppressed."
                ),
            })

    return {
        "status": "success",
        "processed_updates": count,
        "kpis": {
            "tier1_p99_ms": max(s1.latency_tier1.p99, 0.05),
            "fpr": s1.confusion.fpr,
            "precision": s1.confusion.precision,
            "recall": s1.confusion.recall,
            "f1_score": s1.confusion.f1_score,
        },
        "telemetry": {
            "tier1_alerts": rule_engine.stats["short_circuited"],
            "whitelisted": rule_engine.stats["whitelisted_suppressed"],
            "tier2_alerts": ml_engine.stats["anomalies_flagged"],
        },
        "events": events_payload[:50],
    }


# ---------------------------------------------------------------------------
# Direct Uvicorn Server Launcher
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Starts the Uvicorn server when you run: python api.py
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
