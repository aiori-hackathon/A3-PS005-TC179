"""
main.py -- End-to-End Pipeline Orchestrator for Explainable BGP Monitor
"""

import time
from data_engine import SyntheticBGPFeedGenerator
from rule_engine import RuleEngine, build_default_vrp_trie, build_default_inventory
from features import FeatureExtractor
from ml_engine import MLEngine
from evaluator import Evaluator, build_detection_records

def run_pipeline(num_updates: int = 1000):
    print("=" * 70)
    print("  EXPLAINABLE BGP ANOMALY MONITOR -- PIPELINE ORCHESTRATOR")
    print("=" * 70)

    # 1. Initialize Subsystems
    print("[1/5] Initializing Subsystems...")
    feed_generator = SyntheticBGPFeedGenerator(seed=42)
    rule_engine = RuleEngine(
        vrp_trie=build_default_vrp_trie(),
        inventory=build_default_inventory()
    )
    feature_extractor = FeatureExtractor()
    ml_engine = MLEngine(theta_anomaly=0.80)

    # 2. Ingest Seed Stream & Train Tier 2 Isolation Forest
    print("[2/5] Training Tier 2 Isolation Forest on Benign Feature Vectors...")
    seed_updates = feed_generator.generate_batch(count=1000)
    train_vectors = []
    
    for update in seed_updates:
        res1 = rule_engine.evaluate(update)
        if not res1.short_circuit:
            fv = feature_extractor.process(res1)
            if fv is not None:
                train_vectors.append(fv)

    # Train Tier 2 Model
    ml_engine.train(train_vectors)
    print(f"      -> Model trained on {len(train_vectors)} benign feature vectors.")

    # 3. Process Live / Replayed Stream
    print(f"[3/5] Ingesting & Scoring {num_updates} Mixed BGP Updates...")
    live_updates = feed_generator.generate_batch(count=num_updates)

    rule_results = []
    tier2_results = []

    start_time = time.time()
    for update in live_updates:
        # Tier 1 Fast-Path Check
        r1 = rule_engine.evaluate(update)
        rule_results.append(r1)

        # Tier 2 Behavioral Check (only if unverified / not short-circuited)
        if not r1.short_circuit:
            fv = feature_extractor.process(r1)
            if fv is not None and ml_engine.is_trained:
                r2 = ml_engine.score(fv)
                tier2_results.append(r2)

    total_time = (time.time() - start_time) * 1000
    avg_latency = total_time / num_updates
    print(f"      -> Processed {num_updates} updates in {total_time:.2f} ms ({avg_latency:.3f} ms/update).")

    # 4. Evaluate Benchmark Performance (Suites S1-S5)
    print("[4/5] Running Benchmark Evaluation Engine...")
    records = build_detection_records(rule_results, tier2_results if tier2_results else None)
    evaluator = Evaluator()
    suite_reports = evaluator.run_all_suites(records)

    # 5. Display Benchmark Results
    print("\n" + evaluator.summary_table())
    print("=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
