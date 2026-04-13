import json
import os
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_production_data(n=500):
    """Simulate production text statistics."""
    np.random.seed(42)
    return {
        "avg_text_length": float(np.random.normal(85, 20, n).mean()),
        "avg_entities_per_text": float(np.random.normal(2.3, 0.8, n).mean()),
        "entity_distribution": {
            "PER": int(np.random.poisson(487, 1)[0]),
            "ORG": int(np.random.poisson(423, 1)[0]),
            "LOC": int(np.random.poisson(337, 1)[0]),
        },
        "avg_confidence": float(np.random.beta(8, 2, n).mean()),
        "pii_rate": float(np.random.beta(6, 4, n).mean()),
    }


def simulate_reference_data(n=500):
    """Simulate reference/training distribution."""
    np.random.seed(0)
    return {
        "avg_text_length": float(np.random.normal(80, 15, n).mean()),
        "avg_entities_per_text": float(np.random.normal(2.1, 0.6, n).mean()),
        "entity_distribution": {
            "PER": 500,
            "ORG": 400,
            "LOC": 300,
        },
        "avg_confidence": float(np.random.beta(9, 2, n).mean()),
        "pii_rate": 0.60,
    }


def detect_drift(reference, production, threshold=0.1):
    """Detect drift between reference and production distributions."""
    drifted = []

    checks = {
        "avg_text_length": (reference["avg_text_length"], production["avg_text_length"]),
        "avg_entities_per_text": (reference["avg_entities_per_text"], production["avg_entities_per_text"]),
        "avg_confidence": (reference["avg_confidence"], production["avg_confidence"]),
        "pii_rate": (reference["pii_rate"], production["pii_rate"]),
    }

    drift_scores = {}
    for metric, (ref_val, prod_val) in checks.items():
        relative_change = abs(prod_val - ref_val) / (abs(ref_val) + 1e-9)
        drift_scores[metric] = round(relative_change, 4)
        if relative_change > threshold:
            drifted.append(metric)
            logger.warning(f"Drift detected in {metric}: {ref_val:.3f} -> {prod_val:.3f} ({relative_change:.1%} change)")

    return drifted, drift_scores


def run_drift_report():
    logger.info("Running NLP drift detection...")
    reference = simulate_reference_data()
    production = simulate_production_data()
    drifted, drift_scores = detect_drift(reference, production)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": len(drifted) > 0,
        "drifted_metrics": drifted,
        "drift_scores": drift_scores,
        "reference_stats": reference,
        "production_stats": production,
    }

    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift detected in: {drifted if drifted else 'None'}")
    return summary


if __name__ == "__main__":
    summary = run_drift_report()
    print(f"\n✅ Drift Report Generated!")
    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Drifted metrics: {summary['drifted_metrics']}")
    print(f"Drift scores: {summary['drift_scores']}")