import os
import json
import numpy as np
import mlflow
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import (
    precision_score, recall_score, f1_score,
    classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "model").replace("\\", "/")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer,
                   aggregation_strategy="simple")
    return nlp


def evaluate_on_test(nlp, n_samples=200):
    logger.info("Loading test data...")
    dataset = load_dataset("wikiann", "en")
    test_data = dataset["test"].select(range(n_samples))

    with open("data/processed/labels.json") as f:
        info = json.load(f)
    id2label = info["id2label"]

    true_labels = []
    pred_labels = []

    logger.info(f"Evaluating on {n_samples} test samples...")
    for i, sample in enumerate(test_data):
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        text = " ".join(tokens)

        # Get true labels
        true = [id2label[str(tag)] for tag in ner_tags]
        true_labels.append(true)

        # Get predictions
        preds = nlp(text)
        pred = ["O"] * len(tokens)
        for p in preds:
            # Map character positions back to token indices
            char_count = 0
            for j, token in enumerate(tokens):
                if char_count >= p["start"] and char_count < p["end"]:
                    if pred[j] == "O":
                        pred[j] = f"B-{p['entity_group']}"
                    else:
                        pred[j] = f"I-{p['entity_group']}"
                char_count += len(token) + 1
        pred_labels.append(pred)

        if (i + 1) % 50 == 0:
            logger.info(f"Evaluated {i+1}/{n_samples} samples")

    return true_labels, pred_labels


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pii-detection")

    logger.info("Loading model...")
    nlp = load_model()

    with mlflow.start_run(run_name="evaluation-test-set"):
        true_labels, pred_labels = evaluate_on_test(nlp, n_samples=200)

        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels)

        mlflow.log_metrics({
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        })

        # Save report
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/eval_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("data/processed/eval_report.txt")

        print(f"\n✅ Evaluation Complete!")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"\nDetailed Report:\n{report}")