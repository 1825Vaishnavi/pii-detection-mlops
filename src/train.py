import os
import json
import numpy as np
import mlflow
import mlflow.pytorch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MLFLOW_EXPERIMENT = "pii-detection"


def load_label_info():
    with open("data/processed/labels.json") as f:
        info = json.load(f)
    return info["labels"], info["label2id"], info["id2label"]


def tokenize_and_align(examples, tokenizer, label_all_tokens=False):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        padding="max_length"
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_preds = [
        [id2label[str(p)] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[str(l)] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
        "f1":        f1_score(true_labels, true_preds),
    }


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    labels, label2id, id2label = load_label_info()
    num_labels = len(labels)

    logger.info("Loading dataset...")
    dataset = load_dataset("wikiann", "en")

    # Use small subset for faster training
    train_ds = dataset["train"].select(range(2000))
    val_ds   = dataset["validation"].select(range(500))

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_tok = train_ds.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=True, remove_columns=train_ds.column_names
    )
    val_tok = val_ds.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=True, remove_columns=val_ds.column_names
    )

    logger.info("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="data/processed/model",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
        use_cpu=True
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    with mlflow.start_run(run_name="distilbert-pii-ner"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "train_samples": len(train_ds),
        })

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=lambda p: compute_metrics(p, id2label),
        )

        logger.info("Training started...")
        trainer.train()

        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        logger.info(f"Metrics: {metrics}")

        mlflow.log_metrics({
            "f1":        metrics.get("eval_f1", 0),
            "precision": metrics.get("eval_precision", 0),
            "recall":    metrics.get("eval_recall", 0),
            "eval_loss": metrics.get("eval_loss", 0),
        })

        # Save model
        os.makedirs("data/processed/model", exist_ok=True)
        trainer.save_model("data/processed/model")
        tokenizer.save_pretrained("data/processed/model")
        mlflow.log_artifacts("data/processed/model", artifact_path="model")

        print(f"\n✅ Training complete!")
        print(f"F1: {metrics.get('eval_f1', 0):.4f}")
        print(f"Precision: {metrics.get('eval_precision', 0):.4f}")
        print(f"Recall: {metrics.get('eval_recall', 0):.4f}")
        print("Run 'mlflow ui' to see results!")