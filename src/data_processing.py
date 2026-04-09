from datasets import load_dataset
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Loading WikiANN English NER dataset...")
    dataset = load_dataset("wikiann", "en")
    logger.info(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return dataset


def save_label_info(dataset):
    os.makedirs("data/processed", exist_ok=True)
    labels = dataset["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    with open("data/processed/labels.json", "w") as f:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f)
    logger.info(f"Labels: {labels}")
    return labels, label2id, id2label


def show_sample(dataset):
    sample = dataset["train"][0]
    logger.info("Sample:")
    logger.info(f"  Tokens: {sample['tokens']}")
    logger.info(f"  NER tags: {sample['ner_tags']}")
    logger.info(f"  Spans: {sample['spans']}")


if __name__ == "__main__":
    dataset = load_data()
    labels, label2id, id2label = save_label_info(dataset)
    show_sample(dataset)
    print(f"\n✅ Data processing complete!")
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Labels: {labels}")
    print(f"Saved label info to data/processed/labels.json")