import os
import json
import numpy as np
import mlflow
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 10000
EMBED_DIM    = 128
NUM_FILTERS  = 64
FILTER_SIZES = [2, 3, 4]
DROPOUT      = 0.5
EPOCHS       = 5
BATCH_SIZE   = 32
LR           = 1e-3
MAX_LEN      = 64


# ── Tokenizer ────────────────────────────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, sentences):
        counter = Counter()
        for s in sentences:
            counter.update(s.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 2):
            self.word2idx[word] = len(self.word2idx)
        logger.info(f"Vocab size: {len(self.word2idx)}")

    def encode(self, sentence, max_len=MAX_LEN):
        tokens = sentence.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids


# ── Dataset ───────────────────────────────────────────────────────────────────
class PIIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = [tokenizer.encode(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ── TextCNN Model ─────────────────────────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)


# ── Data prep ─────────────────────────────────────────────────────────────────
def prepare_data():
    logger.info("Loading WikiANN dataset...")
    dataset = load_dataset("wikiann", "en")

    pii_texts, non_pii_texts = [], []

    for split in ["train", "validation"]:
        for sample in dataset[split]:
            text = " ".join(sample["tokens"])
            has_pii = any(t != 0 for t in sample["ner_tags"])
            if has_pii:
                pii_texts.append(text)
            else:
                non_pii_texts.append(text)

    # Since WikiANN has mostly PII, create non-PII samples synthetically
    generic_sentences = [
        "the weather is nice today",
        "the car is parked outside",
        "the book is on the table",
        "the train arrives at noon",
        "the store opens at nine",
        "the food was delicious yesterday",
        "the meeting starts in ten minutes",
        "the report is due on friday",
        "the sky is clear and blue",
        "the project deadline was extended",
    ]

    # Balance dataset: equal PII and non-PII
    n = min(len(pii_texts), 5000)
    pii_texts = pii_texts[:n]
    non_pii_texts = (generic_sentences * (n // len(generic_sentences) + 1))[:n]

    texts  = pii_texts + non_pii_texts
    labels = [1] * n + [0] * n

    # Shuffle
    combined = list(zip(texts, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts, labels = zip(*combined)

    logger.info(f"Total: {len(texts)} | PII: {sum(labels)} | No PII: {len(labels)-sum(labels)}")
    return list(texts), list(labels)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # F1 score
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return total_loss / len(loader), correct / len(loader.dataset), precision, recall, f1


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pii-detection")
    device = torch.device("cpu")

    texts, labels = prepare_data()

    # Split
    split = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts)

    # Save tokenizer vocab
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/cnn_vocab.json", "w") as f:
        json.dump(tokenizer.word2idx, f)

    # Datasets
    train_ds = PIIDataset(train_texts, train_labels, tokenizer)
    val_ds   = PIIDataset(val_texts,   val_labels,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    with mlflow.start_run(run_name="TextCNN-PII-Classifier"):
        mlflow.log_params({
            "model": "TextCNN",
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_filters": NUM_FILTERS,
            "filter_sizes": str(FILTER_SIZES),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
        })

        model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_FILTERS, FILTER_SIZES, 2, DROPOUT).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_f1 = 0
        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device)
            val_loss, val_acc, precision, recall, f1 = eval_epoch(model, val_dl, criterion, device)

            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f}")
            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_loss,  "val_acc": val_acc,
                "precision": precision, "recall": recall, "f1": f1
            }, step=epoch)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "data/processed/cnn_model.pt")

        # Log final metrics
        mlflow.log_metrics({"best_f1": best_f1, "final_precision": precision, "final_recall": recall})
        mlflow.log_artifact("data/processed/cnn_model.pt")
        mlflow.log_artifact("data/processed/cnn_vocab.json")

        print(f"\n✅ TextCNN Training Complete!")
        print(f"Best F1:    {best_f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print("\nBoth models logged to MLflow — run 'python -m mlflow ui --backend-store-uri sqlite:///mlflow.db' to compare!")