# PII Detection & Redaction - Production NLP MLOps Pipeline

> An end-to-end production NLP system that detects and redacts Personally Identifiable Information (PII) from text using fine-tuned DistilBERT and TextCNN - with full MLOps infrastructure including experiment tracking, model serving, CI/CD, and monitoring.

---

##  Why This Project

PII/PHI detection is critical in healthcare, finance, and legal domains. This system:
- **Detects** PERSON, ORG, LOCATION entities using transformer-based NER
- **Redacts** PII with configurable styles (`[PERSON]`, `****`, `[REDACTED]`)
- **Scores risk** - PERSON=HIGH, ORG/LOC=MEDIUM
- **Compares** two architectures: DistilBERT (precise extraction) vs TextCNN (fast screening)

---

## Architecture

```
WikiANN Dataset (HuggingFace)
         │
         ▼
┌─────────────────────┐
│   Data Processing    │  src/data_processing.py
│   WikiANN English    │  20,000 samples, 7 NER labels
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
DistilBERT  TextCNN
NER Model   Classifier
F1: 0.6448  F1: 0.9995
    │         │
    └────┬────┘
         ▼
  MLflow Tracking
  (Experiment Registry)
         │
         ▼
┌─────────────────────┐
│    FastAPI Server    │
│  /detect            │  Token-level NER
│  /redact            │  PII replacement
│  /batch_detect      │  Bulk processing
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Docker    GitHub Actions
Container  CI/CD Pipeline
         │
         ▼
┌─────────────────────┐
│    Monitoring        │
│  Evidently Drift     │
│  Streamlit Dashboard │
└─────────────────────┘
```

---

##  Model Comparison

| Model | Task | F1 | Precision | Recall | Speed | Use Case |
|---|---|---|---|---|---|---|
| **DistilBERT** | Token NER | 0.6448 | 0.6066 | 0.6881 | ~120ms | Precise extraction |
| **TextCNN** | Binary Classification | 0.9995 | 1.0000 | 0.9990 | ~5ms | Fast screening |

> **Production Strategy:** Use TextCNN for fast PII screening, then DistilBERT only on flagged texts — reduces latency by 80%!

---

##  MLOps Layers

| Layer | Technology | Description |
|---|---|---|
| **Data** | HuggingFace Datasets | WikiANN English NER dataset |
| **Training** | PyTorch + Transformers | Fine-tune DistilBERT + TextCNN |
| **Tracking** | MLflow | Log hyperparameters, metrics, artifacts |
| **API** | FastAPI | `/detect`, `/redact`, `/batch_detect` |
| **Container** | Docker | Containerized API |
| **CI/CD** | GitHub Actions | Auto test → build on every push |
| **Monitoring** | Evidently AI + Streamlit | Drift detection + live dashboard |
| **Repro** | MLproject | One-command pipeline reproduction |

---

##  Project Structure

```
pii-detection-mlops/
├── src/
│   ├── data_processing.py   # Load WikiANN, save label info
│   ├── train.py             # Fine-tune DistilBERT NER
│   ├── train_cnn.py         # Train TextCNN classifier
│   ├── evaluate.py          # Evaluate on test set
│   └── predict.py           # Standalone prediction + redaction
├── api/
│   ├── main.py              # FastAPI app
│   └── schemas.py           # Pydantic schemas with risk scoring
├── monitoring/
│   ├── drift_detection.py   # NLP drift detection
│   └── dashboard.py         # 5-page Streamlit dashboard
├── tests/
│   └── test_api.py          # API tests with mock model
├── .github/workflows/
│   └── ci_cd.yml            # GitHub Actions pipeline
├── Dockerfile
├── MLproject
└── requirements.txt
```

---

##  Quick Start

### 1. Clone and install
```bash
git clone https://github.com/1825Vaishnavi/pii-detection-mlops.git
cd pii-detection-mlops
pip install -r requirements.txt
```

### 2. Process data
```bash
python src/data_processing.py
```

### 3. Train models
```bash
python src/train.py        # DistilBERT NER (~34 mins CPU)
python src/train_cnn.py    # TextCNN (~5 mins CPU)
```

### 4. Start API
```bash
python -m uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs
```

### 5. View MLflow experiments
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://127.0.0.1:5000
```

### 6. Run monitoring dashboard
```bash
python monitoring/drift_detection.py
python -m streamlit run monitoring/dashboard.py
# Visit http://localhost:8501
```

---

##  API Endpoints

### `POST /detect` — Detect PII entities
```json
Request:  {"text": "John Smith works at Google in New York."}
Response: {
  "entities": [
    {"text": "john smith", "label": "PER", "risk_level": "HIGH", "confidence": 0.99},
    {"text": "google",     "label": "ORG", "risk_level": "MEDIUM", "confidence": 0.98},
    {"text": "new york",   "label": "LOC", "risk_level": "MEDIUM", "confidence": 0.97}
  ],
  "pii_found": true,
  "high_risk_count": 1
}
```

### `POST /redact` — Redact PII from text
```json
Request:  {"text": "John Smith works at Google.", "replacement_style": "label"}
Response: {
  "original_text": "John Smith works at Google.",
  "redacted_text": "[PER] works at [ORG]."
}
```

### Redaction styles:
| Style | Example |
|---|---|
| `label` | `[PER] works at [ORG]` |
| `asterisk` | `********** works at ******` |
| `redacted` | `[REDACTED] works at [REDACTED]` |

---

##  CI/CD Pipeline

Every push to `main` triggers:
1. Install dependencies
2. Run tests (`pytest`)
3. Build Docker image

---

##  Monitoring Dashboard

5-page Streamlit dashboard:
- **Model Comparison** — DistilBERT vs TextCNN side by side
- **Live PII Detection** — real-time detection and redaction
- **Entity Analytics** — distribution of entity types and risk levels
- **Drift Monitoring** — detect when production data shifts
- **System Health** — API status, response times, CI/CD status

---

##  Reproduce Training

```bash
mlflow run . -e train
mlflow run . -e train_cnn
mlflow run . -e evaluate
```

---

##  Dataset

**WikiANN English NER Dataset**
- 20,000 training samples
- 7 NER labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
- Source: [HuggingFace — WikiANN](https://huggingface.co/datasets/wikiann)

---

##  Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| NLP | HuggingFace Transformers, PyTorch |
| Models | DistilBERT, TextCNN |
| Tracking | MLflow |
| API | FastAPI, Pydantic v2 |
| Container | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI, Streamlit |

---

##  Author

**Vaishnavi Mallikarjun Gajarla**
Master's in Data Analytics Engineering - Northeastern University | GPA: 3.7

[GitHub](https://github.com/1825Vaishnavi) | [LinkedIn](https://linkedin.com/in/vaishnavi)
