import os
import logging
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from api.schemas import (
    DetectRequest, DetectResponse,
    BatchDetectRequest, BatchDetectResponse,
    RedactRequest, RedactResponse, PIIEntity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Detection & Redaction API",
    description="Detect and redact PII entities (PERSON, ORG, LOCATION) using fine-tuned DistilBERT",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "model").replace("\\", "/")
nlp_pipeline = None

# Risk levels for each entity type
RISK_LEVELS = {
    "PER": "HIGH",
    "ORG": "MEDIUM",
    "LOC": "MEDIUM",
    "MISC": "LOW"
}


def load_model():
    global nlp_pipeline
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        nlp_pipeline = pipeline(
            "ner", model=model, tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        logger.info("PII model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        nlp_pipeline = None


load_model()


def detect_pii(text: str) -> DetectResponse:
    results = nlp_pipeline(text)
    entities = []
    for r in results:
        entities.append(PIIEntity(
            text=r["word"],
            label=r["entity_group"],
            start=r["start"],
            end=r["end"],
            confidence=round(float(r["score"]), 4),
            risk_level=RISK_LEVELS.get(r["entity_group"], "LOW")
        ))
    return DetectResponse(
        text=text,
        entities=entities,
        pii_found=len(entities) > 0,
        total_entities=len(entities),
        high_risk_count=sum(1 for e in entities if e.risk_level == "HIGH")
    )


def redact_text(text: str, entities, replacement_style: str = "label") -> str:
    redacted = text
    # Process in reverse order to preserve indices
    for entity in sorted(entities, key=lambda x: x.start, reverse=True):
        if replacement_style == "label":
            replacement = f"[{entity.label}]"
        elif replacement_style == "asterisk":
            replacement = "*" * len(entity.text)
        else:
            replacement = "[REDACTED]"
        redacted = redacted[:entity.start] + replacement + redacted[entity.end:]
    return redacted


@app.get("/")
def root():
    return {"message": "PII Detection & Redaction API is running!", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": nlp_pipeline is not None}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    if nlp_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return detect_pii(request.text)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redact", response_model=RedactResponse)
def redact(request: RedactRequest):
    if nlp_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = detect_pii(request.text)
        redacted = redact_text(request.text, result.entities, request.replacement_style)
        return RedactResponse(
            original_text=request.text,
            redacted_text=redacted,
            entities=result.entities,
            pii_found=result.pii_found
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_detect", response_model=BatchDetectResponse)
def batch_detect(request: BatchDetectRequest):
    if nlp_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = [detect_pii(t) for t in request.texts]
        return BatchDetectResponse(
            results=results,
            total_texts=len(results),
            texts_with_pii=sum(1 for r in results if r.pii_found)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))