import os
import logging
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from api.schemas import (
    DetectRequest, DetectResponse,
    BatchDetectRequest, BatchDetectResponse, PIIEntity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Detection API",
    description="Detect PII entities (PERSON, ORG, LOCATION) using fine-tuned DistilBERT",
    version="1.0.0"
)

MODEL_PATH = "data/processed/model"
nlp_pipeline = None


def load_model():
    global nlp_pipeline
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        nlp_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
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
            end=r["end"]
        ))
    return DetectResponse(
        text=text,
        entities=entities,
        pii_found=len(entities) > 0
    )


@app.get("/")
def root():
    return {"message": "PII Detection API is running!", "version": "1.0.0"}


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