import json
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "model").replace("\\", "/")

RISK_LEVELS = {
    "PER": "HIGH",
    "ORG": "MEDIUM",
    "LOC": "MEDIUM",
    "MISC": "LOW"
}


def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    return pipeline("ner", model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple")


def predict(text: str, nlp=None):
    if nlp is None:
        nlp = load_pipeline()
    results = nlp(text)
    entities = []
    for r in results:
        entities.append({
            "text": r["word"],
            "label": r["entity_group"],
            "start": r["start"],
            "end": r["end"],
            "confidence": round(float(r["score"]), 4),
            "risk_level": RISK_LEVELS.get(r["entity_group"], "LOW")
        })
    return {"text": text, "entities": entities, "pii_found": len(entities) > 0}


def redact(text: str, entities: list, style: str = "label") -> str:
    redacted = text
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        if style == "label":
            rep = f"[{e['label']}]"
        elif style == "asterisk":
            rep = "*" * len(e["text"])
        else:
            rep = "[REDACTED]"
        redacted = redacted[:e["start"]] + rep + redacted[e["end"]:]
    return redacted


if __name__ == "__main__":
    samples = [
        "John Smith works at Google in New York.",
        "Dr. Sarah Connor from MIT published a paper on AI safety.",
        "The meeting between Apple and Microsoft will be in San Francisco.",
    ]

    nlp = load_pipeline()
    print("\n🔍 PII Detection Results\n" + "="*50)
    for text in samples:
        result = predict(text, nlp)
        redacted = redact(text, result["entities"], style="label")
        print(f"\nOriginal : {text}")
        print(f"Redacted : {redacted}")
        print(f"Entities : {[(e['text'], e['label'], e['risk_level']) for e in result['entities']]}")
    print("\n✅ Done!")