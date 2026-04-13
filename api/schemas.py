from pydantic import BaseModel
from typing import List, Optional


class PIIEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float
    risk_level: str


class DetectRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "John Smith works at Google in New York."
            }
        }


class DetectResponse(BaseModel):
    text: str
    entities: List[PIIEntity]
    pii_found: bool
    total_entities: int
    high_risk_count: int


class RedactRequest(BaseModel):
    text: str
    replacement_style: str = "label"  # "label", "asterisk", "redacted"

    class Config:
        json_schema_extra = {
            "example": {
                "text": "John Smith works at Google in New York.",
                "replacement_style": "label"
            }
        }


class RedactResponse(BaseModel):
    original_text: str
    redacted_text: str
    entities: List[PIIEntity]
    pii_found: bool


class BatchDetectRequest(BaseModel):
    texts: List[str]


class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]
    total_texts: int
    texts_with_pii: int