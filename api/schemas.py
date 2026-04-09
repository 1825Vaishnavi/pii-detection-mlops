from pydantic import BaseModel
from typing import List


class PIIEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int


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


class BatchDetectRequest(BaseModel):
    texts: List[str]


class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]
    total_texts: int
    texts_with_pii: int