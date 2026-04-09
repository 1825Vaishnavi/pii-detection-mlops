import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the NLP pipeline before importing app
mock_pipeline = MagicMock()
mock_pipeline.return_value = [
    {"word": "john smith", "entity_group": "PER", "start": 0, "end": 10, "score": 0.99},
    {"word": "google", "entity_group": "ORG", "start": 20, "end": 26, "score": 0.98},
    {"word": "new york", "entity_group": "LOC", "start": 30, "end": 38, "score": 0.97},
]

import api.main as main_module
main_module.nlp_pipeline = mock_pipeline

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

sample_text = {"text": "John Smith works at Google in New York."}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["message"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_detect():
    r = client.post("/detect", json=sample_text)
    assert r.status_code == 200
    data = r.json()
    assert "entities" in data
    assert "pii_found" in data
    assert data["pii_found"] is True
    assert len(data["entities"]) > 0


def test_batch_detect():
    r = client.post("/batch_detect", json={"texts": [
        "John Smith works at Google.",
        "No PII here."
    ]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_texts"] == 2
    assert len(data["results"]) == 2