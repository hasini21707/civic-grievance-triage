import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_api_predict():
    payload = {
        "category": "Storm Water Drain(SWD)",
        "sub_category": "Open Manhole",
        "description": "Open manhole dangerous exposed wire live cable",
        "ward": "Jnanabharathi Ward",
        "complaint_date": "2026-08-20T10:00:00"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "severity" in data
    assert "priority_score" in data
    assert "priority_level" in data
    assert 0.0 <= data["priority_score"] <= 100.0


def test_api_ingest_complaint():
    import uuid
    dynamic_id = f"TEST-API-{str(uuid.uuid4())[:8]}"
    payload = {
        "complaint_id": dynamic_id,
        "category": "Road Maintenance",
        "sub_category": "Potholes",
        "ward_name": "Horamavu",
        "staff_remarks": "Severe pothole blocking traffic",
        "staff_name": "Officer Smith"
    }
    response = client.post("/complaints", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["complaint_id"] == dynamic_id
    assert "priority_score" in data
