from fastapi.testclient import TestClient
from app.main import app
import pytest
import os

# Ensure we can run tests even if models are missing by mocking, 
# but for this assignment we expect models to exist.
# If they don't, tests might fail on 503 Service Unavailable which is correct behavior.

client = TestClient(app)

def test_health_check():
    # If model loads, it's 200/healthy. If not, 200/unhealthy.
    # The endpoint returns 200 OK with json status.
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert "status" in json_resp

def test_predict_legitimate_transaction():
    # Note: Requires model to be loaded. 
    # If model loading failed in startup (e.g. invalid path), this will return 503.
    # Check health first
    health = client.get("/health").json()
    if health["status"] != "healthy":
        pytest.skip("Model not loaded, skipping prediction test")

    payload = {
        "amount": 100.0,
        "merchant_category": "grocery",
        "distance_from_home": 5.0,
        "timestamp": "2025-09-15T14:30:00Z"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "anomaly_score" in data

def test_predict_anomaly_transaction():
    health = client.get("/health").json()
    if health["status"] != "healthy":
        pytest.skip("Model not loaded, skipping prediction test")

    payload = {
        "amount": 500000.0,
        "merchant_category": "electronics",
        "distance_from_home": 5000.0,
        "timestamp": "2025-09-15T03:00:00Z" 
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["is_fraud"], bool)

def test_invalid_input():
    payload = {
        "amount": -100, 
        "merchant_category": "unknown", 
        "distance_from_home": -5,
        "timestamp": "invalid"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
