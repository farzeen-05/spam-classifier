from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] == True


def test_predict_spam():
    response = client.post("/predict", json={"text": "Win free money now!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_ham():
    response = client.post("/predict", json={"text": "Hey, are we still meeting tomorrow?"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["spam", "ham"]


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_response_has_latency():
    response = client.post("/predict", json={"text": "Call me now to claim your prize"})
    assert response.status_code == 200
    assert "latency_ms" in response.json()


def test_confidence_is_float():
    response = client.post("/predict", json={"text": "Free entry in a weekly competition"})
    data = response.json()
    assert isinstance(data["confidence"], float)