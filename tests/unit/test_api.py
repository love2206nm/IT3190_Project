"""Unit tests for FastAPI endpoints (Presentation Layer)."""

from fastapi.testclient import TestClient

from source.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "UIT-VSMEC" in data["app_name"]


def test_predict_sentiment_scores_enjoyment() -> None:
    payload = {"text": "Hôm nay nhận được học bổng vui quá!"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] == "Enjoyment"
    assert data["intensity"] > 0.8
    assert not data["is_mixed"]
    assert "scores" in data
    assert len(data["scores"]) == 7
    assert data["scores"]["Enjoyment"] > 0.8


def test_predict_sentiment_scores_sadness() -> None:
    payload = {"text": "Chia tay rồi buồn quá đi mất"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] == "Sadness"
    assert data["intensity"] > 0.7
    assert "scores" in data


def test_predict_sentiment_mixed_emotions() -> None:
    payload = {"text": "Shop làm ăn gian dối, giao hàng rách nát bực cả mình"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] == "Anger"
    assert "scores" in data
