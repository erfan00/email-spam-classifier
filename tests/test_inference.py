"""Tests for the inference wrapper and the FastAPI HTTP layer.

Edge cases explicitly covered: empty text, non-ASCII text, and very long text,
as well as the "model not trained yet" failure mode.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import email_spam_classifier.predict as predict_module
from email_spam_classifier.api.main import app
from email_spam_classifier.predict import ModelNotTrainedError, predict_text


@pytest.mark.usefixtures("wired_baseline_model")
class TestPredictText:
    def test_obvious_spam_is_classified_as_spam(self) -> None:
        result = predict_text(
            "WINNER!! Claim your FREE prize now, click here urgent limited offer cash"
        )
        assert result.label == "spam"
        assert result.spam_probability > 0.5

    def test_obvious_ham_is_classified_as_ham(self) -> None:
        result = predict_text("Hey, are we still meeting for lunch tomorrow at noon? Let me know.")
        assert result.label == "ham"
        assert result.spam_probability < 0.5

    def test_empty_text_does_not_raise(self) -> None:
        result = predict_text("")
        assert result.label in ("spam", "ham")
        assert 0.0 <= result.spam_probability <= 1.0

    def test_none_like_whitespace_text_does_not_raise(self) -> None:
        result = predict_text("   \n\t  ")
        assert result.label in ("spam", "ham")

    def test_non_ascii_text_does_not_raise(self) -> None:
        result = predict_text("日本語のメールです。これはテストです。café naïve 🎉")
        assert result.label in ("spam", "ham")
        assert 0.0 <= result.spam_probability <= 1.0

    def test_very_long_text_does_not_raise(self) -> None:
        result = predict_text("free money winner click now " * 5000)
        assert result.label in ("spam", "ham")

    def test_model_name_is_echoed_back(self) -> None:
        result = predict_text("hello there", model_name="baseline")
        assert result.model_name == "baseline"


class TestModelNotTrainedError:
    def test_raises_clear_error_when_model_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        predict_module.clear_model_cache()
        bogus_path = predict_module.Path("/nonexistent/model.joblib")
        monkeypatch.setitem(predict_module._MODEL_PATHS, "baseline", bogus_path)
        with pytest.raises(ModelNotTrainedError):
            predict_text("hello", model_name="baseline")
        predict_module.clear_model_cache()


@pytest.mark.usefixtures("wired_baseline_model")
class TestPredictEndpoint:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_predict_spam_like_text(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            json={
                "text": "WINNER!! Claim your FREE prize now, click here urgent cash",
                "model": "baseline",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["label"] == "spam"
        assert 0.0 <= body["spam_probability"] <= 1.0
        assert body["model"] == "baseline"

    def test_predict_ham_like_text(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            json={"text": "Let's meet for lunch tomorrow at noon.", "model": "baseline"},
        )
        assert response.status_code == 200
        assert response.json()["label"] == "ham"

    def test_predict_empty_text(self, client: TestClient) -> None:
        response = client.post("/predict", json={"text": "", "model": "baseline"})
        assert response.status_code == 200
        body = response.json()
        assert body["label"] in ("spam", "ham")

    def test_predict_missing_text_field_defaults_to_empty(self, client: TestClient) -> None:
        response = client.post("/predict", json={"model": "baseline"})
        assert response.status_code == 200

    def test_predict_rejects_invalid_model_name(self, client: TestClient) -> None:
        response = client.post("/predict", json={"text": "hello", "model": "not_a_real_model"})
        assert response.status_code == 422

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "baseline_available" in body

    def test_demo_front_end_is_served_at_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestPredictEndpointModelNotTrained:
    def test_returns_503_when_embedding_model_not_trained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        predict_module.clear_model_cache()
        bogus_path = predict_module.Path("/nonexistent/model.joblib")
        monkeypatch.setitem(predict_module._MODEL_PATHS, "embedding", bogus_path)
        client = TestClient(app)
        response = client.post("/predict", json={"text": "hello", "model": "embedding"})
        assert response.status_code == 503
        predict_module.clear_model_cache()
