"""Unit tests for SentimentPipelineOrchestrator and CLI runner (Services Layer)."""

import argparse
from typing import Any, Dict, List

import numpy as np

from source.domain.entities.sentiment_sample import EmotionLabel, EmotionScoreProfile
from source.domain.interface.base_classifer import BaseSentimentClassifier
from source.domain.interface.base_preprocess import BasePreprocess
from source.services.cli.main import handle_health, handle_predict
from source.services.cli.orchestrator import SentimentPipelineOrchestrator


class MockPreprocess(BasePreprocess):
    """Mock cài đặt BasePreprocess để kiểm tra Dependency Injection."""

    def clean_text(self, text: str) -> str:
        return text.strip().lower()

    def clean_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return samples


class MockClassifier(BaseSentimentClassifier):
    """Mock cài đặt BaseSentimentClassifier để kiểm tra Dependency Injection."""

    def train(self, X_train: Any, y_train: Any) -> None:
        pass

    def predict_scores(self, X_test: Any) -> np.ndarray:
        # Giả lập trả về vector 7 điểm số với Enjoyment cao nhất
        return np.array([[0.90, 0.02, 0.02, 0.02, 0.01, 0.02, 0.01]])

    def predict(self, X_test: Any) -> np.ndarray:
        return np.array(["Enjoyment"])

    def save(self, file_path: str) -> None:
        pass

    def load(self, file_path: str) -> None:
        pass


def test_orchestrator_fallback_scoring():
    orchestrator = SentimentPipelineOrchestrator()
    profile = orchestrator.analyze_text("Hôm nay nhận được học bổng vui quá!")
    assert isinstance(profile, EmotionScoreProfile)
    assert profile.dominant_emotion == EmotionLabel.ENJOYMENT
    assert profile.intensity > 0.8
    assert len(profile.scores) == 7


def test_orchestrator_batch_analyze():
    orchestrator = SentimentPipelineOrchestrator()
    texts = ["Vui quá", "Buồn quá", "Bực mình ghê"]
    results = orchestrator.batch_analyze(texts)
    assert len(results) == 3
    assert results[0].dominant_emotion == EmotionLabel.ENJOYMENT
    assert results[1].dominant_emotion == EmotionLabel.SADNESS
    assert results[2].dominant_emotion == EmotionLabel.ANGER


def test_orchestrator_with_injected_dependencies():
    # Kiểm tra cơ chế Dependency Injection của Clean Architecture
    mock_cleaner = MockPreprocess()
    mock_model = MockClassifier()

    orchestrator = SentimentPipelineOrchestrator(
        classifier=mock_model,
        preprocessor=mock_cleaner,
    )
    profile = orchestrator.analyze_text("  CÂU TEST VUI  ")
    assert profile.dominant_emotion == EmotionLabel.ENJOYMENT
    assert profile.intensity == 0.90


def test_cli_predict_handler():
    args = argparse.Namespace(text="Hôm nay thi qua môn vui quá")
    status = handle_predict(args)
    assert status == 0


def test_cli_health_handler():
    args = argparse.Namespace()
    status = handle_health(args)
    assert status == 0
