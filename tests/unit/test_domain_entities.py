"""Unit tests for Domain Entities and Interfaces (Clean Architecture Layer 1)."""

from dataclasses import FrozenInstanceError

import pytest

from source.domain.entities.sentiment_sample import (
    CleanedTextSample,
    EmotionLabel,
    EmotionScoreProfile,
    SentimentPrediction,
)
from source.domain.interface.base_classifer import BaseSentimentClassifier
from source.domain.interface.base_preprocess import BasePreprocess


def test_emotion_label_from_string() -> None:
    assert EmotionLabel.from_string("enjoyment") == EmotionLabel.ENJOYMENT
    assert EmotionLabel.from_string("Sadness") == EmotionLabel.SADNESS
    assert EmotionLabel.from_string("ANGER") == EmotionLabel.ANGER
    assert EmotionLabel.from_string("unknown_emotion") == EmotionLabel.OTHER


def test_emotion_score_profile_single_dominant() -> None:
    scores = {
        "Enjoyment": 0.85,
        "Sadness": 0.05,
        "Anger": 0.02,
        "Fear": 0.02,
        "Disgust": 0.01,
        "Surprise": 0.03,
        "Other": 0.02,
    }
    profile = EmotionScoreProfile(text="Vui quá xá!", scores=scores)
    assert profile.dominant_emotion == EmotionLabel.ENJOYMENT
    assert profile.intensity == 0.85
    assert not profile.is_mixed
    assert profile.ranking[0][0] == EmotionLabel.ENJOYMENT


def test_emotion_score_profile_mixed_emotions() -> None:
    # Câu có cảm xúc pha trộn (Vừa giận dữ vừa ghê tởm)
    scores = {
        "Anger": 0.55,
        "Disgust": 0.30,  # >= 0.20 -> is_mixed phải là True
        "Sadness": 0.05,
        "Fear": 0.04,
        "Surprise": 0.03,
        "Other": 0.02,
        "Enjoyment": 0.01,
    }
    profile = EmotionScoreProfile(text="Lừa đảo bẩn thỉu tức điên người", scores=scores)
    assert profile.dominant_emotion == EmotionLabel.ANGER
    assert profile.intensity == 0.55
    assert profile.is_mixed
    ranking = profile.ranking
    assert ranking[0][0] == EmotionLabel.ANGER
    assert ranking[1][0] == EmotionLabel.DISGUST


def test_sentiment_prediction_immutability() -> None:
    pred = SentimentPrediction(
        text="Vui quá",
        primary_emotion=EmotionLabel.ENJOYMENT,
        confidence=0.95,
        probabilities={"Enjoyment": 0.95, "Other": 0.05},
    )
    assert pred.primary_emotion == EmotionLabel.ENJOYMENT
    assert pred.confidence == 0.95
    # Dataclass frozen nên ném FrozenInstanceError khi cố tình sửa thuộc tính
    with pytest.raises(FrozenInstanceError):
        pred.confidence = 0.5  # type: ignore[misc]


def test_cleaned_text_sample() -> None:
    sample = CleanedTextSample(
        raw_text="Dm vui vcl",
        cleaned_text="vui quá",
        label=EmotionLabel.ENJOYMENT,
    )
    assert sample.raw_text == "Dm vui vcl"
    assert sample.cleaned_text == "vui quá"
    assert sample.label == EmotionLabel.ENJOYMENT


def test_interfaces_are_abstract() -> None:
    # Không thể khởi tạo trực tiếp abstract base class
    with pytest.raises(TypeError):
        BaseSentimentClassifier()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        BasePreprocess()  # type: ignore[abstract]
