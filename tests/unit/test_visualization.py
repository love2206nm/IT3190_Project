"""Unit tests for Emotion Visualization Subsystem (Infrastructure Layer)."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Sử dụng backend Agg không cần màn hình hiển thị để chạy headless an toàn
matplotlib.use("Agg")

from source.domain.entities.sentiment_sample import EmotionScoreProfile
from source.infrastructure.interact.visualization.emotion_charts import EmotionVisualizer


def test_plot_radar_chart(tmp_path: Path) -> None:
    scores = {
        "Enjoyment": 0.70,
        "Sadness": 0.10,
        "Anger": 0.05,
        "Fear": 0.05,
        "Disgust": 0.02,
        "Surprise": 0.05,
        "Other": 0.03,
    }
    profile = EmotionScoreProfile(text="Bài thi đạt điểm A vui quá", scores=scores)
    save_file = tmp_path / "test_radar.png"

    fig = EmotionVisualizer.plot_radar_chart(profile, save_path=save_file)
    assert fig is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)


def test_plot_spectrum_bar(tmp_path: Path) -> None:
    scores = {
        "Anger": 0.60,
        "Disgust": 0.25,
        "Sadness": 0.05,
        "Fear": 0.04,
        "Surprise": 0.03,
        "Other": 0.02,
        "Enjoyment": 0.01,
    }
    profile = EmotionScoreProfile(text="Bực mình thật sự", scores=scores)
    save_file = tmp_path / "test_bar.png"

    fig = EmotionVisualizer.plot_spectrum_bar(profile, save_path=save_file)
    assert fig is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    matrix = np.array(
        [
            [50, 2, 1, 0, 0, 1, 3],
            [3, 40, 2, 1, 1, 0, 2],
            [1, 1, 45, 2, 3, 0, 1],
            [0, 2, 1, 35, 1, 2, 2],
            [0, 1, 4, 1, 38, 0, 1],
            [2, 0, 0, 1, 0, 30, 2],
            [4, 3, 2, 1, 1, 2, 60],
        ]
    )
    labels = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
    save_file = tmp_path / "test_cm.png"

    fig = EmotionVisualizer.plot_confusion_matrix(matrix, labels, save_path=save_file)
    assert fig is not None
    assert save_file.exists()
    plt.close(fig)
