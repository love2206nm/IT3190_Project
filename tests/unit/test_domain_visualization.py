"""Unit tests for Domain Visualization Subsystem."""

from pathlib import Path

import numpy as np

from source.domain.visualization.benchmark_visualizer import BenchmarkVisualizer
from source.domain.visualization.eda_visualizer import EDAVisualizer
from source.domain.visualization.tuning_visualizer import TuningVisualizer


def test_eda_visualizer_generates_plots(tmp_path: Path) -> None:
    """Kiểm tra sinh 3 biểu đồ EDA không bị lỗi và sinh đúng file PNG."""
    raw_texts = ["câu ngắn", "câu này dài hơn một chút để test biểu đồ", "vui quá xá :))"]
    clean_texts = ["câu ngắn", "câu này dài hơn một chút để test biểu đồ", "vui quá xá icon_vui"]

    p1 = tmp_path / "01_len.png"
    EDAVisualizer.plot_sentence_length_distribution(raw_texts, clean_texts, p1)
    assert p1.exists()
    assert p1.stat().st_size > 1000

    p2 = tmp_path / "02_vocab.png"
    EDAVisualizer.plot_vocabulary_and_sentiment_tokens(
        raw_vocab_size=100,
        clean_vocab_size=80,
        emoji_counts={"icon_vui": 15, "icon_buon": 8},
        output_path=p2,
    )
    assert p2.exists()
    assert p2.stat().st_size > 1000

    p3 = tmp_path / "03_labels.png"
    train_lbls = ["Enjoyment", "Enjoyment", "Sadness", "Other"]
    val_lbls = ["Enjoyment", "Anger"]
    test_lbls = ["Sadness", "Fear"]
    EDAVisualizer.plot_dataset_label_distribution(train_lbls, val_lbls, test_lbls, p3)
    assert p3.exists()
    assert p3.stat().st_size > 1000


def test_tuning_visualizer_generates_plots(tmp_path: Path) -> None:
    """Kiểm tra sinh biểu đồ tuning siêu tham số."""
    p_curve = tmp_path / "tuning_curve.png"
    TuningVisualizer.plot_hyperparameter_curve(
        param_name="alpha",
        param_values=[0.1, 0.5, 1.0, 2.0],
        train_scores=[0.85, 0.82, 0.80, 0.78],
        val_scores=[0.60, 0.65, 0.64, 0.61],
        model_name="Naive Bayes",
        output_path=p_curve,
    )
    assert p_curve.exists()
    assert p_curve.stat().st_size > 1000

    p_heat = tmp_path / "tuning_heatmap.png"
    summary = {
        "Naive Bayes": {"train_macro_f1": 0.82, "val_macro_f1": 0.65, "val_accuracy": 0.68, "train_time_sec": 0.05},
        "Logistic Regression": {
            "train_macro_f1": 0.88,
            "val_macro_f1": 0.69,
            "val_accuracy": 0.71,
            "train_time_sec": 0.20,
        },
    }
    TuningVisualizer.plot_all_models_tuning_summary_heatmap(summary, p_heat)
    assert p_heat.exists()
    assert p_heat.stat().st_size > 1000


def test_benchmark_visualizer_generates_plots(tmp_path: Path) -> None:
    """Kiểm tra sinh 5 biểu đồ đối đầu Best Models."""
    labels = ["Enjoyment", "Disgust", "Anger", "Sadness", "Fear", "Surprise", "Other"]
    metrics_dict = {
        "Naive Bayes": {"macro_f1": 0.62, "weighted_f1": 0.65, "accuracy": 0.66, "train_time_sec": 0.05},
        "Linear SVM": {"macro_f1": 0.71, "weighted_f1": 0.73, "accuracy": 0.74, "train_time_sec": 0.40},
    }

    # 1. Benchmark Bar
    p1 = tmp_path / "benchmark_bar.png"
    BenchmarkVisualizer.plot_best_models_benchmark(metrics_dict, p1)
    assert p1.exists()

    # 2. Trade-off
    p2 = tmp_path / "tradeoff.png"
    BenchmarkVisualizer.plot_training_latency_tradeoff(metrics_dict, p2)
    assert p2.exists()

    # 3. Confusion Matrix
    p3 = tmp_path / "cm.png"
    cm1 = np.eye(7) * 50
    cm2 = np.eye(7) * 40 + 2
    BenchmarkVisualizer.plot_confusion_matrix_comparison(cm1, cm2, labels, "Linear SVM", "Naive Bayes", p3)
    assert p3.exists()

    # 4. ROC Curves
    p4 = tmp_path / "roc.png"
    roc_data = {lbl: {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.85} for lbl in labels}
    BenchmarkVisualizer.plot_multiclass_roc_curves(roc_data, "Linear SVM", p4)
    assert p4.exists()

    # 5. Radar Chart
    p5 = tmp_path / "radar.png"
    models_per_class = {
        "Linear SVM": {lbl: 0.7 for lbl in labels},
        "Naive Bayes": {lbl: 0.6 for lbl in labels},
    }
    BenchmarkVisualizer.plot_per_class_f1_radar(models_per_class, labels, p5)
    assert p5.exists()
