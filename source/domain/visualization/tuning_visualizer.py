"""Hyperparameter Tuning Visualizer (Domain Visualization Layer).

Cung cấp các hàm vẽ biểu đồ phân tích và so sánh các bộ siêu tham số cho 5 mô hình ML:
1. Đường cong Validation Curve so sánh điểm Train vs Validation theo từng tham số.
2. Heatmap tổng hợp hiệu năng các tổ hợp siêu tham số tối ưu.
Lưu biểu đồ vào docs/figures/hyperparameter_tuning/.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class TuningVisualizer:
    """Công cụ trực quan hóa kết quả thử nghiệm và tối ưu hóa siêu tham số."""

    @staticmethod
    def _ensure_dir(output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def plot_hyperparameter_curve(
        cls,
        param_name: str,
        param_values: List[Any],
        train_scores: List[float],
        val_scores: List[float],
        model_name: str,
        output_path: Union[str, Path],
        metric_name: str = "Macro-F1",
    ) -> None:
        """Vẽ đường cong biến thiên của chỉ số đánh giá theo siêu tham số (Validation Curve)."""
        target_path = cls._ensure_dir(output_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.set_theme(style="whitegrid")

        # Ép kiểu x dạng chuỗi nếu có None hoặc dạng categorical
        x_str = [str(v) for v in param_values]
        x_indices = np.arange(len(param_values))

        ax.plot(
            x_indices, train_scores, "o--", label=f"Train {metric_name}", color="#2980b9", linewidth=2, markersize=7
        )
        ax.plot(
            x_indices, val_scores, "s-", label=f"Validation {metric_name}", color="#e67e22", linewidth=2.5, markersize=8
        )

        # Đánh dấu điểm tối ưu nhất trên tập Validation
        best_idx = int(np.argmax(val_scores))
        best_val = val_scores[best_idx]
        best_param = x_str[best_idx]

        ax.plot(best_idx, best_val, "r*", markersize=16, label=f"Tối ưu: {param_name}={best_param} ({best_val:.4f})")
        ax.annotate(
            f"Điểm cao nhất: {best_val:.4f}\n({param_name}={best_param})",
            xy=(best_idx, best_val),
            xytext=(best_idx, best_val + 0.02),
            ha="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656"),
        )

        ax.set_title(
            f"ĐỒ THỊ TỐI ƯU SIÊU THAM SỐ {param_name.upper()} - {model_name.upper()}", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel(f"Giá trị siêu tham số: {param_name}", fontsize=11)
        ax.set_ylabel(f"Chỉ số {metric_name}", fontsize=11)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_str, fontsize=10)

        # Giới hạn trục y hợp lý
        y_min = max(0.0, min(min(train_scores), min(val_scores)) - 0.05)
        y_max = min(1.05, max(max(train_scores), max(val_scores)) + 0.06)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_all_models_tuning_summary_heatmap(
        cls,
        tuning_summary: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ Heatmap so sánh tổng quan hiệu năng tốt nhất của từng mô hình qua các khâu tuning."""
        target_path = cls._ensure_dir(output_path)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Chuẩn bị ma trận: hàng là mô hình, cột là các chỉ số
        models = list(tuning_summary.keys())
        metrics = ["train_macro_f1", "val_macro_f1", "val_accuracy", "train_time_sec"]
        metric_labels = ["Train Macro-F1", "Val Macro-F1", "Val Accuracy", "Train Time (s)"]

        matrix = np.zeros((len(models), len(metrics)))
        for i, m in enumerate(models):
            for j, met in enumerate(metrics):
                matrix[i, j] = tuning_summary[m].get(met, 0.0)

        # Vẽ Heatmap cho 3 cột chỉ số F1 và Accuracy (chuẩn hóa màu theo dải 0.0 -> 1.0)
        sns.heatmap(
            matrix[:, :3],
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            yticklabels=models,
            xticklabels=metric_labels[:3],
            ax=ax,
            cbar_kws={"label": "Điểm số"},
            linewidths=1,
            linecolor="white",
        )

        ax.set_title("TỔNG HỢP HIỆU NĂNG CÁC TỔ HỢP SIÊU THAM SỐ TỐI ƯU CỦA 5 MÔ HÌNH", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)
