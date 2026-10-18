"""Model Benchmark & Comparison Visualizer (Domain Visualization Layer).

Cung cấp các hàm vẽ biểu đồ đối đầu chuyên sâu giữa 5 Best Models đã được tối ưu tham số:
1. Bar Chart so sánh Macro-F1, Weighted-F1, Accuracy trên tập Test.
2. Scatter Plot phân tích đánh đổi tài nguyên (Training Time vs Macro-F1).
3. Confusion Matrix Heatmap đối đầu giữa Best Model (LinearSVC) vs Baseline (Naive Bayes).
4. Đường cong One-vs-Rest Multiclass ROC & AUC cho 7 nhãn cảm xúc.
5. Radar Chart so sánh F1-Score 7 nhãn giữa các Best Models.
Lưu biểu đồ vào docs/figures/model_comparison/.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BenchmarkVisualizer:
    """Công cụ trực quan hóa kết quả đối đầu và đánh giá mô hình học máy."""

    @staticmethod
    def _ensure_dir(output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def plot_best_models_benchmark(
        cls,
        metrics_dict: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ Bar Chart so sánh các chỉ số chất lượng chính (Macro-F1, Weighted-F1, Accuracy)."""
        target_path = cls._ensure_dir(output_path)
        models = list(metrics_dict.keys())

        macro_f1s = [metrics_dict[m].get("macro_f1", 0.0) for m in models]
        weighted_f1s = [metrics_dict[m].get("weighted_f1", 0.0) for m in models]
        accuracies = [metrics_dict[m].get("accuracy", 0.0) for m in models]

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.set_theme(style="whitegrid")

        r1 = ax.bar(x - width, macro_f1s, width, label="Macro-F1 (Trọng tâm)", color="#2ecc71", edgecolor="black")
        ax.bar(x, weighted_f1s, width, label="Weighted-F1", color="#3498db", edgecolor="black")
        ax.bar(x + width, accuracies, width, label="Accuracy", color="#9b59b6", edgecolor="black")

        ax.set_ylabel("Điểm số [0.0 - 1.0]", fontsize=12)
        ax.set_title("ĐỐI ĐẦU 5 MÔ HÌNH HỌC MÁY TỐI ƯU TRÊN TẬP TEST (UIT-VSMEC)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", fontsize=11)

        # Hiển thị số liệu trên đầu cột Macro-F1
        for rect in r1:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_training_latency_tradeoff(
        cls,
        metrics_dict: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ phân tích đánh đổi giữa Thời gian huấn luyện (s) và Macro-F1."""
        target_path = cls._ensure_dir(output_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        models = list(metrics_dict.keys())
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, m in enumerate(models):
            t_time = metrics_dict[m].get("train_time_sec", 0.1)
            macro_f1 = metrics_dict[m].get("macro_f1", 0.0)
            ax.scatter(t_time, macro_f1, color=colors[i % len(colors)], s=250, edgecolors="black", label=m, zorder=5)
            ax.annotate(
                f"{m}\n({t_time:.2f}s | F1: {macro_f1:.3f})",
                xy=(t_time, macro_f1),
                xytext=(t_time, macro_f1 + 0.015),
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
            )

        ax.set_title("PHÂN TÍCH ĐÁNH ĐỔI TÀI NGUYÊN (TRAINING TIME VS MACRO-F1)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Thời gian huấn luyện (giây) - Trục Log", fontsize=11)
        ax.set_ylabel("Chỉ số Macro-F1", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="--")
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_confusion_matrix_comparison(
        cls,
        cm_best: np.ndarray,
        cm_baseline: np.ndarray,
        labels: List[str],
        best_name: str,
        baseline_name: str,
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ Heatmap Ma trận nhầm lẫn so sánh chuẩn hóa % giữa Best Model và Baseline."""
        target_path = cls._ensure_dir(output_path)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Chuẩn hóa tỷ lệ theo hàng (%)
        cm_base_pct = cm_baseline.astype("float") / np.maximum(cm_baseline.sum(axis=1, keepdims=True), 1) * 100
        cm_best_pct = cm_best.astype("float") / np.maximum(cm_best.sum(axis=1, keepdims=True), 1) * 100

        # 1. Baseline Model
        sns.heatmap(
            cm_base_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=axes[0],
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            linewidths=0.5,
        )
        axes[0].set_title(f"Baseline: {baseline_name} (Nhầm lẫn %)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Nhãn Thực tế (Ground Truth)")
        axes[0].set_xlabel("Nhãn Dự đoán (Predicted)")

        # 2. Best Model
        sns.heatmap(
            cm_best_pct,
            annot=True,
            fmt=".1f",
            cmap="Greens",
            ax=axes[1],
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            linewidths=0.5,
        )
        axes[1].set_title(f"Best Model: {best_name} (Nhầm lẫn %)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Nhãn Thực tế (Ground Truth)")
        axes[1].set_xlabel("Nhãn Dự đoán (Predicted)")

        plt.suptitle("MA TRẬN NHẦM LẪN ĐỐI ĐẦU (TẬP TEST UIT-VSMEC)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_multiclass_roc_curves(
        cls,
        roc_data: Dict[str, Dict[str, Any]],
        model_name: str,
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ đường cong One-vs-Rest ROC & diện tích AUC cho 7 nhãn cảm xúc."""
        target_path = cls._ensure_dir(output_path)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.set_theme(style="whitegrid")

        colors = sns.color_palette("Set1", len(roc_data))

        for (label, metrics), color in zip(roc_data.items(), colors, strict=False):
            fpr = metrics.get("fpr", [0, 1])
            tpr = metrics.get("tpr", [0, 1])
            auc_val = metrics.get("auc", 0.5)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Đoán ngẫu nhiên (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Tỷ lệ Dương tính giả (False Positive Rate)", fontsize=11)
        ax.set_ylabel("Tỷ lệ Dương tính thật (True Positive Rate)", fontsize=11)
        ax.set_title(
            f"ĐƯỜNG CONG ONE-VS-REST ROC 7 CHIỀU CẢM XÚC - {model_name.upper()}", fontsize=13, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_per_class_f1_radar(
        cls,
        models_per_class_f1: Dict[str, Dict[str, float]],
        labels: List[str],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ mạng nhện (Radar Chart) so sánh F1-Score 7 nhãn giữa các Best Models."""
        target_path = cls._ensure_dir(output_path)

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Khép kín đường tròn

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        for (model_name, class_scores), color in zip(models_per_class_f1.items(), colors, strict=False):
            values = [class_scores.get(lbl, 0.0) for lbl in labels]
            values += values[:1]  # Khép kín
            ax.plot(angles, values, color=color, linewidth=2, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=0, fontsize=8)
        ax.set_title("SO SÁNH F1-SCORE TRÊN 7 NHÃN CẢM XÚC (RADAR CHART)", fontsize=13, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)
