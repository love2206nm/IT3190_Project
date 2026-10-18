"""Emotion Visualization Subsystem (Infrastructure Layer).

Cung cấp công cụ trực quan hóa Thang đo phổ điểm 7 chiều cảm xúc (UIT-VSMEC):
- Biểu đồ Mạng nhện (Radar / Spider Chart).
- Biểu đồ Thanh quang phổ cường độ (Spectrum Intensity Bar).
- Biểu đồ Ma trận nhầm lẫn (Confusion Matrix Heatmap).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from source.domain.entities.sentiment_sample import EmotionLabel, EmotionScoreProfile

# Bảng mã màu chuẩn nhận diện từng sắc thái cảm xúc
EMOTION_COLOR_MAP: Dict[str, str] = {
    EmotionLabel.ENJOYMENT.value: "#2ecc71",  # Xanh lá cây
    EmotionLabel.SADNESS.value: "#3498db",  # Xanh dương
    EmotionLabel.ANGER.value: "#e74c3c",  # Đỏ
    EmotionLabel.FEAR.value: "#9b59b6",  # Tím
    EmotionLabel.DISGUST.value: "#e67e22",  # Cam
    EmotionLabel.SURPRISE.value: "#f1c40f",  # Vàng
    EmotionLabel.OTHER.value: "#95a5a6",  # Xám
}

# Nhãn song ngữ hiển thị trên đồ thị
EMOTION_DISPLAY_NAMES: Dict[str, str] = {
    EmotionLabel.ENJOYMENT.value: "Thích thú\n(Enjoyment)",
    EmotionLabel.SADNESS.value: "Buồn bã\n(Sadness)",
    EmotionLabel.ANGER.value: "Giận dữ\n(Anger)",
    EmotionLabel.FEAR.value: "Sợ hãi\n(Fear)",
    EmotionLabel.DISGUST.value: "Ghê tởm\n(Disgust)",
    EmotionLabel.SURPRISE.value: "Bất ngờ\n(Surprise)",
    EmotionLabel.OTHER.value: "Khác\n(Other)",
}


class EmotionVisualizer:
    """Bộ công cụ trực quan hóa dữ liệu cảm xúc cho bài toán UIT-VSMEC."""

    @staticmethod
    def plot_radar_chart(
        profile: EmotionScoreProfile,
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ Biểu đồ Mạng nhện (Radar Chart) biểu diễn thang đo phổ điểm 7 chiều cảm xúc."""
        ordered_labels = [e.value for e in EmotionLabel]
        num_vars = len(ordered_labels)

        # Lấy điểm số theo thứ tự 7 nhãn chuẩn
        scores = [float(profile.scores.get(lbl, 0.0)) for lbl in ordered_labels]

        # Tính toán góc cho 7 đỉnh trên đường tròn
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Đóng kín đường đa giác (nối điểm cuối về điểm đầu)
        scores_closed = scores + scores[:1]
        angles_closed = angles + angles[:1]

        # Khởi tạo đồ thị tọa độ cực (polar plot)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

        # Vẽ đường viền đa giác và đổ màu trong suốt
        dominant_color = EMOTION_COLOR_MAP.get(profile.dominant_emotion.value, "#3498db")
        ax.plot(angles_closed, scores_closed, color=dominant_color, linewidth=2.5, linestyle="solid")
        ax.fill(angles_closed, scores_closed, color=dominant_color, alpha=0.35)

        # Gán nhãn cho từng trục cảm xúc
        display_labels = [EMOTION_DISPLAY_NAMES.get(lbl, lbl) for lbl in ordered_labels]
        ax.set_xticks(angles)
        ax.set_xticklabels(display_labels, size=10, weight="bold")

        # Thiết lập các vòng tròn đồng tâm từ 0.0 đến 1.0
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
        ax.set_ylim(0, 1.0)
        ax.grid(color="#bdc3c7", linestyle="--", linewidth=0.7)

        chart_title = title or f"Phổ Cảm xúc: {profile.dominant_emotion.value} ({profile.intensity * 100:.1f}%)"
        ax.set_title(chart_title, size=13, weight="bold", pad=25, color="#2c3e50")

        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_spectrum_bar(
        profile: EmotionScoreProfile,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Vẽ biểu đồ thanh ngang so sánh cường độ 7 sắc thái cảm xúc."""
        ordered_labels = [e.value for e in EmotionLabel]
        scores = [float(profile.scores.get(lbl, 0.0)) for lbl in ordered_labels]
        colors = [EMOTION_COLOR_MAP.get(lbl, "#3498db") for lbl in ordered_labels]

        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(ordered_labels))

        bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor="#2c3e50", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl for lbl in ordered_labels], size=10, weight="bold")
        ax.invert_yaxis()  # Nhãn đầu tiên nằm ở trên cùng
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Cường độ / Xác suất", size=10, weight="bold")
        ax.set_title("Thang đo Cường độ 7 Chiều Cảm xúc (UIT-VSMEC)", size=12, weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Hiển thị số phần trăm ở đầu mỗi thanh
        for bar, score in zip(bars, scores, strict=False):
            width = bar.get_width()
            ax.text(
                width + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{score * 100:.1f}%",
                va="center",
                ha="left",
                size=9,
                weight="bold",
                color="#2c3e50",
            )

        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_confusion_matrix(
        matrix: np.ndarray,
        class_names: List[str],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Vẽ Heatmap Ma trận nhầm lẫn cho việc đánh giá mô hình."""
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="left", weight="bold")
        ax.set_yticklabels(class_names, weight="bold")

        ax.set_xlabel("Nhãn Dự đoán (Predicted)", weight="bold", labelpad=10)
        ax.set_ylabel("Nhãn Thực tế (Actual)", weight="bold", labelpad=10)
        ax.set_title("Confusion Matrix Heatmap", weight="bold", pad=20)

        # Ghi giá trị số vào từng ô
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > (matrix.max() / 2) else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, weight="bold")

        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
