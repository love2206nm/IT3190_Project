"""EDA & Data Preprocessing Visualizer (Domain Visualization Layer).

Cung cấp các hàm vẽ biểu đồ phân tích và so sánh dữ liệu thô (Raw) vs dữ liệu sạch (Cleaned):
1. Phân bố độ dài câu & số lượng từ (KDE/Boxplot).
2. Quy mô không gian từ vựng và tần suất trích xuất semantic emoji tokens.
3. Phân bố 7 nhãn cảm xúc trên Train/Valid/Test & phân tích mất cân bằng dữ liệu.
Lưu biểu đồ vào docs/figures/data_preprocessing/.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class EDAVisualizer:
    """Công cụ trực quan hóa phân tích dữ liệu trước và sau tiền xử lý."""

    @staticmethod
    def _ensure_dir(output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def plot_sentence_length_distribution(
        cls,
        raw_texts: List[str],
        clean_texts: List[str],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ so sánh độ dài câu (Ký tự và Từ) trước vs sau làm sạch."""
        target_path = cls._ensure_dir(output_path)

        raw_char_lens = [len(t) for t in raw_texts]
        clean_char_lens = [len(t) for t in clean_texts]

        raw_word_lens = [len(t.split()) for t in raw_texts]
        clean_word_lens = [len(t.split()) for t in clean_texts]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.set_theme(style="whitegrid")

        # 1. Biểu đồ phân bố số từ (Word Count KDE)
        sns.kdeplot(raw_word_lens, ax=axes[0], label="Dữ liệu thô (Raw)", color="#e74c3c", fill=True, alpha=0.3)
        sns.kdeplot(clean_word_lens, ax=axes[0], label="Dữ liệu sạch (Cleaned)", color="#2ecc71", fill=True, alpha=0.4)
        axes[0].set_title("So sánh Phân bố Số từ trong Câu (Word Count)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Số lượng từ / câu")
        axes[0].set_ylabel("Mật độ phân bố (Density)")
        axes[0].set_xlim(0, 50)
        axes[0].legend(loc="upper right")

        # 2. Biểu đồ Boxplot phân bố số ký tự
        char_data = [raw_char_lens, clean_char_lens]
        box = axes[1].boxplot(
            char_data,
            tick_labels=["Dữ liệu thô", "Dữ liệu sạch"],
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )
        colors = ["#f39c12", "#3498db"]
        for patch, color in zip(box["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_title("Độ biến thiên Số ký tự (Character Count Boxplot)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Số lượng ký tự")

        plt.suptitle("ĐÁNH GIÁ TÁC ĐỘNG CỦA TIỀN XỬ LÝ LÊN ĐỘ DÀI VĂN BẢN (UIT-VSMEC)", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_vocabulary_and_sentiment_tokens(
        cls,
        raw_vocab_size: int,
        clean_vocab_size: int,
        emoji_counts: Dict[str, int],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ so sánh kích thước từ vựng và tần suất các semantic emoji tokens."""
        target_path = cls._ensure_dir(output_path)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.set_theme(style="whitegrid")

        # 1. So sánh quy mô từ vựng
        labels = ["Trước tiền xử lý (Raw)", "Sau tiền xử lý (Cleaned)"]
        sizes = [raw_vocab_size, clean_vocab_size]
        reduction_pct = ((raw_vocab_size - clean_vocab_size) / raw_vocab_size) * 100 if raw_vocab_size > 0 else 0

        bars = axes[0].bar(labels, sizes, color=["#95a5a6", "#27ae60"], width=0.5, edgecolor="black")
        for bar, size in zip(bars, sizes, strict=False):
            yval = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 100,
                f"{size:,}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        axes[0].set_title(f"Thu gọn Không gian Từ vựng (Giảm {reduction_pct:.1f}%)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Số lượng từ vựng độc nhất (Unique Words)")

        # 2. Tần suất trích xuất semantic emoji tokens
        if emoji_counts:
            sorted_tokens = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)
            tok_names = [t[0] for t in sorted_tokens]
            tok_freqs = [t[1] for t in sorted_tokens]

            token_palette = sns.color_palette("viridis", len(tok_names))
            bars_tok = axes[1].barh(tok_names, tok_freqs, color=token_palette, edgecolor="black")
            for bar in bars_tok:
                xval = bar.get_width()
                axes[1].text(
                    xval + 5,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{int(xval)}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )
            axes[1].invert_yaxis()
            axes[1].set_title("Tần suất Trích xuất Semantic Emoji Tokens", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Số lần xuất hiện trong kho ngữ liệu")
        else:
            axes[1].text(0.5, 0.5, "Không có dữ liệu Emoji", ha="center", va="center")

        plt.suptitle("PHÂN TÍCH TỪ VỰNG & BẢO TOÀN ĐẶC TRƯNG CẢM XÚC (UIT-VSMEC)", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)

    @classmethod
    def plot_dataset_label_distribution(
        cls,
        train_labels: List[str],
        valid_labels: List[str],
        test_labels: List[str],
        output_path: Union[str, Path],
    ) -> None:
        """Vẽ biểu đồ cột ghép tỷ lệ % 7 nhãn cảm xúc trên Train/Valid/Test."""
        target_path = cls._ensure_dir(output_path)

        train_counts = Counter(train_labels)
        valid_counts = Counter(valid_labels)
        test_counts = Counter(test_labels)

        # Lấy danh sách 7 nhãn chuẩn
        classes = ["Enjoyment", "Disgust", "Other", "Sadness", "Anger", "Fear", "Surprise"]

        train_total = max(len(train_labels), 1)
        valid_total = max(len(valid_labels), 1)
        test_total = max(len(test_labels), 1)

        train_pcts = [train_counts.get(c, 0) / train_total * 100 for c in classes]
        valid_pcts = [valid_counts.get(c, 0) / valid_total * 100 for c in classes]
        test_pcts = [test_counts.get(c, 0) / test_total * 100 for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.set_theme(style="whitegrid")

        rects1 = ax.bar(x - width, train_pcts, width, label=f"Train ({train_total} mẫu)", color="#3498db")
        ax.bar(x, valid_pcts, width, label=f"Valid ({valid_total} mẫu)", color="#f1c40f")
        ax.bar(x + width, test_pcts, width, label=f"Test ({test_total} mẫu)", color="#e74c3c")

        ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=12)
        ax.set_title(
            "PHÂN BỐ 7 NHÃN CẢM XÚC TRÊN CÁC TẬP DỮ LIỆU & PHÂN TÍCH MẤT CÂN BẰNG", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
        ax.legend(fontsize=11)

        # Thêm ghi chú tỷ lệ trên cột train
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close(fig)
