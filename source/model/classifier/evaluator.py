"""Model Evaluator (Model Classifier Layer).

Đánh giá toàn diện chất lượng mô hình phân loại cảm xúc 7 chiều (UIT-VSMEC):
- Macro-F1 (Thước đo ưu tiên hàng đầu, giải quyết mất cân bằng dữ liệu)
- Weighted-F1, Accuracy
- Log-Loss (Multi-class Cross Entropy) & Brier Score
- Confusion Matrix (Ma trận nhầm lẫn 7x7)
- One-vs-Rest ROC & AUC cho 7 nhãn
- Thời gian huấn luyện (s) và độ trễ suy luận (ms/sample)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("model_training.evaluator")


class ModelEvaluator:
    """Bộ đánh giá chất lượng toàn diện cho các mô hình học máy."""

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        train_time_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Tính toán toàn bộ các chỉ số đánh giá cho một mô hình.

        Args:
            y_true: Mảng nhãn thực tế [N] (số nguyên 0..6)
            y_pred: Mảng nhãn dự đoán [N] (số nguyên 0..6)
            y_scores: Ma trận xác suất dự đoán [N, 7] (tùy chọn)
            labels: Danh sách 7 tên nhãn chuẩn
            train_time_sec: Thời gian huấn luyện (giây)

        Returns:
            Dict chứa đầy đủ các chỉ số đánh giá
        """
        if labels is None:
            labels = ["Enjoyment", "Disgust", "Anger", "Sadness", "Fear", "Surprise", "Other"]

        # 1. Các chỉ số cơ bản
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # 2. Per-class F1
        per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = {labels[i]: float(per_class_f1_arr[i]) for i in range(len(labels))}

        # 3. Ma trận nhầm lẫn
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

        # 4. Xác suất đa lớp (Log-Loss, Brier Score, ROC curves)
        log_loss_val = 0.0
        brier_val = 0.0
        roc_data: Dict[str, Dict[str, Any]] = {}

        if y_scores is not None and y_scores.shape[1] == len(labels):
            try:
                log_loss_val = float(log_loss(y_true, y_scores, labels=list(range(len(labels)))))
            except Exception as e:
                logger.warning(f"Lỗi tính log_loss: {e}")
                log_loss_val = 0.0

            # Brier Score đa lớp (trung bình One-vs-Rest)
            y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

            brier_scores: List[float] = []
            for i in range(len(labels)):
                bs = brier_score_loss(y_true_bin[:, i], y_scores[:, i])
                brier_scores.append(bs)
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = float(auc(fpr, tpr))
                roc_data[labels[i]] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": roc_auc,
                }
            brier_val = float(np.mean(brier_scores))

        results: Dict[str, Any] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1,
            "log_loss": log_loss_val,
            "brier_score": brier_val,
            "train_time_sec": train_time_sec,
            "confusion_matrix": cm,
            "roc_data": roc_data,
        }

        logger.info(
            f"Kết quả đánh giá: Macro-F1={macro_f1:.4f}, Accuracy={acc:.4f}, "
            f"Log-Loss={log_loss_val:.4f}, TrainTime={train_time_sec:.2f}s"
        )
        return results
