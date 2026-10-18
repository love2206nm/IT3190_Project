"""Base Sentiment Classifier Interface (Domain Layer).

Quy định hợp đồng (Contract) chuẩn cho tất cả các mô hình học máy:
Trọng tâm là ước lượng Thang đo phổ điểm 7 chiều cảm xúc (UIT-VSMEC).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np


class BaseSentimentClassifier(ABC):
    """Lớp trừu tượng định nghĩa các phương thức bắt buộc của mô hình phân loại."""

    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện mô hình trên tập dữ liệu đặc trưng và nhãn."""
        pass

    @abstractmethod
    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận thang đo điểm số [N, 7] cho 7 nhãn cảm xúc UIT-VSMEC.

        Mỗi hàng đại diện cho một mẫu, các cột tương ứng với điểm số [0.0, 1.0]
        của 7 nhãn: Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other.
        """
        pass

    def predict_proba(self, X_test: Any) -> np.ndarray:
        """Alias cho predict_scores để tương thích với scikit-learn convention."""
        return self.predict_scores(X_test)

    @abstractmethod
    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc chiếm ưu thế nhất (dominant label) cho mỗi mẫu."""
        pass

    def precdict(self, X_test: Any) -> np.ndarray:
        """Alias cho predict để tương thích ngược."""
        return self.predict(X_test)

    @abstractmethod
    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu trọng số mô hình ra file (vd: joblib, pickle)."""
        pass

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Any:
        """Tải mô hình đã huấn luyện từ file."""
        pass


# Alias giữ nguyên tên cũ có typo để code của thành viên khác không bị vỡ
BaseSentimentClassifer = BaseSentimentClassifier
