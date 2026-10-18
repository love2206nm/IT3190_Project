"""Tree and Ensemble Sentiment Classifiers (Model Layer).

Cài đặt 2 mô hình phi tuyến tính và quần thể cây:
1. DecisionTreeModel: Cây quyết định đơn lẻ có khống chế độ sâu chống overfit.
2. RandomForestModel: Quần thể Bagging 100 cây quyết định.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from source.domain.interface.base_classifer import BaseSentimentClassifier

logger = logging.getLogger("model_training.tree_models")


class DecisionTreeModel(BaseSentimentClassifier):
    """Mô hình Cây quyết định (Decision Tree) cho phân loại cảm xúc 7 lớp."""

    def __init__(
        self,
        max_depth: Optional[int] = 25,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: Union[str, dict, None] = "balanced",
    ) -> None:
        """Khởi tạo DecisionTree với các tham số khống chế độ phức tạp."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=42,
        )
        self._is_trained = False

    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện cây quyết định."""
        logger.info(f"Huấn luyện DecisionTree (max_depth={self.max_depth}) trên {X_train.shape[0]} mẫu...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Huấn luyện DecisionTree hoàn tất.")

    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận xác suất [N, 7]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        probs = self.model.predict_proba(X_test)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc [0..6]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        return self.model.predict(X_test)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu cây quyết định ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "max_depth": self.max_depth}, path)
        logger.info(f"Đã lưu DecisionTree vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "DecisionTreeModel":
        """Tải cây quyết định từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.max_depth = data.get("max_depth", 25)
        self._is_trained = True
        logger.info(f"Đã tải DecisionTree từ: {path}")
        return self


class RandomForestModel(BaseSentimentClassifier):
    """Mô hình Quần thể Rừng ngẫu nhiên (Random Forest) với 100 cây Bagging."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 30,
        class_weight: Union[str, dict, None] = "balanced",
        n_jobs: int = -1,
    ) -> None:
        """Khởi tạo RandomForestClassifier."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self._is_trained = False

    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện quần thể rừng ngẫu nhiên."""
        logger.info(
            f"Huấn luyện RandomForest ({self.n_estimators} trees, max_depth={self.max_depth}) "
            f"trên {X_train.shape[0]} mẫu..."
        )
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Huấn luyện RandomForest hoàn tất.")

    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận xác suất trung bình từ 100 cây [N, 7]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        probs = self.model.predict_proba(X_test)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc [0..6]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        return self.model.predict(X_test)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu RandomForest ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "n_estimators": self.n_estimators}, path)
        logger.info(f"Đã lưu RandomForest vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "RandomForestModel":
        """Tải RandomForest từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.n_estimators = data.get("n_estimators", 100)
        self._is_trained = True
        logger.info(f"Đã tải RandomForest từ: {path}")
        return self
