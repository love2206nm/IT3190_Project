"""Linear and Probabilistic Sentiment Classifiers (Model Layer).

Cài đặt 3 mô hình tuyến tính và xác suất kế thừa BaseSentimentClassifier:
1. MultinomialNBClassifier: Mô hình Naive Bayes đa thức (Baseline tốc độ cao).
2. LogisticRegressionClassifier: Hồi quy Logistic đa lớp (class_weight='balanced').
3. LinearSVCClassifier: Support Vector Machine tuyến tính bọc CalibratedClassifierCV để xuất xác suất 7 chiều.
"""

import logging
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from source.domain.interface.base_classifer import BaseSentimentClassifier

logger = logging.getLogger("model_training.linear_models")


class MultinomialNBClassifier(BaseSentimentClassifier):
    """Mô hình Naive Bayes đa thức cho bài toán phân loại cảm xúc 7 lớp."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Khởi tạo với tham số làm mịn Laplace alpha."""
        self.alpha = alpha
        self.model = MultinomialNB(alpha=self.alpha)
        self._is_trained = False

    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện Naive Bayes trên ma trận đặc trưng và nhãn."""
        logger.info(f"Huấn luyện MultinomialNB (alpha={self.alpha}) trên {X_train.shape[0]} mẫu...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Huấn luyện MultinomialNB hoàn tất.")

    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận xác suất [N, 7] chuẩn hóa tổng hàng bằng 1.0."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        probs = self.model.predict_proba(X_test)
        # Chuẩn hóa an toàn tổng hàng = 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc số nguyên có xác suất cao nhất."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        return self.model.predict(X_test)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu trọng số mô hình ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "alpha": self.alpha}, path)
        logger.info(f"Đã lưu MultinomialNB vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "MultinomialNBClassifier":
        """Tải mô hình đã lưu từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.alpha = data.get("alpha", 1.0)
        self._is_trained = True
        logger.info(f"Đã tải MultinomialNB từ: {path}")
        return self


class LogisticRegressionClassifier(BaseSentimentClassifier):
    """Mô hình Hồi quy Logistic đa lớp có trọng số cân bằng nhãn."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Union[str, dict, None] = "balanced",
    ) -> None:
        """Khởi tạo Logistic Regression với tham số điều chuẩn C."""
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
            random_state=42,
        )
        self._is_trained = False

    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện Logistic Regression trên ma trận đặc trưng và nhãn."""
        logger.info(f"Huấn luyện LogisticRegression (C={self.C}) trên {X_train.shape[0]} mẫu...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Huấn luyện LogisticRegression hoàn tất.")

    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận xác suất [N, 7]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        probs = self.model.predict_proba(X_test)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc số nguyên [0..6]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        return self.model.predict(X_test)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu trọng số mô hình ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "C": self.C}, path)
        logger.info(f"Đã lưu LogisticRegression vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "LogisticRegressionClassifier":
        """Tải mô hình đã lưu từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.C = data.get("C", 1.0)
        self._is_trained = True
        logger.info(f"Đã tải LogisticRegression từ: {path}")
        return self


class LinearSVCClassifier(BaseSentimentClassifier):
    """Mô hình Linear Support Vector Machine bọc CalibratedClassifierCV để ước lượng xác suất."""

    def __init__(
        self,
        C: float = 1.0,
        cv: int = 3,
        class_weight: Union[str, dict, None] = "balanced",
    ) -> None:
        """Khởi tạo LinearSVC bọc CalibratedClassifierCV."""
        self.C = C
        self.cv = cv
        self.class_weight = class_weight
        self.base_svc = LinearSVC(
            C=self.C,
            loss="squared_hinge",
            class_weight=self.class_weight,
            random_state=42,
            max_iter=3000,
        )
        self.calibrated_model = CalibratedClassifierCV(estimator=self.base_svc, cv=self.cv)
        self._is_trained = False

    def train(self, X_train: Any, y_train: Any) -> None:
        """Huấn luyện LinearSVC kèm hiệu chuẩn xác suất Platt Scaling."""
        logger.info(f"Huấn luyện Calibrated LinearSVC (C={self.C}, cv={self.cv}) trên {X_train.shape[0]} mẫu...")
        self.calibrated_model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Huấn luyện Calibrated LinearSVC hoàn tất.")

    def predict_scores(self, X_test: Any) -> np.ndarray:
        """Trả về ma trận xác suất [N, 7]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        probs = self.calibrated_model.predict_proba(X_test)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, X_test: Any) -> np.ndarray:
        """Dự đoán nhãn cảm xúc số nguyên [0..6]."""
        if not self._is_trained:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi train() trước.")
        return self.calibrated_model.predict(X_test)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu trọng số mô hình ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"calibrated_model": self.calibrated_model, "C": self.C}, path)
        logger.info(f"Đã lưu Calibrated LinearSVC vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "LinearSVCClassifier":
        """Tải mô hình đã lưu từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        data = joblib.load(path)
        self.calibrated_model = data["calibrated_model"]
        self.C = data.get("C", 1.0)
        self._is_trained = True
        logger.info(f"Đã tải Calibrated LinearSVC từ: {path}")
        return self
