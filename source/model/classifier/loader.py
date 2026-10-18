"""Model Loader Utility (Model Layer).

Cung cấp hàm nạp tự động bất kỳ mô hình phân loại cảm xúc nào được lưu trữ dưới định dạng .joblib
thành đối tượng kế thừa BaseSentimentClassifier tương ứng.
"""

from pathlib import Path
from typing import Any, Dict, Union

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from source.domain.interface.base_classifer import BaseSentimentClassifier
from source.model.classifier.algorithms.linear_models import (
    LinearSVCClassifier,
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
)
from source.model.classifier.algorithms.tree_models import (
    DecisionTreeModel,
    RandomForestModel,
)


def load_saved_classifier(file_path: Union[str, Path]) -> BaseSentimentClassifier:
    """Tải và khởi tạo đúng lớp Classifier bọc ngoài từ file .joblib đã lưu."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {path}")

    data: Dict[str, Any] = joblib.load(path)

    if "calibrated_model" in data:
        svc = LinearSVCClassifier(C=data.get("C", 1.0))
        svc.calibrated_model = data["calibrated_model"]
        svc._is_trained = True
        return svc

    inner_model = data.get("model")
    if inner_model is None:
        raise ValueError(f"File mô hình {path} không chứa đối tượng hợp lệ.")

    if isinstance(inner_model, LogisticRegression):
        lr = LogisticRegressionClassifier(C=data.get("C", 1.0))
        lr.model = inner_model
        lr._is_trained = True
        return lr

    if isinstance(inner_model, MultinomialNB):
        nb = MultinomialNBClassifier(alpha=data.get("alpha", 1.0))
        nb.model = inner_model
        nb._is_trained = True
        return nb

    if isinstance(inner_model, DecisionTreeClassifier):
        dt = DecisionTreeModel(max_depth=data.get("max_depth", 20))
        dt.model = inner_model
        dt._is_trained = True
        return dt

    if isinstance(inner_model, RandomForestClassifier):
        rf = RandomForestModel(n_estimators=data.get("n_estimators", 100))
        rf.model = inner_model
        rf._is_trained = True
        return rf

    raise TypeError(f"Không nhận diện được loại mô hình: {type(inner_model)}")
