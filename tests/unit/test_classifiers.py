"""Unit tests for Classifier Subsystem (5 ML Models)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from source.model.classifier.algorithms.linear_models import (
    LinearSVCClassifier,
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
)
from source.model.classifier.algorithms.tree_models import (
    DecisionTreeModel,
    RandomForestModel,
)


@pytest.fixture
def dummy_train_data() -> tuple[csr_matrix, np.ndarray]:
    """Tạo dữ liệu huấn luyện giả lập 14 mẫu với 7 nhãn (mỗi nhãn 2 mẫu)."""
    np.random.seed(42)
    # 14 samples, 20 features
    X_dense = np.random.rand(14, 20)
    X = csr_matrix(X_dense)
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    return X, y


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (MultinomialNBClassifier, {"alpha": 1.0}),
        (LogisticRegressionClassifier, {"C": 1.0}),
        (LinearSVCClassifier, {"C": 1.0, "cv": 2}),
        (DecisionTreeModel, {"max_depth": 5}),
        (RandomForestModel, {"n_estimators": 10, "max_depth": 5}),
    ],
)
def test_all_models_lifecycle(
    dummy_train_data: tuple[csr_matrix, np.ndarray],
    model_cls: type,
    kwargs: dict,
    tmp_path: Path,
) -> None:
    """Kiểm tra train, predict_scores [N, 7], predict, save, load cho cả 5 mô hình."""
    X, y = dummy_train_data
    model = model_cls(**kwargs)

    # 1. Train
    model.train(X, y)

    # 2. Predict scores
    scores = model.predict_scores(X)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (14, 7)
    # Tổng xác suất mỗi hàng xấp xỉ 1.0
    row_sums = scores.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # 3. Predict dominant class
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 14

    # 4. Save and Load
    model_path = tmp_path / f"model_{model_cls.__name__}.joblib"
    model.save(model_path)
    assert model_path.exists()

    loaded = model_cls(**kwargs)
    loaded.load(model_path)
    loaded_scores = loaded.predict_scores(X)
    assert loaded_scores.shape == (14, 7)
