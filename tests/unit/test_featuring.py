"""Unit tests for Featuring Subsystem (TfidfFeatureExtractor & LabelEncoderHelper)."""

from pathlib import Path

import numpy as np

from source.data_process.featuring.label_encoder import (
    ORDERED_EMOTIONS,
    LabelEncoderHelper,
)
from source.data_process.featuring.tfidf_extractor import TfidfFeatureExtractor


def test_label_encoder_bidirectional_mapping() -> None:
    """Kiểm tra mã hóa và giải mã 7 nhãn cảm xúc."""
    assert len(ORDERED_EMOTIONS) == 7

    encoded = LabelEncoderHelper.encode_labels(["Enjoyment", "Anger", "Other"])
    assert isinstance(encoded, np.ndarray)
    assert len(encoded) == 3
    assert encoded[0] == 0
    assert encoded[1] == 2
    assert encoded[2] == 6

    decoded = LabelEncoderHelper.decode_labels(encoded)
    assert decoded == ["Enjoyment", "Anger", "Other"]


def test_tfidf_feature_extractor_lifecycle(tmp_path: Path) -> None:
    """Kiểm tra vòng đời fit, transform, save, load của TfidfFeatureExtractor."""
    corpus = [
        "hôm nay hoàn thành bài tập vui quá",
        "thật là thất vọng buồn bã quá",
        "dịch vụ tệ hại tức giận",
        "sợ hãi không dám đi một mình",
    ]

    extractor = TfidfFeatureExtractor(max_features=100, min_df=1)
    matrix = extractor.fit_transform(corpus)

    assert matrix.shape[0] == len(corpus)
    assert matrix.shape[1] > 0
    assert extractor.is_fitted

    # Save and Load
    save_path = tmp_path / "tfidf.joblib"
    extractor.save(save_path)

    loaded_extractor = TfidfFeatureExtractor()
    loaded_extractor.load(save_path)
    assert loaded_extractor.is_fitted

    new_matrix = loaded_extractor.transform(["vui quá hôm nay"])
    assert new_matrix.shape[1] == matrix.shape[1]
