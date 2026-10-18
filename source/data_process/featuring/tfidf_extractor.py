"""TF-IDF Feature Extractor (Data Process Layer).

Trích xuất ma trận đặc trưng TF-IDF N-grams (1, 2) cho văn bản tiếng Việt UIT-VSMEC.
Nguyên tắc chống rò rỉ dữ liệu (No Data Leakage):
- fit() DUY NHẤT trên tập huấn luyện (train.json).
- transform() trên tập validation, test và suy luận mẫu mới.
"""

import logging
from pathlib import Path
from typing import List, Union

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("data_pipeline.tfidf_extractor")


class TfidfFeatureExtractor:
    """Bộ trích xuất đặc trưng văn bản bằng TF-IDF N-grams (1, 2)."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ) -> None:
        """Khởi tạo cấu hình TfidfVectorizer."""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Kiểm tra vectorizer đã được fit hay chưa."""
        return self._is_fitted

    def fit(self, texts: List[str]) -> "TfidfFeatureExtractor":
        """Huấn luyện từ vựng TF-IDF trên tập văn bản (CHỈ DÙNG TRÊN TẬP TRAIN)."""
        logger.info(f"Bắt đầu fit TfidfVectorizer trên {len(texts)} văn bản...")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Fit hoàn tất. Kích thước từ vựng thực tế: {vocab_size} n-grams.")
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Chuyển đổi danh sách văn bản thành ma trận thưa TF-IDF CSR."""
        if not self._is_fitted:
            raise RuntimeError("TfidfVectorizer chưa được fit. Hãy gọi fit() trước khi transform().")
        matrix = self.vectorizer.transform(texts)
        return matrix

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Vừa fit vừa transform (tiện ích cho tập train)."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu trọng số và từ điển vectorizer ra file .joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"Đã lưu TfidfVectorizer vào: {path}")

    def load(self, file_path: Union[str, Path]) -> "TfidfFeatureExtractor":
        """Tải TfidfVectorizer đã lưu từ file .joblib."""
        path = Path(file_path)
        if not path.exists():
            msg = f"Không tìm thấy file TfidfVectorizer: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        self.vectorizer = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Đã nạp TfidfVectorizer từ: {path} ({len(self.vectorizer.vocabulary_)} n-grams)")
        return self
