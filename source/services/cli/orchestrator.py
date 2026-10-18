"""Sentiment Pipeline Orchestrator (Services Layer).

Đóng vai trò bộ não điều phối trung tâm của toàn bộ hệ thống:
Tương tác trực tiếp với Domain Layer qua các lớp trừu tượng (BasePreprocess, BaseSentimentClassifier).
Điều phối: Raw text -> TextCleaner -> TfidfFeatureExtractor -> Classifier -> EmotionScoreProfile.
"""

from pathlib import Path
from typing import Dict, List, Optional

from source.data_process.featuring.label_encoder import ORDERED_EMOTIONS
from source.data_process.featuring.tfidf_extractor import TfidfFeatureExtractor
from source.data_process.preprocessing.text_cleaner import TextCleaner
from source.domain.entities.sentiment_sample import EmotionLabel, EmotionScoreProfile
from source.domain.interface.base_classifer import BaseSentimentClassifier
from source.domain.interface.base_preprocess import BasePreprocess
from source.infrastructure.logging import LogChannel, get_channel_logger
from source.model.classifier.loader import load_saved_classifier

logger = get_channel_logger(LogChannel.APP_SERVICE, name="orchestrator")


class SentimentPipelineOrchestrator:
    """Bộ điều phối chính của hệ thống phân loại cảm xúc UIT-VSMEC."""

    def __init__(
        self,
        classifier: Optional[BaseSentimentClassifier] = None,
        preprocessor: Optional[BasePreprocess] = None,
        vectorizer: Optional[TfidfFeatureExtractor] = None,
    ) -> None:
        """Khởi tạo Orchestrator với Dependency Injection."""
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer

    @classmethod
    def build_default(cls, project_root: Optional[Path] = None) -> "SentimentPipelineOrchestrator":
        """Khởi tạo Orchestrator mặc định, tự động nạp best_model.joblib nếu có."""
        if project_root is None:
            root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            root = project_root

        saved_models_dir = root / "source" / "model" / "classifier" / "saved_models"
        best_model_path = saved_models_dir / "best_model.joblib"
        tfidf_path = saved_models_dir / "tfidf_vectorizer.joblib"

        preprocessor = TextCleaner()
        classifier = None
        vectorizer = None

        if best_model_path.exists() and tfidf_path.exists():
            try:
                # Nạp TF-IDF
                vec = TfidfFeatureExtractor()
                vec.load(tfidf_path)
                vectorizer = vec

                # Nạp mô hình vô địch đã lưu
                classifier = load_saved_classifier(best_model_path)
                logger.info(f"Orchestrator đã nạp thành công mô hình thật từ: {best_model_path}")
            except Exception as e:
                logger.warning(f"Không thể nạp mô hình đã lưu: {e}. Sử dụng fallback rule-based.")
                classifier = None
                vectorizer = None
        else:
            logger.info("Chưa tìm thấy best_model.joblib. Orchestrator hoạt động ở chế độ heuristic fallback.")

        return cls(classifier=classifier, preprocessor=preprocessor, vectorizer=vectorizer)

    def analyze_text(self, text: str) -> EmotionScoreProfile:
        """Điều phối toàn bộ đường ống: Text thô -> Làm sạch -> Vectorize -> Thang đo 7 chiều."""
        cleaned_text = self.preprocessor.clean_text(text) if self.preprocessor else text.strip()

        if self.classifier:
            if self.vectorizer and self.vectorizer.is_fitted:
                feat_matrix = self.vectorizer.transform([cleaned_text])
                scores_matrix = self.classifier.predict_scores(feat_matrix)
            else:
                # Hỗ trợ MockClassifier trong unit test nhận raw list
                scores_matrix = self.classifier.predict_scores([cleaned_text])

            labels = ORDERED_EMOTIONS
            scores_dict = {lbl: float(scores_matrix[0][i]) for i, lbl in enumerate(labels)}
        else:
            scores_dict = self._fallback_heuristic_scoring(cleaned_text)

        return EmotionScoreProfile(text=text, scores=scores_dict)

    def batch_analyze(self, texts: List[str]) -> List[EmotionScoreProfile]:
        """Điều phối phân tích hàng loạt mẫu văn bản."""
        return [self.analyze_text(t) for t in texts]

    def _fallback_heuristic_scoring(self, text: str) -> Dict[str, float]:
        """Quy tắc tính điểm dự phòng khi chưa nạp trọng số mô hình đã huấn luyện."""
        lower_text = text.lower()
        if any(w in lower_text for w in ["vui", "sướng", "tuyệt", "thích", "haha", "yêu", "icon_vui"]):
            return {
                EmotionLabel.ENJOYMENT.value: 0.82,
                EmotionLabel.SURPRISE.value: 0.08,
                EmotionLabel.OTHER.value: 0.04,
                EmotionLabel.SADNESS.value: 0.02,
                EmotionLabel.FEAR.value: 0.02,
                EmotionLabel.ANGER.value: 0.01,
                EmotionLabel.DISGUST.value: 0.01,
            }
        if any(w in lower_text for w in ["buồn", "khóc", "chán", "thất vọng", "cô đơn", "icon_buon"]):
            return {
                EmotionLabel.SADNESS.value: 0.78,
                EmotionLabel.FEAR.value: 0.09,
                EmotionLabel.OTHER.value: 0.05,
                EmotionLabel.ANGER.value: 0.04,
                EmotionLabel.DISGUST.value: 0.02,
                EmotionLabel.SURPRISE.value: 0.01,
                EmotionLabel.ENJOYMENT.value: 0.01,
            }
        if any(w in lower_text for w in ["tức", "bực", "chửi", "điên", "ghét", "icon_tuc_gian"]):
            return {
                EmotionLabel.ANGER.value: 0.75,
                EmotionLabel.DISGUST.value: 0.16,
                EmotionLabel.OTHER.value: 0.04,
                EmotionLabel.SADNESS.value: 0.02,
                EmotionLabel.FEAR.value: 0.02,
                EmotionLabel.SURPRISE.value: 0.01,
                EmotionLabel.ENJOYMENT.value: 0.00,
            }
        return {e.value: round(1.0 / 7, 3) for e in EmotionLabel}
