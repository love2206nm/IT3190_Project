"""Pure Domain Entities for Sentiment Analysis (UIT-VSMEC).

Clean Architecture: Tầng Domain độc lập hoàn toàn, không phụ thuộc thư viện ngoài.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EmotionLabel(str, Enum):
    """7 nhãn cảm xúc chuẩn của tập dữ liệu UIT-VSMEC."""

    ENJOYMENT = "Enjoyment"
    SADNESS = "Sadness"
    ANGER = "Anger"
    FEAR = "Fear"
    DISGUST = "Disgust"
    SURPRISE = "Surprise"
    OTHER = "Other"

    @classmethod
    def from_string(cls, value: str) -> "EmotionLabel":
        """Chuyển đổi chuỗi (không phân biệt hoa thường) thành EmotionLabel."""
        val_clean = value.strip().capitalize()
        for member in cls:
            if member.value.lower() == val_clean.lower():
                return member
        return cls.OTHER


@dataclass(frozen=True)
class CleanedTextSample:
    """Mẫu văn bản sau khi qua khối tiền xử lý."""

    raw_text: str
    cleaned_text: str
    label: Optional[EmotionLabel] = None


@dataclass(frozen=True)
class EmotionScoreProfile:
    """Thang đo phổ điểm 7 chiều cảm xúc cho một câu văn bản (UIT-VSMEC)."""

    text: str
    scores: Dict[str, float]

    @property
    def dominant_emotion(self) -> EmotionLabel:
        """Cảm xúc chiếm ưu thế nhất (argmax điểm số)."""
        top_name = max(self.scores.keys(), key=lambda k: self.scores[k])
        return EmotionLabel.from_string(top_name)

    @property
    def intensity(self) -> float:
        """Cường độ (điểm số) của cảm xúc chiếm ưu thế."""
        return self.scores.get(self.dominant_emotion.value, 0.0)

    @property
    def is_mixed(self) -> bool:
        """Xác định cảm xúc pha trộn (nếu nhãn cao thứ nhì đạt ít nhất 20%)."""
        sorted_vals = sorted(self.scores.values(), reverse=True)
        return len(sorted_vals) > 1 and sorted_vals[1] >= 0.20

    @property
    def ranking(self) -> List[Tuple[EmotionLabel, float]]:
        """Danh sách 7 cảm xúc xếp thứ tự điểm số từ cao xuống thấp."""
        sorted_items = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return [(EmotionLabel.from_string(name), score) for name, score in sorted_items]


@dataclass(frozen=True)
class SentimentPrediction:
    """Kết quả phân loại cảm xúc (Tương thích ngược)."""

    text: str
    primary_emotion: EmotionLabel
    confidence: float
    probabilities: Dict[str, float]
