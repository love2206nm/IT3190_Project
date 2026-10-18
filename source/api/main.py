"""FastAPI Serving Application for UIT-VSMEC Sentiment Analysis.

Clean Architecture: Tầng Presentation mỏng nhẹ, điều phối 100% qua SentimentPipelineOrchestrator.
"""

from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from source.config import settings
from source.domain.entities.sentiment_sample import EmotionLabel
from source.services.cli import SentimentPipelineOrchestrator

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API phục vụ Thang đo phổ điểm 7 chiều cảm xúc tiếng Việt (UIT-VSMEC)",
)

# Khởi tạo bộ điều phối chính duy nhất từ tầng Services (nạp mô hình tốt nhất nếu có)
orchestrator = SentimentPipelineOrchestrator.build_default()


class PredictRequest(BaseModel):
    """Schema yêu cầu dự đoán cảm xúc."""

    text: str = Field(..., min_length=1, json_schema_extra={"example": "Hôm nay trời đẹp và vui quá!"})


class PredictResponse(BaseModel):
    """Schema kết quả Thang đo 7 chiều cảm xúc."""

    text: str
    scores: Dict[str, float] = Field(..., description="Thang đo điểm số [0.0, 1.0] của 7 cảm xúc")
    dominant_emotion: EmotionLabel = Field(..., description="Cảm xúc chiếm ưu thế nhất")
    intensity: float = Field(..., description="Cường độ của cảm xúc chiếm ưu thế")
    is_mixed: bool = Field(..., description="Cờ cảnh báo cảm xúc pha trộn")
    # Các trường tương thích ngược:
    primary_emotion: EmotionLabel
    confidence: float
    probabilities: Dict[str, float]


@app.get("/health", summary="Kiểm tra trạng thái server")
def health_check() -> Dict[str, str]:
    """Trả về trạng thái hoạt động của hệ thống."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
    }


@app.post("/api/v1/predict", response_model=PredictResponse, summary="Dự đoán thang đo phổ điểm 7 chiều cảm xúc")
def predict_sentiment(request: PredictRequest) -> PredictResponse:
    """Ước lượng thang đo phổ điểm 7 chiều cảm xúc UIT-VSMEC qua bộ điều phối trung tâm."""
    profile = orchestrator.analyze_text(request.text)

    return PredictResponse(
        text=request.text,
        scores=profile.scores,
        dominant_emotion=profile.dominant_emotion,
        intensity=profile.intensity,
        is_mixed=profile.is_mixed,
        primary_emotion=profile.dominant_emotion,
        confidence=profile.intensity,
        probabilities=profile.scores,
    )
