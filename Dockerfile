# Multi-stage optimized Dockerfile for UIT-VSMEC Sentiment Analysis
FROM python:3.11-slim as base

# Ngăn Python tạo file .pyc và buffer output ra stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000

WORKDIR /app

# Cài đặt các gói hệ thống tối thiểu cần thiết để biên dịch (nếu có C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies trước để tối ưu hóa Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tạo user không có đặc quyền root để bảo mật
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Sao chép mã nguồn ứng dụng, cấu hình và trọng số mô hình đã huấn luyện
COPY --chown=appuser:appuser source/ ./source/
COPY --chown=appuser:appuser config.py ./config.py
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml
COPY --chown=appuser:appuser .env.example ./.env.example

USER appuser

# Expose cả cổng API (8000) và Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck xác thực trạng thái container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Mặc định chạy FastAPI service (Tương thích chuẩn Render Web Service)
CMD ["sh", "-c", "uvicorn source.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
