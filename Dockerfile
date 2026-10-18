# Multi-stage optimized Dockerfile for UIT-VSMEC Sentiment Analysis (Frontend UI & API)
FROM python:3.11-slim as base

# Ngăn Python tạo file .pyc và buffer output ra stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8501

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

# Expose cả cổng Streamlit Frontend (8501) và API (8000)
EXPOSE 8501 8000

# Healthcheck xác thực trạng thái Streamlit Frontend
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || curl -f http://localhost:${PORT:-8501}/ || exit 1

# Mặc định khởi chạy Frontend Web UI (source/ui/app.py) trên cổng PORT do Render cấp phát
CMD ["sh", "-c", "python -m streamlit run source/ui/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"]
