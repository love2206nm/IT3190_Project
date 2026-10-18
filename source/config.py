"""Configuration and Path Management for IT3190 Project.

Provides centralized paths and project settings to prevent hardcoding.
Positioned at root source level to be cleanly accessible across layers.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Tự động nạp biến môi trường từ .env nếu có
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class ProjectPaths:
    """Định nghĩa tất cả đường dẫn chuẩn trong dự án."""

    root: Path = BASE_DIR
    source: Path = BASE_DIR / "source"
    data: Path = BASE_DIR / "data"
    data_raw: Path = BASE_DIR / "data" / "raw_data"
    data_clean: Path = BASE_DIR / "data" / "clean_data"
    models_saved: Path = BASE_DIR / "source" / "model" / "classifier" / "saved_models"
    docs: Path = BASE_DIR / "docs"
    figures: Path = BASE_DIR / "docs" / "figures"
    tests: Path = BASE_DIR / "tests"


@dataclass(frozen=True)
class Settings:
    """Cấu hình ứng dụng toàn cục."""

    app_name: str = "UIT-VSMEC Sentiment Analysis"
    app_version: str = "1.0.0"
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton instances
paths = ProjectPaths()
settings = Settings()
