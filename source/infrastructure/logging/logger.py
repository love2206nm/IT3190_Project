"""Multi-channel Structured Logging Subsystem (Infrastructure Layer).

Cung cấp hệ thống ghi log đa kênh tách biệt hoàn toàn giữa các nghiệp vụ:
- DATA_PIPELINE -> logs/data_pipeline.log
- MODEL_TRAINING -> logs/model_training.log
- APP_SERVICE -> logs/app_service.log
- TEST_EXECUTION -> logs/test_execution.log
"""

import logging
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class LogChannel(str, Enum):
    """Các kênh logging tách biệt theo ranh giới nghiệp vụ."""

    DATA_PIPELINE = "data_pipeline"
    MODEL_TRAINING = "model_training"
    APP_SERVICE = "app_service"
    TEST_EXECUTION = "test_execution"


CHANNEL_FILE_MAP: Dict[LogChannel, str] = {
    LogChannel.DATA_PIPELINE: "data_pipeline.log",
    LogChannel.MODEL_TRAINING: "model_training.log",
    LogChannel.APP_SERVICE: "app_service.log",
    LogChannel.TEST_EXECUTION: "test_execution.log",
}

_LOGGERS_CACHE: Dict[str, logging.Logger] = {}


def ensure_logs_directory() -> Path:
    """Đảm bảo thư mục logs/ tồn tại và có file .gitkeep."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep_file = LOGS_DIR / ".gitkeep"
    if not gitkeep_file.exists():
        gitkeep_file.touch(exist_ok=True)
    return LOGS_DIR


def get_channel_logger(
    channel: LogChannel,
    name: str = "",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Khởi tạo hoặc lấy Logger theo kênh chuyên biệt với RotatingFileHandler."""
    logger_name = f"{channel.value}.{name}" if name else channel.value

    if logger_name in _LOGGERS_CACHE:
        return _LOGGERS_CACHE[logger_name]

    ensure_logs_directory()
    log_filename = CHANNEL_FILE_MAP.get(channel, f"{channel.value}.log")
    log_filepath = LOGS_DIR / log_filename

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=str(log_filepath),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    _LOGGERS_CACHE[logger_name] = logger
    return logger
