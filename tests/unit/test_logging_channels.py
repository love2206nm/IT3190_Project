"""Unit tests for Structured Multi-channel Logging Subsystem."""

from pathlib import Path

from source.infrastructure.logging import LogChannel, get_channel_logger


def test_channel_loggers_create_distinct_files(tmp_path: Path) -> None:
    """Kiểm tra 4 kênh logging ghi vào 4 file tách biệt."""
    for channel in LogChannel:
        logger = get_channel_logger(channel, name=f"test_{channel.value}")
        test_msg = f"Test message for channel {channel.value}"
        logger.info(test_msg)

        # Kiểm tra file log tương ứng tồn tại trong thư mục logs
        root = Path(__file__).resolve().parent.parent.parent
        log_file = root / "logs" / f"{channel.value}.log"
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert test_msg in content
