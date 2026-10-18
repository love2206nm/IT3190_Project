"""JSON Dataset Repository Implementation (Infrastructure Layer).

Thực thi đọc và ghi dữ liệu mẫu UIT-VSMEC từ các file JSON trên đĩa:
- Dữ liệu thô: data/raw_data/{split}.json
- Dữ liệu sạch: data/clean_data/{split}.json
Ghi log quá trình vào kênh LogChannel.DATA_PIPELINE.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from source.config import paths
from source.domain.interface.base_dataset_repo import BaseDatasetRepository
from source.infrastructure.logging import LogChannel, get_channel_logger

logger = get_channel_logger(LogChannel.DATA_PIPELINE, name="json_dataset_repo")


class JSONDatasetRepository(BaseDatasetRepository):
    """Kho lưu trữ đọc/ghi dữ liệu JSON cho bộ ngữ liệu UIT-VSMEC."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Khởi tạo với đường dẫn gốc dự án."""
        if project_root is None:
            self.root_dir = paths.root
        else:
            self.root_dir = project_root

        self.raw_data_dir = self.root_dir / "data" / "raw_data"
        self.clean_data_dir = self.root_dir / "data" / "clean_data"

        # Đảm bảo thư mục đích tồn tại
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Đọc tập dữ liệu thô từ data/raw_data/{split}.json."""
        file_path = self.raw_data_dir / f"{split}.json"
        if not file_path.exists():
            msg = f"Không tìm thấy file dữ liệu thô: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Bắt đầu đọc dữ liệu thô từ: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Đọc thành công tập thô '{split}': {len(data)} mẫu văn bản.")
        return data

    def save_clean_dataset(self, split: str, samples: List[Dict[str, Any]]) -> None:
        """Lưu tập dữ liệu đã làm sạch vào data/clean_data/{split}.json."""
        file_path = self.clean_data_dir / f"{split}.json"
        logger.info(f"Bắt đầu ghi {len(samples)} mẫu sạch vào: {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Ghi thành công tập sạch '{split}' với {len(samples)} mẫu.")

    def load_clean_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Đọc tập dữ liệu đã làm sạch từ data/clean_data/{split}.json."""
        file_path = self.clean_data_dir / f"{split}.json"
        if not file_path.exists():
            msg = f"Chưa có dữ liệu sạch cho split '{split}' tại: {file_path}. Hãy chạy tiền xử lý trước."
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Đọc dữ liệu sạch từ: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Đọc thành công tập sạch '{split}': {len(data)} mẫu văn bản.")
        return data
