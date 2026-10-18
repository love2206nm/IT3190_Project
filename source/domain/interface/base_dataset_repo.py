"""Base Dataset Repository Interface (Domain Layer).

Quy định hợp đồng trừu tượng cho việc đọc và ghi dữ liệu mẫu văn bản cảm xúc (UIT-VSMEC).
Tuân thủ Clean Architecture: Domain không phụ thuộc vào tầng ngoài hay hệ thống lưu trữ vật lý.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDatasetRepository(ABC):
    """Giao diện trừu tượng cho kho lưu trữ dữ liệu tập mẫu UIT-VSMEC."""

    @abstractmethod
    def load_raw_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Đọc tập dữ liệu thô ban đầu theo phân chia ('train', 'valid', 'test')."""
        pass

    @abstractmethod
    def save_clean_dataset(self, split: str, samples: List[Dict[str, Any]]) -> None:
        """Lưu tập dữ liệu đã làm sạch theo phân chia ('train', 'valid', 'test')."""
        pass

    @abstractmethod
    def load_clean_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Đọc tập dữ liệu đã làm sạch theo phân chia ('train', 'valid', 'test')."""
        pass
