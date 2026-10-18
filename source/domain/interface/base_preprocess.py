"""Base Preprocessing Interface (Domain Layer).

Quy định hợp đồng chuẩn cho khối làm sạch dữ liệu văn bản tiếng Việt.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePreprocess(ABC):
    """Lớp trừu tượng cho các thuật toán tiền xử lý văn bản."""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Làm sạch và chuẩn hóa một mẫu văn bản đơn lẻ."""
        pass

    @abstractmethod
    def clean_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Làm sạch và chuẩn hóa toàn bộ tập dữ liệu mẫu."""
        pass
