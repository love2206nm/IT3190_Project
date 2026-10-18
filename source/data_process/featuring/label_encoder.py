"""Label Encoder Helper (Data Process Layer).

Mã hóa cố định 7 nhãn cảm xúc UIT-VSMEC giữa dạng Chuỗi (String) và Số nguyên (Integer [0..6]).
Đảm bảo đồng nhất tuyệt đối giữa khâu huấn luyện, đánh giá và suy luận thực tế.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from source.domain.entities.sentiment_sample import EmotionLabel

# Danh sách 7 nhãn chuẩn theo thứ tự cố định
ORDERED_EMOTIONS: List[str] = [
    EmotionLabel.ENJOYMENT.value,  # 0
    EmotionLabel.DISGUST.value,  # 1
    EmotionLabel.ANGER.value,  # 2
    EmotionLabel.SADNESS.value,  # 3
    EmotionLabel.FEAR.value,  # 4
    EmotionLabel.SURPRISE.value,  # 5
    EmotionLabel.OTHER.value,  # 6
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(ORDERED_EMOTIONS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(ORDERED_EMOTIONS)}


class LabelEncoderHelper:
    """Công cụ hỗ trợ mã hóa nhãn cảm xúc 7 lớp."""

    @staticmethod
    def get_labels() -> List[str]:
        """Lấy danh sách 7 tên nhãn chuẩn."""
        return list(ORDERED_EMOTIONS)

    @staticmethod
    def label_to_id(label: str) -> int:
        """Chuyển đổi tên nhãn chuỗi sang id số nguyên [0..6]."""
        clean_label = label.strip().capitalize()
        return LABEL_TO_ID.get(clean_label, LABEL_TO_ID[EmotionLabel.OTHER.value])

    @staticmethod
    def id_to_label(id_: int) -> str:
        """Chuyển đổi id số nguyên sang tên nhãn chuỗi."""
        return ID_TO_LABEL.get(id_, EmotionLabel.OTHER.value)

    @classmethod
    def encode_labels(cls, labels: List[str]) -> np.ndarray:
        """Mã hóa một danh sách nhãn chuỗi thành mảng numpy số nguyên 1D."""
        return np.array([cls.label_to_id(lbl) for lbl in labels], dtype=np.int64)

    @classmethod
    def decode_labels(cls, ids: Union[np.ndarray, List[int]]) -> List[str]:
        """Giải mã mảng id số nguyên thành danh sách nhãn chuỗi."""
        return [cls.id_to_label(int(i)) for i in ids]

    @staticmethod
    def export_mapping(file_path: Union[str, Path]) -> None:
        """Xuất metadata ánh xạ ra file JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ordered_labels": ORDERED_EMOTIONS,
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
