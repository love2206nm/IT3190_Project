"""Text Cleaner Implementation (Data Processing Layer).

Kế thừa BasePreprocess từ Domain Layer:
1. Chuẩn hóa bảng mã Unicode tiếng Việt sang chuẩn NFC.
2. Chuyển chữ thường (lowercase).
3. Loại bỏ URLs, email, thẻ HTML.
4. Ánh xạ Emoji & Emoticon sang Semantic Tokens (icon_vui, icon_buon, ...).
5. Rút gọn ký tự lặp kéo dài cảm xúc (buồnnnn -> buồn, ngonnn -> ngon).
6. Thay thế từ điển Teencode và Viết tắt tiếng Việt (TEENCODE_DICT).
7. Loại bỏ ký tự đặc biệt vô nghĩa, bảo vệ chữ tiếng Việt có dấu và token icon.
8. Chuẩn hóa khoảng trắng.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List

from source.data_process.preprocessing.emoji_dict import EMOJI_DICT
from source.data_process.preprocessing.teencode_dict import TEENCODE_DICT
from source.domain.interface.base_preprocess import BasePreprocess

logger = logging.getLogger("data_pipeline.text_cleaner")


class TextCleaner(BasePreprocess):
    """Bộ tiền xử lý và làm sạch văn bản mạng xã hội tiếng Việt UIT-VSMEC."""

    def __init__(self) -> None:
        """Khởi tạo các biểu thức chính quy (Regex patterns) biên dịch sẵn."""
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
        self._email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self._html_pattern = re.compile(r"<.*?>")
        # Rút gọn 3+ ký tự liên tiếp về 1 ký tự (ngonnnn -> ngon, buồnnnn -> buồn)
        self._repeat_pattern = re.compile(r"([a-zà-ỹ])\1{2,}", flags=re.IGNORECASE)
        # Giữ lại chữ tiếng Việt có dấu, số, khoảng trắng và dấu gạch dưới (cho token icon_*)
        self._special_char_pattern = re.compile(
            r"[^a-z0-9\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]"
        )
        self._space_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """Làm sạch và chuẩn hóa một chuỗi văn bản đơn lẻ.

        Args:
            text: Văn bản thô ban đầu

        Returns:
            Văn bản đã qua quy chuẩn làm sạch, sẵn sàng đưa vào TF-IDF
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Chuẩn hóa Unicode NFC
        clean = unicodedata.normalize("NFC", text)

        # 2. Xóa URL, Email, HTML
        clean = self._url_pattern.sub(" ", clean)
        clean = self._email_pattern.sub(" ", clean)
        clean = self._html_pattern.sub(" ", clean)

        # 3. Ánh xạ Emoji & Emoticons trước khi lowercase hoặc lọc ký tự
        for emoji_char, token in EMOJI_DICT.items():
            if emoji_char in clean:
                clean = clean.replace(emoji_char, f" {token} ")

        # 4. Lowercase
        clean = clean.lower()

        # 5. Rút gọn ký tự lặp kéo dài
        clean = self._repeat_pattern.sub(r"\1", clean)

        # 6. Thay thế Teencode & Viết tắt qua từ điển
        words = clean.split()
        normalized_words = [TEENCODE_DICT.get(w, w) for w in words]
        clean = " ".join(normalized_words)

        # 7. Lọc ký tự đặc biệt dư thừa
        clean = self._special_char_pattern.sub(" ", clean)

        # 8. Chuẩn hóa khoảng trắng
        clean = self._space_pattern.sub(" ", clean).strip()

        # Fallback tránh câu rỗng hoàn toàn sau khi làm sạch
        if not clean:
            clean = "văn_bản_rỗng"

        return clean

    def clean_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Làm sạch toàn bộ tập dữ liệu mẫu UIT-VSMEC.

        Args:
            samples: Danh sách các dict [{"Sentence": "...", "Emotion": "..."}]

        Returns:
            Danh sách các dict chứa câu đã làm sạch
        """
        logger.info(f"Bắt đầu làm sạch tập dữ liệu quy mô {len(samples)} mẫu.")
        cleaned_samples: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples):
            raw_sentence = sample.get("Sentence", "")
            emotion = sample.get("Emotion", "Other")

            clean_sentence = self.clean_text(raw_sentence)
            cleaned_samples.append(
                {
                    "Sentence": clean_sentence,
                    "Emotion": emotion,
                }
            )

            if (idx + 1) % 1000 == 0 or (idx + 1) == len(samples):
                logger.info(f"Đã xử lý: {idx + 1}/{len(samples)} mẫu...")

        logger.info("Hoàn tất làm sạch tập dữ liệu.")
        return cleaned_samples
