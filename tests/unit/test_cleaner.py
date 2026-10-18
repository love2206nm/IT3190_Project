"""Unit tests for TextCleaner (Preprocessing Subsystem)."""

from source.data_process.preprocessing.text_cleaner import TextCleaner


def test_text_cleaner_teencode_normalization() -> None:
    """Kiểm tra chuẩn hóa teencode và từ viết tắt."""
    cleaner = TextCleaner()
    raw = "ko bt lm j vs mik het trơn"
    cleaned = cleaner.clean_text(raw)
    assert "không" in cleaned
    assert "biết" in cleaned
    assert "làm" in cleaned
    assert "gì" in cleaned
    assert "với" in cleaned
    assert "mình" in cleaned


def test_text_cleaner_emoji_to_semantic_tokens() -> None:
    """Kiểm tra chuyển đổi emoji sang semantic tokens."""
    cleaner = TextCleaner()
    raw_enjoy = "Món ăn ngon quá ❤️ :))"
    clean_enjoy = cleaner.clean_text(raw_enjoy)
    assert "icon_vui" in clean_enjoy

    raw_sad = "Buồn quá trời 😭😭 :("
    clean_sad = cleaner.clean_text(raw_sad)
    assert "icon_buon" in clean_sad

    raw_anger = "Thái độ lồi lõm 😡 😠"
    clean_anger = cleaner.clean_text(raw_anger)
    assert "icon_tuc_gian" in clean_anger


def test_text_cleaner_repeated_letters_shortening() -> None:
    """Kiểm tra rút gọn ký tự kéo dài."""
    cleaner = TextCleaner()
    raw = "Ăn ngonnnn lắm buồn nnnnn"
    cleaned = cleaner.clean_text(raw)
    assert "ngon" in cleaned
    assert "buồn" in cleaned


def test_text_cleaner_urls_and_empty_fallback() -> None:
    """Kiểm tra lọc URL và fallback chuỗi rỗng."""
    cleaner = TextCleaner()
    raw_url = "Xem thêm tại https://facebook.com/review và www.test.vn"
    cleaned_url = cleaner.clean_text(raw_url)
    assert "http" not in cleaned_url
    assert "www" not in cleaned_url

    raw_empty = "!@#$%^&*"
    cleaned_empty = cleaner.clean_text(raw_empty)
    assert cleaned_empty == "văn_bản_rỗng"
