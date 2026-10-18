"""Unit tests for JSONDatasetRepository (Infrastructure Layer)."""

from pathlib import Path

from source.infrastructure.database.repo.json_dataset_repo import JSONDatasetRepository


def test_dataset_repository_load_raw() -> None:
    """Kiểm tra đọc dữ liệu thô train/valid/test từ đĩa."""
    repo = JSONDatasetRepository()
    train_data = repo.load_raw_dataset("train")
    valid_data = repo.load_raw_dataset("valid")
    test_data = repo.load_raw_dataset("test")

    assert len(train_data) == 5548
    assert len(valid_data) == 686
    assert len(test_data) == 693
    assert "Sentence" in train_data[0]
    assert "Emotion" in train_data[0]


def test_dataset_repository_save_and_load_clean(tmp_path: Path) -> None:
    """Kiểm tra lưu và nạp dữ liệu sạch qua mock repo directory."""
    repo = JSONDatasetRepository(project_root=tmp_path)
    sample_data = [{"Sentence": "câu test sạch", "Emotion": "Enjoyment"}]

    repo.save_clean_dataset("test_split", sample_data)
    loaded = repo.load_clean_dataset("test_split")

    assert len(loaded) == 1
    assert loaded[0]["Sentence"] == "câu test sạch"
    assert loaded[0]["Emotion"] == "Enjoyment"
