"""Domain Interfaces."""

from source.domain.interface.base_classifer import BaseSentimentClassifier
from source.domain.interface.base_dataset_repo import BaseDatasetRepository
from source.domain.interface.base_preprocess import BasePreprocess

__all__ = [
    "BaseSentimentClassifier",
    "BasePreprocess",
    "BaseDatasetRepository",
]
