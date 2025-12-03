import json
from typing import IO

from delft.textClassification.config import (
    ModelConfig as _ModelConfig,
    TrainingConfig as _TrainingConfig
)

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager


class DefaultConfig:
    EMBEDDNGS_NAME = 'glove.6B.50d'


class AppConfig:
    def __init__(
            self,
            download_manager: DownloadManager = None,
            embedding_manager: EmbeddingManager = None):
        self.download_manager = download_manager
        self.embedding_manager = embedding_manager


class ModelConfig(_ModelConfig):
    def save_fp(self, fp: IO):
        json.dump(vars(self), fp, sort_keys=False, indent=4)

    @classmethod
    def load_fp(cls, fp: IO):
        variables = json.load(fp)
        self = cls()
        for key, val in variables.items():
            setattr(self, key, val)
        return self


class TrainingConfig(_TrainingConfig):
    def __init__(
        self,
        log_dir: str = None,
        learning_rate: float = 0.001,
        **kwargs
    ):
        self.log_dir = log_dir
        super().__init__(
            learning_rate=learning_rate,
            **kwargs
        )
