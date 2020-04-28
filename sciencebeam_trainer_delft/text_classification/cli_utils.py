import logging
from typing import List

# import numpy as np
# import pandas as pd

import delft.textClassification.models
import delft.textClassification.wrapper
from delft.textClassification import Classifier

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.models.Attention import Attention
from sciencebeam_trainer_delft.text_classification.models import (
    train_model
)

from sciencebeam_trainer_delft.text_classification.reader import (
    load_texts_and_classes_pandas
)
from sciencebeam_trainer_delft.text_classification.config import (
    ModelConfig,
    TrainingConfig
)


LOGGER = logging.getLogger(__name__)


def get_downloaded_input_paths(
        input_paths: List[str],
        download_manager: DownloadManager) -> List[str]:
    return [
        download_manager.download_if_url(input_path)
        for input_path in input_paths
    ]


def load_input_data(
        input_paths: List[str],
        download_manager: DownloadManager,
        limit: int = None):
    assert len(input_paths) == 1
    LOGGER.info('loading data: %s', input_paths)
    downloaded_input_paths = get_downloaded_input_paths(
        input_paths,
        download_manager=download_manager
    )
    xtr, y, y_names = load_texts_and_classes_pandas(
        downloaded_input_paths[0],
        limit=limit
    )
    LOGGER.info('loaded data: %d rows', len(xtr))
    # LOGGER.info('y:\n%s', y)
    # LOGGER.info('y value_counts:\n%s', np.bincount(y))
    # raise RuntimeError('dummy')
    return xtr, y, y_names


def _patch_delft():
    delft.textClassification.models.Attention = Attention
    delft.textClassification.wrapper.train_model = train_model


def train(
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_input_texts: List[str],
        train_input_labels: List[List[str]],
        model_path: str):

    _patch_delft()

    model = Classifier(embeddings_name=model_config.embeddings_name)
    model.embeddings_name = model_config.embeddings_name
    model.model_config = model_config
    model.model_config.word_embedding_size = model.embeddings.embed_size
    model.training_config = training_config

    model.train(train_input_texts, train_input_labels)
    model.save(model_path)
