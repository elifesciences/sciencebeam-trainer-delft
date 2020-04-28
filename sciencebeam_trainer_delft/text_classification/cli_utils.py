import logging
import time
from typing import List, Tuple

# import numpy as np
# import pandas as pd

import delft.textClassification.models
import delft.textClassification.wrapper

from sciencebeam_trainer_delft.text_classification.wrapper import Classifier
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.models.Attention import Attention
from sciencebeam_trainer_delft.text_classification.models import (
    train_model
)

from sciencebeam_trainer_delft.text_classification.evaluation import (
    ClassificationResult
)

from sciencebeam_trainer_delft.text_classification.reader import (
    load_texts_and_classes_pandas,
    load_classes_pandas
)
from sciencebeam_trainer_delft.text_classification.config import (
    AppConfig,
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
        limit: int = None) -> Tuple[List[str], List[List[int]], List[str]]:
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
    return xtr, y, y_names


def load_label_data(
        input_paths: List[str],
        download_manager: DownloadManager,
        limit: int = None) -> Tuple[List[List[int]], List[str]]:
    assert len(input_paths) == 1
    LOGGER.info('loading data: %s', input_paths)
    downloaded_input_paths = get_downloaded_input_paths(
        input_paths,
        download_manager=download_manager
    )
    y, y_names = load_classes_pandas(
        downloaded_input_paths[0],
        limit=limit
    )
    LOGGER.info('loaded data: %d rows', len(y))
    return y, y_names


def _patch_delft():
    delft.textClassification.models.Attention = Attention
    delft.textClassification.wrapper.train_model = train_model


def train(
        app_config: AppConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_input_texts: List[str],
        train_input_labels: List[List[str]],
        model_path: str):

    _patch_delft()

    model = Classifier(
        embeddings_name=model_config.embeddings_name,
        download_manager=app_config.download_manager,
        embedding_manager=app_config.embedding_manager
    )
    model.embeddings_name = model_config.embeddings_name
    model.model_config = model_config
    model.model_config.word_embedding_size = model.embeddings.embed_size
    model.training_config = training_config

    model.train(train_input_texts, train_input_labels)
    LOGGER.info('saving model to: %s', model_path)
    model.save_to(model_path)


def predict(
        app_config: AppConfig,
        eval_input_texts: List[str],
        model_path: str):
    model = Classifier(
        download_manager=app_config.download_manager,
        embedding_manager=app_config.embedding_manager
    )
    model.load_from(model_path)

    LOGGER.info('number of texts to classify: %s', len(eval_input_texts))
    start_time = time.time()
    result = model.predict(eval_input_texts, output_format="csv")
    LOGGER.info("runtime: %s seconds", round(time.time() - start_time, 3))
    return result


def evaluate(
        app_config: AppConfig,
        eval_input_texts: List[str],
        eval_input_labels: List[List[int]],
        model_path: str):
    model = Classifier(
        download_manager=app_config.download_manager,
        embedding_manager=app_config.embedding_manager
    )
    model.load_from(model_path)

    LOGGER.info('number of texts to classify: %s', len(eval_input_texts))
    start_time = time.time()
    result = model.predict(eval_input_texts, output_format="csv")
    LOGGER.info("runtime: %s seconds", round(time.time() - start_time, 3))
    return ClassificationResult(
        y_true=eval_input_labels,
        y_pred=result,
        label_names=model.model_config.list_classes
    )
