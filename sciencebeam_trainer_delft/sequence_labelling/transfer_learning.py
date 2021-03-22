import argparse
import logging
from typing import Dict, List, Optional, NamedTuple

import keras
import numpy as np

from delft.sequenceLabelling.preprocess import WordPreprocessor
from delft.sequenceLabelling.models import BaseModel

from sciencebeam_trainer_delft.utils.misc import parse_comma_separated_str
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader
from sciencebeam_trainer_delft.sequence_labelling.models import (
    get_model
)


LOGGER = logging.getLogger(__name__)


class TransferLearningConfig(NamedTuple):
    source_model_path: Optional[str] = None
    copy_layers: Optional[List[str]] = None
    copy_preprocessor_fields: Optional[List[str]] = None
    freeze_layers: Optional[List[str]] = None


class TransferModelWrapper:
    def __init__(self, model: BaseModel):
        self.model = model
        self.keras_model: keras.Model = model.model
        self.keras_layers_by_name: Dict[str, keras.layers.Layer] = {
            layer.name: layer
            for layer in self.keras_model.layers
        }
        self.layer_names = set(self.keras_layers_by_name.keys())

    def get_layer_weights(self, layer_name: str) -> List[np.ndarray]:
        return self.keras_layers_by_name[layer_name].get_weights()

    def set_layer_weights(self, layer_name: str, weights: List[np.ndarray]):
        LOGGER.info('setting weights of layer: %r', layer_name)
        self.keras_layers_by_name[layer_name].set_weights(weights)

    def freeze_layer(self, layer_name: str):
        LOGGER.info('freezing layer: %r', layer_name)
        self.keras_layers_by_name[layer_name].trainable = False


class TransferLearningSource:
    def __init__(
        self,
        transfer_learning_config: TransferLearningConfig,
        source_model: BaseModel,
        source_preprocessor: WordPreprocessor
    ):
        self.transfer_learning_config = transfer_learning_config
        self.source_model = source_model
        self.source_preprocessor = source_preprocessor

    @staticmethod
    def from_config(
        transfer_learning_config: TransferLearningConfig,
        download_manager: DownloadManager = None
    ) -> Optional['TransferLearningSource']:
        if not transfer_learning_config:
            LOGGER.info('no transfer learning config specified')
            return None
        if not transfer_learning_config.source_model_path:
            LOGGER.info('no transfer learning source model specified')
            return None
        LOGGER.info('transfer learning config: %s', transfer_learning_config)
        model_loader = ModelLoader(download_manager=download_manager)
        directory = model_loader.download_model(transfer_learning_config.source_model_path)
        source_model_config = model_loader.load_model_config_from_directory(directory)
        source_preprocessor = model_loader.load_preprocessor_from_directory(directory)
        source_model: BaseModel = get_model(
            source_model_config,
            source_preprocessor,
            ntags=len(source_preprocessor.vocab_tag)
        )
        return TransferLearningSource(
            transfer_learning_config=transfer_learning_config,
            source_model=source_model,
            source_preprocessor=source_preprocessor
        )

    def apply_preprocessor(self, target_preprocessor: WordPreprocessor):
        if not self.transfer_learning_config.copy_preprocessor_fields:
            LOGGER.info('no transfer learning preprocessor fields specified')
            return
        for field_name in self.transfer_learning_config.copy_preprocessor_fields:
            LOGGER.info('copying preprocessor field: %r', field_name)
            value = getattr(self.source_preprocessor, field_name)
            setattr(target_preprocessor, field_name, value)

    def apply_weights(self, target_model: BaseModel):
        if not self.transfer_learning_config.copy_layers:
            LOGGER.info('no transfer learning source layers specified')
            return
        wrapped_source_model = TransferModelWrapper(self.source_model)
        wrapped_target_model = TransferModelWrapper(target_model)
        requested_layers = self.transfer_learning_config.copy_layers
        missing_source_layers = set(requested_layers) - set(wrapped_source_model.layer_names)
        if missing_source_layers:
            raise ValueError('missing source layers for transfer learning: %s (available: %s)' % (
                missing_source_layers, wrapped_source_model.layer_names
            ))
        missing_target_layers = set(requested_layers) - set(wrapped_target_model.layer_names)
        if missing_target_layers:
            raise ValueError('missing target layers for transfer learning: %s (available: %s)' % (
                missing_target_layers, wrapped_target_model.layer_names
            ))
        for layer_name in requested_layers:
            wrapped_target_model.set_layer_weights(
                layer_name,
                wrapped_source_model.get_layer_weights(layer_name)
            )


def freeze_model_layers(target_model: BaseModel, layers: Optional[List[str]]):
    if not layers:
        return
    wrapped_target_model = TransferModelWrapper(target_model)
    for layer_name in layers:
        wrapped_target_model.freeze_layer(layer_name)


def add_transfer_learning_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--transfer-source-model-path',
        type=str,
        help='path to model, that learned layers or parameters should be transfered from'
    )
    parser.add_argument(
        '--transfer-copy-layers',
        type=parse_comma_separated_str,
        help='the layers to transfer'
    )
    parser.add_argument(
        '--transfer-copy-preprocessor-fields',
        type=parse_comma_separated_str,
        help='the preprocessor fields to transfer (e.g. "vocab_char")'
    )
    parser.add_argument(
        '--transfer-freeze-layers',
        type=parse_comma_separated_str,
        help='the layers to freeze'
    )


def get_transfer_learning_config_for_parsed_args(
    args: argparse.Namespace
) -> TransferLearningConfig:
    return TransferLearningConfig(
        source_model_path=args.transfer_source_model_path,
        copy_layers=args.transfer_copy_layers,
        copy_preprocessor_fields=args.transfer_copy_preprocessor_fields,
        freeze_layers=args.transfer_freeze_layers
    )
