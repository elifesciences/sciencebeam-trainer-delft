import logging
import json
import os
from datetime import datetime
from abc import ABC
from typing import Callable, Dict

import joblib

from delft.sequenceLabelling.models import Model
from delft.sequenceLabelling.preprocess import (
    FeaturesPreprocessor as DelftFeaturesPreprocessor,
    WordPreprocessor as DelftWordPreprocessor
)

from sciencebeam_trainer_delft.utils.typing import T, U, V
from sciencebeam_trainer_delft.utils.cloud_support import auto_upload_from_local_file
from sciencebeam_trainer_delft.utils.io import open_file, write_text, read_text
from sciencebeam_trainer_delft.utils.json import to_json, from_json

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    T_FeaturesPreprocessor
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_trainer_delft.sequence_labelling.tools.install_models import (
    copy_directory_with_source_meta
)


LOGGER = logging.getLogger(__name__)


class _BaseModelSaverLoader(ABC):
    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_pickle_file = 'preprocessor.pkl'
    preprocessor_json_file = 'preprocessor.json'
    meta_file = 'meta.json'


def _convert_keys(
    d: Dict[T, V],
    convert_fn: Callable[[T], U]
) -> Dict[U, V]:
    return {
        convert_fn(key): value
        for key, value in d.items()
    }


def get_feature_preprocessor_json(
        feature_preprocessor: T_FeaturesPreprocessor) -> dict:
    if not isinstance(feature_preprocessor, DelftFeaturesPreprocessor):
        return feature_preprocessor.__getstate__()
    feature_preprocessor_dict = vars(feature_preprocessor).copy()
    feature_preprocessor_dict['features_map_to_index'] = _convert_keys(
        feature_preprocessor_dict['features_map_to_index'],
        str
    )
    return feature_preprocessor_dict


def get_preprocessor_json(preprocessor: DelftWordPreprocessor) -> dict:
    if type(preprocessor) != DelftWordPreprocessor:  # pylint: disable=unidiomatic-typecheck
        return to_json(preprocessor)
    preprocessor_dict = vars(preprocessor).copy()
    feature_preprocessor = preprocessor_dict.get('feature_preprocessor')
    if feature_preprocessor:
        if type(feature_preprocessor) != DelftFeaturesPreprocessor:  # noqa pylint: disable=unidiomatic-typecheck
            return to_json(preprocessor)
        preprocessor_dict['feature_preprocessor'] = get_feature_preprocessor_json(
            feature_preprocessor
        )
    return to_json(preprocessor_dict, plain_json=True)


def get_feature_preprocessor_for_json(feature_preprocessor_json: dict) -> T_FeaturesPreprocessor:
    if not feature_preprocessor_json:
        return None
    LOGGER.debug('feature_preprocessor_json: %s', feature_preprocessor_json)
    feature_preprocessor = from_json(feature_preprocessor_json, DelftFeaturesPreprocessor)
    if isinstance(feature_preprocessor, DelftFeaturesPreprocessor):
        if isinstance(feature_preprocessor.features_map_to_index, dict):
            # features_map_to_index is initialized as a list (but then used as a dict)
            feature_preprocessor.features_map_to_index = _convert_keys(
                feature_preprocessor.features_map_to_index,
                int
            )
    return feature_preprocessor


def get_preprocessor_for_json(preprocessor_json: dict) -> DelftWordPreprocessor:
    preprocessor = from_json(preprocessor_json, DelftWordPreprocessor)
    LOGGER.debug('preprocessor type: %s', type(preprocessor))
    if isinstance(preprocessor, str):
        LOGGER.debug('preprocessor: %r', preprocessor)
    if isinstance(preprocessor.feature_preprocessor, dict):
        preprocessor.feature_preprocessor = get_feature_preprocessor_for_json(
            preprocessor.feature_preprocessor
        )
    return preprocessor


class ModelSaver(_BaseModelSaverLoader):
    def __init__(
            self,
            preprocessor: DelftWordPreprocessor,
            model_config: ModelConfig):
        self.preprocessor = preprocessor
        self.model_config = model_config

    def _save_preprocessor_json(self, preprocessor: DelftWordPreprocessor, filepath: str):
        write_text(
            filepath,
            json.dumps(get_preprocessor_json(preprocessor), sort_keys=False, indent=4)
        )
        LOGGER.info('preprocessor json saved to %s', filepath)

    def _save_preprocessor_pickle(self, preprocessor: DelftWordPreprocessor, filepath: str):
        with open_file(filepath, 'wb') as fp:
            joblib.dump(preprocessor, fp)
        LOGGER.info('preprocessor pickle saved to %s', filepath)

    def _save_model_config(self, model_config: ModelConfig, filepath: str):
        LOGGER.debug('model_config: %s', model_config)
        with open_file(filepath, 'w') as fp:
            model_config.save(fp)
        LOGGER.info('model config file saved to %s', filepath)

    def _save_model(self, model: Model, filepath: str):
        with auto_upload_from_local_file(filepath) as local_filepath:
            model.save(local_filepath)
        LOGGER.info('model saved to %s', filepath)

    def _save_meta(self, meta: dict, filepath: str):
        with open_file(filepath, 'w') as fp:
            json.dump(meta, fp, sort_keys=False, indent=4)
        LOGGER.info('model meta saved to %s', filepath)

    def _update_checkpoints_meta_file(self, filepath: str, checkpoint_directory: str, epoch: int):
        try:
            with open_file(filepath, 'r') as fp:
                meta = json.load(fp)
        except FileNotFoundError:
            meta = {}
        checkpoint_meta = {
            'epoch': (1 + epoch),
            'path': checkpoint_directory,
            'timestamp': datetime.utcnow().isoformat()
        }
        meta['checkpoints'] = meta.get('checkpoints', [])
        meta['checkpoints'].append(checkpoint_meta)
        meta['last_checkpoint'] = checkpoint_meta
        with open_file(filepath, 'w') as fp:
            json.dump(meta, fp, sort_keys=False, indent=4)
        LOGGER.info('updated checkpoints meta: %s', filepath)

    def save_to(self, directory: str, model: Model, meta: dict = None):
        os.makedirs(directory, exist_ok=True)
        self._save_preprocessor_json(
            self.preprocessor, os.path.join(directory, self.preprocessor_json_file)
        )
        self._save_preprocessor_pickle(
            self.preprocessor, os.path.join(directory, self.preprocessor_pickle_file)
        )
        self._save_model_config(self.model_config, os.path.join(directory, self.config_file))
        self._save_model(model, os.path.join(directory, self.weight_file))
        if meta:
            self._save_meta(meta, os.path.join(directory, self.meta_file))

    def add_checkpoint_meta(self, checkpoint_directory: str, epoch: int):
        self._update_checkpoints_meta_file(
            os.path.join(os.path.dirname(checkpoint_directory), 'checkpoints.json'),
            checkpoint_directory=checkpoint_directory,
            epoch=epoch
        )


class ModelLoader(_BaseModelSaverLoader):
    def __init__(
            self,
            download_manager: DownloadManager = None):
        if download_manager is None:
            download_manager = DownloadManager()
        self.download_manager = download_manager

    def download_model(self, dir_path: str) -> str:
        if not dir_path.endswith('.tar.gz'):
            return dir_path
        local_dir_path = str(self.download_manager.get_local_file(
            dir_path, auto_uncompress=False
        )).replace('.tar.gz', '')
        copy_directory_with_source_meta(dir_path, local_dir_path)
        return local_dir_path

    def load_preprocessor_from_directory(self, directory: str):
        try:
            return self.load_preprocessor_from_json_file(
                os.path.join(directory, self.preprocessor_json_file)
            )
        except FileNotFoundError:
            LOGGER.info('preprocessor json not found, falling back to pickle')
            return self.load_preprocessor_from_pickle_file(
                os.path.join(directory, self.preprocessor_pickle_file)
            )

    def load_preprocessor_from_pickle_file(self, filepath: str):
        LOGGER.info('loading preprocessor pickle from %s', filepath)
        with open_file(filepath, 'rb') as fp:
            return joblib.load(fp)

    def load_preprocessor_from_json_file(self, filepath: str):
        LOGGER.info('loading preprocessor json from %s', filepath)
        return get_preprocessor_for_json(json.loads(
            read_text(filepath)
        ))

    def load_model_config_from_directory(self, directory: str):
        return self.load_model_config_from_file(os.path.join(directory, self.config_file))

    def load_model_config_from_file(self, filepath: str):
        LOGGER.info('loading model config from %s', filepath)
        with open_file(filepath, 'r') as fp:
            return ModelConfig.load(fp)

    def load_model_from_directory(self, directory: str, model: Model):
        return self.load_model_from_file(
            os.path.join(directory, self.weight_file),
            model=model
        )

    def load_model_from_file(self, filepath: str, model: Model):
        LOGGER.info('loading model from %s', filepath)
        # we need a seekable file, ensure we download the file first
        local_filepath = self.download_manager.download_if_url(filepath)
        # using load_weights to avoid print statement in load method
        model.model.load_weights(local_filepath)
