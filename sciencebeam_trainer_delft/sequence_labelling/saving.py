import logging
import json
import os
from datetime import datetime
from abc import ABC

from delft.sequenceLabelling.models import Model
from delft.sequenceLabelling.preprocess import WordPreprocessor as DefaultWordPreprocessor

from sciencebeam_trainer_delft.utils.cloud_support import auto_upload_from_local_file
from sciencebeam_trainer_delft.utils.io import open_file, write_text, read_text
from sciencebeam_trainer_delft.utils.json import to_json, from_json

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


LOGGER = logging.getLogger(__name__)


class _BaseModelSaverLoader(ABC):
    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_pickle_file = 'preprocessor.pkl'
    preprocessor_json_file = 'preprocessor.json'
    meta_file = 'meta.json'


def get_preprocessor_json(preprocessor: Preprocessor) -> dict:
    return to_json(preprocessor)


def get_preprocessor_for_json(preprocessor_json: dict) -> Preprocessor:
    return from_json(preprocessor_json, DefaultWordPreprocessor)


class ModelSaver(_BaseModelSaverLoader):
    def __init__(
            self,
            preprocessor: Preprocessor,
            model_config: ModelConfig):
        self.preprocessor = preprocessor
        self.model_config = model_config

    def _save_preprocessor_json(self, preprocessor: Preprocessor, filepath: str):
        write_text(
            filepath,
            json.dumps(get_preprocessor_json(preprocessor), sort_keys=False, indent=4)
        )
        LOGGER.info('preprocessor json saved to %s', filepath)

    def _save_preprocessor_pickle(self, preprocessor: Preprocessor, filepath: str):
        with open_file(filepath, 'wb') as fp:
            preprocessor.save(fp)
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
            return Preprocessor.load(fp)

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
