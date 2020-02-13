import logging
import json
import os
from datetime import datetime
from abc import ABC

from delft.sequenceLabelling.models import Model

from sciencebeam_trainer_delft.utils.cloud_support import auto_upload_from_local_file
from sciencebeam_trainer_delft.utils.io import open_file

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor


LOGGER = logging.getLogger(__name__)


class _BaseModelSaverLoader(ABC):
    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_file = 'preprocessor.pkl'
    meta_file = 'meta.json'


class ModelSaver(_BaseModelSaverLoader):
    def __init__(
            self,
            preprocessor: Preprocessor,
            model_config: ModelConfig):
        self.preprocessor = preprocessor
        self.model_config = model_config

    def _save_preprocessor(self, preprocessor: Preprocessor, filepath: str):
        with open_file(filepath, 'wb') as fp:
            preprocessor.save(fp)
        LOGGER.info('preprocessor saved to %s', filepath)

    def _save_model_config(self, model_config: ModelConfig, filepath: str):
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
        self._save_preprocessor(self.preprocessor, os.path.join(directory, self.preprocessor_file))
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
    def load_preprocessor_from_directory(self, directory: str):
        return self.load_preprocessor_from_file(os.path.join(directory, self.preprocessor_file))

    def load_preprocessor_from_file(self, filepath: str):
        LOGGER.info('loading preprocessor from %s', filepath)
        with open_file(filepath, 'rb') as fp:
            return Preprocessor.load(fp)

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
        with open_file(filepath, 'rb') as fp:
            # using load_weights to avoid print statement in load method
            model.model.load_weights(fp)
