import logging
import os

from delft.sequenceLabelling.models import Model

from sciencebeam_trainer_delft.config import ModelConfig
from sciencebeam_trainer_delft.preprocess import Preprocessor
from sciencebeam_trainer_delft.utils import open_file


LOGGER = logging.getLogger(__name__)


class _BaseModelSaverLoader:
    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_file = 'preprocessor.pkl'


class ModelSaver(_BaseModelSaverLoader):
    def __init__(
            self,
            preprocessor: Preprocessor,
            model_config: ModelConfig):
        self.preprocessor = preprocessor
        self.model_config = model_config

    def _save_preprocessor(self, preprocessor: Preprocessor, filepath: str):
        preprocessor.save(filepath)
        LOGGER.info('preprocessor saved to %s', filepath)

    def _save_model_config(self, model_config: ModelConfig, filepath: str):
        model_config.save(filepath)
        LOGGER.info('model config file saved to %s', filepath)

    def _save_model(self, model: Model, filepath: str):
        model.save(filepath)
        LOGGER.info('model saved to %s', filepath)

    def save_to(self, directory: str, model: Model):
        os.makedirs(directory, exist_ok=True)
        self._save_preprocessor(self.preprocessor, os.path.join(directory, self.preprocessor_file))
        self._save_model_config(self.model_config, os.path.join(directory, self.config_file))
        self._save_model(model, os.path.join(directory, self.weight_file))


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
            model.load(fp)
