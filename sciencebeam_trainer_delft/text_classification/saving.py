import logging

from keras.models import Model

from sciencebeam_trainer_delft.utils.cloud_support import auto_upload_from_local_file
from sciencebeam_trainer_delft.utils.io import open_file

from sciencebeam_trainer_delft.text_classification.config import ModelConfig
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


LOGGER = logging.getLogger(__name__)


class ModelSaver:
    def save_model_config(self, model_config: ModelConfig, filepath: str):
        LOGGER.debug('model_config: %s', model_config)
        with open_file(filepath, 'w') as fp:
            model_config.save_fp(fp)
        LOGGER.info('model config file saved to %s', filepath)

    def save_model_weights(self, model: Model, filepath: str):
        with auto_upload_from_local_file(filepath) as local_filepath:
            model.save(local_filepath)
        LOGGER.info('model saved to %s', filepath)


class ModelLoader:
    def __init__(
            self,
            download_manager: DownloadManager = None):
        if download_manager is None:
            download_manager = DownloadManager()
        self.download_manager = download_manager

    def load_model_config_from_file(self, filepath: str):
        LOGGER.info('loading model config from %s', filepath)
        with open_file(filepath, 'r') as fp:
            return ModelConfig.load_fp(fp)

    def load_model_weights_from_file(self, filepath: str, model: Model):
        LOGGER.info('loading model from %s', filepath)
        # we need a seekable file, ensure we download the file first
        local_filepath = self.download_manager.download_if_url(filepath)
        # using load_weights to avoid print statement in load method
        model.load_weights(local_filepath)
