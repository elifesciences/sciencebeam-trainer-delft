import os
import logging
from functools import wraps
from contextlib import contextmanager
from tempfile import TemporaryDirectory, mkdtemp
from pathlib import Path

from six import string_types

import keras

import delft
import delft.sequenceLabelling.trainer

import sciencebeam_trainer_delft.trainer
from sciencebeam_trainer_delft.utils import copy_file


LOGGER = logging.getLogger(__name__)


def _is_cloud_location(filepath):
    return isinstance(filepath, string_types) and filepath.startswith('gs://')


def _copy_file_to_cloud(source_filepath, target_filepath, overwrite=True):
    copy_file(source_filepath, target_filepath, overwrite=overwrite)


def _copy_directory_to_cloud(source_filepath, target_filepath, overwrite=True):
    for temp_file_path in Path(source_filepath).glob('**/*'):
        if not temp_file_path.is_file():
            continue
        relative_filename = temp_file_path.relative_to(source_filepath)
        cloud_path = os.path.join(target_filepath, relative_filename)
        LOGGER.info('copying %s to %s', temp_file_path, cloud_path)
        _copy_file_to_cloud(temp_file_path, cloud_path, overwrite=overwrite)


def _copy_to_cloud(source_filepath, target_filepath, overwrite=True):
    if Path(source_filepath).is_file():
        _copy_file_to_cloud(source_filepath, target_filepath, overwrite=overwrite)
        return
    if Path(source_filepath).is_dir():
        _copy_directory_to_cloud(source_filepath, target_filepath, overwrite=overwrite)
        return


def _get_temp_path(filepath):
    return mkdtemp(suffix=os.path.basename(filepath))


@contextmanager
def _cloud_location_as_temp_context(filepath):
    with TemporaryDirectory(suffix=os.path.basename(filepath)) as temp_dir:
        temp_path = os.path.join(temp_dir, os.path.basename(filepath))
        LOGGER.info('temp_path: %s', temp_dir)
        yield temp_path
        _copy_to_cloud(temp_path, filepath)


def wrap_get_callbacks(get_callbacks_fn: callable):
    @wraps(get_callbacks_fn)
    def wrapped_get_callbacks(  # pylint: disable=keyword-arg-before-vararg
            log_dir=None, *args, **kwargs):
        LOGGER.info('log_dir: %s', log_dir)
        if log_dir is None or not _is_cloud_location(log_dir):
            return get_callbacks_fn(log_dir, *args, **kwargs)
        local_log_dir = _get_temp_path(log_dir)
        # pass in a local path, to avoid local directory creation on a cloud path
        callbacks = get_callbacks_fn(local_log_dir, *args, **kwargs)
        for callback in callbacks:
            if isinstance(callback, keras.callbacks.ModelCheckpoint):
                # revert back the filepath to a cloud location since we patched save_weights
                callback.filepath = os.path.join(
                    log_dir, Path(callback.filepath).relative_to(local_log_dir)
                )
                LOGGER.info('callback.filepath: %s', callback.filepath)
        return callbacks
    return wrapped_get_callbacks


def wrap_network_save_weights_instance_model(save_fn: callable):
    @wraps(save_fn)
    def wrapped_network_save_weights(self, filepath=None, overwrite=True):
        LOGGER.info('wrapped_network_save_weights, filepath: %s', filepath)
        if not filepath or not _is_cloud_location(filepath):
            return save_fn(self, filepath, overwrite=overwrite)
        with _cloud_location_as_temp_context(filepath) as temp_path:
            return save_fn(self, temp_path, overwrite=overwrite)
    return wrapped_network_save_weights


def wrap_delft_sequence_save_instance_method(sequence_save_fn: callable):
    @wraps(sequence_save_fn)
    def wrapped_sequence_save(self, dir_path=None):
        LOGGER.info('wrapped_sequence_save, dir_path: %s', dir_path)
        if dir_path is None:
            return sequence_save_fn(self)
        if not _is_cloud_location(dir_path):
            return sequence_save_fn(self, dir_path)
        with _cloud_location_as_temp_context(dir_path) as temp_path:
            return sequence_save_fn(self, temp_path)
    return wrapped_sequence_save


def wrap_save_instance_method(save_fn: callable):
    @wraps(save_fn)
    def wrapped_save(*args, **kwargs):
        print('wrapped_save, args:', args, kwargs)
        return save_fn(*args, **kwargs)
    return wrapped_save


def wrap_save_model(save_model_fn: callable):
    @wraps(save_model_fn)
    def wrapped_save_model(model, filepath, overwrite=True, include_optimizer=True):
        print('wrapped_save_model, output_path:', filepath)
        return save_model_fn(model, filepath, overwrite, include_optimizer)
    return wrapped_save_model


def patch_cloud_support():
    keras.engine.saving.save_model = wrap_save_model(keras.engine.saving.save_model)
    keras.models.save_model = keras.engine.saving.save_model
    keras.engine.network.Network.save_weights = wrap_network_save_weights_instance_model(
        keras.engine.network.Network.save_weights
    )
    delft.sequenceLabelling.Sequence.save = wrap_delft_sequence_save_instance_method(
        delft.sequenceLabelling.Sequence.save
    )
    delft.sequenceLabelling.trainer.get_callbacks = wrap_get_callbacks(
        delft.sequenceLabelling.trainer.get_callbacks
    )
    sciencebeam_trainer_delft.trainer.get_callbacks = wrap_get_callbacks(
        sciencebeam_trainer_delft.trainer.get_callbacks
    )
