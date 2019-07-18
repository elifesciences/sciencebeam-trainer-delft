import os
import logging
from contextlib import contextmanager
from tempfile import TemporaryDirectory, mkdtemp
from pathlib import Path

from six import string_types

from sciencebeam_trainer_delft.utils.io import copy_file, path_join


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
        cloud_path = path_join(target_filepath, relative_filename)
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


@contextmanager
def auto_upload_from_local_path(filepath: str):
    if not filepath or not _is_cloud_location(filepath):
        os.makedirs(filepath, exist_ok=True)
        yield filepath
    else:
        with _cloud_location_as_temp_context(filepath) as temp_path:
            yield temp_path


@contextmanager
def auto_upload_from_local_file(filepath: str):
    if not filepath or not _is_cloud_location(filepath):
        yield filepath
    else:
        with _cloud_location_as_temp_context(filepath) as local_path:
            yield local_path


def patch_cloud_support():
    # deprecated
    pass
