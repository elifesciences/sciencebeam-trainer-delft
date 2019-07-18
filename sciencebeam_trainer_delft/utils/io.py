import logging
import os
from abc import ABC, abstractmethod
from shutil import copyfileobj
from contextlib import contextmanager
from gzip import GzipFile
from lzma import LZMAFile
from urllib.request import urlopen
from typing import List

from six import string_types, text_type

import numpy as np

try:
    from tensorflow.python.lib.io import file_io as tf_file_io
    from tensorflow.python.framework.errors_impl import NotFoundError as tf_NotFoundError
except ImportError:
    tf_file_io = None
    tf_NotFoundError = None


LOGGER = logging.getLogger(__name__)


def is_external_location(filepath: str):
    return isinstance(filepath, string_types) and '://' in filepath


def path_join(parent, child):
    return os.path.join(str(parent), str(child))


def is_gzip_filename(filepath: str):
    return filepath.endswith('.gz')


def is_xz_filename(filepath: str):
    return filepath.endswith('.xz')


def strip_gzip_filename_ext(filepath: str):
    if not is_gzip_filename(filepath):
        raise ValueError('not a gzip filename: %s' % filepath)
    return os.path.splitext(filepath)[0]


def strip_xz_filename_ext(filepath: str):
    if not is_xz_filename(filepath):
        raise ValueError('not a xz filename: %s' % filepath)
    return os.path.splitext(filepath)[0]


class CompressionWrapper(ABC):
    @abstractmethod
    def strip_compression_filename_ext(self, filepath: str):
        pass

    @abstractmethod
    def wrap_fileobj(self, filename: str, fileobj):
        pass


class GzipCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_gzip_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj):
        return GzipFile(filename=filename, fileobj=fileobj)


class XzCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_xz_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj):
        return LZMAFile(filename=fileobj)


class DummyCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return filepath

    def wrap_fileobj(self, filename: str, fileobj):
        return fileobj


GZIP_COMPRESSION_WRAPPER = GzipCompressionWrapper()
XZ_COMPRESSION_WRAPPER = XzCompressionWrapper()
DUMMY_COMPRESSION_WRAPPER = DummyCompressionWrapper()


def get_compression_wrapper(filepath: str):
    if is_gzip_filename(filepath):
        return GZIP_COMPRESSION_WRAPPER
    if is_xz_filename(filepath):
        return XZ_COMPRESSION_WRAPPER
    return DUMMY_COMPRESSION_WRAPPER


@contextmanager
def _open_raw(filepath: str, mode: str):
    if filepath.startswith('https://'):
        with urlopen(filepath) as fp:
            yield fp
    else:
        try:
            with tf_file_io.FileIO(filepath, mode=mode) as fp:
                yield fp
        except tf_NotFoundError as e:
            raise FileNotFoundError('file not found: %s' % filepath) from e


@contextmanager
def open_file(filepath: str, mode: str, compression_wrapper: CompressionWrapper = None):
    if compression_wrapper is None:
        compression_wrapper = get_compression_wrapper(filepath)
    if mode in {'rb', 'r'}:
        with _open_raw(filepath, mode=mode) as source_fp:
            yield compression_wrapper.wrap_fileobj(filename=filepath, fileobj=source_fp)
    elif mode in {'wb', 'w'}:
        tf_file_io.recursive_create_dir(os.path.dirname(filepath))
        with _open_raw(filepath, mode=mode) as target_fp:
            yield compression_wrapper.wrap_fileobj(filename=filepath, fileobj=target_fp)
    else:
        raise ValueError('unsupported mode: %s' % mode)


def copy_file(source_filepath: str, target_filepath: str, overwrite: bool = True):
    if tf_file_io is None:
        raise ImportError('Cloud storage file transfer requires TensorFlow.')
    if not overwrite and tf_file_io.file_exists(target_filepath):
        LOGGER.info('skipping already existing file: %s', target_filepath)
        return
    with open_file(text_type(source_filepath), mode='rb') as source_fp:
        with open_file(text_type(target_filepath), mode='wb') as target_fp:
            copyfileobj(source_fp, target_fp)


def concatenate_or_none(arrays: List[np.array], **kwargs) -> np.array:
    if arrays[0] is None:
        return None
    return np.concatenate(arrays, **kwargs)
