import logging
import os
from shutil import copyfileobj
from contextlib import contextmanager
from gzip import GzipFile
from urllib.request import urlopen
from typing import List

from six import string_types, text_type

import numpy as np

try:
    from tensorflow.python.lib.io import file_io as tf_file_io
except ImportError:
    tf_file_io = None


LOGGER = logging.getLogger(__name__)


def is_external_location(filepath: str):
    return isinstance(filepath, string_types) and '://' in filepath


def is_gzip_filename(filepath: str):
    return filepath.endswith('.gz')


def strip_gzip_filename_ext(filepath: str):
    if not is_gzip_filename(filepath):
        raise ValueError('not a gzip filename: %s' % filepath)
    return os.path.splitext(filepath)[0]


@contextmanager
def _open_raw(filepath: str, mode: str):
    if filepath.startswith('https://'):
        with urlopen(filepath) as fp:
            yield fp
    else:
        with tf_file_io.FileIO(filepath, mode=mode) as fp:
            yield fp


@contextmanager
def open_file(filepath: str, mode: str, gzip_compression=None):
    if gzip_compression is None:
        gzip_compression = is_gzip_filename(filepath)
    if mode == 'rb':
        with _open_raw(filepath, mode=mode) as source_fp:
            if gzip_compression:
                yield GzipFile(filename=filepath, fileobj=source_fp)
            else:
                yield source_fp
    elif mode == 'wb':
        tf_file_io.recursive_create_dir(os.path.dirname(filepath))
        with _open_raw(filepath, mode=mode) as target_fp:
            if gzip_compression:
                yield GzipFile(filename=filepath, fileobj=target_fp)
            else:
                yield target_fp
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


def parse_number_range(expr: str) -> List[int]:
    fragments = expr.split('-')
    if len(fragments) == 1:
        return [int(expr)]
    if len(fragments) == 2:
        return list(range(int(fragments[0]), int(fragments[1]) + 1))
    raise ValueError('invalid number range: %s' % fragments)


def parse_number_ranges(expr: str) -> List[int]:
    if not expr:
        return []
    numbers = []
    for fragment in expr.split(','):
        numbers.extend(parse_number_range(fragment))
    return numbers


def concatenate_or_none(arrays: List[np.array], **kwargs) -> np.array:
    if arrays[0] is None:
        return None
    return np.concatenate(arrays, **kwargs)
