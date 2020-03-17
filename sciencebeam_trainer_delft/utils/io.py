import logging
import os
import tempfile
from abc import ABC, abstractmethod
from shutil import copyfileobj
from contextlib import contextmanager
from gzip import GzipFile
from lzma import LZMAFile
from urllib.error import HTTPError
from urllib.request import urlopen
from typing import List, IO

from six import string_types, text_type

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
    def wrap_fileobj(self, filename: str, fileobj: IO, mode: str = None):
        pass

    def open(self, filename: str, mode: str):
        return self.wrap_fileobj(
            filename=filename,
            fileobj=_open_raw(filename, mode=mode),
            mode=mode
        )


class ClosingGzipFile(GzipFile):
    # GzipFile doesn't close the underlying fileobj, we will do that here
    def close(self):
        fileobj = self.fileobj
        LOGGER.debug('ClosingGzipFile.close, fileobj: %s', fileobj)
        try:
            super().close()
        finally:
            if fileobj is not None:
                LOGGER.debug('closing: %s', fileobj)
                fileobj.close()


class GzipCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_gzip_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: str = None):
        return ClosingGzipFile(filename=filename, fileobj=fileobj, mode=mode)

    @contextmanager
    def open(self, filename: str, mode: str):
        if is_external_location(filename):
            # there seem to be an issue with GzipFile and fileobj
            with tempfile.TemporaryDirectory(suffix='-gzip') as gzip_dir:
                local_gzip_file = os.path.join(gzip_dir, os.path.basename(filename))
                with ClosingGzipFile(filename=local_gzip_file, mode=mode) as local_fp:
                    yield local_fp
                tf_file_io.copy(local_gzip_file, filename, overwrite=True)
        else:
            with ClosingGzipFile(filename=filename, mode=mode) as local_fp:
                yield local_fp


class XzCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_xz_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: str = None):
        return LZMAFile(filename=fileobj, mode=mode)


class DummyCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return filepath

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: str = None):
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
        try:
            with urlopen(filepath) as fp:
                yield fp
        except HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError('file not found: %s' % filepath) from error
            raise
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
    LOGGER.debug(
        'open_file, filepath=%s, mode=%s, compression_wrapper=%s',
        filepath, mode, compression_wrapper
    )
    if mode in {'rb', 'r'}:
        with _open_raw(filepath, mode=mode) as source_fp:
            yield compression_wrapper.wrap_fileobj(
                filename=filepath,
                fileobj=source_fp,
                mode=mode
            )
    elif mode in {'wb', 'w'}:
        tf_file_io.recursive_create_dir(os.path.dirname(filepath))
        with compression_wrapper.open(filepath, mode=mode) as target_fp:
            yield target_fp
    else:
        raise ValueError('unsupported mode: %s' % mode)


def _require_tf_file_io():
    if tf_file_io is None:
        raise ImportError('Cloud storage file transfer requires TensorFlow.')


def copy_file(source_filepath: str, target_filepath: str, overwrite: bool = True):
    _require_tf_file_io()
    if not overwrite and tf_file_io.file_exists(target_filepath):
        LOGGER.info('skipping already existing file: %s', target_filepath)
        return
    with open_file(text_type(source_filepath), mode='rb') as source_fp:
        with open_file(text_type(target_filepath), mode='wb') as target_fp:
            copyfileobj(source_fp, target_fp)


def list_files(directory_path: str) -> List[str]:
    _require_tf_file_io()
    return tf_file_io.list_directory(directory_path)
