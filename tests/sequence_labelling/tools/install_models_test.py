import gzip
import tarfile
import zipfile
from io import BytesIO
from pickle import UnpicklingError
from pathlib import Path
from unittest.mock import patch

import pytest

import sciencebeam_trainer_delft.sequence_labelling.tools.install_models as install_models_module
from sciencebeam_trainer_delft.sequence_labelling.tools.install_models import (
    copy_directory_with_source_meta,
    main,
    SOURCE_URL_META_FILENAME
)


MODEL_NAME_1 = 'model1'
MODEL_FILE_1 = 'file.bin'
MODEL_PICKLE_FILE_1 = 'file.pkl'
MODEL_DATA_1 = b'model data 1'
MODEL_DATA_2 = b'model data 2'


@pytest.fixture(name='list_files_mock')
def _list_files_mock():
    with patch.object(install_models_module, 'list_files') as mock:
        mock.return_value = [MODEL_FILE_1]
        yield mock


@pytest.fixture(name='copy_file_mock')
def _copy_file_mock():
    with patch.object(install_models_module, 'copy_file') as mock:
        yield mock


@pytest.fixture(name='copy_directory_with_source_meta_mock')
def _copy_directory_with_source_meta():
    with patch.object(install_models_module, 'copy_directory_with_source_meta') as mock:
        yield mock


@pytest.fixture(name='models_path')
def _models_path(temp_dir: Path) -> Path:
    return temp_dir.joinpath('models')


@pytest.fixture(name='target_directory')
def _target_directory(models_path: Path) -> Path:
    return models_path.joinpath('models', MODEL_NAME_1)


@pytest.fixture(name='source_path')
def _source_path(temp_dir: Path) -> Path:
    return temp_dir.joinpath('source')


class TestCopySourceDirectoryWithSourceMeta:
    def test_should_copy_to_not_yet_existing_directory(
            self,
            target_directory: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_1
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(source_path)
        )

    def test_should_skip_if_source_url_matches(
            self,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text(str(source_path))
        target_directory.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_2)
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_1

    def test_should_not_skip_if_source_url_is_different(
            self,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text("other")
        target_directory.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_2)
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_2

    def test_should_not_skip_if_force_is_true(
            self,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text(str(source_path))
        target_directory.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_2)
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=True
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_2

    def test_should_extract_from_tar_gz(
            self,
            target_directory: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        tar_gz_file_path = source_path.joinpath('archive1.tar.gz')
        with tarfile.open(str(tar_gz_file_path), mode='w:gz') as tar_file:
            buf = BytesIO(MODEL_DATA_1)
            tar_info = tarfile.TarInfo(name=MODEL_FILE_1)
            tar_info.size = len(buf.getvalue())
            tar_file.addfile(tar_info, buf)
        copy_directory_with_source_meta(
            source_url=str(tar_gz_file_path),
            target_directory=str(target_directory),
            force=False
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_1
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(tar_gz_file_path)
        )

    def test_should_extract_from_zip(
            self,
            target_directory: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        zip_file_path = source_path.joinpath('archive1.zip')
        with zipfile.ZipFile(str(zip_file_path), mode='w') as zip_file:
            zip_file.writestr(MODEL_FILE_1, MODEL_DATA_1)
        copy_directory_with_source_meta(
            source_url=str(zip_file_path),
            target_directory=str(target_directory),
            force=False
        )
        assert target_directory.joinpath(MODEL_FILE_1).read_bytes() == MODEL_DATA_1
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(zip_file_path)
        )


class TestMain:
    def test_should_copy_files(
            self,
            models_path: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        target_directory = models_path.joinpath(MODEL_NAME_1)
        main([
            '--model-base-path=%s' % models_path,
            '--install', '%s=%s' % (MODEL_NAME_1, source_path)
        ])
        assert (
            target_directory.joinpath(MODEL_FILE_1).read_bytes()
            == MODEL_DATA_1
        )
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(source_path)
        )

    def test_should_copy_and_decompress_gzipped_file(
            self,
            models_path: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(source_path.joinpath(MODEL_FILE_1 + '.gz')), 'wb') as out_fp:
            out_fp.write(MODEL_DATA_1)
        target_directory = models_path.joinpath(MODEL_NAME_1)
        main([
            '--model-base-path=%s' % models_path,
            '--install', '%s=%s' % (MODEL_NAME_1, source_path)
        ])
        assert (
            target_directory.joinpath(MODEL_FILE_1).read_bytes()
            == MODEL_DATA_1
        )
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(source_path)
        )

    def test_should_remove_or_rename_existing_files(
            self,
            models_path: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_FILE_1).write_bytes(MODEL_DATA_1)
        target_directory = models_path.joinpath(MODEL_NAME_1)
        target_directory.mkdir(parents=True)
        existing_file = target_directory.joinpath('existing.data')
        existing_file.write_text('existing file')
        assert existing_file.exists()
        main([
            '--model-base-path=%s' % models_path,
            '--install', '%s=%s' % (MODEL_NAME_1, source_path)
        ])
        assert not existing_file.exists()
        assert (
            target_directory.joinpath(MODEL_FILE_1).read_bytes()
            == MODEL_DATA_1
        )
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(source_path)
        )

    def test_should_validate_invalid_pickles_files(
            self,
            models_path: Path,
            source_path: Path):
        source_path.mkdir(parents=True, exist_ok=True)
        source_path.joinpath(MODEL_PICKLE_FILE_1).write_bytes(MODEL_DATA_1)
        with pytest.raises(UnpicklingError):
            main([
                '--model-base-path=%s' % models_path,
                '--install', '%s=%s' % (MODEL_NAME_1, source_path),
                '--validate-pickles'
            ])
