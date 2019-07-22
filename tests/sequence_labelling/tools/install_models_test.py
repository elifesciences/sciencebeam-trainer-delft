from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sciencebeam_trainer_delft.sequence_labelling.tools.install_models as install_models_module
from sciencebeam_trainer_delft.sequence_labelling.tools.install_models import (
    copy_directory_with_source_meta,
    main,
    SOURCE_URL_META_FILENAME
)


MODEL_NAME_1 = 'model1'
MODEL_FILE_1 = 'file.bin'
MODEL_DATA_1 = b'model data 1'


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
            list_files_mock: MagicMock,
            copy_file_mock: MagicMock,
            target_directory: Path,
            source_path: Path):
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        list_files_mock.assert_called_with(str(source_path))
        copy_file_mock.assert_called_with(
            str(source_path.joinpath(MODEL_FILE_1)),
            str(target_directory.joinpath(MODEL_FILE_1))
        )
        assert (
            target_directory.joinpath(SOURCE_URL_META_FILENAME).read_text()
            == str(source_path)
        )

    def test_should_skip_if_source_url_matches(
            self,
            list_files_mock: MagicMock,
            copy_file_mock: MagicMock,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text(str(source_path))
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        list_files_mock.assert_not_called()
        copy_file_mock.assert_not_called()

    def test_should_not_skip_if_source_url_is_different(
            self,
            list_files_mock: MagicMock,
            copy_file_mock: MagicMock,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text("other")
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=False
        )
        list_files_mock.assert_called()
        copy_file_mock.assert_called()

    def test_should_not_skip_if_force_is_true(
            self,
            list_files_mock: MagicMock,
            copy_file_mock: MagicMock,
            target_directory: Path,
            source_path: Path):
        target_directory.mkdir(parents=True, exist_ok=True)
        target_directory.joinpath(SOURCE_URL_META_FILENAME).write_text(str(source_path))
        copy_directory_with_source_meta(
            source_url=str(source_path),
            target_directory=str(target_directory),
            force=True
        )
        list_files_mock.assert_called()
        copy_file_mock.assert_called()


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
