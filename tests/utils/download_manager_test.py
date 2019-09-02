import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sciencebeam_trainer_delft.utils.download_manager as download_manager_module
from sciencebeam_trainer_delft.utils.download_manager import (
    DownloadManager
)


EMBEDDING_NAME_1 = 'embedding1'
EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1
EXTERNAL_TXT_GZ_URL_1 = EXTERNAL_TXT_URL_1 + '.gz'


@pytest.fixture(name='copy_file_mock', autouse=True)
def _copy_file_mock():
    with patch.object(download_manager_module, 'copy_file') as mock:
        yield mock


@pytest.fixture(name='data_dir')
def _data_dir(temp_dir: Path):
    return temp_dir.joinpath('data')


@pytest.fixture(name='download_dir')
def _download_dir(data_dir: Path):
    return data_dir.joinpath('download')


class TestDownloadManager:
    def test_should_download(
            self,
            copy_file_mock: MagicMock,
            download_dir: Path):
        download_file = str(download_dir.joinpath(os.path.basename(EXTERNAL_TXT_URL_1)))
        download_manager = DownloadManager(download_dir=download_dir)
        assert download_manager.download(EXTERNAL_TXT_URL_1) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)

    def test_should_download_using_passed_in_local_file(
            self,
            copy_file_mock: MagicMock,
            download_dir: Path):
        download_file = str(download_dir.joinpath('custom.file'))
        download_manager = DownloadManager(download_dir=download_dir)
        assert download_manager.download(
            EXTERNAL_TXT_URL_1, local_file=download_file
        ) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)

    def test_should_unzip_embedding(
            self,
            copy_file_mock: MagicMock,
            download_dir: Path):
        download_file = str(download_dir.joinpath(os.path.basename(EXTERNAL_TXT_URL_1)))
        download_manager = DownloadManager(download_dir=download_dir)
        assert download_manager.download(EXTERNAL_TXT_GZ_URL_1) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_GZ_URL_1, download_file)
