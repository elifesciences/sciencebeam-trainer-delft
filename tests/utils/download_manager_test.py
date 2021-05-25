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


@pytest.fixture(name='download_manager')
def _download_manager(download_dir: Path):
    return DownloadManager(download_dir=str(download_dir))


class TestDownloadManager:
    def test_get_local_file_should_return_same_path_for_same_urls(
            self,
            download_manager: DownloadManager):
        assert (
            download_manager.get_local_file(EXTERNAL_TXT_URL_1)
            == download_manager.get_local_file(EXTERNAL_TXT_URL_1)
        )

    def test_get_local_file_should_return_different_path_for_different_url_paths(
            self,
            download_manager: DownloadManager):
        assert (
            download_manager.get_local_file('http://host1/file.txt')
            != download_manager.get_local_file('http://host2/file.txt')
        )

    def test_should_download(
            self,
            copy_file_mock: MagicMock,
            download_manager: DownloadManager):
        download_file = download_manager.download(EXTERNAL_TXT_URL_1)
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)
        assert str(download_file).endswith(os.path.basename(EXTERNAL_TXT_URL_1))

    def test_should_download_using_passed_in_local_file(
            self,
            copy_file_mock: MagicMock,
            download_dir: Path):
        download_file = str(download_dir.joinpath('custom.file'))
        download_manager = DownloadManager(download_dir=str(download_dir))
        assert download_manager.download(
            EXTERNAL_TXT_URL_1, local_file=download_file
        ) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)

    def test_should_unzip_embedding(
            self,
            copy_file_mock: MagicMock,
            download_manager: DownloadManager):
        download_file = download_manager.download(EXTERNAL_TXT_GZ_URL_1)
        copy_file_mock.assert_called_with(EXTERNAL_TXT_GZ_URL_1, download_file)
        assert str(download_file).endswith(os.path.basename(EXTERNAL_TXT_URL_1))
