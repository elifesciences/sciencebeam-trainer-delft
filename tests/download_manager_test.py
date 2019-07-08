import os
from unittest.mock import patch, MagicMock

import pytest
from py._path.local import LocalPath

import sciencebeam_trainer_delft.download_manager as embedding_manager_module
from sciencebeam_trainer_delft.download_manager import (
    DownloadManager
)


EMBEDDING_NAME_1 = 'embedding1'
EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1
EXTERNAL_TXT_GZ_URL_1 = EXTERNAL_TXT_URL_1 + '.gz'


@pytest.fixture(name='copy_file_mock', autouse=True)
def _copy_file_mock():
    with patch.object(embedding_manager_module, 'copy_file') as mock:
        yield mock

@pytest.fixture(name='data_dir')
def _data_dir(tmpdir):
    return tmpdir.join('data')


@pytest.fixture(name='download_dir')
def _download_dir(data_dir):
    return data_dir.join('download')


class TestDownloadManager:
    def test_should_download(
            self,
            copy_file_mock: MagicMock,
            download_dir: LocalPath):
        download_file = download_dir.join(os.path.basename(EXTERNAL_TXT_URL_1))
        download_manager = DownloadManager(download_dir=download_dir)
        assert download_manager.download(EXTERNAL_TXT_URL_1) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)

    def test_should_unzip_embedding(
            self,
            copy_file_mock: MagicMock,
            download_dir: LocalPath):
        download_file = download_dir.join(os.path.basename(EXTERNAL_TXT_URL_1))
        download_manager = DownloadManager(download_dir=download_dir)
        assert download_manager.download(EXTERNAL_TXT_GZ_URL_1) == download_file
        copy_file_mock.assert_called_with(EXTERNAL_TXT_GZ_URL_1, download_file)
