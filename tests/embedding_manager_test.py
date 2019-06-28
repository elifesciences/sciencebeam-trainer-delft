import os
from unittest.mock import patch, MagicMock

import pytest
from py._path.local import LocalPath

import sciencebeam_trainer_delft.embedding_manager as embedding_manager_module
from sciencebeam_trainer_delft.embedding_manager import (
    EmbeddingManager
)


EMBEDDING_NAME_1 = 'embedding1'
EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1
EXTERNAL_TXT_GZ_URL_1 = EXTERNAL_TXT_URL_1 + '.gz'


@pytest.fixture(name='copy_file_mock', autouse=True)
def _copy_file_mock():
    with patch.object(embedding_manager_module, 'copy_file') as mock:
        yield mock


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(tmpdir):
    return tmpdir.join('embedding-registry.json')


@pytest.fixture(name='data_dir')
def _data_dir(tmpdir):
    return tmpdir.join('data')


@pytest.fixture(name='download_dir')
def _download_dir(data_dir):
    return data_dir.join('download')


class TestEmbeddingManager:
    def test_should_download_and_install_embedding(
            self,
            copy_file_mock: MagicMock,
            embedding_registry_path: LocalPath,
            download_dir: LocalPath):
        download_file = download_dir.join(os.path.basename(EXTERNAL_TXT_URL_1))
        embedding_manager = EmbeddingManager(embedding_registry_path, download_dir=download_dir)
        embedding_manager.download_and_install_embedding(EXTERNAL_TXT_URL_1)
        copy_file_mock.assert_called_with(EXTERNAL_TXT_URL_1, download_file)

        embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
        assert embedding_config
        assert embedding_config['name'] == EMBEDDING_NAME_1
        assert embedding_config['path'] == str(download_file)

    def test_should_unzip_embedding(
            self,
            copy_file_mock: MagicMock,
            embedding_registry_path: LocalPath,
            download_dir: LocalPath):
        download_file = download_dir.join(os.path.basename(EXTERNAL_TXT_URL_1))
        embedding_manager = EmbeddingManager(embedding_registry_path, download_dir=download_dir)
        embedding_manager.download_and_install_embedding(EXTERNAL_TXT_GZ_URL_1)
        copy_file_mock.assert_called_with(EXTERNAL_TXT_GZ_URL_1, download_file)

        embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
        assert embedding_config
        assert embedding_config['name'] == EMBEDDING_NAME_1
        assert embedding_config['path'] == str(download_file)
