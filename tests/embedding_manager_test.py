from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sciencebeam_trainer_delft.embedding_manager import (
    EmbeddingManager
)


EMBEDDING_NAME_1 = 'embedding1'
EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1
EXTERNAL_TXT_GZ_URL_1 = EXTERNAL_TXT_URL_1 + '.gz'
DOWNLOAD_PATH_1 = '/path/to/download/%s.txt' % EMBEDDING_NAME_1


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(temp_dir: Path):
    return temp_dir.joinpath('embedding-registry.json')


@pytest.fixture(name='download_manager')
def _download_manager():
    download_manager = MagicMock(name='download_manager')
    download_manager.download_if_url.return_value = DOWNLOAD_PATH_1
    return download_manager


class TestEmbeddingManager:
    def test_should_disable_lmdb_cache(
            self,
            download_manager: MagicMock,
            embedding_registry_path: Path):
        embedding_manager = EmbeddingManager(
            embedding_registry_path, download_manager=download_manager
        )
        embedding_manager.disable_embedding_lmdb_cache()
        assert embedding_manager.get_embedding_lmdb_path() is None

    def test_should_download_and_install_embedding(
            self,
            download_manager: MagicMock,
            embedding_registry_path: Path):
        embedding_manager = EmbeddingManager(
            embedding_registry_path, download_manager=download_manager
        )
        embedding_manager.download_and_install_embedding(EXTERNAL_TXT_URL_1)
        download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_URL_1)

        embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
        assert embedding_config
        assert embedding_config['name'] == EMBEDDING_NAME_1
        assert embedding_config['path'] == DOWNLOAD_PATH_1

    def test_should_unzip_embedding(
            self,
            download_manager: MagicMock,
            embedding_registry_path: Path):
        embedding_manager = EmbeddingManager(
            embedding_registry_path, download_manager=download_manager
        )
        embedding_manager.download_and_install_embedding(EXTERNAL_TXT_GZ_URL_1)
        download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_GZ_URL_1)

        embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
        assert embedding_config
        assert embedding_config['name'] == EMBEDDING_NAME_1
        assert embedding_config['path'] == DOWNLOAD_PATH_1
