from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager

import sciencebeam_trainer_delft.embedding.cli as cli_module
from sciencebeam_trainer_delft.embedding.cli import main


EMBEDDING_NAME_1 = 'embedding1'

EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(temp_dir: Path):
    return temp_dir.joinpath('resources-registry.json')


@pytest.fixture(name='embedding_lmdb_path')
def _embedding_lmdb_path(temp_dir: Path):
    return temp_dir.joinpath('data/db')


@pytest.fixture(name='download_manager')
def _download_manager():
    download_manager = MagicMock(name='download_manager')
    return download_manager


@pytest.fixture(name='embedding_manager')
def _embedding_manager(
        download_manager: MagicMock,
        embedding_registry_path: Path,
        embedding_lmdb_path: Path):
    embedding_manager = EmbeddingManager(
        str(embedding_registry_path),
        download_manager=download_manager,
        default_embedding_lmdb_path=str(embedding_lmdb_path),
        min_lmdb_cache_size=0
    )
    return embedding_manager


@pytest.fixture(name='embedding_manager_class_mock')
def _embedding_manager_class_mock():
    with patch.object(cli_module, 'EmbeddingManager') as mock:
        yield mock


@pytest.fixture(name='embedding_manager_mock')
def _embedding_manager_mock(embedding_manager_class_mock: MagicMock):
    return embedding_manager_class_mock.return_value


class TestMain:
    def test_should_clear_lmdb_path(
            self,
            embedding_registry_path: Path,
            embedding_manager: EmbeddingManager):
        main([
            'disable-lmdb-cache',
            '--registry-path=%s' % embedding_registry_path
        ])
        assert embedding_manager.get_embedding_lmdb_path() is None

    def test_should_set_lmdb_path(
            self,
            embedding_registry_path: Path,
            embedding_manager: EmbeddingManager):
        main([
            'set-lmdb-path',
            '--registry-path=%s' % embedding_registry_path,
            '--lmdb-cache-path=data/updated-path'
        ])
        assert embedding_manager.get_embedding_lmdb_path() == 'data/updated-path'

    def test_should_set_embedding_url(
            self,
            embedding_registry_path: Path,
            embedding_manager: EmbeddingManager):
        embedding_manager.add_embedding_config({
            'name': EMBEDDING_NAME_1,
            'url': 'other'
        })
        main([
            'override-embedding-url',
            '--registry-path=%s' % embedding_registry_path,
            '--override-url=%s=%s' % (EMBEDDING_NAME_1, EXTERNAL_TXT_URL_1)
        ])
        embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
        assert embedding_config
        assert embedding_config['url'] == EXTERNAL_TXT_URL_1

    def test_should_preload_embedding(
            self,
            embedding_registry_path: Path,
            embedding_manager_mock: MagicMock):
        embedding_manager_mock.ensure_available.return_value = EMBEDDING_NAME_1
        main([
            'preload',
            '--registry-path=%s' % embedding_registry_path,
            '--embedding=%s' % EMBEDDING_NAME_1
        ])
        embedding_manager_mock.ensure_available.assert_called_with(EMBEDDING_NAME_1)
        embedding_manager_mock.ensure_lmdb_cache_if_enabled.assert_called_with(EMBEDDING_NAME_1)
