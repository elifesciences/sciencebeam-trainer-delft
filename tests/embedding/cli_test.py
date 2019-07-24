from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager

import sciencebeam_trainer_delft.embedding.cli as cli_module
from sciencebeam_trainer_delft.embedding.cli import main


EMBEDDING_NAME_1 = 'embedding1'


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(temp_dir: Path):
    return temp_dir.joinpath('embedding-registry.json')


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
        embedding_registry_path,
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

    def test_should_preload_embedding(
            self,
            embedding_registry_path: Path,
            embedding_manager_mock: MagicMock):
        main([
            'preload',
            '--registry-path=%s' % embedding_registry_path,
            '--embedding=%s' % EMBEDDING_NAME_1
        ])
        embedding_manager_mock.ensure_available.assert_called_with(EMBEDDING_NAME_1)
