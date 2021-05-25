from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sciencebeam_trainer_delft.embedding import (
    EmbeddingManager
)

from .test_data import TEST_DATA_PATH


EMBEDDING_NAME_1 = 'embedding1'
EMBEDDING_ALIAS_1 = 'alias1'
EXTERNAL_TXT_URL_1 = 'http://host/%s.txt' % EMBEDDING_NAME_1
EXTERNAL_TXT_GZ_URL_1 = EXTERNAL_TXT_URL_1 + '.gz'
EXTERNAL_MDB_URL_1 = 'http://host/%s.mdb' % EMBEDDING_NAME_1
EXTERNAL_MDB_GZ_URL_1 = EXTERNAL_MDB_URL_1 + '.gz'
DOWNLOAD_FILENAME_1 = 'xyz-%s.txt' % EMBEDDING_NAME_1


EMBEDDING_1 = {
    'name': EMBEDDING_NAME_1,
    'path': str(Path(TEST_DATA_PATH).joinpath('%s.txt' % EMBEDDING_NAME_1)),
    'url': EXTERNAL_TXT_URL_1,
    'type': 'glove',
    'format': 'vec',
    'lang': 'en'
}


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(temp_dir: Path):
    return temp_dir.joinpath('embedding-registry.json')


@pytest.fixture(name='download_path')
def _download_path(temp_dir: Path):
    p = temp_dir.joinpath('download')
    p.mkdir()
    return p


@pytest.fixture(name='embedding_lmdb_path')
def _embedding_lmdb_path(temp_dir: Path):
    return temp_dir.joinpath('data/db')


@pytest.fixture(name='download_path_1')
def _download_path_1(download_path: Path):
    return download_path.joinpath(DOWNLOAD_FILENAME_1)


@pytest.fixture(name='download_manager')
def _download_manager(download_path: Path):
    download_manager = MagicMock(name='download_manager')
    download_manager.download_if_url.return_value = str(
        download_path.joinpath(DOWNLOAD_FILENAME_1)
    )
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


def _create_dummy_lmdb_cache_file(embedding_manager: EmbeddingManager, embedding_name: str):
    embedding_cache_file = Path(
        embedding_manager.get_embedding_lmdb_path()
    ).joinpath(embedding_name).joinpath('data.mdb')
    embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
    embedding_cache_file.touch()


class TestEmbeddingManager:
    class TestDisableEmbeddingLmbdCache:
        def test_should_disable_lmdb_cache(
                self,
                embedding_manager: EmbeddingManager):
            embedding_manager.disable_embedding_lmdb_cache()
            assert embedding_manager.get_embedding_lmdb_path() is None

    class TestDownloadAndInstallEmbedding:
        def test_should_download_and_install_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager,
                download_path_1: Path):
            embedding_manager.download_and_install_embedding(EXTERNAL_TXT_URL_1)
            download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_URL_1)

            embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
            assert embedding_config
            assert embedding_config['name'] == EMBEDDING_NAME_1
            assert embedding_config['path'] == str(download_path_1)

        def test_should_unzip_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager,
                download_path_1: Path):
            embedding_manager.download_and_install_embedding(EXTERNAL_TXT_GZ_URL_1)
            download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_GZ_URL_1)

            embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
            assert embedding_config
            assert embedding_config['name'] == EMBEDDING_NAME_1
            assert embedding_config['path'] == str(download_path_1)

        def test_should_unzip_mdb_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager):
            embedding_manager.download_and_install_embedding(EXTERNAL_MDB_GZ_URL_1)
            download_manager.download.assert_called_with(
                EXTERNAL_MDB_GZ_URL_1,
                local_file=str(
                    embedding_manager.get_embedding_lmdb_cache_data_path(EMBEDDING_NAME_1)
                )
            )

            embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
            assert embedding_config
            assert embedding_config['name'] == EMBEDDING_NAME_1

    class TestEnsureAvailable:
        def test_should_download_and_register_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager,
                download_path_1: Path):
            assert embedding_manager.ensure_available(EXTERNAL_TXT_URL_1) == EMBEDDING_NAME_1
            download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_URL_1)

            embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
            assert embedding_config
            assert embedding_config['name'] == EMBEDDING_NAME_1
            assert embedding_config['path'] == str(download_path_1)

        def test_should_not_download_if_already_downloaded(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager):
            embedding_manager.download_and_install_embedding(EXTERNAL_TXT_URL_1)
            download_manager.reset_mock()
            embedding_config = embedding_manager.get_embedding_config(EMBEDDING_NAME_1)
            assert embedding_config
            Path(embedding_config['path']).touch()

            assert embedding_manager.ensure_available(EXTERNAL_TXT_URL_1) == EMBEDDING_NAME_1
            download_manager.download_if_url.assert_not_called()

        def test_should_not_download_if_not_downloaded_but_has_lmdb_cache(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager):
            _create_dummy_lmdb_cache_file(embedding_manager, EMBEDDING_NAME_1)
            assert embedding_manager.ensure_available(EXTERNAL_TXT_URL_1) == EMBEDDING_NAME_1
            download_manager.download_if_url.assert_not_called()
            embedding_manager.validate_embedding(EMBEDDING_NAME_1)

        def test_should_download_if_config_exists_but_not_downloaded(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager):
            embedding_manager.download_and_install_embedding(EXTERNAL_TXT_URL_1)
            download_manager.reset_mock()

            assert embedding_manager.ensure_available(EXTERNAL_TXT_URL_1) == EMBEDDING_NAME_1
            download_manager.download_if_url.assert_called_with(EXTERNAL_TXT_URL_1)

        def test_should_download_registered_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager,
                download_path_1: Path):
            embedding_manager.add_embedding_config({
                'name': EMBEDDING_NAME_1,
                'path': str(download_path_1),
                'url': EXTERNAL_TXT_URL_1
            })
            assert embedding_manager.ensure_available(EMBEDDING_NAME_1) == EMBEDDING_NAME_1
            download_manager.download.assert_called_with(
                EXTERNAL_TXT_URL_1, local_file=str(download_path_1)
            )

        def test_should_resolve_registered_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager,
                download_path_1: Path):
            embedding_manager.set_embedding_aliases({
                EMBEDDING_ALIAS_1: EMBEDDING_NAME_1
            })
            embedding_manager.add_embedding_config({
                'name': EMBEDDING_NAME_1,
                'path': str(download_path_1),
                'url': EXTERNAL_TXT_URL_1
            })
            assert embedding_manager.ensure_available(EMBEDDING_ALIAS_1) == EMBEDDING_NAME_1
            download_manager.download.assert_called_with(
                EXTERNAL_TXT_URL_1, local_file=str(download_path_1)
            )

        def test_should_download_registered_mdb_embedding(
                self,
                download_manager: MagicMock,
                embedding_manager: EmbeddingManager):
            embedding_manager.add_embedding_config({
                'name': EMBEDDING_NAME_1,
                'url': EXTERNAL_MDB_GZ_URL_1
            })
            assert embedding_manager.ensure_available(EMBEDDING_NAME_1) == EMBEDDING_NAME_1
            download_manager.download.assert_called_with(
                EXTERNAL_MDB_GZ_URL_1,
                local_file=str(
                    embedding_manager.get_embedding_lmdb_cache_data_path(
                        EMBEDDING_NAME_1
                    )
                )
            )

    class TestEnsureLmdbCacheIfEnabled:
        def test_should_generate_lmdb_cache(
                self,
                embedding_manager: EmbeddingManager):
            embedding_manager.add_embedding_config(EMBEDDING_1)
            assert embedding_manager.ensure_available(EMBEDDING_NAME_1) == EMBEDDING_NAME_1
            assert not embedding_manager.has_lmdb_cache(EMBEDDING_NAME_1)
            embedding_manager.ensure_lmdb_cache_if_enabled(EMBEDDING_NAME_1)
            assert embedding_manager.has_lmdb_cache(EMBEDDING_NAME_1)

        def test_should_skip_if_lmdb_cache_is_disabled(
                self,
                embedding_manager: EmbeddingManager):
            embedding_manager.add_embedding_config(EMBEDDING_1)
            embedding_manager.disable_embedding_lmdb_cache()
            assert embedding_manager.ensure_available(EMBEDDING_NAME_1) == EMBEDDING_NAME_1
            assert not embedding_manager.has_lmdb_cache(EMBEDDING_NAME_1)
            embedding_manager.ensure_lmdb_cache_if_enabled(EMBEDDING_NAME_1)
            assert not embedding_manager.has_lmdb_cache(EMBEDDING_NAME_1)

    class TestResolveAlias:
        def test_should_return_passed_in_embedding_name_by_default(
                self,
                embedding_manager: EmbeddingManager):
            assert embedding_manager.resolve_alias(EMBEDDING_NAME_1) == EMBEDDING_NAME_1

        def test_should_resolve_embedding_alias(
                self,
                embedding_manager: EmbeddingManager):
            embedding_manager.set_embedding_aliases({
                EMBEDDING_ALIAS_1: EMBEDDING_NAME_1
            })
            assert embedding_manager.resolve_alias(EMBEDDING_ALIAS_1) == EMBEDDING_NAME_1
