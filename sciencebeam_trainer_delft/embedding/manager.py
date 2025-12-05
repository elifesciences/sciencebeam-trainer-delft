import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.embedding.embedding import Embeddings

from sciencebeam_trainer_delft.utils.io import (
    is_external_location,
    strip_compression_filename_ext
)


LOGGER = logging.getLogger(__name__)


DEFAULT_EMBEDDING_REGISTRY = 'delft/resources-registry.json'
DEFAULT_DOWNLOAD_DIR = 'data/download'
DEFAULT_EMBEDDING_LMDB_PATH = 'data/db'

DEFAULT_MIN_LMDB_CACHE_SIZE = 1024 * 1024


def _find_embedding_index(embedding_list: List[dict], name: str) -> int:
    matching_indices = [i for i, x in enumerate(embedding_list) if x['name'] == name]
    if matching_indices:
        return matching_indices[0]
    return -1


def _get_embedding_name_for_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    if name.endswith('.mdb'):
        return os.path.splitext(name)[0]
    return name


def _get_embedding_format_for_filename(filename: str) -> str:
    if '.bin' in filename:
        return 'bin'
    return 'vec'


def _get_embedding_type_for_filename(filename: str) -> str:
    if 'glove' in filename.lower():
        return 'glove'
    return filename


def _get_embedding_config_for_filename(filename: str) -> dict:
    return {
        'name': _get_embedding_name_for_filename(filename),
        'format': _get_embedding_format_for_filename(filename),
        'type': _get_embedding_type_for_filename(filename),
        'lang': 'en'
    }


class EmbeddingManager:
    def __init__(
            self,
            path: str = DEFAULT_EMBEDDING_REGISTRY,
            download_manager: DownloadManager = None,
            download_dir: str = DEFAULT_DOWNLOAD_DIR,
            default_embedding_lmdb_path: str = DEFAULT_EMBEDDING_LMDB_PATH,
            min_lmdb_cache_size: int = DEFAULT_MIN_LMDB_CACHE_SIZE):
        assert download_manager
        self.path = path
        self.download_manager = download_manager
        self.download_dir = download_dir
        self.default_embedding_lmdb_path = default_embedding_lmdb_path
        self.min_lmdb_cache_size = min_lmdb_cache_size

    def _load(self) -> dict:
        return json.loads(Path(self.path).read_text())

    def _save(self, registry_data: dict):
        LOGGER.debug('saving registry data: %s', registry_data)
        return Path(self.path).write_text(json.dumps(registry_data, indent=4))

    def _get_registry_data(self) -> dict:
        try:
            return self._load()
        except FileNotFoundError:
            return {}

    def get_embedding_lmdb_path(self):
        registry_data = self._get_registry_data()
        return registry_data.get('embedding-lmdb-path', self.default_embedding_lmdb_path)

    def is_embedding_lmdb_cache_enabled(self):
        return self.get_embedding_lmdb_path()

    def set_embedding_lmdb_cache_path(self, embedding_lmdb_cache_path: str):
        registry_data = self._get_registry_data()
        registry_data['embedding-lmdb-path'] = embedding_lmdb_cache_path
        self._save(registry_data)

    def disable_embedding_lmdb_cache(self):
        self.set_embedding_lmdb_cache_path(None)

    def add_embedding_config(self, embedding_config: dict):
        LOGGER.debug('adding config: %s', embedding_config)
        embedding_name = embedding_config['name']
        registry_data = self._get_registry_data()
        if 'embeddings' not in registry_data:
            registry_data['embeddings'] = []
        embedding_list = registry_data['embeddings']
        index = _find_embedding_index(embedding_list, embedding_name)
        if index < 0:
            embedding_list.append(embedding_config)
        else:
            embedding_list[index] = embedding_config
        if 'embedding-lmdb-path' not in registry_data:
            registry_data['embedding-lmdb-path'] = self.default_embedding_lmdb_path
        self._save(registry_data)

    def get_embedding_config(self, embedding_name: str) -> Optional[dict]:
        embedding_list = self._get_registry_data().get('embeddings', [])
        index = _find_embedding_index(embedding_list, embedding_name)
        if index < 0:
            LOGGER.info('embedding not found with name "%s" in %s', embedding_name, embedding_list)
            return None
        return embedding_list[index]

    def set_embedding_aliases(self, embedding_aliases: Dict[str, str]):
        registry_data = self._get_registry_data()
        registry_data['embedding-aliases'] = embedding_aliases
        self._save(registry_data)

    def get_embedding_aliases(self) -> dict:
        registry_data = self._get_registry_data()
        return registry_data.get('embedding-aliases', {})

    def resolve_alias(self, embedding_name: str):
        return self.get_embedding_aliases().get(embedding_name, embedding_name)

    def is_lmdb_cache_file(self, embedding_url: str) -> bool:
        return strip_compression_filename_ext(
            embedding_url
        ).endswith('.mdb')

    def download_and_install_source_embedding(self, embedding_url: str) -> str:
        download_file = self.download_manager.download_if_url(embedding_url)
        filename = os.path.basename(strip_compression_filename_ext(embedding_url))
        embedding_config = _get_embedding_config_for_filename(filename)
        embedding_name = embedding_config['name']
        self.add_embedding_config({
            **embedding_config,
            'path': str(download_file)
        })
        return embedding_name

    def _download_lmdb_cache_embedding(self, embedding_name: str, embedding_url: str):
        self.download_manager.download(
            embedding_url,
            local_file=str(self.get_embedding_lmdb_cache_data_path(embedding_name))
        )

    def download_and_install_lmdb_cache_embedding(self, embedding_url: str) -> str:
        embedding_config = _get_embedding_config_for_filename(embedding_url)
        embedding_name = embedding_config['name']
        self._download_lmdb_cache_embedding(embedding_name, embedding_url)
        self.add_embedding_config(embedding_config)
        return embedding_name

    def download_and_install_embedding(self, embedding_url: str) -> str:
        if self.is_lmdb_cache_file(embedding_url):
            return self.download_and_install_lmdb_cache_embedding(embedding_url)
        return self.download_and_install_source_embedding(embedding_url)

    def download_and_install_embedding_if_url(self, embedding_url_or_name: str):
        if is_external_location(embedding_url_or_name):
            return self.download_and_install_embedding(embedding_url_or_name)
        return embedding_url_or_name

    def get_embedding_lmdb_cache_data_path(self, embedding_name: str):
        embedding_lmdb_path = self.get_embedding_lmdb_path()
        if not embedding_lmdb_path:
            return None
        embedding_lmdb_dir = Path(embedding_lmdb_path).joinpath(embedding_name)
        return embedding_lmdb_dir.joinpath('data.mdb')

    def has_lmdb_cache(self, embedding_name: str):
        embedding_lmdb_file = self.get_embedding_lmdb_cache_data_path(
            embedding_name
        )
        if not embedding_lmdb_file:
            return False
        exists = embedding_lmdb_file.is_file()
        size = exists and embedding_lmdb_file.stat().st_size
        valid = exists and size >= self.min_lmdb_cache_size
        LOGGER.debug(
            'embedding_lmdb_file: %s (exists: %s, valid: %s, size: %s)',
            embedding_lmdb_file, exists, valid, size
        )
        if valid:
            LOGGER.info(
                'has already lmdb cache: %s (%s bytes)',
                embedding_lmdb_file, size
            )
        return valid

    def is_downloaded(self, embedding_name: str):
        embedding_config = self.get_embedding_config(embedding_name)
        if not embedding_config:
            return False
        embedding_path = embedding_config.get('path')
        if not embedding_path or not Path(embedding_path).exists():
            return False
        LOGGER.info('already downloaded: %s', embedding_name)
        return True

    def is_downloaded_or_has_lmdb_cache(self, embedding_name: str):
        return self.is_downloaded(embedding_name) or self.has_lmdb_cache(embedding_name)

    def _ensure_external_url_available(self, embedding_url: str):
        embedding_name = _get_embedding_name_for_filename(embedding_url)
        if not self.is_downloaded_or_has_lmdb_cache(embedding_name):
            return self.download_and_install_embedding(embedding_url)
        if not self.get_embedding_config(embedding_name):
            self.add_embedding_config(
                _get_embedding_config_for_filename(embedding_url)
            )
        return embedding_name

    def _ensure_registered_embedding_available(self, embedding_name: str):
        self.validate_embedding(embedding_name)
        if self.is_downloaded_or_has_lmdb_cache(embedding_name):
            return embedding_name
        embedding_config = self.get_embedding_config(embedding_name)
        assert embedding_config, "embedding_config required for %s" % embedding_name
        embedding_url = embedding_config.get('url')
        assert embedding_url, "embedding_url required for %s" % embedding_name

        if self.is_lmdb_cache_file(embedding_url):
            self._download_lmdb_cache_embedding(embedding_name, embedding_url)
            return embedding_name

        embedding_path = embedding_config.get('path')
        assert embedding_path, "embedding_path required for %s" % embedding_name
        self.download_manager.download(embedding_url, local_file=embedding_path)
        return embedding_name

    def ensure_lmdb_cache_if_enabled(self, embedding_name: str):
        if not self.get_embedding_lmdb_path():
            return
        self.get_embeddings_for_name(embedding_name)
        assert self.has_lmdb_cache(embedding_name)

    def ensure_available(self, embedding_url_or_name: str):
        if is_external_location(embedding_url_or_name):
            return self._ensure_external_url_available(embedding_url_or_name)
        return self._ensure_registered_embedding_available(self.resolve_alias(
            embedding_url_or_name
        ))

    def validate_embedding(self, embedding_name):
        if not self.get_embedding_config(embedding_name):
            raise ValueError('invalid embedding name: %s' % embedding_name)

    def get_embeddings_for_name(self, embedding_name: str, **kwargs) -> Embeddings:
        self.validate_embedding(embedding_name)
        registry = self._get_registry_data()
        return Embeddings(embedding_name, resource_registry=registry, **kwargs)
