import json
import logging
import os
from pathlib import Path
from typing import List

from sciencebeam_trainer_delft.utils import (
    copy_file,
    is_external_location,
    is_gzip_filename,
    strip_gzip_filename_ext
)


LOGGER = logging.getLogger(__name__)


DEFAULT_EMBEDDING_REGISTRY = 'embedding-registry.json'
DEFAULT_DOWNLOAD_DIR = 'data/download'


def _find_embedding_index(embedding_list: List[dict], name: str) -> int:
    matching_indices = [i for i, x in enumerate(embedding_list) if x['name'] == name]
    if matching_indices:
        return matching_indices[0]
    return -1


def _get_embedding_name_for_filename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def _get_embedding_format_for_filename(filename: str) -> str:
    if '.bin' in filename:
        return 'bin'
    return 'vec'


def _get_embedding_type_for_filename(filename: str) -> str:
    if 'glove' in filename.lower():
        return 'glove'
    return filename


def _get_embedding_config_for_filename(filename: str) -> str:
    return {
        'name': _get_embedding_name_for_filename(filename),
        'format': _get_embedding_format_for_filename(filename),
        'type': _get_embedding_type_for_filename(filename),
        'lang': 'en'
    }


class EmbeddingManager:
    def __init__(
            self, path: str = DEFAULT_EMBEDDING_REGISTRY,
            download_dir: str = DEFAULT_DOWNLOAD_DIR):
        self.path = path
        self.download_dir = download_dir

    def _load(self) -> dict:
        return json.loads(Path(self.path).read_text())

    def _save(self, registry_data: dict):
        LOGGER.debug('saving registry data: %s', registry_data)
        return Path(self.path).write_text(json.dumps(registry_data))

    def _get_registry_data(self) -> dict:
        try:
            return self._load()
        except FileNotFoundError:
            return {}

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
        self._save(registry_data)

    def get_embedding_config(self, embedding_name: str) -> dict:
        embedding_list = self._get_registry_data().get('embeddings', [])
        index = _find_embedding_index(embedding_list, embedding_name)
        if index < 0:
            LOGGER.info('embedding not found with name "%s" in %s', embedding_name, embedding_list)
            return None
        return embedding_list[index]

    def download_and_install_embedding(self, embedding_url: str) -> str:
        filename = os.path.basename(embedding_url)
        if is_gzip_filename(filename):
            filename = strip_gzip_filename_ext(filename)
        download_file = os.path.join(self.download_dir, filename)
        embedding_config = _get_embedding_config_for_filename(filename)
        embedding_name = embedding_config['name']
        if os.path.exists(download_file):
            LOGGER.info('file already exists: %s', download_file)
        else:
            LOGGER.info('copying %s to %s', embedding_url, download_file)
            copy_file(embedding_url, download_file)
        self.add_embedding_config({
            **embedding_config,
            'path': str(download_file)
        })
        return embedding_name

    def download_and_install_embedding_if_url(self, embedding_url_or_name: str):
        if is_external_location(embedding_url_or_name):
            return self.download_and_install_embedding(embedding_url_or_name)
        return embedding_url_or_name

    def validate_embedding(self, embedding_name):
        if not self.get_embedding_config(embedding_name):
            raise ValueError('invalid embedding name: %s')
