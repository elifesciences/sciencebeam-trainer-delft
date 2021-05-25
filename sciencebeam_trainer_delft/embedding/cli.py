import argparse
import logging
from typing import Dict, List

from sciencebeam_trainer_delft.utils.misc import parse_dict, merge_dicts

from sciencebeam_trainer_delft.utils.cli import (
    initialize_and_call_main,
    SubCommand,
    SubCommandProcessor
)

from sciencebeam_trainer_delft.embedding.manager import (
    EmbeddingManager,
    DownloadManager,
    DEFAULT_EMBEDDING_REGISTRY
)


LOGGER = logging.getLogger(__name__)


class Commands:
    DISABLE_LMDB_CACHE = 'disable-lmdb-cache'
    SET_LMDB_PATH = 'set-lmdb-path'
    PRELOAD = 'preload'
    OVERRIDE_EMBEDDING_URL = 'override-embedding-url'


def _add_registry_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--registry-path",
        default=DEFAULT_EMBEDDING_REGISTRY,
        help="Path to the embedding registry"
    )


def _get_embedding_manager(args: argparse.Namespace) -> EmbeddingManager:
    return EmbeddingManager(
        path=args.registry_path,
        download_manager=DownloadManager()
    )


class DisableLmdbCacheSubCommand(SubCommand):
    def __init__(self):
        super().__init__(Commands.DISABLE_LMDB_CACHE, 'Disable LMDB cache')

    def add_arguments(self, parser: argparse.ArgumentParser):
        _add_registry_path_argument(parser)

    def run(self, args: argparse.Namespace):
        embedding_manager = _get_embedding_manager(args)
        embedding_manager.disable_embedding_lmdb_cache()


class SetLmdbPathSubCommand(SubCommand):
    def __init__(self):
        super().__init__(Commands.SET_LMDB_PATH, 'Set LMDB cache path')

    def add_arguments(self, parser: argparse.ArgumentParser):
        _add_registry_path_argument(parser)
        parser.add_argument(
            "--lmdb-cache-path",
            required=True,
            help="Path to the LMDB cache"
        )

    def run(self, args: argparse.Namespace):
        embedding_manager = _get_embedding_manager(args)
        embedding_manager.set_embedding_lmdb_cache_path(
            args.lmdb_cache_path
        )


class PreloadSubCommand(SubCommand):
    def __init__(self):
        super().__init__(Commands.PRELOAD, 'Ensure embedding(s) are ready to use')

    def add_arguments(self, parser: argparse.ArgumentParser):
        _add_registry_path_argument(parser)
        parser.add_argument(
            "--embedding",
            required=True,
            help="Name of embedding(s) to preload"
        )

    def run(self, args: argparse.Namespace):
        embedding_manager = _get_embedding_manager(args)
        embedding_name = embedding_manager.ensure_available(args.embedding)
        embedding_manager.ensure_lmdb_cache_if_enabled(embedding_name)


def parse_embedding_url_override_expr(embedding_url_override_expr: str) -> Dict[str, str]:
    LOGGER.debug('embedding_url_override_expr: %s', embedding_url_override_expr)
    return parse_dict(embedding_url_override_expr, delimiter='|')


class OverrideEmbeddingUrlSubCommand(SubCommand):
    def __init__(self):
        super().__init__(
            Commands.OVERRIDE_EMBEDDING_URL,
            'Override the URL of embeddings so that they can be loaded from another location'
        )

    def add_arguments(self, parser: argparse.ArgumentParser):
        _add_registry_path_argument(parser)
        parser.add_argument(
            "--override-url",
            nargs='+',
            required=True,
            type=parse_embedding_url_override_expr,
            help=(
                "The urls to override, in the form: <embedding name>=<url>"
                "\n (multiple urls can be specified by using the pipe ('|') separator"
                " or using the --override-url parameter multiple times"
            )
        )

    def run(self, args: argparse.Namespace):
        url_by_embedding_name = merge_dicts(args.override_url)
        LOGGER.debug('url_by_embedding_name: %s', url_by_embedding_name)
        embedding_manager = _get_embedding_manager(args)
        for embedding_name, embedding_url in url_by_embedding_name.items():
            LOGGER.info('setting url of embedding %s to %s', embedding_name, embedding_url)
            embedding_config = embedding_manager.get_embedding_config(embedding_name)
            assert embedding_config
            embedding_manager.add_embedding_config({
                **embedding_config,
                'url': embedding_url
            })


SUB_COMMANDS = [
    DisableLmdbCacheSubCommand(),
    SetLmdbPathSubCommand(),
    PreloadSubCommand(),
    OverrideEmbeddingUrlSubCommand()
]


def main(argv: List[str] = None):
    processor = SubCommandProcessor(
        SUB_COMMANDS,
        description='Manage Embeddings'
    )
    processor.main(argv)


if __name__ == "__main__":
    initialize_and_call_main(main)
