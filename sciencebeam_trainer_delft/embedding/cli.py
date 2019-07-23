import argparse
import logging
from typing import List


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


SUB_COMMANDS = [
    DisableLmdbCacheSubCommand(),
    SetLmdbPathSubCommand()
]


def main(argv: List[str] = None):
    processor = SubCommandProcessor(
        SUB_COMMANDS,
        description='Manage Embeddings'
    )
    processor.main(argv)


if __name__ == "__main__":
    initialize_and_call_main(main)
