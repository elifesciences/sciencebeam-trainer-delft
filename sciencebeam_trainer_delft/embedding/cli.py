import argparse
import logging
from abc import abstractmethod, ABC
from typing import List


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


class SubCommand(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace):
        pass


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


SUB_COMMAND_BY_NAME = {
    sub_command.name: sub_command
    for sub_command in SUB_COMMANDS
}


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage Embeddings"
    )

    subparsers = parser.add_subparsers(
        dest='command', required=True
    )
    for sub_command in SUB_COMMANDS:
        sub_parser = subparsers.add_parser(
            sub_command.name, help=sub_command.description
        )
        sub_command.add_arguments(sub_parser)

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)
    return args


def run(args: argparse.Namespace):
    sub_command = SUB_COMMAND_BY_NAME[args.command]
    sub_command.run(args)


def main(argv: List[str] = None):
    LOGGER.debug('argv: %s', argv)
    args = parse_args(argv)

    if args.debug:
        LOGGER.setLevel('DEBUG')
        logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')

    run(args)


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(level='INFO')

    main()
