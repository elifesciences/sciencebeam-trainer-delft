import argparse
import logging
import sys
from abc import abstractmethod, ABC
from typing import List, Callable


LOGGER = logging.getLogger(__name__)


def add_debug_argument(parser: argparse.ArgumentParser):
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")


def add_default_arguments(parser: argparse.ArgumentParser):
    add_debug_argument(parser)


def get_project_package():
    return 'sciencebeam_trainer_delft'


def process_debug_argument(args: argparse.Namespace):
    if args.debug:
        logging.getLogger('__main__').setLevel('DEBUG')
        logging.getLogger(get_project_package()).setLevel('DEBUG')


def process_default_args(args: argparse.Namespace):
    process_debug_argument(args)


def default_main(
        parse_args: Callable[[List[str]], argparse.Namespace],
        run: Callable[[argparse.Namespace], None],
        argv: List[str] = None):
    LOGGER.debug('argv: %s', argv)
    args = parse_args(argv)
    process_default_args(args)
    run(args)


def configure_main_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')


def initialize_and_call_main(main: callable):
    configure_main_logging()
    main()


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


class SubCommandProcessor:
    def __init__(self, sub_commands: List[SubCommand], description: str = None):
        self.sub_commands = sub_commands
        self.sub_command_by_name = {
            sub_command.name: sub_command
            for sub_command in sub_commands
        }
        self.description = description

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.description
        )
        self.add_sub_command_parsers(parser)
        return parser

    def parse_args(self, argv: List[str] = None) -> argparse.Namespace:
        return self.get_parser().parse_args(argv)

    def add_sub_command_parsers(self, parser: argparse.ArgumentParser):
        kwargs = {}
        if sys.version_info >= (3, 7):
            kwargs['required'] = True
        subparsers = parser.add_subparsers(dest='command', **kwargs)
        subparsers.required = True
        self.add_sub_command_parsers_to_subparsers(subparsers)

    def add_sub_command_parsers_to_subparsers(self, subparsers: argparse.ArgumentParser):
        for sub_command in self.sub_commands:
            sub_parser = subparsers.add_parser(
                sub_command.name, help=sub_command.description
            )
            sub_command.add_arguments(sub_parser)
            add_default_arguments(sub_parser)

    def run(self, args: argparse.Namespace):
        sub_command = self.sub_command_by_name[args.command]
        sub_command.run(args)

    def main(self, argv: List[str] = None):
        args = self.parse_args(argv)
        process_default_args(args)
        self.run(args)
