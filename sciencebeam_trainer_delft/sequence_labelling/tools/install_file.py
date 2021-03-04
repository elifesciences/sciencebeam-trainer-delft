import argparse
import logging
from pathlib import Path
from typing import List

from sciencebeam_trainer_delft.utils.io import copy_file

from sciencebeam_trainer_delft.utils.cli import (
    add_default_arguments,
    process_default_args,
    initialize_and_call_main
)


LOGGER = logging.getLogger(__name__)

SOURCE_URL_META_FILENAME_EXT = '.source-url'


def install_file(source_file_path: str, target_file_path: str, force: bool):
    _target_file_path = Path(target_file_path)
    _target_file_path.parent.mkdir(parents=True, exist_ok=True)
    _target_source_url_file_path = Path(target_file_path + SOURCE_URL_META_FILENAME_EXT)
    if not force and _target_source_url_file_path.exists():
        current_source_url = _target_source_url_file_path.read_text().strip()
        if current_source_url == str(source_file_path):
            LOGGER.debug(
                'current source_url of %s already (skipping): %s',
                target_file_path, current_source_url
            )
            return
    copy_file(str(source_file_path), str(target_file_path))
    _target_source_url_file_path.write_text(str(source_file_path))


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install file")
    parser.add_argument('--source', required=True, help='Source file url or path')
    parser.add_argument('--target', required=True, help='Target file path')
    parser.add_argument('--force', action='store_true', help='Force override files')
    add_default_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    install_file(source_file_path=args.source, target_file_path=args.target, force=args.force)


def main(argv: List[str] = None):
    args = parse_args(argv)
    process_default_args(args)
    run(args)


if __name__ == "__main__":
    initialize_and_call_main(main)
