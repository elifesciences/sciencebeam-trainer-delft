import argparse
import concurrent.futures
import logging
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from sciencebeam_trainer_delft.utils.io import open_file
from sciencebeam_trainer_delft.utils.cli import (
    add_default_arguments,
    process_default_args,
    initialize_and_call_main
)


LOGGER = logging.getLogger(__name__)


class OutputFormats:
    TEXT = 'text'
    JSON = 'json'


ALL_OUTPUT_FORMATS = [OutputFormats.TEXT, OutputFormats.JSON]


def read_json(path: str) -> dict:
    with open_file(path, mode='r') as fp:
        return json.load(fp)


def get_checkpoints_json(checkpoint_path: str) -> dict:
    return read_json(os.path.join(checkpoint_path, 'checkpoints.json'))


def get_checkpoint_urls(checkpoints_json: dict) -> List[str]:
    return sorted({
        checkpoint['path']
        for checkpoint in checkpoints_json.get('checkpoints', {})
    })


def get_last_checkpoint_url(checkpoints_json: dict) -> Optional[str]:
    checkpoint_urls = get_checkpoint_urls(checkpoints_json)
    return checkpoint_urls[-1] if checkpoint_urls else None


def get_checkpoint_meta(checkpoint_path: str) -> dict:
    return read_json(os.path.join(checkpoint_path, 'meta.json'))


def get_checkpoint_meta_map(
        checkpoint_urls: List[str],
        max_workers: int) -> Dict[str, dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda path: (path, get_checkpoint_meta(path)),
            checkpoint_urls
        )
        return dict(tqdm(results, total=len(checkpoint_urls)))


def get_checkpoint_meta_map_sorted_by_f1(checkpoint_meta_map: Dict[str, dict]):
    return OrderedDict(sorted(
        checkpoint_meta_map.items(),
        key=lambda item: item[1].get('f1') or 0
    ))


def get_checkpoint_summary_list(
    checkpoint_meta_map_sorted_by_f1: Dict[str, dict],
    last_checkpoint: dict,
    limit: int
) -> List[dict]:
    last_checkpoint_path = last_checkpoint.get('path')
    best_meta = list(checkpoint_meta_map_sorted_by_f1.values())[-1]
    best_f1 = best_meta.get('f1')
    return [
        {
            **meta,
            'path': path,
            'is_last': path == last_checkpoint_path,
            'is_best': meta.get('f1') == best_f1
        }
        for path, meta in (
            list(checkpoint_meta_map_sorted_by_f1.items())[-limit:]
        )
    ]


def format_checkpoint_summary_as_text(
        checkpoint_summary_list: List[dict]) -> str:
    return 'best checkpoints:\n%s' % '\n\n'.join([
        '%05d: %s (%s)%s%s' % (
            int(checkpoint_summary.get('epoch', 0)),
            checkpoint_summary.get('f1'),
            checkpoint_summary.get('path'),
            ' (last)' if checkpoint_summary.get('is_last') else '',
            ' (best)' if checkpoint_summary.get('is_best') else ''
        )
        for checkpoint_summary in checkpoint_summary_list
    ])


def format_checkpoint_summary(
        checkpoint_summary_list: List[dict],
        output_format: str) -> str:
    if output_format == OutputFormats.TEXT:
        return format_checkpoint_summary_as_text(
            checkpoint_summary_list
        )
    if output_format == OutputFormats.JSON:
        return json.dumps(checkpoint_summary_list, indent=2)
    raise ValueError('unsupported output format: %s' % output_format)


def checkpoint_summary(
        checkpoint_path: str,
        max_workers: int,
        limit: int,
        output_format: str):
    LOGGER.info('checkpoint_path: %s', checkpoint_path)
    checkpoints_json = get_checkpoints_json(checkpoint_path)
    LOGGER.debug('checkpoints_json: %s', checkpoints_json)
    checkpoint_urls = get_checkpoint_urls(checkpoints_json)
    LOGGER.debug('checkpoint_urls: %s', checkpoint_urls)
    last_checkpoint = checkpoints_json.get('last_checkpoint')
    if last_checkpoint:
        LOGGER.info('last checkpoint: %s', last_checkpoint)
    if not checkpoint_urls:
        raise RuntimeError('no checkpoints found')
    checkpoint_meta_map = get_checkpoint_meta_map(
        checkpoint_urls,
        max_workers=max_workers
    )
    LOGGER.debug('checkpoint_meta_map: %s', checkpoint_meta_map)
    checkpoint_meta_map_sorted_by_f1 = get_checkpoint_meta_map_sorted_by_f1(
        checkpoint_meta_map
    )
    checkpoint_summary_list = get_checkpoint_summary_list(
        checkpoint_meta_map_sorted_by_f1=checkpoint_meta_map_sorted_by_f1,
        last_checkpoint=last_checkpoint,
        limit=limit,
    )
    formatted_summary = format_checkpoint_summary(
        checkpoint_summary_list=checkpoint_summary_list,
        output_format=output_format
    )
    print(formatted_summary)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoints related summary"
    )

    parser.add_argument(
        "--output-format",
        choices=ALL_OUTPUT_FORMATS,
        default=OutputFormats.TEXT,
        help="The desired output format."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="The base path of the checkpoints."
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum number of workers for IO requests"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number results to show"
    )

    add_default_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    checkpoint_summary(
        checkpoint_path=args.checkpoint,
        max_workers=args.max_workers,
        limit=args.limit,
        output_format=args.output_format
    )


def main(argv: List[str] = None):
    args = parse_args(argv)
    process_default_args(args)
    run(args)


if __name__ == "__main__":
    initialize_and_call_main(main)
