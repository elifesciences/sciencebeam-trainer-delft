import logging
import sys
from itertools import islice
from typing import List, Iterable

import subprocess


LOGGER = logging.getLogger(__name__)


def format_feature_line(feature_line: List[str]) -> str:
    return '\t'.join(feature_line)


class WapitiModel:
    def __init__(self, process: subprocess.Popen):
        self.process = process

    def iter_read_lines(self) -> Iterable[str]:
        while self.process.poll() is None:
            yield self.process.stdout.readline().decode('utf-8').rstrip()

    def iter_label(self, data: str) -> str:
        self.process.stdin.write((data + '\n\n\n').encode('utf-8'))
        self.process.stdin.flush()
        yield from self.iter_read_lines()

    def label_lines(self, lines: List[str]) -> List[str]:
        LOGGER.debug('lines: %s', lines)
        self.process.stdin.write(('\n'.join(lines) + '\n\n').encode('utf-8'))
        self.process.stdin.flush()
        labelled_lines = list(islice(self.iter_read_lines(), len(lines) + 1))
        LOGGER.debug('labelled_lines: %s', labelled_lines)
        return labelled_lines[:-1]

    def label_raw_text(self, data: str) -> str:
        return '\n'.join(self.label_lines(data.splitlines()))

    def label_features(self, features: List[List[str]]) -> str:
        lines = [
            format_feature_line(feature_line)
            for feature_line in features
        ]
        return [
            labelled_line.rsplit('\t', maxsplit=1)[-1]
            for labelled_line in self.label_lines(lines)
        ]


class WapitiWrapper:
    def __init__(self, wapiti_binary_path: str = 'wapiti'):
        self.wapiti_binary_path = wapiti_binary_path

    def check_available(self):
        self.run_wapiti(['--version'])

    def load_model(self, model_path: str, output_only_labels: bool = True) -> WapitiModel:
        args = [
            'label',
            '--model',
            str(model_path)
        ]
        if output_only_labels:
            args.append('--label')
        command = [self.wapiti_binary_path] + args
        return WapitiModel(process=subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ))

    def run_wapiti(self, args: List[str]):
        command = [self.wapiti_binary_path] + args
        LOGGER.debug('calling wapiti: %s', command)
        subprocess.run(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True
        )

    def train(
            self,
            data_path: str,
            output_model_path: str,
            template_path: str = None):
        args = ['train']
        if template_path:
            args.append('--pattern')
            args.append(str(template_path))
        args.append(str(data_path))
        args.append(str(output_model_path))
        self.run_wapiti(args)
