import logging
import threading
import os
import sys
from collections import Counter
from itertools import islice
from multiprocessing import cpu_count
from typing import IO, List, Iterable, Optional, cast

import subprocess


LOGGER = logging.getLogger(__name__)


DEFAULT_STOP_EPSILON_VALUE = '0.00001'
DEFAULT_STOP_WINDOW_SIZE = 20


DEFAULT_INVALID_CHARACTER_PLACEHOLDER = '?'

INVAID_CHARACTER_START_ORD = 0x6EE80


def format_feature_line(feature_line: List[str]) -> str:
    return '\t'.join(feature_line)


def replace_invalid_characters(text: str, placeholder: str = DEFAULT_INVALID_CHARACTER_PLACEHOLDER):
    return ''.join((
        c if ord(c) < INVAID_CHARACTER_START_ORD else placeholder
        for c in text
    ))


def lines_to_log(logger: logging.Logger, level: int, message: str, lines: Iterable[str]):
    LOGGER.debug('lines: %s', lines)
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        line = line.rstrip()
        logger.log(level, message, line)


class WapitiModel:
    def __init__(self, process: subprocess.Popen):
        self.process = process

    @property
    def process_stdin(self) -> IO:
        stdin = self.process.stdin
        assert stdin
        return stdin

    @property
    def process_stdout(self) -> IO:
        stdout = self.process.stdout
        assert stdout
        return stdout

    def iter_read_lines(self) -> Iterable[str]:
        while self.process.poll() is None:
            line = self.process_stdout.readline().decode('utf-8').rstrip()
            LOGGER.debug('read line: %s', line)
            yield line

    def iter_label(self, data: str) -> Iterable[str]:
        self.process_stdin.write((data + '\n\n\n').encode('utf-8'))
        self.process_stdin.flush()
        yield from self.iter_read_lines()

    def label_lines(self, lines: List[str], clean_input: bool = False) -> List[str]:
        LOGGER.debug('lines: %s', lines)
        for line in lines + ['', '']:
            if clean_input:
                cleaned_line = replace_invalid_characters(line, placeholder='?')
            else:
                cleaned_line = line
            try:
                LOGGER.debug('writing line: %s', cleaned_line)
                LOGGER.debug('line counts: %s', Counter(cleaned_line))
                self.process_stdin.write(
                    (cleaned_line + '\n').encode('utf-8')
                )
                self.process_stdin.flush()
            except BrokenPipeError:
                LOGGER.error('failed to write line: %s', [(c, hex(ord(c))) for c in cleaned_line])
                raise
        self.process_stdin.flush()
        labelled_lines = list(islice(self.iter_read_lines(), len(lines) + 1))
        LOGGER.debug('labelled_lines: %s', labelled_lines)
        return labelled_lines[:-1]

    def label_raw_text(self, data: str) -> str:
        return '\n'.join(self.label_lines(data.splitlines()))

    def label_features(self, features: List[List[str]]) -> List[List[str]]:
        lines = [
            format_feature_line(feature_line)
            for feature_line in features
        ]
        return [
            [
                token_features[0],
                labelled_line.rsplit('\t', maxsplit=1)[-1]
            ]
            for labelled_line, token_features in zip(self.label_lines(lines), features)
        ]


class WapitiWrapper:
    def __init__(self, wapiti_binary_path: str = None):
        self.wapiti_binary_path = wapiti_binary_path or 'wapiti'

    def check_available(self):
        self.run_wapiti(['--version'])

    def load_model(
            self,
            model_path: str,
            output_only_labels: bool = True,
            stderr_to_log_enabled: bool = True) -> WapitiModel:
        if not os.path.isfile(str(model_path)):
            raise FileNotFoundError('wapiti model not found: %s' % model_path)
        args = [
            'label',
            '--model',
            str(model_path)
        ]
        if output_only_labels:
            args.append('--label')
        command = [self.wapiti_binary_path] + args
        LOGGER.debug('running wapiti: %s', command)
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr if not stderr_to_log_enabled else subprocess.PIPE
        )
        process.poll()
        if stderr_to_log_enabled:
            t = threading.Thread(target=lambda: lines_to_log(
                LOGGER, logging.INFO, 'wapiti, stderr: %s',
                cast(Iterable[str], process.stderr)
            ))
            t.daemon = True
            t.start()
        return WapitiModel(process=process)

    def run_wapiti(self, args: List[str]):
        command = [self.wapiti_binary_path] + args
        LOGGER.info('calling wapiti: %s', command)
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        ) as process:
            assert process.stdout
            with process.stdout:
                lines_to_log(
                    LOGGER,
                    logging.INFO,
                    'wapiti: %s',
                    cast(Iterable[str], process.stdout)
                )
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command
                )
            LOGGER.debug('wapiti call succeeded')

    def label(
            self,
            model_path: str,
            data_path: str,
            output_data_path: str,
            output_only_labels: bool = True):
        if not os.path.isfile(str(model_path)):
            raise FileNotFoundError('model file not found: %s' % model_path)
        if not os.path.isfile(str(data_path)):
            raise FileNotFoundError('data file not found: %s' % data_path)
        args = [
            'label',
            '--model',
            str(model_path)
        ]
        if output_only_labels:
            args.append('--label')
        args.append(str(data_path))
        args.append(str(output_data_path))
        self.run_wapiti(args)

    def train(
        self,
        data_path: str,
        output_model_path: str,
        template_path: Optional[str] = None,
        max_iter: Optional[int] = None,
        num_threads: Optional[int] = None,
        stop_epsilon_value: Optional[str] = None,
        stop_window_size: Optional[int] = None
    ):
        if not os.path.isfile(str(data_path)):
            raise FileNotFoundError('data file not found: %s' % data_path)

        if not num_threads:
            num_threads = cpu_count()
        if not stop_epsilon_value:
            stop_epsilon_value = DEFAULT_STOP_EPSILON_VALUE
        if not stop_window_size:
            stop_window_size = DEFAULT_STOP_WINDOW_SIZE

        args = ['train']
        if template_path:
            if not os.path.isfile(str(template_path)):
                raise FileNotFoundError('template file not found: %s' % template_path)
            args.append('--pattern')
            args.append(str(template_path))
        if max_iter:
            args.append('--maxiter')
            args.append(str(max_iter))

        args.append('--nthread')
        args.append(str(num_threads))

        args.append('--stopeps')
        args.append(str(stop_epsilon_value))

        args.append('--stopwin')
        args.append(str(stop_window_size))

        args.append(str(data_path))
        args.append(str(output_model_path))
        self.run_wapiti(args)
