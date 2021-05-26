from __future__ import absolute_import

import logging
import sys
from io import StringIO
from contextlib import contextmanager
from typing import Callable, IO, List, Optional, Sequence, TextIO, cast


LOGGER = logging.getLogger(__name__)


def configure_logging(level='INFO', secondary_level='WARN'):
    logging.basicConfig(level=secondary_level)
    logging.getLogger('delft').setLevel(level)
    logging.getLogger('sciencebeam_trainer_delft').setLevel(level)


def reset_logging(**kwargs):
    logging.root.handlers = []
    configure_logging(**kwargs)


class TeeStreamToLineWriter:
    def __init__(
            self,
            *line_writers: Callable[[str], None],
            raw_fp: IO = None,
            append_line_feed: bool = False):
        self.line_writers = line_writers
        self.raw_fp = raw_fp
        self.line_buffer = StringIO()
        self.append_line_feed = append_line_feed

    def _write_line(self, line: str):
        if self.append_line_feed:
            line += '\n'
        for line_writer in self.line_writers:
            line_writer(line)

    def _flush_message(self, message: str):
        self._write_line(message.split('\r')[-1].rstrip())

    def write(self, message: str):
        if self.raw_fp:
            self.raw_fp.write(message)
        if not message:
            return
        if message.startswith('\n'):
            self._flush_message(self.line_buffer.getvalue())
            self.line_buffer = StringIO()
            message = message[1:]
        if not message:
            return
        lines = message.split('\n')
        complete_lines = lines[:-1]
        remaining_message = lines[-1]
        if complete_lines:
            self.line_buffer.write(complete_lines[0])
            complete_lines[0] = self.line_buffer.getvalue()
            self.line_buffer = StringIO()
        else:
            self.line_buffer.write(remaining_message)
        for complete_line in complete_lines:
            self._flush_message(complete_line)

    def flush(self):
        if self.raw_fp:
            self.raw_fp.flush()


@contextmanager
def tee_stdout_lines_to(
        *line_writers: Callable[[str], None],
        **kwargs):
    prev_stdout = sys.stdout
    try:
        sys.stdout = cast(TextIO, TeeStreamToLineWriter(
            *line_writers,
            raw_fp=prev_stdout,
            **kwargs
        ))
        yield sys.stdout
    finally:
        sys.stdout = prev_stdout


@contextmanager
def tee_stderr_lines_to(
        *line_writers: Callable[[str], None],
        **kwargs):
    prev_stderr = sys.stderr
    try:
        sys.stderr = cast(TextIO, TeeStreamToLineWriter(
            *line_writers,
            raw_fp=prev_stderr,
            **kwargs
        ))
        yield sys.stderr
    finally:
        sys.stderr = prev_stderr


@contextmanager
def tee_stdout_and_stderr_lines_to(
        *line_writers: Callable[[str], None],
        **kwargs):
    with tee_stdout_lines_to(*line_writers, **kwargs) as stdout:
        with tee_stderr_lines_to(*line_writers, **kwargs) as stderr:
            yield (stdout, stderr)


class LineWriterLoggingHandler(logging.Handler):
    def __init__(
            self,
            *line_writers: Callable[[str], None],
            append_line_feed: bool = False,
            **kwargs):
        self.line_writers: Sequence[Callable[[str], None]] = line_writers
        self.append_line_feed = append_line_feed
        self._logging = False
        super().__init__(**kwargs)

    def _write_line(self, line: str):
        if self.append_line_feed:
            line += '\n'
        for line_writer in self.line_writers:
            try:
                line_writer(line)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    'failed to write: %r due to %s', line, exc, exc_info=exc
                )

    def emit(self, record: logging.LogRecord):
        if self._logging:
            return
        try:
            self._logging = True
            self._write_line(self.format(record))
        finally:
            self._logging = False


def get_default_logging_formatter() -> Optional[logging.Formatter]:
    for root_handler in logging.root.handlers:
        if isinstance(root_handler, logging.StreamHandler):
            return root_handler.formatter
    return None


def flush_logging_handlers(handlers: List[logging.Handler]):
    for handler in handlers:
        handler.flush()


@contextmanager
def tee_logging_lines_to(
        *line_writers: Callable[[str], None],
        logger: logging.Logger = None,
        formatter: logging.Formatter = None,
        **kwargs):
    if logger is None:
        logger = logging.root
    if formatter is None:
        formatter = get_default_logging_formatter()
    prev_handlers = logger.handlers
    try:
        handler = LineWriterLoggingHandler(*line_writers, **kwargs)
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)
        yield logger
    finally:
        flush_logging_handlers(logger.handlers)
        flush_logging_handlers(logging.root.handlers)
        logger.handlers = prev_handlers
