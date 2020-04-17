import logging
import sys

from sciencebeam_trainer_delft.utils.logging import (
    TeeStreamToLineWriter,
    tee_stdout_lines_to,
    tee_stderr_lines_to,
    tee_stdout_and_stderr_lines_to,
    tee_logging_lines_to
)


class TestTeeStreamToLineWriter:
    def test_should_only_log_the_last_message_when_cursor_is_reset(self):
        lines = []
        out_fp = TeeStreamToLineWriter(
            lines.append
        )
        out_fp.write('test\rupdate 1\rupdate 2\n')
        assert lines == ['update 2']

    def test_should_only_log_the_last_message_when_cursor_is_reset_separate_calls(self):
        lines = []
        out_fp = TeeStreamToLineWriter(
            lines.append
        )
        for c in 'test\rupdate 1\rupdate 2\n':
            out_fp.write(c)
        assert lines == ['update 2']


class TestTeeStdoutLinesTo:
    def test_should_redirect_stdout_and_restore_stdout(self, capsys):
        prev_stdout = sys.stdout
        lines = []
        with tee_stdout_lines_to(lines.append):
            sys.stdout.write('test\n')
            assert lines == ['test']
            captured = capsys.readouterr()
            assert captured.out == 'test\n'
            assert captured.err == ''
        assert sys.stdout == prev_stdout


class TestTeeStderrLinesTo:
    def test_should_redirect_stderr_and_restore_stderr(self, capsys):
        prev_stderr = sys.stderr
        lines = []
        with tee_stderr_lines_to(lines.append):
            sys.stderr.write('test\n')
            assert lines == ['test']
            captured = capsys.readouterr()
            assert captured.out == ''
            assert captured.err == 'test\n'
        assert sys.stderr == prev_stderr


class TestTeeStdoutAndStderrLinesTo:
    def test_should_redirect_stdout_and_stderr(self, capsys):
        prev_stdout = sys.stdout
        prev_stderr = sys.stderr
        lines = []
        with tee_stdout_and_stderr_lines_to(lines.append):
            sys.stdout.write('info\n')
            sys.stderr.write('error\n')
            assert lines == ['info', 'error']
            captured = capsys.readouterr()
            assert captured.out == 'info\n'
            assert captured.err == 'error\n'
        assert sys.stdout == prev_stdout
        assert sys.stderr == prev_stderr


class TestTeeLoggingLinesTo:
    def test_should_redirect_logging_to_lines(self, caplog):
        lines = []
        with tee_logging_lines_to(lines.append):
            logging.getLogger('logger_name').info('test')
            assert caplog.messages == ['test']
            assert len(lines) == 1
            assert lines[0].endswith('test')
