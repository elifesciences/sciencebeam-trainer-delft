import logging
from io import StringIO

from sciencebeam_trainer_delft.utils.progress_logger import logging_tqdm


class TestLoggingTqdm:
    def test_should_log_tqdm_output(self):
        logger = logging.Logger('test')
        out = StringIO()
        stream_handler = logging.StreamHandler(out)
        logger.addHandler(stream_handler)
        with logging_tqdm(total=2, logger=logger, mininterval=0) as pbar:
            pbar.update(1)
        last_log_line = out.getvalue().splitlines()[-1]
        assert '50%' in last_log_line
        assert '1/2' in last_log_line
