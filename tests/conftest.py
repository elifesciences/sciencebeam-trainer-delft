import logging
from pathlib import Path

import pytest
from py._path.local import LocalPath

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    logging.getLogger('tests').setLevel('DEBUG')
    logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


@pytest.fixture
def temp_dir(tmpdir: LocalPath):
    # convert to standard Path
    return Path(str(tmpdir))


@pytest.fixture(scope='session', autouse=True)
def tf_eager_mode():
    try:
        tf.compat.v1.enable_eager_execution()
    except (ValueError, AttributeError) as e:
        LOGGER.debug('failed to switch to eager mode due to %s', e)
