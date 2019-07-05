import logging
import os

import pytest

import tensorflow as tf

from .test_data import TEST_DATA_PATH


LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    logging.getLogger('tests').setLevel('DEBUG')
    logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


@pytest.fixture
def sample_train_file():
    return os.path.join(
        TEST_DATA_PATH,
        'test-header.train'
    )

@pytest.fixture(scope='session', autouse=True)
def tf_eager_mode():
    try:
        tf.compat.v1.enable_eager_execution()
    except (ValueError, AttributeError) as e:
        LOGGER.debug('failed to switch to eager mode due to %s', e)
