import logging
import os

import pytest

from .test_data import TEST_DATA_PATH


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


@pytest.fixture(name='sample_train_file')
def _sample_train_file():
    return os.path.join(
        TEST_DATA_PATH,
        'test-header.train'
    )
