import logging
import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


@pytest.fixture(name='sample_train_file')
def _sample_train_file():
    return os.path.join(
        os.path.dirname(__file__),
        'test_data/test-header.train'
    )
