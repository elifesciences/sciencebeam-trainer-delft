import os

import pytest

from sciencebeam_trainer_delft.data import load_data_and_labels_crf_file


@pytest.fixture(name='sample_train_file')
def _sample_train_file():
    return os.path.join(
        os.path.dirname(__file__),
        'test_data/test-header.train'
    )


class TestLoadDataAndLabelsCrfFile:
    def test_should_load_sample_file(self, sample_train_file: str):
        x_all, y_all, f_all = load_data_and_labels_crf_file(sample_train_file)
        assert len(x_all) == len(y_all) == len(f_all)

    def test_should_apply_limit(self, sample_train_file: str):
        x_all, y_all, f_all = load_data_and_labels_crf_file(sample_train_file, limit=1)
        assert len(x_all) == len(y_all) == len(f_all) == 1
