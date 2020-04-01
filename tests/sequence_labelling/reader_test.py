import logging
from pathlib import Path

from sciencebeam_trainer_delft.sequence_labelling.reader import (
    load_data_and_labels_crf_file,
    load_data_crf_string
)


LOGGER = logging.getLogger(__name__)


UNICODE_VALUE_1 = 'unicode:\u1234'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'

FEATURE_1 = 'feature1'
FEATURE_2 = 'feature2'

LABEL_1 = 'label1'


class TestLoadDataAndLabelsCrfFile:
    def test_should_load_sample_file(self, sample_train_file: str):
        x_all, y_all, f_all = load_data_and_labels_crf_file(sample_train_file)
        assert len(x_all) == len(y_all) == len(f_all)

    def test_should_apply_limit(self, sample_train_file: str):
        x_all, y_all, f_all = load_data_and_labels_crf_file(sample_train_file, limit=1)
        assert len(x_all) == len(y_all) == len(f_all) == 1

    def test_should_load_unicode_file(self, temp_dir: Path):
        sample_train_file = temp_dir.joinpath('sample.train')
        sample_train_file.write_text(' '.join([UNICODE_VALUE_1] * 12), encoding='utf-8')
        x_all, y_all, f_all = load_data_and_labels_crf_file(str(sample_train_file))
        LOGGER.debug('x_all: %s', x_all)
        LOGGER.debug('y_all: %s', y_all)
        LOGGER.debug('f_all: %s', f_all)
        assert len(x_all) == len(y_all) == len(f_all)
        assert x_all.tolist() == [[UNICODE_VALUE_1]]
        assert y_all.tolist() == [[UNICODE_VALUE_1]]
        assert f_all.tolist() == [[[UNICODE_VALUE_1] * 10]]

    def test_should_load_multi_tokens(self, temp_dir: Path):
        sample_train_file = temp_dir.joinpath('sample.train')
        sample_train_file.write_text(' '.join([
            TOKEN_1, TOKEN_2, FEATURE_1, FEATURE_2, LABEL_1
        ]))
        x_all, y_all, f_all = load_data_and_labels_crf_file(
            str(sample_train_file),
            multi_token_count=2
        )
        LOGGER.debug('x_all: %s', x_all)
        LOGGER.debug('y_all: %s', y_all)
        LOGGER.debug('f_all: %s', f_all)
        assert len(x_all) == len(y_all) == len(f_all)
        assert x_all.tolist() == [[[TOKEN_1, TOKEN_2]]]
        assert y_all.tolist() == [[LABEL_1]]
        assert f_all.tolist() == [[[FEATURE_1, FEATURE_2]]]


class TestLoadDataCrfString:
    def test_should_load_unicode_str(self):
        crf_string = ' '.join([UNICODE_VALUE_1] * 11)
        x_all, f_all = load_data_crf_string(crf_string)
        assert len(x_all) == len(f_all)
        assert x_all.tolist() == [[UNICODE_VALUE_1]]
        assert f_all.tolist() == [[[UNICODE_VALUE_1] * 10]]

    def test_should_load_multi_tokens(self):
        crf_string = ' '.join([
            TOKEN_1, TOKEN_2, FEATURE_1, FEATURE_2
        ])
        x_all, f_all = load_data_crf_string(
            crf_string,
            multi_token_count=2
        )
        LOGGER.debug('x_all: %s', x_all)
        LOGGER.debug('f_all: %s', f_all)
        assert len(x_all) == len(f_all)
        assert x_all.tolist() == [[[TOKEN_1, TOKEN_2]]]
        assert f_all.tolist() == [[[FEATURE_1, FEATURE_2]]]
