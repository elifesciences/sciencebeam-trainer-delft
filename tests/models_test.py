from unittest.mock import MagicMock

import pytest

from sciencebeam_trainer_delft.models import CustomBidLSTM_CRF


@pytest.fixture(name='model_config')
def _model_config():
    config = MagicMock(name='config')
    config.word_embedding_size = 10
    config.max_char_length = 10
    config.char_vocab_size = 10
    config.char_embedding_size = 10
    config.num_word_lstm_units = 5
    config.num_char_lstm_units = 5
    config.dropout = 0.5
    config.recurrent_dropout = 0.0
    return config


class TestCustomBidLSTM_CRF:
    def test_should_be_able_to_build(self, model_config):
        CustomBidLSTM_CRF(model_config, ntags=5)
