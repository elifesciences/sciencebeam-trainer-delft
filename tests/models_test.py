import pytest

from sciencebeam_trainer_delft.config import ModelConfig
from sciencebeam_trainer_delft.models import CustomBidLSTM_CRF


@pytest.fixture(name='model_config')
def _model_config():
    config = ModelConfig(
        word_embedding_size=11,
        max_char_length=12,
        max_feature_size=15,
        dropout=0.5,
        recurrent_dropout=0.0
    )
    config.char_vocab_size = 13
    config.char_embedding_size = 14
    config.num_word_lstm_units = 5
    config.num_char_lstm_units = 6
    return config


class TestCustomBidLSTM_CRF:
    def test_should_be_able_to_build_without_features(self, model_config):
        CustomBidLSTM_CRF(model_config, ntags=5, use_features=False)

    def test_should_be_able_to_build_with_features(self, model_config):
        CustomBidLSTM_CRF(model_config, ntags=5, use_features=True)
