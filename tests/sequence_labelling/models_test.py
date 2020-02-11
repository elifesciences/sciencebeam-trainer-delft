import pytest

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import (
    CustomBidLSTM_CRF,
    BidLSTM_CRF_FEATURES
)


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


@pytest.mark.slow
class TestCustomBidLSTM_CRF:
    def test_should_be_able_to_build_without_features(self, model_config: ModelConfig):
        model_config.use_features = False
        CustomBidLSTM_CRF(model_config, ntags=5)

    def test_should_be_able_to_build_with_features(self, model_config: ModelConfig):
        model_config.use_features = True
        CustomBidLSTM_CRF(model_config, ntags=5)

    def test_should_be_able_to_build_with_feature_embedding(self, model_config: ModelConfig):
        model_config.use_features = True
        model_config.feature_embedding_size = 11
        CustomBidLSTM_CRF(model_config, ntags=5)

@pytest.mark.slow
class TestBidLSTM_CRF_FEATURES:
    def test_should_be_able_to_build_model(self, model_config: ModelConfig):
        model_config.features_indices = [1, 2, 3]
        model_config.features_vocabulary_size = 11
        model_config.features_embedding_size = 12
        model_config.features_lstm_units = 13
        BidLSTM_CRF_FEATURES(model_config, ntags=5)
