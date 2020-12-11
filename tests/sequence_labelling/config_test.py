from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig


FEATURE_INDICES_1 = [9, 10, 11]

FEATURES_EMBEDDING_SIZE_1 = 13


class TestModelConfig:
    def test_should_be_able_to_pass_in_feature_indices(self):
        model_config = ModelConfig(feature_indices=FEATURE_INDICES_1)
        assert model_config.feature_indices == FEATURE_INDICES_1
        assert model_config.features_indices == FEATURE_INDICES_1

    def test_should_be_able_to_pass_in_features_indices(self):
        model_config = ModelConfig(features_indices=FEATURE_INDICES_1)
        assert model_config.feature_indices == FEATURE_INDICES_1
        assert model_config.features_indices == FEATURE_INDICES_1

    def test_should_be_able_to_pass_in_feature_embedding_size(self):
        model_config = ModelConfig(feature_embedding_size=FEATURES_EMBEDDING_SIZE_1)
        assert model_config.feature_embedding_size == FEATURES_EMBEDDING_SIZE_1
        assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1

    def test_should_be_able_to_pass_in_features_embedding_size(self):
        model_config = ModelConfig(features_embedding_size=FEATURES_EMBEDDING_SIZE_1)
        assert model_config.feature_embedding_size == FEATURES_EMBEDDING_SIZE_1
        assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1
