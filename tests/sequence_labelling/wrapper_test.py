
from delft.sequenceLabelling.preprocess import (
    WordPreprocessor as DelftWordPreprocessor,
    FeaturesPreprocessor as DelftFeaturesPreprocessor
)

from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    Preprocessor as ScienceBeamPreprocessor,
    FeaturesPreprocessor as ScienceBeamFeaturesPreprocessor
)

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_preprocessor,
    Sequence,
    DEFAULT_EMBEDDINGS_PATH
)


MODEL_NAME_1 = 'DummyModel1'


class TestGetPreprocessor:
    def test_should_use_default_preprocessor_if_not_using_features(self):
        model_config = ModelConfig(use_features=False)
        preprocessor = get_preprocessor(model_config, has_features=False)
        assert isinstance(preprocessor, DelftWordPreprocessor)
        assert not isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is None

    def test_should_use_default_preprocessor_if_using_features_indices_input(self):
        model_config = ModelConfig(use_features=True, use_features_indices_input=True)
        preprocessor = get_preprocessor(model_config, has_features=True)
        assert isinstance(preprocessor, DelftWordPreprocessor)
        assert not isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is not None
        assert isinstance(preprocessor.feature_preprocessor, DelftFeaturesPreprocessor)

    def test_should_create_preprocessor_with_feature_preprocessor(self):
        model_config = ModelConfig(use_features=True, use_features_indices_input=False)
        preprocessor = get_preprocessor(model_config, has_features=True)
        assert isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is not None
        assert isinstance(preprocessor.feature_preprocessor, ScienceBeamFeaturesPreprocessor)


class TestSequence:
    def test_should_create_embedding_manager_with_default_regisry_path(self):
        model = Sequence(MODEL_NAME_1)
        assert model.embedding_registry_path == DEFAULT_EMBEDDINGS_PATH
        assert model.embedding_manager.path == DEFAULT_EMBEDDINGS_PATH
