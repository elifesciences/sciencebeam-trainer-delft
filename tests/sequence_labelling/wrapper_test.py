from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    Sequence,
    DEFAULT_EMBEDDINGS_PATH
)


MODEL_NAME_1 = 'DummyModel1'


class TestSequence:
    def test_should_create_embedding_manager_with_default_regisry_path(self):
        model = Sequence(MODEL_NAME_1)
        assert model.embedding_registry_path == DEFAULT_EMBEDDINGS_PATH
        assert model.embedding_manager.path == DEFAULT_EMBEDDINGS_PATH
