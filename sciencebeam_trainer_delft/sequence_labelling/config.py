import json
from typing import List

from delft.sequenceLabelling.config import (
    ModelConfig as _ModelConfig,
    TrainingConfig as _TrainingConfig
)


class ModelConfig(_ModelConfig):
    DEFAULT_FEATURES_VOCABULARY_SIZE = 12
    DEFAULT_FEATURES_EMBEDDING_SIZE = 4

    def __init__(
            self,
            *args,
            use_word_embeddings: bool = True,
            use_features: bool = False,
            max_feature_size: int = 50,
            feature_indices: List[int] = None,
            feature_embedding_size: int = DEFAULT_FEATURES_EMBEDDING_SIZE,
            features_vocabulary_size: int = DEFAULT_FEATURES_VOCABULARY_SIZE,
            features_lstm_units: int = None,
            use_features_indices_input: bool = False,
            stateful: bool = False,
            **kwargs):
        super().__init__(*args)
        self.use_word_embeddings = use_word_embeddings
        self.use_features = use_features
        self.max_feature_size = max_feature_size
        self.feature_indices = feature_indices
        self.feature_embedding_size = feature_embedding_size
        self.features_vocabulary_size = features_vocabulary_size
        self.features_lstm_units = features_lstm_units
        self.use_features_indices_input = use_features_indices_input
        self.stateful = stateful
        for key, val in kwargs.items():
            setattr(self, key, val)

    def save(self, file):
        try:
            super().save(file)
        except TypeError:
            json.dump(vars(self), file, sort_keys=False, indent=4)

    @classmethod
    def load(cls, file):
        variables = json.load(file)
        self = cls()
        for key, val in variables.items():
            setattr(self, key, val)
        return self

    # alias due to properties having been renamed in upstream implementation
    @property
    def features_indices(self):
        return self.feature_indices

    @features_indices.setter
    def features_indices(self, feature_indices: List[int]):
        self.feature_indices = feature_indices

    @property
    def features_embedding_size(self):
        return self.feature_embedding_size

    @features_embedding_size.setter
    def features_embedding_size(self, feature_embedding_size: List[int]):
        self.feature_embedding_size = feature_embedding_size


class TrainingConfig(_TrainingConfig):
    def __init__(
            self,
            *args,
            input_window_stride: int = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.input_window_stride = input_window_stride
