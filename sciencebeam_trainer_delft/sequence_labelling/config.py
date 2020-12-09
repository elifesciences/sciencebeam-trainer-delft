import json
from typing import List

from delft.sequenceLabelling.config import (
    ModelConfig as _ModelConfig,
    TrainingConfig as _TrainingConfig
)


FIRST_MODEL_VERSION = 1
MODEL_VERSION = 2


NOT_SET = -1


class ModelConfig(_ModelConfig):
    def __init__(
            self,
            *args,
            use_word_embeddings: bool = True,
            use_features: bool = False,
            max_feature_size: int = 50,
            additional_token_feature_indices: List[int] = None,
            text_feature_indices: List[int] = None,
            concatenated_embeddings_token_count: int = None,
            features_lstm_units: int = None,
            use_features_indices_input: bool = False,
            stateful: bool = False,
            model_version: int = MODEL_VERSION,
            # deprecated
            feature_indices: List[int] = None,
            feature_embedding_size: int = NOT_SET,
            **kwargs):
        if feature_indices:
            kwargs['features_indices'] = feature_indices
        if feature_embedding_size != NOT_SET:
            kwargs['features_embedding_size'] = feature_embedding_size
        super().__init__(*args)
        self.use_word_embeddings = use_word_embeddings
        self.additional_token_feature_indices = additional_token_feature_indices
        self.text_feature_indices = text_feature_indices
        self.concatenated_embeddings_token_count = concatenated_embeddings_token_count
        self.use_features = use_features
        self.max_feature_size = max_feature_size
        self.features_lstm_units = features_lstm_units
        self.use_features_indices_input = use_features_indices_input
        self.stateful = stateful
        self.model_version = model_version
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def is_deprecated_padded_batch_text_list_enabled(self):
        return bool(
            self.model_version < 2
            and self.text_feature_indices
        )

    def save(self, file):
        try:
            super().save(file)
        except TypeError:
            json.dump(vars(self), file, sort_keys=False, indent=4)

    @classmethod
    def load(cls, file):
        variables = json.load(file)
        self = cls()
        # model version is assumed to the first version if not saved
        self.model_version = FIRST_MODEL_VERSION
        for key, val in variables.items():
            setattr(self, key, val)
        return self

    # alias due to properties having been renamed in upstream implementation
    @property
    def feature_indices(self) -> List[int]:
        return (
            self.features_indices
            or self.__dict__.get('feature_indices')
        )

    @feature_indices.setter
    def feature_indices(self, feature_indices: List[int]):
        self.features_indices = feature_indices

    @property
    def feature_embedding_size(self):
        return (
            self.features_embedding_size
            or self.__dict__.get('feature_embedding_size')
        )

    @feature_embedding_size.setter
    def feature_embedding_size(self, feature_embedding_size: List[int]):
        self.features_embedding_size = feature_embedding_size


class TrainingConfig(_TrainingConfig):
    def __init__(
            self,
            *args,
            initial_epoch: int = None,
            input_window_stride: int = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_epoch = initial_epoch
        self.input_window_stride = input_window_stride
