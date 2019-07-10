import json
from typing import List

from delft.sequenceLabelling.config import ModelConfig as _ModelConfig


class ModelConfig(_ModelConfig):
    def __init__(
            self,
            *args,
            use_features: bool = False,
            max_feature_size: int = 50,
            feature_indices: List[int] = None,
            feature_embedding_size: int = None,
            **kwargs):
        super().__init__(*args)
        self.use_features = use_features
        self.max_feature_size = max_feature_size
        self.feature_indices = feature_indices
        self.feature_embedding_size = feature_embedding_size
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def load(cls, file):
        variables = json.load(file)
        self = cls()
        for key, val in variables.items():
            setattr(self, key, val)
        return self
