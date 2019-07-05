
from typing import List

from delft.sequenceLabelling.config import ModelConfig as _ModelConfig


class ModelConfig(_ModelConfig):
    def __init__(
            self,
            *args,
            use_features: bool = False,
            max_feature_size: int = 50,
            feature_indices: List[int] = None,
            **kwargs):
        super().__init__(*args)
        self.use_features = use_features
        self.max_feature_size = max_feature_size
        self.feature_indices = feature_indices
        for key, val in kwargs.items():
            setattr(self, key, val)
