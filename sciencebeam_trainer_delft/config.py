from delft.sequenceLabelling.config import ModelConfig as _ModelConfig


class ModelConfig(_ModelConfig):
    def __init__(
            self,
            *args,
            max_feature_size: int = 50,
            **kwargs):
        super().__init__(*args)
        self.max_feature_size = max_feature_size
        for key, val in kwargs.items():
            setattr(self, key, val)
