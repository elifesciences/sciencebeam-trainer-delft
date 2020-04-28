from delft.textClassification.config import (
    ModelConfig as _ModelConfig,
    TrainingConfig as _TrainingConfig
)


class DefaultConfig:
    EMBEDDNGS_NAME = 'glove.6B.50d'


class ModelConfig(_ModelConfig):
    pass


class TrainingConfig(_TrainingConfig):
    pass
