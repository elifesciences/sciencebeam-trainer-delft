import logging
from typing import List

import numpy as np

from delft.sequenceLabelling.wrapper import Sequence as _Sequence

from sciencebeam_trainer_delft.config import ModelConfig
from sciencebeam_trainer_delft.trainer import Trainer
from sciencebeam_trainer_delft.models import get_model
from sciencebeam_trainer_delft.preprocess import Preprocessor, FeaturesPreprocessor


LOGGER = logging.getLogger(__name__)


def prepare_preprocessor(X, y, model_config, features: np.array = None):
    feature_preprocessor = None
    if features is not None:
        feature_preprocessor = FeaturesPreprocessor(
            feature_indices=model_config.feature_indices
        )
    preprocessor = Preprocessor(
        max_char_length=model_config.max_char_length,
        feature_preprocessor=feature_preprocessor
    )
    preprocessor.fit(X, y)
    if features is not None:
        preprocessor.fit_features(features)
    return preprocessor


class Sequence(_Sequence):
    def __init__(
            self, *args,
            use_features: bool = False,
            feature_indices: List[int] = None,
            feature_embedding_size: int = None,
            multiprocessing: bool = False,
            **kwargs):
        LOGGER.info('Sequence, args=%s, kwargs=%s', args, kwargs)
        super().__init__(*args, **kwargs)
        LOGGER.info('use_features=%s', use_features)
        self.model_config = ModelConfig(
            **vars(self.model_config),
            use_features=use_features,
            feature_indices=feature_indices,
            feature_embedding_size=feature_embedding_size
        )
        self.multiprocessing = multiprocessing

    def train(  # pylint: disable=arguments-differ
            self, x_train, y_train, x_valid=None, y_valid=None,
            features_train: np.array = None,
            features_valid: np.array = None):
        # TBD if valid is None, segment train to get one
        x_all = np.concatenate((x_train, x_valid), axis=0)
        y_all = np.concatenate((y_train, y_valid), axis=0)
        self.p = prepare_preprocessor(
            x_all, y_all,
            features=features_train,
            model_config=self.model_config
        )
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

        if features_train is not None:
            LOGGER.info('x_train.shape: %s', x_train.shape)
            LOGGER.info('features_train.shape: %s', features_train.shape)
            sample_transformed_features = self.p.transform_features(features_train[:1])
            self.model_config.max_feature_size = sample_transformed_features.shape[-1]
            LOGGER.info('max_feature_size: %s', self.model_config.max_feature_size)

        self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            self.training_config,
            multiprocessing=self.multiprocessing,
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid
        )
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()
