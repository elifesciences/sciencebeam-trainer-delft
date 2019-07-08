import logging
from typing import List

import numpy as np

from delft.sequenceLabelling.wrapper import Sequence as _Sequence
from delft.sequenceLabelling.trainer import Scorer

from sciencebeam_trainer_delft.config import ModelConfig
from sciencebeam_trainer_delft.data_generator import DataGenerator
from sciencebeam_trainer_delft.trainer import Trainer
from sciencebeam_trainer_delft.models import get_model
from sciencebeam_trainer_delft.preprocess import Preprocessor, FeaturesPreprocessor
from sciencebeam_trainer_delft.utils import concatenate_or_none


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
        features_all = concatenate_or_none((features_train, features_valid), axis=0)
        self.p = prepare_preprocessor(
            x_all, y_all,
            features=features_all,
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

    def train_nfold(  # pylint: disable=arguments-differ
            self, x_train, y_train, x_valid=None, y_valid=None, fold_number=10,
            features_train: np.array = None,
            features_valid: np.array = None):
        if x_valid is not None and y_valid is not None:
            x_all = np.concatenate((x_train, x_valid), axis=0)
            y_all = np.concatenate((y_train, y_valid), axis=0)
            features_all = concatenate_or_none((features_train, features_valid), axis=0)
            self.p = prepare_preprocessor(
                x_all, y_all,
                features=features_all,
                model_config=self.model_config
            )
        else:
            self.p = prepare_preprocessor(
                x_train, y_train,
                features=features_train,
                model_config=self.model_config
            )
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)
        self.p.return_lengths = True

        self.models = []

        for _ in range(0, fold_number):
            model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
            self.models.append(model)

        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            self.training_config,
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train_nfold(
            x_train, y_train,
            x_valid, y_valid,
            features_train=features_train,
            features_valid=features_valid
        )
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

    def eval(  # pylint: disable=arguments-differ
            self, x_test, y_test, features: np.array = None):
        should_eval_nfold = (
            self.model_config.fold_number > 1
            and self.models
            and len(self.models) == self.model_config.fold_number
        )
        if should_eval_nfold:
            self.eval_nfold(x_test, y_test, features=features)
        else:
            self.eval_single(x_test, y_test, features=features)

    def eval_single(  # pylint: disable=arguments-differ
            self, x_test, y_test, features: np.array = None):
        if self.model:
            # Prepare test data(steps, generator)
            test_generator = DataGenerator(
                x_test, y_test,
                features=features,
                batch_size=self.training_config.batch_size, preprocessor=self.p,
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False
            )

            # Build the evaluator and evaluate the model
            scorer = Scorer(test_generator, self.p, evaluation=True)
            scorer.model = self.model
            scorer.on_epoch_end(epoch=-1)
        else:
            raise OSError('Could not find a model.')

    def eval_nfold(  # pylint: disable=arguments-differ
            self, x_test, y_test, features: np.array = None):
        if self.models is not None:
            total_f1 = 0
            best_f1 = 0
            best_index = 0
            worst_f1 = 1
            worst_index = 0
            reports = []
            total_precision = 0
            total_recall = 0
            for i in range(0, self.model_config.fold_number):
                print(
                    '\n------------------------ fold %s --------------------------------------'
                    % i
                )

                # Prepare test data(steps, generator)
                test_generator = DataGenerator(
                    x_test, y_test,
                    features=features,
                    batch_size=self.training_config.batch_size, preprocessor=self.p,
                    char_embed_size=self.model_config.char_embedding_size,
                    max_sequence_length=self.model_config.max_sequence_length,
                    embeddings=self.embeddings, shuffle=False
                )

                # Build the evaluator and evaluate the model
                scorer = Scorer(test_generator, self.p, evaluation=True)
                scorer.model = self.models[i]
                scorer.on_epoch_end(epoch=-1)
                f1 = scorer.f1
                precision = scorer.precision
                recall = scorer.recall
                reports.append(scorer.report)

                if best_f1 < f1:
                    best_f1 = f1
                    best_index = i
                if worst_f1 > f1:
                    worst_f1 = f1
                    worst_index = i
                total_f1 += f1
                total_precision += precision
                total_recall += recall

            macro_f1 = total_f1 / self.model_config.fold_number
            macro_precision = total_precision / self.model_config.fold_number
            macro_recall = total_recall / self.model_config.fold_number

            print("\naverage over", self.model_config.fold_number, "folds")
            print("\tmacro f1 =", macro_f1)
            print("\tmacro precision =", macro_precision)
            print("\tmacro recall =", macro_recall, "\n")

            print("\n** Worst ** model scores - \n")
            print(reports[worst_index])

            self.model = self.models[best_index]
            print("\n** Best ** model scores - \n")
            print(reports[best_index])
