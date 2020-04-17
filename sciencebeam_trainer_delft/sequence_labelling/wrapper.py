import logging
import os
import time
from typing import List

import numpy as np

from delft.sequenceLabelling.wrapper import Sequence as _Sequence

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.numpy import concatenate_or_none

from sciencebeam_trainer_delft.embedding import Embeddings, EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.data_generator import (
    DataGenerator,
    get_all_batch_tokens,
    concatenate_all_batch_tokens
)
from sciencebeam_trainer_delft.sequence_labelling.trainer import (
    get_model_results,
    Scorer,
    Trainer
)
from sciencebeam_trainer_delft.sequence_labelling.models import (
    get_model,
    updated_implicit_model_config_props
)
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    Preprocessor,
    T_FeaturesPreprocessor,
    FeaturesPreprocessor,
    FeaturesIndicesInputPreprocessor
)
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver, ModelLoader
from sciencebeam_trainer_delft.sequence_labelling.tagger import Tagger
from sciencebeam_trainer_delft.sequence_labelling.evaluation import ClassificationResult

from sciencebeam_trainer_delft.sequence_labelling.debug import get_tag_debug_reporter_if_enabled


LOGGER = logging.getLogger(__name__)


DEFAUT_MODEL_PATH = 'data/models/sequenceLabelling/'
DEFAULT_EMBEDDINGS_PATH = './embedding-registry.json'


def get_features_preprocessor(model_config) -> T_FeaturesPreprocessor:
    if model_config.use_features_indices_input:
        return FeaturesIndicesInputPreprocessor(
            features_indices=model_config.feature_indices,
            features_vocabulary_size=model_config.features_vocabulary_size
        )
    return FeaturesPreprocessor(
        feature_indices=model_config.feature_indices
    )


def prepare_preprocessor(X, y, model_config, features: np.array = None):
    feature_preprocessor = None
    if features is not None:
        feature_preprocessor = get_features_preprocessor(model_config)
    preprocessor = Preprocessor(
        max_char_length=model_config.max_char_length,
        feature_preprocessor=feature_preprocessor
    )
    concatenated_batch_tokens = concatenate_all_batch_tokens(
        get_all_batch_tokens(
            X, features,
            additional_token_feature_indices=model_config.additional_token_feature_indices
        )
    )
    preprocessor.fit(concatenated_batch_tokens, y)
    if features is not None:
        preprocessor.fit_features(features)
        if model_config.features_indices != preprocessor.feature_preprocessor.features_indices:
            LOGGER.info('revised features_indices: %s', model_config.features_indices)
            model_config.features_indices = preprocessor.feature_preprocessor.features_indices
        model_config.features_map_to_index = preprocessor.feature_preprocessor.features_map_to_index
    return preprocessor


def get_model_directory(model_name: str, dir_path: str = None):
    return os.path.join(dir_path or DEFAUT_MODEL_PATH, model_name)


class Sequence(_Sequence):
    def __init__(
            self, *args,
            use_features: bool = False,
            feature_indices: List[int] = None,
            feature_embedding_size: int = None,
            multiprocessing: bool = False,
            embedding_registry_path: str = None,
            embedding_manager: EmbeddingManager = None,
            config_props: dict = None,
            training_props: dict = None,
            eval_max_sequence_length: int = None,
            **kwargs):
        # initialise logging if not already initialised
        logging.basicConfig(level='INFO')
        LOGGER.debug('Sequence, args=%s, kwargs=%s', args, kwargs)
        self.embedding_registry_path = embedding_registry_path or DEFAULT_EMBEDDINGS_PATH
        if embedding_manager is None:
            embedding_manager = EmbeddingManager(
                path=self.embedding_registry_path,
                download_manager=DownloadManager()
            )
        self.embedding_manager = embedding_manager
        self.embeddings = None
        self.max_sequence_length = kwargs.get('max_sequence_length')
        self.eval_max_sequence_length = eval_max_sequence_length
        self.model_path = None
        super().__init__(*args, **kwargs)
        LOGGER.debug('use_features=%s', use_features)
        self.model_config = ModelConfig(
            **{
                **vars(self.model_config),
                **(config_props or {})
            },
            use_features=use_features,
            feature_indices=feature_indices,
            feature_embedding_size=feature_embedding_size
        )
        self.update_model_config_word_embedding_size()
        updated_implicit_model_config_props(self.model_config)
        self.training_config = TrainingConfig(
            **vars(self.training_config),
            **(training_props or {})
        )
        LOGGER.info('training_config: %s', vars(self.training_config))
        self.multiprocessing = multiprocessing
        self.tag_debug_reporter = get_tag_debug_reporter_if_enabled()

    def update_model_config_word_embedding_size(self):
        if self.embeddings:
            additional_token_feature_count = len(
                self.model_config.additional_token_feature_indices or []
            )
            self.model_config.word_embedding_size = (
                self.embeddings.embed_size * (1 + additional_token_feature_count)
            )

    def clear_embedding_cache(self):
        if not self.embeddings:
            return
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

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
            try:
                if isinstance(sample_transformed_features, tuple):
                    sample_transformed_features = sample_transformed_features[0]
                LOGGER.info(
                    'sample_transformed_features.shape: %s',
                    sample_transformed_features.shape
                )
                self.model_config.max_feature_size = sample_transformed_features.shape[-1]
                LOGGER.info('max_feature_size: %s', self.model_config.max_feature_size)
            except Exception:  # pylint: disable=broad-except
                LOGGER.info('features do not implement shape, set max_feature_size manually')

        self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            training_config=self.training_config,
            model_saver=self.get_model_saver(),
            multiprocessing=self.multiprocessing,
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid
        )
        self.clear_embedding_cache()

    def get_model_saver(self):
        return ModelSaver(
            preprocessor=self.p,
            model_config=self.model_config
        )

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
            training_config=self.training_config,
            model_saver=self.get_model_saver(),
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

    def create_eval_data_generator(self, *args, **kwargs) -> DataGenerator:
        return DataGenerator(
            *args,
            batch_size=self.training_config.batch_size,
            preprocessor=self.p,
            additional_token_feature_indices=self.model_config.additional_token_feature_indices,
            char_embed_size=self.model_config.char_embedding_size,
            max_sequence_length=self.eval_max_sequence_length,
            embeddings=self.embeddings,
            **kwargs
        )

    def get_evaluation_result(
            self,
            x_test: List[List[str]],
            y_test: List[List[str]],
            features: List[List[List[str]]] = None) -> ClassificationResult:
        self._require_model()
        # Prepare test data(steps, generator)
        test_generator = self.create_eval_data_generator(
            x_test, y_test,
            features=features,
            shuffle=False
        )

        prediction_result = get_model_results(
            model=self.model,
            valid_batches=test_generator,
            preprocessor=self.p
        )
        return ClassificationResult(
            y_pred=prediction_result.y_pred,
            y_true=prediction_result.y_true
        )

    def eval_single(  # pylint: disable=arguments-differ
            self,
            x_test: List[List[str]],
            y_test: List[List[str]],
            features: List[List[List[str]]] = None):
        classification_result = self.get_evaluation_result(
            x_test=x_test,
            y_test=y_test,
            features=features
        )
        print(classification_result.get_formatted_report(digits=4))

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
                test_generator = self.create_eval_data_generator(
                    x_test, y_test,
                    features=features,
                    shuffle=False
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

    def tag(  # pylint: disable=arguments-differ
            self, texts, output_format, features=None):
        # annotate a list of sentences, return the list of annotations in the
        # specified output_format
        self._require_model()
        if self.model_config.use_features and features is None:
            raise ValueError('features required')
        tagger = Tagger(
            self.model, self.model_config, self.embeddings,
            max_sequence_length=self.max_sequence_length,
            preprocessor=self.p
        )
        start_time = time.time()
        annotations = tagger.tag(
            list(texts), output_format,
            features=features
        )
        runtime = round(time.time() - start_time, 3)
        if output_format == 'json':
            annotations["runtime"] = runtime
        if self.tag_debug_reporter:
            self.tag_debug_reporter.report_tag_results(
                texts=texts,
                features=features,
                annotations=annotations,
                model_name=self._get_model_name()
            )
        return annotations

    def _require_model(self):
        if not self.model:
            raise OSError('Model not loaded: %s' % self._get_model_name())

    def _get_model_name(self):
        return self.model_config.model_name

    @property
    def model_summary_props(self) -> str:
        return {
            'model_type': 'delft',
            'architecture': self.model_config.model_type,
            'model_config': vars(self.model_config)
        }

    def get_model_output_path(self, dir_path: str = None) -> str:
        return get_model_directory(model_name=self.model_config.model_name, dir_path=dir_path)

    def _get_model_directory(self, dir_path: str = None) -> str:
        return self.get_model_output_path(dir_path=dir_path)

    def get_embedding_for_model_config(self, model_config: ModelConfig):
        embedding_name = model_config.embeddings_name
        if not model_config.use_word_embeddings or not embedding_name:
            return None
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        embeddings = Embeddings(
            embedding_name,
            path=self.embedding_registry_path,
            use_ELMo=model_config.use_ELMo,
            use_BERT=model_config.use_BERT
        )
        if not embeddings.embed_size > 0:
            raise AssertionError(
                'invalid embedding size, embeddings not loaded? %s' % embedding_name
            )
        return embeddings

    def get_meta(self):
        return {
            'training_config': vars(self.training_config)
        }

    def save(self, dir_path=None):
        # create subfolder for the model if not already exists
        directory = self._get_model_directory(dir_path)
        os.makedirs(directory, exist_ok=True)
        self.get_model_saver().save_to(directory, model=self.model, meta=self.get_meta())

    def load(self, dir_path=None):
        directory = self._get_model_directory(dir_path)
        self.load_from(directory)

    def load_from(self, directory: str):
        model_loader = ModelLoader()
        self.model_path = directory
        self.p = model_loader.load_preprocessor_from_directory(directory)
        self.model_config = model_loader.load_model_config_from_directory(directory)

        # load embeddings
        LOGGER.info('loading embeddings: %s', self.model_config.embeddings_name)
        self.embeddings = self.get_embedding_for_model_config(self.model_config)
        self.update_model_config_word_embedding_size()

        self.model = get_model(self.model_config, self.p, ntags=len(self.p.vocab_tag))
        model_loader.load_model_from_directory(directory, model=self.model)
