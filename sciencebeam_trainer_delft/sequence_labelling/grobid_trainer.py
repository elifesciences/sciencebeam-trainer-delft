# pylint: disable=too-many-lines
# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
import time
import tempfile
import os
from abc import abstractmethod
from collections import Counter
from itertools import islice
from typing import List, Tuple, Union

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
import sciencebeam_trainer_delft.utils.no_keras_backend_message  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import numpy as np

from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf

from sciencebeam_trainer_delft.utils.misc import parse_number_ranges
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.cloud_support import patch_cloud_support
from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays
from sciencebeam_trainer_delft.utils.tf import get_tf_info
from sciencebeam_trainer_delft.utils.io import (
    copy_file,
    auto_uploading_output_file,
    write_text
)
from sciencebeam_trainer_delft.utils.logging import (
    tee_stdout_and_stderr_lines_to,
    tee_logging_lines_to
)

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.wrapper import Sequence
from sciencebeam_trainer_delft.sequence_labelling.models import get_model_names, patch_get_model
from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_and_labels_crf_file

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import (
    DEFAULT_STOP_EPSILON_VALUE,
    DEFAULT_STOP_WINDOW_SIZE
)
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    WapitiModelAdapter,
    WapitiModelTrainAdapter
)
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_install import (
    install_wapiti_and_get_path_or_none
)

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TAG_OUTPUT_FORMATS,
    get_tag_result,
    format_tag_result
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    EvaluationOutputFormats,
    EVALUATION_OUTPUT_FORMATS,
    ClassificationResult
)

from sciencebeam_trainer_delft.sequence_labelling.input_info import (
    iter_flat_batch_tokens,
    iter_flat_features,
    get_quantiles,
    get_quantiles_feature_value_length_by_index,
    get_feature_counts,
    get_suggested_feature_indices,
    format_dict,
    format_indices
)

from sciencebeam_trainer_delft.utils.cli import (
    SubCommand,
    SubCommandProcessor
)


LOGGER = logging.getLogger(__name__)


WAPITI_MODEL_NAME = 'wapiti'

GROBID_MODEL_NAMES = [
    'affiliation-address', 'citation', 'date', 'figure', 'fulltext', 'header',
    'name', 'name-citation', 'name-header', 'patent', 'reference-segmenter',
    'segmentation', 'software', 'table'
]


class Tasks:
    TRAIN = 'train'
    TRAIN_EVAL = 'train_eval'
    EVAL = 'eval'
    TAG = 'tag'
    WAPITI_TRAIN = 'wapiti_train'
    WAPITI_TRAIN_EVAL = 'wapiti_train_eval'
    WAPITI_EVAL = 'wapiti_eval'
    WAPITI_TAG = 'wapiti_tag'
    INPUT_INFO = 'input_info'


DEFAULT_RANDOM_SEED = 42

DEFAULT_TAG_OUTPUT_FORMAT = TagOutputFormats.XML


def set_random_seeds(random_seed: int):
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)


def get_default_training_data(model: str) -> str:
    return 'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'


def log_data_info(x: np.array, y: np.array, features: np.array):
    LOGGER.info('x sample: %s (y: %s)', x[:1][:10], y[:1][:1])
    LOGGER.info(
        'feature dimensions of first sample, word: %s',
        [{index: value for index, value in enumerate(features[0][0])}]  # noqa pylint: disable=unnecessary-comprehension
    )


def _load_data_and_labels_crf_files(
        input_paths: List[str], limit: int = None) -> Tuple[np.array, np.array, np.array]:
    if len(input_paths) == 1:
        return load_data_and_labels_crf_file(input_paths[0], limit=limit)
    x_list = []
    y_list = []
    features_list = []
    for input_path in input_paths:
        LOGGER.debug('calling load_data_and_labels_crf_file: %s', input_path)
        x, y, f = load_data_and_labels_crf_file(
            input_path,
            limit=limit
        )
        x_list.append(x)
        y_list.append(y)
        features_list.append(f)
    return np.concatenate(x_list), np.concatenate(y_list), np.concatenate(features_list)


def get_clean_features_mask(features_all: np.array) -> List[bool]:
    feature_lengths = Counter((
        len(features_vector)
        for features_doc in features_all
        for features_vector in features_doc
    ))
    if len(feature_lengths) <= 1:
        return [True] * len(features_all)
    expected_feature_length = next(feature_lengths.keys().__iter__())
    LOGGER.info('cleaning features, expected_feature_length=%s', expected_feature_length)
    return [
        all(len(features_vector) == expected_feature_length for features_vector in features_doc)
        for features_doc in features_all
    ]


def get_clean_x_y_features(x: np.array, y: np.array, features: np.array):
    clean_features_mask = get_clean_features_mask(features)
    if sum(clean_features_mask) != len(clean_features_mask):
        LOGGER.info(
            'ignoring %d documents with inconsistent features',
            len(clean_features_mask) - sum(clean_features_mask)
        )
        return (
            x[clean_features_mask],
            y[clean_features_mask],
            features[clean_features_mask]
        )
    return x, y, features


def load_data_and_labels(
        model: str, input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
        clean_features: bool = True,
        random_seed: int = DEFAULT_RANDOM_SEED,
        download_manager: DownloadManager = None):
    assert download_manager
    if not input_paths:
        input_paths = [get_default_training_data(model)]
    LOGGER.info('loading data from: %s', input_paths)
    downloaded_input_paths = [
        download_manager.download_if_url(input_path)
        for input_path in input_paths
    ]
    x_all, y_all, f_all = _load_data_and_labels_crf_files(
        downloaded_input_paths,
        limit=limit
    )
    if shuffle_input:
        shuffle_arrays([x_all, y_all, f_all], random_seed=random_seed)
    log_data_info(x_all, y_all, f_all)
    if clean_features:
        (x_all, y_all, f_all) = get_clean_x_y_features(
            x_all, y_all, f_all
        )
    return x_all, y_all, f_all


def do_train(
        model: Union[Sequence, WapitiModelTrainAdapter],
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        download_manager: DownloadManager = None):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_all, y_all, features_all, test_size=0.1
    )

    LOGGER.info('%d train sequences', len(x_train))
    LOGGER.info('%d validation sequences', len(x_valid))

    start_time = time.time()
    model.train(
        x_train, y_train, x_valid, y_valid,
        features_train=features_train, features_valid=features_valid
    )
    runtime = round(time.time() - start_time, 3)
    LOGGER.info("training runtime: %s seconds ", runtime)

    # saving the model
    if output_path:
        LOGGER.info('saving model to: %s', output_path)
        model.save(output_path)
    else:
        model.save()


# train a GROBID model with all available data
def train(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = 100,
        max_epoch=100,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if use_ELMo:
        model_name += '-with_ELMo'

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        **kwargs
    )

    do_train(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )


def wapiti_train(
        model: str,
        template_path: str,
        output_path: str,
        input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_epoch: int = 100,
        download_manager: DownloadManager = None,
        gzip_enabled: bool = False,
        wapiti_binary_path: str = None,
        wapiti_train_args: dict = None):
    with tempfile.TemporaryDirectory(suffix='-wapiti') as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.wapiti')
        model = WapitiModelTrainAdapter(
            model_name=model,
            template_path=template_path,
            temp_model_path=temp_model_path,
            max_epoch=max_epoch,
            download_manager=download_manager,
            gzip_enabled=gzip_enabled,
            wapiti_binary_path=wapiti_binary_path,
            wapiti_train_args=wapiti_train_args
        )
        do_train(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            download_manager=download_manager
        )


def output_classification_result(
        classification_result: ClassificationResult,
        output_format: str,
        output_path: str = None):
    if output_path:
        LOGGER.info('writing evaluation to: %s', output_path)
        write_text(output_path, classification_result.get_json_formatted_report())
    if output_format == EvaluationOutputFormats.TEXT:
        print("\nEvaluation:\n%s" % classification_result.get_text_formatted_report(
            digits=4
        ))
    else:
        print(classification_result.get_formatted_report(
            output_format=output_format
        ))


def do_train_eval(
        model: Union[Sequence, WapitiModelTrainAdapter],
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        eval_input_paths: List[str] = None,
        eval_limit: int = None,
        eval_output_format: str = None,
        eval_output_path: str = None,
        fold_count: int = 1,
        download_manager: DownloadManager = None):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )

    if eval_input_paths:
        x_eval, y_eval, features_eval = load_data_and_labels(
            model=model,
            input_paths=eval_input_paths, limit=eval_limit,
            download_manager=download_manager
        )
        x_train_all, y_train_all, features_train_all = (
            x_all, y_all, features_all
        )
    else:
        x_train_all, x_eval, y_train_all, y_eval, features_train_all, features_eval = (
            train_test_split(x_all, y_all, features_all, test_size=0.1)
        )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_train_all, y_train_all, features_train_all, test_size=0.1
    )

    LOGGER.info('%d train sequences', len(x_train))
    LOGGER.info('%d validation sequences', len(x_valid))
    LOGGER.info('%d evaluation sequences', len(x_eval))

    start_time = time.time()

    if fold_count == 1:
        model.train(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid
        )
    else:
        model.train_nfold(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid,
            fold_number=fold_count
        )

    runtime = round(time.time() - start_time, 3)
    LOGGER.info("training runtime: %s seconds ", runtime)

    # evaluation
    classification_result = model.get_evaluation_result(
        x_eval, y_eval, features=features_eval
    )
    output_classification_result(
        classification_result,
        output_format=eval_output_format,
        output_path=eval_output_path
    )

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()


# split data, train a GROBID model and evaluate it
def train_eval(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        eval_input_paths: List[str] = None,
        eval_limit: int = None,
        eval_max_sequence_length: int = None,
        eval_output_format: str = None,
        eval_output_path: str = None,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if use_ELMo:
        model_name += '-with_ELMo'
        if model_name in {'software-with_ELMo', 'grobid-software-with_ELMo'}:
            batch_size = 3

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        eval_max_sequence_length=eval_max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )
    do_train_eval(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        eval_input_paths=eval_input_paths,
        eval_limit=eval_limit,
        eval_output_format=eval_output_format,
        eval_output_path=eval_output_path,
        download_manager=download_manager
    )


def wapiti_train_eval(
        model: str,
        template_path: str,
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        eval_input_paths: List[str] = None,
        eval_limit: int = None,
        eval_output_format: str = None,
        eval_output_path: str = None,
        fold_count: int = 1,
        max_epoch: int = 100,
        download_manager: DownloadManager = None,
        gzip_enabled: bool = False,
        wapiti_binary_path: str = None,
        wapiti_train_args: dict = None):
    assert fold_count == 1, 'only fold_count == 1 supported'
    with tempfile.TemporaryDirectory(suffix='-wapiti') as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.wapiti')
        model = WapitiModelTrainAdapter(
            model_name=model,
            template_path=template_path,
            temp_model_path=temp_model_path,
            max_epoch=max_epoch,
            download_manager=download_manager,
            gzip_enabled=gzip_enabled,
            wapiti_binary_path=wapiti_binary_path,
            wapiti_train_args=wapiti_train_args
        )
        do_train_eval(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            eval_input_paths=eval_input_paths,
            eval_limit=eval_limit,
            eval_output_format=eval_output_format,
            eval_output_path=eval_output_path,
            download_manager=download_manager
        )


def do_eval_model(
        model: Union[Sequence, WapitiModelAdapter],
        input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
        split_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        eval_output_format: str = None,
        eval_output_path: str = None,
        download_manager: DownloadManager = None):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )

    if split_input:
        _, x_eval, _, y_eval, _, features_eval = train_test_split(
            x_all, y_all, features_all, test_size=0.1
        )
    else:
        x_eval = x_all
        y_eval = y_all
        features_eval = features_all

    LOGGER.info('%d evaluation sequences', len(x_eval))

    # evaluation
    classification_result = model.get_evaluation_result(
        x_eval, y_eval, features=features_eval
    )
    output_classification_result(
        classification_result,
        output_format=eval_output_format,
        output_path=eval_output_path
    )


def get_model_name(
        model: str,
        use_ELMo: bool = False,
        output_path: str = None,
        model_path: str = None):
    if output_path or model_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if use_ELMo:
        model_name += '-with_ELMo'
    return model_name


def load_delft_model(
        model: str,
        use_ELMo: bool = False,
        output_path: str = None,
        model_path: str = None,
        max_sequence_length: int = 100,
        fold_count: int = 1,
        batch_size: int = 20,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    model = Sequence(
        get_model_name(
            model,
            use_ELMo=use_ELMo,
            output_path=output_path,
            model_path=model_path
        ),
        embeddings_name=None,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    assert model_path
    model.load_from(model_path)
    return model


def eval_model(
        model,
        use_ELMo: bool = False,
        input_paths: List[str] = None,
        output_path: str = None,
        model_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        split_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = 100,
        fold_count: int = 1,
        batch_size: int = 20,
        eval_output_format: str = None,
        eval_output_path: str = None,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):

    model = load_delft_model(
        model=model,
        use_ELMo=use_ELMo,
        output_path=output_path,
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        fold_count=fold_count,
        batch_size=batch_size,
        embedding_manager=embedding_manager,
        **kwargs
    )

    do_eval_model(
        model,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        split_input=split_input,
        eval_output_format=eval_output_format,
        eval_output_path=eval_output_path,
        download_manager=download_manager
    )


def wapiti_eval_model(
        model: str = None,
        input_paths: List[str] = None,
        model_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        split_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        fold_count: int = 1,
        eval_output_format: str = None,
        eval_output_path: str = None,
        download_manager: DownloadManager = None,
        wapiti_binary_path: str = None):
    assert fold_count == 1, 'only fold_count == 1 supported'

    model = WapitiModelAdapter.load_from(
        model_path,
        download_manager=download_manager,
        wapiti_binary_path=wapiti_binary_path
    )
    do_eval_model(
        model,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        split_input=split_input,
        eval_output_format=eval_output_format,
        eval_output_path=eval_output_path,
        download_manager=download_manager
    )


def do_tag_input(
        model: Union[Sequence, WapitiModelAdapter],
        tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
        input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        download_manager: DownloadManager = None):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )

    LOGGER.info('%d input sequences', len(x_all))

    tag_result = model.tag(
        x_all,
        output_format=None,
        features=features_all
    )
    expected_tag_result = get_tag_result(
        texts=x_all,
        labels=y_all
    )
    formatted_tag_result = format_tag_result(
        tag_result,
        output_format=tag_output_format,
        expected_tag_result=expected_tag_result,
        texts=x_all,
        features=features_all,
        model_name=model._get_model_name()  # pylint: disable=protected-access
    )
    LOGGER.info('tag_result:')
    print(formatted_tag_result)


def tag_input(
        model: str,
        tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
        use_ELMo: bool = False,
        input_paths: List[str] = None,
        output_path: str = None,
        model_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = None,
        fold_count: int = 1,
        batch_size: int = 20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):

    model = load_delft_model(
        model=model,
        use_ELMo=use_ELMo,
        output_path=output_path,
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        fold_count=fold_count,
        batch_size=batch_size,
        embedding_manager=embedding_manager,
        **kwargs
    )

    do_tag_input(
        model,
        tag_output_format=tag_output_format,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )


def wapiti_tag_input(
        model: str = None,
        tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
        input_paths: List[str] = None,
        model_path: str = None,
        limit: int = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        shuffle_input: bool = False,
        download_manager: DownloadManager = None,
        wapiti_binary_path: str = None):
    model = WapitiModelAdapter.load_from(
        model_path,
        download_manager=download_manager,
        wapiti_binary_path=wapiti_binary_path
    )
    do_tag_input(
        model,
        tag_output_format=tag_output_format,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )


def print_input_info(
        model: str,
        input_paths: List[str],
        limit: int = None,
        download_manager: DownloadManager = None):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit,
        download_manager=download_manager,
        clean_features=False
    )

    seq_lengths = np.array([len(seq) for seq in x_all])
    y_counts = Counter(
        y_row
        for y_doc in y_all
        for y_row in y_doc
    )
    flat_features = list(iter_flat_features(features_all))
    feature_lengths = Counter(map(len, flat_features))

    print('number of input sequences: %d' % len(x_all))
    print('sequence lengths: %s' % format_dict(get_quantiles(seq_lengths)))
    print('token lengths: %s' % format_dict(get_quantiles(
        map(len, iter_flat_batch_tokens(x_all))
    )))
    print('number of features: %d' % len(features_all[0][0]))
    if len(feature_lengths) > 1:
        print('inconsistent feature length counts: %s' % format_dict(feature_lengths))
        for feature_length in feature_lengths:
            print('examples with feature length=%d:\n%s' % (
                feature_length,
                '\n'.join(islice((
                    ' '.join(features_vector)
                    for features_vector in flat_features
                    if len(features_vector) == feature_length
                ), 3))
            ))
        (x_all, y_all, features_all) = get_clean_x_y_features(
            x_all, y_all, features_all
        )
    quantiles_feature_value_lengths = get_quantiles_feature_value_length_by_index(features_all)
    feature_counts = get_feature_counts(features_all)
    print('feature value lengths: %s' % format_dict(quantiles_feature_value_lengths))
    print('feature counts: %s' % format_dict(feature_counts))
    print('suggested feature indices: %s' % format_indices(
        get_suggested_feature_indices(feature_counts)
    ))
    print('label counts: %s' % format_dict(y_counts))


def add_common_arguments(
        parser: argparse.ArgumentParser,
        max_sequence_length_default: int = None):
    input_group = parser.add_argument_group('input')
    input_group.add_argument(
        "--input",
        nargs='+',
        action='append',
        help="provided training file"
    )
    input_group.add_argument(
        "--shuffle-input",
        action="store_true",
        help="Shuffle the input before splitting"
    )
    input_group.add_argument(
        "--limit",
        type=int,
        help=(
            "limit the number of training samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Set the random seed for reproducibility"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="batch size"
    )
    parser.add_argument(
        "--max-sequence-length", type=int,
        default=max_sequence_length_default,
        help="maximum sequence length"
    )
    parser.add_argument(
        "--no-use-lmdb", action="store_true",
        help="Do not use LMDB embedding cache (load embeddings into memory instead)"
    )

    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")

    parser.add_argument("--quiet", action="store_true", help="Only log errors")

    parser.add_argument(
        "--save-input-to-and-exit",
        help=(
            "If set, saves the input to the specified path and exits."
            " This can be useful to retrain the model outside GROBID."
        )
    )

    parser.add_argument(
        "--log-file",
        help=(
            "If set, saves the output to the specified log file."
            " This may also be a file in a bucket, in which case it will be uploaded at the end."
            " Add the .gz extension if you wish to compress the file."
        )
    )

    parser.add_argument("--job-dir", help="job dir (only used when running via ai platform)")


def add_model_path_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--model-path", **kwargs)


def add_fold_count_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--fold-count", type=int, default=1, **kwargs)


def add_eval_output_format_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-format",
        choices=EVALUATION_OUTPUT_FORMATS,
        default=EvaluationOutputFormats.TEXT
    )


def add_eval_output_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-path",
        help='If specified, saves the evaluation to the specified path in the JSON format'
    )


def add_eval_output_arguments(parser: argparse.ArgumentParser):
    add_eval_output_format_argument(parser)
    add_eval_output_path_argument(parser)


def add_eval_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-input",
        nargs='+',
        action='append',
        help=' '.join([
            "Evaluation data at the end of training. If not specified,",
            "it will use a slice of the training data"
        ])
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help=' '.join([
            "Limit the number of documents to use for evaluation.",
            "This is mostly for testing to make evaluation faster."
        ])
    )
    parser.add_argument(
        "--eval-max-sequence-length",
        type=int,
        help=' '.join([
            "Limit the number of documents to use for evaluation.",
            "This is mostly for testing to make evaluation faster."
        ])
    )


def add_tag_output_format_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--tag-output-format",
        default=DEFAULT_TAG_OUTPUT_FORMAT,
        choices=TAG_OUTPUT_FORMATS,
        help="output format for tag results",
        **kwargs
    )


def add_output_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--output", help="directory where to save a trained model", **kwargs)


def add_max_epoch_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--max-epoch", type=int, default=10,
        help="max epoch to train to",
        **kwargs
    )


def add_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--architecture", default='BidLSTM_CRF',
        choices=get_model_names(),
        help="type of model architecture to be used"
    )
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")

    parser.add_argument(
        "--max-char-length",
        type=int,
        default=30,
        help="The maximum number of chars used by the model"
    )

    parser.add_argument(
        "--additional-token-feature-indices",
        type=parse_number_ranges,
        help="".join([
            "Additional feature values that should be used as tokens.",
            " e.g. 0 or 0-3.",
            " If blank, no additional token features will be used."
        ])
    )

    features_group = parser.add_argument_group('features')
    features_group.add_argument("--use-features", action="store_true", help="Use features")
    features_group.add_argument(
        "--feature-indices",
        type=parse_number_ranges,
        help="The feature indices to use. e.g. 7-10. If blank, all of the features will be used."
    )
    features_group.add_argument(
        "--feature-embedding-size", type=int,
        help="size of feature embedding, use 0 to disable embedding"
    )
    features_group.add_argument(
        "--use-features-indices-input", action="store_true",
        help="Use features indices values (should be inferred from the model)"
    )
    features_group.add_argument(
        "--features-lstm-units", type=int,
        help="Number of LSTM units used by the features"
    )

    parser.add_argument(
        "--stateful", action="store_true",
        help="Make RNNs stateful (required for truncated BPTT)"
    )

    parser.add_argument(
        "--input-window-stride",
        type=int,
        help="Should be equal or less than max sequence length"
    )

    output_group = parser.add_argument_group('output')
    add_output_argument(output_group)
    output_group.add_argument("--checkpoint", help="directory where to save a checkpoint model")

    parser.add_argument(
        "--embedding", default="glove-6B-50d",
        help="name of word embedding"
    )
    features_group.add_argument(
        "--no-embedding",
        dest="use_word_embeddings",
        default=True,
        action="store_false",
        help="Disable the use of word embedding"
    )
    parser.add_argument(
        "--word-lstm-units", type=int, default=100,
        help="number of words in lstm units"
    )
    add_max_epoch_argument(parser)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="how many epochs to continue training after the f1 score hasn't improved"
    )


def add_wapiti_train_arguments(parser: argparse.ArgumentParser):
    add_output_argument(parser)
    add_max_epoch_argument(parser)
    parser.add_argument("--wapiti-template", required=True)
    parser.add_argument(
        "--wapiti-gzip",
        action="store_true",
        help="whether to gzip wapiti models before saving"
    )
    parser.add_argument(
        "--wapiti-stop-epsilon-value",
        default=DEFAULT_STOP_EPSILON_VALUE
    )
    parser.add_argument(
        "--wapiti-stop-window-size",
        type=int,
        default=DEFAULT_STOP_WINDOW_SIZE
    )


def get_wapiti_train_args(args: argparse.Namespace) -> dict:
    return dict(
        stop_epsilon_value=args.wapiti_stop_epsilon_value,
        stop_window_size=args.wapiti_stop_window_size
    )


def add_wapiti_install_arguments(parser: argparse.ArgumentParser):
    example_url = "https://github.com/kermitt2/Wapiti/archive/master.tar.gz"
    parser.add_argument(
        "--wapiti-install-source",
        help="source file to install wapiti from, e.g. %s" % example_url
    )


def add_all_non_positional_arguments(parser: argparse.ArgumentParser):
    add_common_arguments(parser)
    add_train_arguments(parser)


def add_model_positional_argument(parser: argparse.ArgumentParser):
    parser.add_argument("model", nargs='?', choices=GROBID_MODEL_NAMES)


def _flatten_input_paths(input_paths_list: List[List[str]]) -> List[str]:
    if not input_paths_list:
        return input_paths_list
    return [input_path for input_paths in input_paths_list for input_path in input_paths]


def process_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input = _flatten_input_paths(args.input)
    try:
        args.eval_input = _flatten_input_paths(args.eval_input)
    except AttributeError:
        pass


def create_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )


def save_input_to(input_paths: List[str], output_path: str):
    assert len(input_paths) == 1, "exactly one input path expected (got: %s)" % input_paths
    input_path = input_paths[0]
    LOGGER.info('saving input (%s) to: %s', input_path, output_path)
    copy_file(input_path, output_path)


def get_eval_output_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_output_format=args.eval_output_format,
        eval_output_path=args.eval_output_path
    )


class GrobidTrainerSubCommand(SubCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_manager = None
        self.embedding_manager = None

    @abstractmethod
    def do_run(self, args: argparse.Namespace):
        pass

    def preload_and_validate_embedding(
            self,
            embedding_name: str,
            use_word_embeddings: bool = True) -> str:
        if not use_word_embeddings:
            return None
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        self.embedding_manager.validate_embedding(embedding_name)
        return embedding_name

    def get_common_args(self, args: argparse.Namespace) -> dict:
        return dict(
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            shuffle_input=args.shuffle_input,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            multiprocessing=args.multiprocessing,
            embedding_manager=self.embedding_manager,
            download_manager=self.download_manager
        )

    def get_train_args(self, args: argparse.Namespace) -> dict:
        return dict(
            architecture=args.architecture,
            use_ELMo=args.use_ELMo,
            output_path=args.output,
            log_dir=args.checkpoint,
            word_lstm_units=args.word_lstm_units,
            max_epoch=args.max_epoch,
            use_features=args.use_features,
            feature_indices=args.feature_indices,
            feature_embedding_size=args.feature_embedding_size,
            patience=args.early_stopping_patience,
            config_props=dict(
                max_char_length=args.max_char_length,
                additional_token_feature_indices=args.additional_token_feature_indices,
                use_word_embeddings=args.use_word_embeddings,
                use_features_indices_input=args.use_features_indices_input,
                features_lstm_units=args.features_lstm_units,
                stateful=args.stateful
            ),
            training_props=dict(
                input_window_stride=args.input_window_stride
            ),
            **self.get_common_args(args)
        )

    def run(self, args: argparse.Namespace):
        if args.save_input_to_and_exit:
            save_input_to(args.input, args.save_input_to_and_exit)
            return

        self.download_manager = DownloadManager()
        self.embedding_manager = EmbeddingManager(
            download_manager=self.download_manager
        )
        if args.no_use_lmdb:
            self.embedding_manager.disable_embedding_lmdb_cache()

        set_random_seeds(args.random_seed)

        self.do_run(args)

        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()


class TrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train(
            embeddings_name=embedding_name,
            **self.get_train_args(args)
        )


class WapitiTrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train(
            model=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args)
        )


class TrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')
        add_fold_count_argument(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train_eval(
            fold_count=args.fold_count,
            embeddings_name=embedding_name,
            eval_input_paths=args.eval_input,
            eval_limit=args.eval_limit,
            eval_max_sequence_length=args.eval_max_sequence_length,
            **get_eval_output_args(args),
            **self.get_train_args(args)
        )


class WapitiTrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train_eval(
            model=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            eval_input_paths=args.eval_input,
            eval_limit=args.eval_limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args),
            **get_eval_output_args(args),
        )


class EvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        parser.add_argument(
            "--use-eval-train-test-split",
            action="store_true",
            help=" ".join([
                "If enabled, split the input when running 'eval'",
                "(in the same way it is split for 'train_eval')"
            ])
        )
        add_eval_output_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        eval_model(
            model_path=args.model_path,
            split_input=args.use_eval_train_test_split,
            **get_eval_output_args(args),
            **self.get_common_args(args)
        )


class WapitiEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_eval_model(
            model_path=args.model_path,
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            **get_eval_output_args(args),
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class TagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)

    def do_run(self, args: argparse.Namespace):
        tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            **self.get_common_args(args)
        )


class WapitiTagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class InputInfoSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        print_input_info(
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager
        )


SUB_COMMANDS = [
    TrainSubCommand(
        Tasks.TRAIN,
        'Train the model using the provided input(s)'
    ),
    TrainEvalSubCommand(
        Tasks.TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    EvalSubCommand(
        Tasks.EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    TagSubCommand(
        Tasks.TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    WapitiTrainSubCommand(
        Tasks.WAPITI_TRAIN,
        'Train the model using the provided input(s)'
    ),
    WapitiTrainEvalSubCommand(
        Tasks.WAPITI_TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    WapitiEvalSubCommand(
        Tasks.WAPITI_EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    WapitiTagSubCommand(
        Tasks.WAPITI_TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    InputInfoSubCommand(
        Tasks.INPUT_INFO,
        'Display input summary information relating to the passed in input(s)'
    )
]


def get_subcommand_processor():
    return SubCommandProcessor(SUB_COMMANDS, command_dest='action')


def parse_args(argv: List[str] = None, subcommand_processor: SubCommandProcessor = None):
    parser = create_parser()
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')

    add_model_positional_argument(parser)
    subcommand_processor.add_sub_command_parsers(parser)

    args = parser.parse_args(argv)
    process_args(args)
    return args


def run(args: argparse.Namespace, subcommand_processor: SubCommandProcessor = None):
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')
    try:
        subcommand_processor.run(args)
    except BaseException as e:
        LOGGER.error('uncaught exception: %s', e, exc_info=1)
        raise


def main(argv: List[str] = None):
    subcommand_processor = get_subcommand_processor()
    args = parse_args(argv, subcommand_processor=subcommand_processor)
    if args.quiet:
        logging.root.setLevel('ERROR')
    elif args.debug:
        for name in [__name__, 'sciencebeam_trainer_delft', 'delft']:
            logging.getLogger(name).setLevel('DEBUG')
    if args.log_file:
        with auto_uploading_output_file(args.log_file, mode='w') as log_fp:
            try:
                with tee_stdout_and_stderr_lines_to(log_fp.write, append_line_feed=True):
                    with tee_logging_lines_to(log_fp.write, append_line_feed=True):
                        run(args, subcommand_processor=subcommand_processor)
            finally:
                logging.shutdown()
    else:
        run(args, subcommand_processor=subcommand_processor)


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()

    main()
