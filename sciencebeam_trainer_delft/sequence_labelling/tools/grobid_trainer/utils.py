# initially based on:
# https://github.com/kermitt2/delft/blob/master/grobidTagger.py

import logging
import time
import tempfile
import os
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from typing import List, Tuple, Union

import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays
from sciencebeam_trainer_delft.utils.io import (
    write_text
)

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    TrainNotificationManager,
    notify_train_success,
    notify_train_error
)

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    Sequence
)
from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_and_labels_crf_file

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    WapitiModelAdapter,
    WapitiModelTrainAdapter
)

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    get_tag_result,
    format_tag_result
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    EvaluationOutputFormats,
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


LOGGER = logging.getLogger(__name__)


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
        train_notification_manager: TrainNotificationManager = None,
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

    notify_train_success(
        train_notification_manager,
        model_path=model.get_model_output_path(output_path),
        last_checkpoint_path=model.last_checkpoint_path
    )


def do_train_with_error_notification(
        model: Union[Sequence, WapitiModelTrainAdapter],
        output_path: str = None,
        train_notification_manager: TrainNotificationManager = None,
        **kwargs):
    model_path = model.get_model_output_path(output_path)
    try:
        do_train(
            model=model,
            output_path=output_path,
            train_notification_manager=train_notification_manager,
            **kwargs
        )
    except BaseException as error:  # pylint: disable=broad-except
        notify_train_error(
            train_notification_manager,
            model_path=model_path,
            error=repr(error)
        )
        raise


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
        resume_train_model_path: str = None,
        train_notification_manager: TrainNotificationManager = None,
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
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        **kwargs
    )
    if resume_train_model_path:
        model.load_from(resume_train_model_path)

    do_train_with_error_notification(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        train_notification_manager=train_notification_manager,
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
        train_notification_manager: TrainNotificationManager = None,
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
        do_train_with_error_notification(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            train_notification_manager=train_notification_manager,
            download_manager=download_manager
        )


def output_classification_result(
        classification_result: ClassificationResult,
        eval_output_args: dict,
        eval_input_paths: List[str] = None,
        model_path: str = None,
        model_summary_props: dict = None):
    eval_output_args = eval_output_args or dict()
    output_format = eval_output_args.get('eval_output_args')
    output_path = eval_output_args.get('eval_output_path')
    eval_first_entity = eval_output_args.get('eval_first_entity')
    if not output_format:
        output_format = EvaluationOutputFormats.TEXT
    if eval_first_entity:
        classification_result = classification_result.with_first_entities()
    meta = {}
    meta['eval_timestamp'] = datetime.now(timezone.utc).isoformat()
    if eval_input_paths:
        meta['eval_input_paths'] = eval_input_paths
    if model_path:
        meta['model_path'] = model_path
    if model_summary_props:
        meta.update(model_summary_props)
    if output_path:
        LOGGER.info('writing evaluation to: %s', output_path)
        write_text(output_path, classification_result.get_json_formatted_report(meta=meta))
    if output_format == EvaluationOutputFormats.TEXT:
        print("\nEvaluation:\n%s" % classification_result.get_text_formatted_report(
            digits=4
        ))
    elif output_format == EvaluationOutputFormats.JSON:
        print(classification_result.get_json_formatted_report(meta=meta))
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
        eval_output_args: dict = None,
        fold_count: int = 1,
        train_notification_manager: TrainNotificationManager = None,
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
        eval_output_args=eval_output_args,
        eval_input_paths=eval_input_paths,
        model_path=model.get_model_output_path(output_path),
        model_summary_props=model.model_summary_props
    )

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()

    notify_train_success(
        train_notification_manager,
        model_path=model.get_model_output_path(output_path),
        last_checkpoint_path=model.last_checkpoint_path,
        classification_result=classification_result
    )


def do_train_eval_with_error_notification(
        model: Union[Sequence, WapitiModelTrainAdapter],
        output_path: str = None,
        train_notification_manager: TrainNotificationManager = None,
        **kwargs):
    model_path = model.get_model_output_path(output_path)
    try:
        do_train_eval(
            model=model,
            output_path=output_path,
            train_notification_manager=train_notification_manager,
            **kwargs
        )
    except BaseException as error:  # pylint: disable=broad-except
        notify_train_error(
            train_notification_manager,
            model_path=model_path,
            error=repr(error)
        )
        raise


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
        eval_output_args: dict = None,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        resume_train_model_path: str = None,
        train_notification_manager: TrainNotificationManager = None,
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
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )
    if resume_train_model_path:
        model.load_from(resume_train_model_path)
    do_train_eval_with_error_notification(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        eval_input_paths=eval_input_paths,
        eval_limit=eval_limit,
        eval_output_args=eval_output_args,
        train_notification_manager=train_notification_manager,
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
        eval_output_args: dict = None,
        fold_count: int = 1,
        max_epoch: int = 100,
        train_notification_manager: TrainNotificationManager = None,
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
        do_train_eval_with_error_notification(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            eval_input_paths=eval_input_paths,
            eval_limit=eval_limit,
            eval_output_args=eval_output_args,
            train_notification_manager=train_notification_manager,
            download_manager=download_manager
        )


def do_eval_model(
        model: Union[Sequence, WapitiModelAdapter],
        input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
        split_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        eval_output_args: dict = None,
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
        eval_output_args=eval_output_args,
        eval_input_paths=input_paths,
        model_path=model.model_path,
        model_summary_props=model.model_summary_props
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
        eval_output_args: dict = None,
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
        eval_output_args=eval_output_args,
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
        eval_output_args: dict = None,
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
        eval_output_args=eval_output_args,
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
        input_window_stride: int = None,
        stateful: bool = None,
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
        input_window_stride=input_window_stride,
        stateful=stateful,
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
