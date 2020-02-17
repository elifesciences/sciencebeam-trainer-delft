# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
import time
from typing import List, Tuple

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
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
from sciencebeam_trainer_delft.utils.io import copy_file

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.wrapper import Sequence
from sciencebeam_trainer_delft.sequence_labelling.models import get_model_names, patch_get_model
from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_and_labels_crf_file
from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TAG_OUTPUT_FORMATS,
    get_tag_result,
    format_tag_result
)


LOGGER = logging.getLogger(__name__)


GROBID_MODEL_NAMES = [
    'affiliation-address', 'citation', 'date', 'header',
    'name-citation', 'name-header', 'software'
]


class Tasks:
    TRAIN = 'train'
    TRAIN_EVAL = 'train_eval'
    EVAL = 'eval'
    TAG = 'tag'


ALL_TASKS = [Tasks.TRAIN, Tasks.TRAIN_EVAL, Tasks.EVAL, Tasks.TAG]

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


def load_data_and_labels(
        model: str, input_paths: List[str] = None,
        limit: int = None,
        shuffle_input: bool = False,
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
    return x_all, y_all, f_all


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
    # model.save = wrap_save(model.save)

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


# split data, train a GROBID model and evaluate it
def train_eval(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_paths: List[str] = None,
        output_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )

    x_train_all, x_eval, y_train_all, y_eval, features_train_all, features_eval = train_test_split(
        x_all, y_all, features_all, test_size=0.1
    )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_train_all, y_train_all, features_train_all, test_size=0.1
    )

    LOGGER.info('%d train sequences', len(x_train))
    LOGGER.info('%d validation sequences', len(x_valid))
    LOGGER.info('%d evaluation sequences', len(x_eval))

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
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

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
    print("\nEvaluation:")
    model.eval(x_eval, y_eval, features=features_eval)

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()


def eval_model(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_paths: List[str] = None,
        output_path: str = None,
        model_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        split_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
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

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if use_ELMo:
        model_name += '-with_ELMo'

    # set embeddings_name to None, it will be loaded from the model
    embeddings_name = None

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    assert model_path
    model.load_from(model_path)

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval, features=features_eval)


def tag_input(
        model,
        tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
        embeddings_name: str = None,
        architecture: str = 'BidLSTM_CRF',
        use_ELMo: bool = False,
        input_paths: List[str] = None,
        output_path: str = None,
        model_path: str = None,
        limit: int = None,
        shuffle_input: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        download_manager=download_manager
    )

    LOGGER.info('%d input sequences', len(x_all))

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if use_ELMo:
        model_name += '-with_ELMo'

    # set embeddings_name to None, it will be loaded from the model
    embeddings_name = None

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    assert model_path
    model.load_from(model_path)

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


def add_all_non_positional_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument(
        "--architecture", default='BidLSTM_CRF',
        choices=get_model_names(),
        help="type of model architecture to be used"
    )
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")
    parser.add_argument("--use-features", action="store_true", help="Use features")
    parser.add_argument(
        "--feature-indices",
        type=parse_number_ranges,
        help="The feature indices to use. e.g. 7:10. If blank, all of the features will be used."
    )
    parser.add_argument(
        "--feature-embedding-size", type=int,
        help="size of feature embedding, use 0 to disable embedding"
    )
    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")

    output_group = parser.add_argument_group('output')
    output_group.add_argument("--output", help="directory where to save a trained model")
    output_group.add_argument("--checkpoint", help="directory where to save a checkpoint model")
    output_group.add_argument(
        "--tag-output-format",
        default=DEFAULT_TAG_OUTPUT_FORMAT,
        choices=TAG_OUTPUT_FORMATS,
        help="output format for tag results"
    )

    parser.add_argument("--model-path", help="directory to the saved or loaded model")

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
        "--use-eval-train-test-split",
        action="store_true",
        help=" ".join([
            "If enabled, split the input when running 'eval'",
            "(in the same way it is split for 'train_eval')"
        ])
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
        "--embedding", default="glove-6B-50d",
        help="name of word embedding"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="batch size"
    )
    parser.add_argument(
        "--word-lstm-units", type=int, default=100,
        help="number of words in lstm units"
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=500,
        help="maximum sequence length"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=10,
        help="max epoch to train to"
    )

    parser.add_argument(
        "--no-use-lmdb", action="store_true",
        help="Do not use LMDB embedding cache (load embeddings into memory instead)"
    )

    parser.add_argument(
        "--save-input-to-and-exit",
        help=(
            "If set, saves the input to the specified path and exits."
            " This can be useful to retrain the model outside GROBID."
        )
    )

    parser.add_argument("--job-dir", help="job dir (only used when running via ai platform)")


def process_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input:
        args.input = [input_path for input_paths in args.input for input_path in input_paths]


def parse_args(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )

    parser.add_argument("model", choices=GROBID_MODEL_NAMES)
    parser.add_argument("action", choices=ALL_TASKS)
    add_all_non_positional_arguments(parser)

    args = parser.parse_args(argv)
    process_args(args)
    return args


def save_input_to(input_paths: List[str], output_path: str):
    assert len(input_paths) == 1, "exactly one input path expected (got: %s)" % input_paths
    input_path = input_paths[0]
    LOGGER.info('saving input (%s) to: %s', input_path, output_path)
    copy_file(input_path, output_path)


def run(args):
    model = args.model
    action = args.action

    if args.save_input_to_and_exit:
        save_input_to(args.input, args.save_input_to_and_exit)
        return

    use_ELMo = args.use_ELMo
    architecture = args.architecture

    download_manager = DownloadManager()

    embedding_manager = EmbeddingManager(download_manager=download_manager)
    if args.no_use_lmdb:
        embedding_manager.disable_embedding_lmdb_cache()
    if action in {Tasks.TRAIN, Tasks.TRAIN_EVAL}:
        embedding_name = embedding_manager.ensure_available(args.embedding)
        LOGGER.info('embedding_name: %s', embedding_name)
        embedding_manager.validate_embedding(embedding_name)
    else:
        embedding_name = embedding_manager.resolve_alias(args.embedding)

    train_args = dict(
        model=model,
        embeddings_name=embedding_name,
        embedding_manager=embedding_manager,
        architecture=architecture, use_ELMo=use_ELMo,
        input_paths=args.input,
        output_path=args.output,
        limit=args.limit,
        shuffle_input=args.shuffle_input,
        random_seed=args.random_seed,
        log_dir=args.checkpoint,
        batch_size=args.batch_size,
        word_lstm_units=args.word_lstm_units,
        max_sequence_length=args.max_sequence_length,
        max_epoch=args.max_epoch,
        use_features=args.use_features,
        feature_indices=args.feature_indices,
        feature_embedding_size=args.feature_embedding_size,
        multiprocessing=args.multiprocessing,
        download_manager=download_manager
    )

    LOGGER.info('get_tf_info: %s', get_tf_info())

    set_random_seeds(args.random_seed)

    if action == Tasks.TRAIN:
        train(**train_args)

    if action == Tasks.TRAIN_EVAL:
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(
            fold_count=args.fold_count,
            **train_args
        )

    if action == Tasks.EVAL:
        if not args.model_path:
            raise ValueError('--model-path required')
        eval_model(
            model_path=args.model_path,
            split_input=args.use_eval_train_test_split,
            **train_args
        )

    if action == Tasks.TAG:
        if not args.model_path:
            raise ValueError('--model-path required')
        tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            **train_args
        )

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


def main(argv: List[str] = None):
    args = parse_args(argv)
    try:
        run(args)
    except BaseException as e:
        LOGGER.error('uncaught exception: %s', e, exc_info=1)
        raise


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()

    main()
