# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
import time
from typing import List

import sciencebeam_trainer_delft.no_warn_if_disabled  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import numpy as np

from sklearn.model_selection import train_test_split
import keras.backend as K

from sciencebeam_trainer_delft.utils.misc import parse_number_ranges
from sciencebeam_trainer_delft.wrapper import Sequence

from sciencebeam_trainer_delft.cloud_support import patch_cloud_support
from sciencebeam_trainer_delft.embedding_manager import EmbeddingManager
from sciencebeam_trainer_delft.models import get_model_names, patch_get_model
from sciencebeam_trainer_delft.reader import load_data_and_labels_crf_file
from sciencebeam_trainer_delft.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.tf import get_tf_info


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


def get_default_training_data(model: str) -> str:
    return 'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'


def log_data_info(x: np.array, y: np.array, features: np.array):
    LOGGER.info('x sample: %s (y: %s)', x[:1][:10], y[:1][:1])
    LOGGER.info(
        'feature dimensions of first sample, word: %s',
        [{index: value for index, value in enumerate(features[0][0])}]
    )


def load_data_and_labels(
        model: str, input_path: str = None,
        limit: int = None,
        download_manager: DownloadManager = None):
    assert download_manager
    if input_path is None:
        input_path = get_default_training_data(model)
    LOGGER.info('loading data from: %s', input_path)
    x_all, y_all, f_all = load_data_and_labels_crf_file(
        download_manager.download_if_url(input_path),
        limit=limit
    )
    log_data_info(x_all, y_all, f_all)
    return x_all, y_all, f_all


# train a GROBID model with all available data
def train(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_path=None, output_path=None,
        limit: int = None,
        max_sequence_length: int = 100,
        max_epoch=100,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_path=input_path, limit=limit,
        download_manager=download_manager
    )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_all, y_all, features_all, test_size=0.1
    )

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

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
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    if output_path:
        print('saving model to:', output_path)
        model.save(output_path)
    else:
        model.save()


# split data, train a GROBID model and evaluate it
def train_eval(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_path=None, output_path=None,
        limit: int = None,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_path=input_path, limit=limit,
        download_manager=download_manager
    )

    x_train_all, x_eval, y_train_all, y_eval, features_train_all, features_eval = train_test_split(
        x_all, y_all, features_all, test_size=0.1
    )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_train_all, y_train_all, features_train_all, test_size=0.1
    )

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

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
    print("training runtime: %s seconds " % (runtime))

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
        input_path=None, output_path=None, model_path: str = None,
        limit: int = None,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, y_all, features_all = load_data_and_labels(
        model=model, input_path=input_path, limit=limit,
        download_manager=download_manager
    )

    _, x_eval, _, y_eval, _, features_eval = train_test_split(
        x_all, y_all, features_all, test_size=0.1
    )

    print(len(x_eval), 'evaluation sequences')

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
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_path=None, output_path=None, model_path: str = None,
        limit: int = None,
        max_sequence_length: int = 100,
        fold_count=1, max_epoch=100, batch_size=20,
        download_manager: DownloadManager = None,
        embedding_manager: EmbeddingManager = None,
        **kwargs):
    x_all, _, features_all = load_data_and_labels(
        model=model, input_path=input_path, limit=limit,
        download_manager=download_manager
    )

    print(len(x_all), 'input sequences')

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

    tag_result = model.tag(x_all, output_format=None, features=features_all)
    LOGGER.info('tag results: %s', tag_result)


def parse_args(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )

    parser.add_argument("model", choices=GROBID_MODEL_NAMES)
    parser.add_argument("action", choices=ALL_TASKS)
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
    parser.add_argument("--output", help="directory where to save a trained model")
    parser.add_argument("--checkpoint", help="directory where to save a checkpoint model")
    parser.add_argument("--model-path", help="directory to the saved model")
    parser.add_argument("--input", help="provided training file")
    parser.add_argument("--limit", type=int, help="limit the number of training samples")
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

    parser.add_argument("--job-dir", help="job dir (only used when running via ai platform)")

    args = parser.parse_args(argv)
    return args


def run(args):
    model = args.model
    action = args.action

    use_ELMo = args.use_ELMo
    architecture = args.architecture

    download_manager = DownloadManager()

    embedding_manager = EmbeddingManager(download_manager=download_manager)
    if args.no_use_lmdb:
        embedding_manager.disable_embedding_lmdb_cache()
    if action in {Tasks.TRAIN, Tasks.TRAIN_EVAL}:
        embedding_name = embedding_manager.ensure_available(args.embedding)
    else:
        embedding_name = embedding_manager.resolve_alias(args.embedding)
    LOGGER.info('embedding_name: %s', embedding_name)
    embedding_manager.validate_embedding(embedding_name)

    train_args = dict(
        model=model,
        embeddings_name=embedding_name,
        embedding_manager=embedding_manager,
        architecture=architecture, use_ELMo=use_ELMo,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
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
        eval_model(model_path=args.model_path, **train_args)

    if action == Tasks.TAG:
        if not args.model_path:
            raise ValueError('--model-path required')
        tag_input(model_path=args.model_path, **train_args)

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


def main(argv: List[str] = None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()

    main()
