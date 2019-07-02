# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
import time
from typing import List

from sklearn.model_selection import train_test_split
import keras.backend as K

from delft.sequenceLabelling import Sequence

from delft.sequenceLabelling.reader import (
    load_data_and_labels_crf_file
)

from sciencebeam_trainer_delft.cloud_support import patch_cloud_support
from sciencebeam_trainer_delft.embedding_manager import EmbeddingManager
from sciencebeam_trainer_delft.models import get_model_names, patch_get_model


GROBID_MODEL_NAMES = [
    'affiliation-address', 'citation', 'date', 'header',
    'name-citation', 'name-header', 'software'
]


def get_default_training_data(model: str) -> str:
    return 'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'


# train a GROBID model with all available data
def train(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_path=None, output_path=None,
        max_epoch=100, **kwargs):
    print('Loading data...')
    if input_path is None:
        input_path = get_default_training_data(model)
    x_all, y_all, _ = load_data_and_labels_crf_file(input_path)
    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-'+model

    if use_ELMo:
        model_name += '-with_ELMo'

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        model_type=architecture,
        use_ELMo=use_ELMo,
        **kwargs
    )
    # model.save = wrap_save(model.save)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
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
        fold_count=1, max_epoch=100, batch_size=20, **kwargs):
    print('Loading data...')
    if input_path is None:
        input_path = get_default_training_data(model)
    x_all, y_all, _ = load_data_and_labels_crf_file(input_path)

    x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-'+model

    if use_ELMo:
        model_name += '-with_ELMo'
        if model_name in {'software-with_ELMo', 'grobid-software-with_ELMo'}:
            batch_size = 3

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        recurrent_dropout=0.50,
        embeddings_name=embeddings_name,
        model_type=architecture,
        use_ELMo=use_ELMo,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    start_time = time.time()

    if fold_count == 1:
        model.train(x_train, y_train, x_valid, y_valid)
    else:
        model.train_nfold(x_train, y_train, x_valid, y_valid, fold_number=fold_count)

    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval)

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()


def parse_args(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )

    parser.add_argument("model", choices=GROBID_MODEL_NAMES)
    parser.add_argument("action", choices=['train', 'train_eval'])
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument(
        "--architecture", default='BidLSTM_CRF',
        choices=get_model_names(),
        help="type of model architecture to be used"
    )
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")
    parser.add_argument("--output", help="directory where to save a trained model")
    parser.add_argument("--checkpoint", help="directory where to save a checkpoint model")
    parser.add_argument("--input", help="provided training file")
    parser.add_argument(
        "--embedding", default="glove-6B-50d",
        help="name of word embedding"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="batch size"
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=500,
        help="maximum sequence length"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=10,
        help="max epoch to train to"
    )

    args = parser.parse_args(argv)
    return args


def run(args):
    model = args.model
    action = args.action

    use_ELMo = args.use_ELMo
    architecture = args.architecture

    embedding_manager = EmbeddingManager()
    embedding_name = embedding_manager.download_and_install_embedding_if_url(
        args.embedding
    )
    embedding_manager.validate_embedding(embedding_name)

    train_args = dict(
        model=model,
        embeddings_name=embedding_name,
        architecture=architecture, use_ELMo=use_ELMo,
        input_path=args.input,
        output_path=args.output,
        log_dir=args.checkpoint,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        max_epoch=args.max_epoch
    )

    if action == 'train':
        train(**train_args)

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(
            fold_count=args.fold_count,
            **train_args
        )

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


def main(argv: List[str] = None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()

    main()
