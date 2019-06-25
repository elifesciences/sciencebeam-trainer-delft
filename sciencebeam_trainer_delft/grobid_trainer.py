# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
import time

from sklearn.model_selection import train_test_split
import keras.backend as K

from delft.sequenceLabelling import Sequence

from delft.sequenceLabelling.reader import (
    load_data_and_labels_crf_file
)


MODELS = [
    'affiliation-address', 'citation', 'date', 'header',
    'name-citation', 'name-header', 'software'
]


# train a GROBID model with all available data
def train(
        model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
        input_path=None, output_path=None,
        max_epoch=100, **kwargs):
    print('Loading data...')
    if input_path is None:
        x_all, y_all, _ = load_data_and_labels_crf_file(
            'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'
        )
    else:
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
        x_all, y_all, _ = load_data_and_labels_crf_file(
            'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'
        )
    else:
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


def main():
    parser = argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )

    actions = ['train', 'tag', 'train_eval', 'eval']
    architectures = ['BidLSTM_CRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU-CRF']

    parser.add_argument("model")
    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture", default='BidLSTM_CRF',
                        help="type of model architecture to be used, one of " + str(architectures))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")
    parser.add_argument("--output", help="directory where to save a trained model")
    parser.add_argument("--input", help="provided training file")

    args = parser.parse_args()

    model = args.model
    if not model in MODELS:
        print('invalid model, should be one of', MODELS)

    action = args.action
    if action not in actions:
        print('action not specified, must be one of ' + str(actions))

    use_ELMo = args.use_ELMo
    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of ' + str(architectures))

    output = args.output
    input_path = args.input

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file
    # embedding-registry.json
    # be sure to use here the same name as in the registry
    # e.g. 'glove-840B', 'fasttext-crawl', 'word2vec',
    # and that the path in the registry to the embedding file is correct on your system
    embeddings_name = "glove-6B-50d"
    default_args = dict(batch_size=1, max_sequence_length=50, max_epoch=1)

    if action == 'train':
        train(
            model, embeddings_name, architecture=architecture, use_ELMo=use_ELMo,
            input_path=input_path, output_path=output,
            **default_args
        )

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(
            model, embeddings_name, architecture=architecture, use_ELMo=use_ELMo,
            input_path=input_path, output_path=output, fold_count=args.fold_count,
            **default_args
        )

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    main()
