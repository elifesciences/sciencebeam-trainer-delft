import logging
from unittest.mock import MagicMock
from typing import List

import pytest

import numpy as np

from delft.sequenceLabelling.preprocess import to_casing_single

from sciencebeam_trainer_delft.sequence_labelling.data_generator import (
    is_batch_multi_tokens,
    get_batch_multi_token_count,
    left_pad_batch_values,
    get_stateless_window_indices_and_offset,
    get_batch_window_indices_and_offset,
    DataGenerator
)


LOGGER = logging.getLogger(__name__)


# Note: don't use numerics in words due to num_norm flag
WORD_1 = 'word_a'
WORD_2 = 'word_b'
WORD_3 = 'word_c'
WORD_4 = 'word_d'

SENTENCE_TOKENS_1 = [WORD_1, WORD_2]

LONG_SENTENCE_TOKENS = SENTENCE_TOKENS_1 + SENTENCE_TOKENS_1 + SENTENCE_TOKENS_1
SHORT_SENTENCE_TOKENS = SENTENCE_TOKENS_1

WORD_FEATURES_1 = ('feature1-1', 'feature1-2')
WORD_FEATURES_2 = ('feature2-1', 'feature2-2')

FEATURE_MAP = {
    WORD_FEATURES_1: np.asarray([1, 0, 0, 1]),
    WORD_FEATURES_2: np.asarray([2, 0, 0, 2])
}

SENTENCE_FEATURES_1 = [WORD_FEATURES_1, WORD_FEATURES_2]

LABEL_1 = 'label1'
LABEL_2 = 'label2'
LABEL_3 = 'label3'

EMBEDDING_MAP = {
    None: [0, 0, 0],
    WORD_1: [1, 1, 1],
    WORD_2: [2, 2, 2],
    WORD_3: [3, 3, 3],
    WORD_4: [4, 4, 4]
}


WORD_INDEX_MAP = {
    None: 0,
    WORD_1: 1,
    WORD_2: 2,
    WORD_3: 3,
    WORD_4: 4
}


LABEL_INDEX_MAP = {
    LABEL_1: 1,
    LABEL_2: 2,
    LABEL_3: 3
}


DEFAULT_ARGS = dict(
    batch_size=1,
    tokenize=False,
    shuffle=False
)


def get_word_vector(word: str):
    return np.asarray(EMBEDDING_MAP[word])


def get_word_vectors(words: List[str]):
    return np.stack([get_word_vector(word) for word in words])


def get_word_indices(words: List[str]):
    if len(words) > 0 and isinstance(words[0], (tuple, list, np.ndarray)):
        LOGGER.debug('multi words: %s', words)
        return np.concatenate([
            get_word_indices([multi_word[i] for multi_word in words])
            for i in range(len(words[0]))
        ], axis=-1)
    return np.asarray([WORD_INDEX_MAP[word] for word in words])


def get_label_indices(labels: List[str]):
    return np.asarray([LABEL_INDEX_MAP[label] for label in labels])


def get_transformed_feature(word_features: str):
    return np.asarray(FEATURE_MAP[tuple(word_features)])


def get_transformed_features(sentence_features: List[str]):
    return np.stack([get_transformed_feature(feature) for feature in sentence_features])


def preprocess_transform(X, y=None, extend=False):
    X_extend = X
    if extend:
        LOGGER.debug('extending X: %s', X)
        extend_value = None
        if is_batch_multi_tokens(X):
            extend_value = [None] * get_batch_multi_token_count(X)
        X_extend = [
            list(sentence_tokens) + [extend_value]
            for sentence_tokens in X
        ]
    x_words_transformed = [get_word_indices(sentence_tokens) for sentence_tokens in X_extend]
    x_lengths_transformed = [len(sentence_tokens) for sentence_tokens in X]
    LOGGER.debug('x_lengths_transformed: %s', x_lengths_transformed)
    if y is None:
        return x_words_transformed, x_lengths_transformed
    y_transformed = [get_label_indices(sentence_labels) for sentence_labels in y]
    return (x_words_transformed, x_lengths_transformed), y_transformed


def preprocess_transform_features(features_batch):
    return np.asarray([get_transformed_features(features) for features in features_batch])


def get_lengths(a):
    return [len(x) for x in a]


@pytest.fixture(name='preprocessor')
def _preprocessor():
    preprocessor = MagicMock(name='preprocessor')
    preprocessor.return_casing = False
    preprocessor.return_lengths = False
    preprocessor.return_features = False
    preprocessor.transform.side_effect = preprocess_transform
    preprocessor.transform_features.side_effect = preprocess_transform_features
    return preprocessor


@pytest.fixture(name='embeddings')
def _embeddings():
    embeddings = MagicMock(name='embeddings')
    embeddings.get_word_vector.side_effect = get_word_vector
    embeddings.use_ELMo = False
    embeddings.use_BERT = False
    embeddings.embed_size = len(get_word_vector(WORD_1))
    return embeddings


def all_close(a, b):
    LOGGER.debug('a: %s', a)
    LOGGER.debug('b: %s', b)
    return np.allclose(a, b)


class TestLeftPadBatchValues:
    def test_should_left_pad_values(self):
        all_close(
            left_pad_batch_values(
                np.asarray([
                    [[1, 1], [2, 2]],
                    [[1, 1]]
                ]),
                2,
                dtype='float32'
            ), np.asarray([
                [[1, 1], [2, 2]],
                [[1, 1], [0, 0]]
            ])
        )

    def test_should_truncate_values(self):
        all_close(
            left_pad_batch_values(
                np.asarray([
                    [[1, 1], [2, 2], [3, 3]],
                    [[1, 1]]
                ]),
                2,
                dtype='float32'
            ), np.asarray([
                [[1, 1], [2, 2]],
                [[1, 1], [0, 0]]
            ])
        )

    def test_should_left_pad_zero_length_values(self):
        all_close(
            left_pad_batch_values(
                np.asarray([
                    [],
                    [[1, 1], [2, 2]],
                    [[1, 1]]
                ]),
                2,
                dtype='float32'
            ), np.asarray([
                [[0, 0], [0, 0]],
                [[1, 1], [2, 2]],
                [[1, 1], [0, 0]]
            ])
        )


class TestGetStatelessWindowIndicesAndOffset:
    def test_should_return_single_sequence_offset_as_is(self):
        assert get_stateless_window_indices_and_offset(
            sequence_lengths=[5],
            window_stride=10
        ) == [(0, 0)]

    def test_should_return_split_sequence_after_window_stride(self):
        assert get_stateless_window_indices_and_offset(
            sequence_lengths=[15],
            window_stride=10
        ) == [(0, 0), (0, 10)]

    def test_should_return_not_split_sequence_if_window_stride_is_none(self):
        assert get_stateless_window_indices_and_offset(
            sequence_lengths=[15],
            window_stride=None
        ) == [(0, 0)]


class TestGetBatchWindowIndicesAndOffset:
    def test_should_return_single_sequence_offset_as_is(self):
        assert get_batch_window_indices_and_offset(
            sequence_lengths=[5],
            window_stride=10,
            batch_size=1
        ) == [[(0, 0)]]

    def test_should_return_multiple_sequence_offset_across_batches(self):
        assert get_batch_window_indices_and_offset(
            sequence_lengths=[1, 2, 3, 4],
            window_stride=10,
            batch_size=2
        ) == [[(0, 0), (1, 0)], [(2, 0), (3, 0)]]

    def test_should_always_shorter_same_size_batches(self):
        assert get_batch_window_indices_and_offset(
            sequence_lengths=[1, 2, 3],
            window_stride=10,
            batch_size=2
        ) == [[(0, 0), (1, 0)], [(2, 0), (1, 10)]]

    def test_should_slice_single_longer_sequence_across_multiple_batches(self):
        assert get_batch_window_indices_and_offset(
            sequence_lengths=[25],
            window_stride=10,
            batch_size=1
        ) == [[(0, 0)], [(0, 10)], [(0, 20)]]

    def test_should_slice_multiple_longer_and_shorter_sequences_across_multiple_batches(self):
        assert get_batch_window_indices_and_offset(
            sequence_lengths=[1, 25, 3],
            window_stride=10,
            batch_size=2
        ) == [[(0, 0), (1, 0)], [(2, 0), (1, 10)], [(2, 10), (1, 20)]]


class TestDataGenerator:
    def test_should_be_able_to_instantiate(self, preprocessor, embeddings):
        DataGenerator(
            np.asarray([[WORD_1, WORD_2]]),
            np.asarray([[LABEL_1]]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )

    def test_should_be_able_to_get_item(self, preprocessor, embeddings):
        item = DataGenerator(
            np.asarray([SENTENCE_TOKENS_1]),
            np.asarray([[LABEL_1]]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )[0]
        LOGGER.debug('item: %s', item)
        assert len(item) == 2
        x, labels = item
        assert all_close(labels, get_label_indices([LABEL_1]))
        assert all_close(x[0], get_word_vectors(SENTENCE_TOKENS_1))
        assert all_close(x[1], [get_word_indices(SENTENCE_TOKENS_1)])
        assert all_close(x[-1], [len(SENTENCE_TOKENS_1)])

    def test_should_generate_windows_and_disable_shuffle(self, preprocessor, embeddings):
        batches = DataGenerator(
            np.asarray([[WORD_1, WORD_2, WORD_3]]),
            np.asarray([[LABEL_1, LABEL_2, LABEL_3]]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            input_window_stride=2,
            max_sequence_length=2,
            **{
                **DEFAULT_ARGS,
                'shuffle': True
            }
        )
        assert not batches.shuffle

        LOGGER.debug('batches: %s', batches)
        assert len(batches) == 2

        batch_0 = batches[0]
        LOGGER.debug('batch_0: %s', batch_0)
        assert len(batch_0) == 2
        x, labels = batch_0
        LOGGER.debug('labels: %s', labels)
        assert all_close(labels, get_label_indices([LABEL_1, LABEL_2]))
        assert all_close(x[0], get_word_vectors([WORD_1, WORD_2]))
        assert all_close(x[1], [get_word_indices([WORD_1, WORD_2])])
        assert all_close(x[-1], [2])

        batch_1 = batches[1]
        LOGGER.debug('batch_1: %s', batch_1)
        # due to extend, the minimum length is 2
        assert len(batch_1) == 2
        x, labels = batch_1
        LOGGER.debug('labels: %s', labels)
        assert all_close(labels, get_label_indices([LABEL_3]))
        assert all_close(x[0], get_word_vectors([WORD_3, None]))
        assert all_close(x[1], [get_word_indices([WORD_3, None])])
        assert all_close(x[-1], [1])

    def test_should_use_dummy_word_embeddings_if_disabled(self, preprocessor):
        preprocessor.return_casing = False
        item = DataGenerator(
            np.asarray([SENTENCE_TOKENS_1]),
            np.asarray([[LABEL_1]]),
            preprocessor=preprocessor,
            use_word_embeddings=False,
            embeddings=None,
            **DEFAULT_ARGS
        )[0]
        LOGGER.debug('item: %s', item)
        assert len(item) == 2
        x, labels = item
        assert all_close(labels, get_label_indices([LABEL_1]))
        assert all_close(x[0], [np.zeros((len(SENTENCE_TOKENS_1), 0), dtype='float32')])
        assert all_close(x[1], [get_word_indices(SENTENCE_TOKENS_1)])
        assert all_close(x[-1], [len(SENTENCE_TOKENS_1)])

    def test_should_concatenate_word_embeddings_if_using_multiple_tokens(
            self, preprocessor, embeddings):
        preprocessor.return_casing = False
        sentence_tokens_1 = [[WORD_1, WORD_2], [WORD_3, WORD_4]]
        item = DataGenerator(
            np.asarray([sentence_tokens_1]),
            np.asarray([[LABEL_1]]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )[0]
        LOGGER.debug('item: %s', item)
        assert len(item) == 2
        x, labels = item
        assert all_close(labels, get_label_indices([LABEL_1]))
        assert all_close(x[0], np.concatenate(
            (get_word_vectors([WORD_1, WORD_3]), get_word_vectors([WORD_2, WORD_4])),
            axis=-1
        ))
        assert all_close(x[1], [get_word_indices(sentence_tokens_1)])
        assert all_close(x[-1], [len(sentence_tokens_1)])

    def test_should_return_casing(self, preprocessor, embeddings):
        preprocessor.return_casing = True
        item = DataGenerator(
            np.asarray([SENTENCE_TOKENS_1]),
            np.asarray([[LABEL_1]]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )[0]
        LOGGER.debug('item: %s', item)
        assert len(item) == 2
        x, labels = item
        assert all_close(labels, get_label_indices([LABEL_1]))
        assert all_close(x[0], get_word_vectors(SENTENCE_TOKENS_1))
        assert all_close(x[1], [get_word_indices(SENTENCE_TOKENS_1)])
        assert all_close(x[2], [
            to_casing_single(SENTENCE_TOKENS_1, maxlen=len(SENTENCE_TOKENS_1))
        ])
        assert all_close(x[-1], [len(SENTENCE_TOKENS_1)])

    def test_should_return_features(self, preprocessor, embeddings):
        preprocessor.return_features = True
        item = DataGenerator(
            np.asarray([SENTENCE_TOKENS_1]),
            np.asarray([[LABEL_1]]),
            features=np.asarray([SENTENCE_FEATURES_1]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )[0]
        LOGGER.debug('item: %s', item)
        assert len(item) == 2
        x, labels = item
        assert all_close(labels, get_label_indices([LABEL_1]))
        assert all_close(x[0], get_word_vectors(SENTENCE_TOKENS_1))
        assert all_close(x[1], [get_word_indices(SENTENCE_TOKENS_1)])
        assert all_close(x[2], [get_transformed_features(SENTENCE_FEATURES_1)])
        assert all_close(x[-1], [len(SENTENCE_TOKENS_1)])

    def test_should_return_left_pad_batch_values(self, preprocessor, embeddings):
        preprocessor.return_features = True
        batch_size = 2
        batch = DataGenerator(
            np.asarray([
                LONG_SENTENCE_TOKENS,
                SHORT_SENTENCE_TOKENS
            ]),
            np.asarray([
                [LABEL_1] * len(LONG_SENTENCE_TOKENS),
                [LABEL_2] * len(SHORT_SENTENCE_TOKENS)
            ]),
            features=np.asarray([
                np.asarray([WORD_FEATURES_1] * len(LONG_SENTENCE_TOKENS)),
                np.asarray([WORD_FEATURES_2] * len(SHORT_SENTENCE_TOKENS))
            ]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **{
                **DEFAULT_ARGS,
                'batch_size': batch_size
            }
        )[0]
        LOGGER.debug('batch: %s', batch)
        assert len(batch) == 2
        inputs, _ = batch
        batch_x = inputs[0]
        batch_features = inputs[2]
        assert get_lengths(batch_x) == [len(LONG_SENTENCE_TOKENS)] * batch_size
        assert get_lengths(batch_features) == [len(LONG_SENTENCE_TOKENS)] * batch_size

    def test_should_truncate_using_max_sequence_length_if_tokenize(
            self, preprocessor, embeddings):
        preprocessor.return_features = True
        batch_size = 2
        LOGGER.debug('SHORT_SENTENCE_TOKENS: %s', SHORT_SENTENCE_TOKENS)
        LOGGER.debug('LONG_SENTENCE_TOKENS: %s', LONG_SENTENCE_TOKENS)
        batch = DataGenerator(
            np.asarray([
                ' '.join(LONG_SENTENCE_TOKENS),
                ' '.join(SHORT_SENTENCE_TOKENS)
            ]),
            np.asarray([
                [LABEL_1] * len(LONG_SENTENCE_TOKENS),
                [LABEL_2] * len(SHORT_SENTENCE_TOKENS)
            ]),
            features=np.asarray([
                np.asarray([WORD_FEATURES_1] * len(LONG_SENTENCE_TOKENS)),
                np.asarray([WORD_FEATURES_2] * len(SHORT_SENTENCE_TOKENS))
            ]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            max_sequence_length=len(SHORT_SENTENCE_TOKENS),
            **{
                **DEFAULT_ARGS,
                'batch_size': batch_size,
                'tokenize': True
            }
        )[0]
        LOGGER.debug('batch: %s', batch)
        assert len(batch) == 2
        inputs, batch_y = batch
        batch_x = inputs[0]
        batch_features = inputs[2]
        assert get_lengths(batch_x) == [len(SHORT_SENTENCE_TOKENS)] * batch_size
        assert get_lengths(batch_features) == [len(SHORT_SENTENCE_TOKENS)] * batch_size
        assert get_lengths(batch_y) == [len(SHORT_SENTENCE_TOKENS)] * batch_size

    def test_should_truncate_using_max_sequence_length_if_already_tokenized(
            self, preprocessor, embeddings):
        preprocessor.return_features = True
        batch_size = 2
        LOGGER.debug('SHORT_SENTENCE_TOKENS: %s', SHORT_SENTENCE_TOKENS)
        LOGGER.debug('LONG_SENTENCE_TOKENS: %s', LONG_SENTENCE_TOKENS)
        batch = DataGenerator(
            np.asarray([
                LONG_SENTENCE_TOKENS,
                SHORT_SENTENCE_TOKENS
            ]),
            np.asarray([
                [LABEL_1] * len(LONG_SENTENCE_TOKENS),
                [LABEL_2] * len(SHORT_SENTENCE_TOKENS)
            ]),
            features=np.asarray([
                np.asarray([WORD_FEATURES_1] * len(LONG_SENTENCE_TOKENS)),
                np.asarray([WORD_FEATURES_2] * len(SHORT_SENTENCE_TOKENS))
            ]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            max_sequence_length=len(SHORT_SENTENCE_TOKENS),
            **{
                **DEFAULT_ARGS,
                'batch_size': batch_size,
                'tokenize': False
            }
        )[0]
        LOGGER.debug('batch: %s', batch)
        assert len(batch) == 2
        inputs, batch_y = batch
        batch_x = inputs[0]
        batch_features = inputs[2]
        assert get_lengths(batch_x) == [len(SHORT_SENTENCE_TOKENS)] * batch_size
        assert get_lengths(batch_features) == [len(SHORT_SENTENCE_TOKENS)] * batch_size
        assert get_lengths(batch_y) == [len(SHORT_SENTENCE_TOKENS)] * batch_size

    def test_should_not_fail_on_shuffle_dataset(self, preprocessor, embeddings):
        data_generator = DataGenerator(
            np.asarray([SENTENCE_TOKENS_1]),
            np.asarray([[LABEL_1]]),
            features=np.asarray([SENTENCE_FEATURES_1]),
            preprocessor=preprocessor,
            embeddings=embeddings,
            **DEFAULT_ARGS
        )
        data_generator._shuffle_dataset()  # pylint: disable=protected-access
