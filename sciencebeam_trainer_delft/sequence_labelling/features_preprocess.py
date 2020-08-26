import logging
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig


LOGGER = logging.getLogger(__name__)

# this is derived from https://github.com/Hironsan/anago/blob/master/anago/preprocess.py
# and then from:
# https://github.com/kermitt2/delft/blob/d2f8390ac01779cab959f57aa6e1a8f1d2723505/
# delft/sequenceLabelling/preprocess.py

UNK = '<UNK>'
PAD = '<PAD>'


def calculate_cardinality(feature_vector, indices=None):
    """
    Calculate cardinality of each features

    :param feature_vector: three dimensional vector with features
    :param indices: list of indices of the features to be extracted
    :return: a map where each key is the index of the feature and the value is a map feature_value,
    value_index.
    For example
     [(0, {'feature1': 1, 'feature2': 2})]

     indicates that the feature is at index 0 and has two values, features1 and features2 with two
     unique index.

     NOTE: the features are indexed from 1 to n + 1. The 0 value is reserved as padding
    """
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    LOGGER.debug('feature_vector[0][0] (%d): %s', len(feature_vector[0][0]), feature_vector[0][0])
    for index_column in range(index, len(feature_vector[0][0])):
        if indices and index_column not in indices:
            index += 1
            continue

        values = set()
        for rows in feature_vector:
            for row in rows:
                try:
                    value = row[index_column]
                except IndexError:
                    raise IndexError('out of index, index=%d, len=%d, row=%s' % (
                        index_column, len(row), row
                    ))
                if value != " ":
                    values.add(value)

        values = sorted(values)
        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            # We reserve the 0 for the unseen features so the indexes
            # will go from 1 to cardinality + 1
            values_to_int[values_list[val_num]] = val_num + 1

        columns_length.append((index, values_to_int))
        index += 1

    return columns_length


def cardinality_to_index_map(columns_length, features_max_vector_size):
    # Filter out the columns that are not fitting
    columns_index = []
    for index, column_content_cardinality in columns_length:
        if len(column_content_cardinality) <= features_max_vector_size:
            columns_index.append((index, column_content_cardinality))
    # print(columns_index)
    index_list = [ind[0] for ind in columns_index if ind[0] >= 0]
    val_to_int_list = {value[0]: value[1] for value in columns_index}

    return index_list, val_to_int_list


def reduce_features_to_indexes(feature_vector, features_max_vector_size, indices=None):
    cardinality = calculate_cardinality(feature_vector, indices=indices)
    index_list, map_to_integers = cardinality_to_index_map(cardinality, features_max_vector_size)

    return index_list, map_to_integers


def reduce_features_vector(feature_vector, features_max_vector_size):
    '''
    Reduce the features vector.
    First it calculates cardinalities for each value that each feature can assume, then
    removes features with cardinality above features_max_vector_size.

    :param feature_vector: feature vector to be reduced
    :param features_max_vector_size maximum size of the one-hot-encoded values
    :return:
    '''

    # Compute frequencies for each column
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    for index_column in range(index, len(feature_vector[0])):
        values = set()
        for rows in feature_vector:
            value = rows[index_column]
            if value != " ":
                values.add(value)

        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            values_to_int[values_list[val_num]] = val_num

        columns_length.append((index, values_to_int))
        index += 1
        # print("Column: " + str(index_column) + " Len:  " + str(len(values)))

    # Filter out the columns that are not fitting
    columns_index = []
    for index, column_content_cardinality in columns_length:
        if len(column_content_cardinality) <= features_max_vector_size:
            columns_index.append((index, column_content_cardinality))
    # print(columns_index)
    index_list = [ind[0] for ind in columns_index if ind[0]]

    # Assign indexes to each feature value
    reduced_features_vector = []
    for index_row in range(0, len(feature_vector)):
        reduced_features_vector.append(
            [
                feature_vector[index_row][index_column]
                for index, index_column in enumerate(index_list)
            ]
        )

    return reduced_features_vector


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok=0, nlevels=1, max_char_length=30):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max_char_length
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(len, sequences))
        sequence_padded, _ = _pad_sequences(
            sequence_padded, [pad_tok] * max_length_word, max_length_sentence
        )
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length


def _convert_keys(d: dict, convert_fn: callable) -> dict:
    return {
        convert_fn(key): value
        for key, value in d.items()
    }


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_indices: Iterable[int] = None,
                 features_vocabulary_size=ModelConfig.DEFAULT_FEATURES_VOCABULARY_SIZE,
                 features_map_to_index=None):
        # feature_indices_set = None
        if features_map_to_index is None:
            features_map_to_index = []
        self.features_vocabulary_size = features_vocabulary_size
        self.features_indices = features_indices

        # List of mappings to integers (corresponding to each feature column) of features values
        # This value could be provided (model has been loaded) or not (first-time-training)
        self.features_map_to_index = features_map_to_index

    def __getstate__(self):
        return {
            'features_vocabulary_size': self.features_vocabulary_size,
            'features_indices': self.features_indices,
            'features_map_to_index': _convert_keys(self.features_map_to_index, str)
        }

    def __setstate__(self, state):
        self.features_vocabulary_size = state['features_vocabulary_size']
        self.features_indices = state['features_indices']
        self.features_map_to_index = _convert_keys(state['features_map_to_index'], int)
        return self

    def fit(self, X):
        if not self.features_indices:
            indexes, mapping = reduce_features_to_indexes(X, self.features_vocabulary_size)
        else:
            indexes, mapping = reduce_features_to_indexes(X, self.features_vocabulary_size,
                                                          indices=self.features_indices)

        self.features_map_to_index = mapping
        self.features_indices = indexes
        return self

    def transform(self, X, extend=False):
        """
        Transform the features into a vector, return the vector and the extracted number of features

        :param extend: when set to true it's adding an additional
           empty feature list in the sequence.
        """
        features_vector = [
            [
                [
                    self.features_map_to_index[index][value]
                    if (
                        index in self.features_map_to_index
                        and value in self.features_map_to_index[index]
                    )
                    else 0
                    for index, value in enumerate(value_list) if index in self.features_indices
                ] for value_list in document
            ] for document in X
        ]

        features_count = len(self.features_indices)

        if extend:
            for out in features_vector:
                out.append([0] * features_count)

        features_vector_padded, _ = pad_sequences(features_vector, [0] * features_count)
        output = np.asarray(features_vector_padded)

        return output, features_count
