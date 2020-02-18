import logging

import numpy as np
import keras
from delft.sequenceLabelling.preprocess import (
    to_vector_single, to_casing_single,
    to_vector_simple_with_elmo, to_vector_simple_with_bert
)
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor


LOGGER = logging.getLogger(__name__)


def left_pad_batch_values(batch_values: np.array, max_sequence_length: int, dtype=None):
    if dtype is None:
        dtype = batch_values.dtype
    batch_size = len(batch_values)
    value_dimension = len(batch_values[0][0])
    result = np.zeros((batch_size, max_sequence_length, value_dimension), dtype=dtype)
    for batch_index in range(batch_size):
        values = batch_values[batch_index]
        if len(values) > max_sequence_length:
            values = values[:max_sequence_length]
        result[batch_index, :len(values)] = values
    return result


def truncate_batch_values(batch_values: list, max_sequence_length: int) -> list:
    return [
        row[:max_sequence_length]
        for row in batch_values
    ]


# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self, x, y,
            batch_size=24,
            preprocessor: Preprocessor = None,
            char_embed_size=25,
            embeddings=None,
            max_sequence_length=None,
            tokenize=False,
            shuffle=True,
            features=None):
        'Initialization'
        self.x = x
        self.y = y
        # features here are optional additional features provided
        # in the case of GROBID input for instance
        self.features = features
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        if preprocessor.return_features and self.features is None:
            raise ValueError('features required')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        if (len(self.x) % self.batch_size) == 0:
            return int(np.floor(len(self.x) / self.batch_size))
        else:
            return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        return self.__data_generation(index)

    def _shuffle_dataset(self):
        arrays_to_shuffle = [self.x]
        if self.y is not None:
            arrays_to_shuffle.append(self.y)
        if self.features is not None:
            arrays_to_shuffle.append(self.features)
        shuffle_arrays(arrays_to_shuffle)

    def on_epoch_end(self):
        # shuffle dataset at each epoch
        if self.shuffle:
            self._shuffle_dataset()

    def __data_generation(self, index):  # pylint: disable=too-many-statements
        'Generates data containing batch_size samples'
        max_iter = min(self.batch_size, len(self.x) - self.batch_size * index)

        # restrict data to index window
        sub_x = self.x[(index * self.batch_size):(index * self.batch_size) + max_iter]

        # tokenize texts in self.x if not already done
        if self.tokenize:
            x_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in sub_x
            ]
        else:
            x_tokenized = sub_x

        max_length_x = max((len(tokens) for tokens in x_tokenized))

        if self.max_sequence_length and max_length_x > self.max_sequence_length:
            max_length_x = self.max_sequence_length
            # truncation of sequence at max_sequence_length
            x_tokenized = truncate_batch_values(x_tokenized, self.max_sequence_length)

        # prevent sequence of length 1 alone in a batch (this causes an error in tf)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
        if self.preprocessor.return_casing:
            batch_a = np.zeros((max_iter, max_length_x), dtype='float32')

        batch_y = None

        if self.embeddings.use_ELMo:
            # batch_x = to_vector_elmo(x_tokenized, self.embeddings, max_length_x)
            batch_x = to_vector_simple_with_elmo(x_tokenized, self.embeddings, max_length_x)
        elif self.embeddings.use_BERT:
            # batch_x = to_vector_bert(x_tokenized, self.embeddings, max_length_x)
            batch_x = to_vector_simple_with_bert(x_tokenized, self.embeddings, max_length_x)

        # generate data
        for i in range(0, max_iter):
            # store sample embeddings
            if not self.embeddings.use_ELMo and not self.embeddings.use_BERT:
                batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x)

            if self.preprocessor.return_casing:
                batch_a[i] = to_casing_single(x_tokenized[i], max_length_x)

        batch_y = None
        # store tag embeddings
        if self.y is not None:
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            max_length_y = max((len(y_row) for y_row in batch_y))
            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                max_length_y = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                batch_y = truncate_batch_values(batch_y, self.max_sequence_length)

            batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend)
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0])

        batch_l = batches[1]

        inputs = [batch_x, batch_c]
        if self.preprocessor.return_casing:
            inputs.append(batch_a)
        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            LOGGER.debug('extend: %s', extend)
            try:
                batch_features, _ = self.preprocessor.transform_features(sub_f, extend=extend)
                batch_features = left_pad_batch_values(batch_features, max_length_x)
            except TypeError:
                batch_features = left_pad_batch_values(
                    self.preprocessor.transform_features(sub_f),
                    max_length_x
                )
            LOGGER.debug('batch_features.shape: %s', batch_features.shape)
            inputs.append(batch_features)
        inputs.append(batch_l)

        return inputs, batch_y
