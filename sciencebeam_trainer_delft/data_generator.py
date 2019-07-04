import logging

import numpy as np
import keras
from delft.sequenceLabelling.preprocess import (
    to_vector_single, to_casing_single,
    to_vector_simple_with_elmo, to_vector_simple_with_bert
)
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.preprocess import WordPreprocessor


LOGGER = logging.getLogger(__name__)


# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self, x, y,
            batch_size=24,
            preprocessor: WordPreprocessor = None,
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

    def shuffle_pair(self, a, b):
        # generate permutation index array
        permutation = np.random.permutation(a.shape[0])
        # shuffle the two arrays
        return a[permutation], b[permutation]

    def on_epoch_end(self):
        # shuffle dataset at each epoch
        if self.shuffle:
            if self.y is None:
                np.random.shuffle(self.x)
            else:
                self.shuffle_pair(self.x, self.y)

    def __data_generation(self, index):
        'Generates data containing batch_size samples'
        max_iter = min(self.batch_size, len(self.x) - self.batch_size * index)

        # restrict data to index window
        sub_x = self.x[(index * self.batch_size):(index * self.batch_size) + max_iter]

        # tokenize texts in self.x if not already done
        max_length_x = 0
        if self.tokenize:
            x_tokenized = []
            for i in range(0, max_iter):
                tokens = tokenizeAndFilterSimple(sub_x[i])
                if len(tokens) > max_length_x:
                    max_length_x = len(tokens)
                x_tokenized.append(tokens)
        else:
            for tokens in sub_x:
                if len(tokens) > max_length_x:
                    max_length_x = len(tokens)
            x_tokenized = sub_x

        # prevent sequence of length 1 alone in a batch (this causes an error in tf)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
        if self.preprocessor.return_casing:
            batch_a = np.zeros((max_iter, max_length_x), dtype='float32')

        batch_y = None
        max_length_y = max_length_x
        if self.y is not None:
            # note: tags are always already "tokenized",
            batch_y = np.zeros((max_iter, max_length_y), dtype='float32')

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

            # store tag embeddings
            if self.y is not None:
                batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]

        if self.y is not None:
            batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend)
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0])

        batch_l = batches[1]

        inputs = [batch_x, batch_c]
        if self.preprocessor.return_casing:
            inputs.append(batch_a)
        if self.preprocessor.return_features:
            batch_features = self.features[
                (index * self.batch_size):(index * self.batch_size) + max_iter
            ]
            inputs.append(batch_features)
        inputs.append(batch_l)

        return inputs, batch_y
