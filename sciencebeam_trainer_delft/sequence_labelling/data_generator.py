import logging
from itertools import zip_longest
from typing import Iterable, List, Tuple

import numpy as np
import keras

from delft.utilities.Embeddings import Embeddings
from delft.sequenceLabelling.preprocess import (
    to_vector_single, to_casing_single,
    to_vector_simple_with_elmo, to_vector_simple_with_bert,
    PAD
)
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor


LOGGER = logging.getLogger(__name__)

NBSP = '\u00A0'


def left_pad_batch_values(batch_values: np.array, max_sequence_length: int, dtype=None):
    if dtype is None:
        dtype = batch_values.dtype
    batch_size = len(batch_values)
    value_dimension = 0
    for batch_value in batch_values:
        try:
            value_dimension = len(batch_value[0])
            continue
        except IndexError:
            pass
    result = np.zeros((batch_size, max_sequence_length, value_dimension), dtype=dtype)
    for batch_index in range(batch_size):
        values = batch_values[batch_index]
        if len(values) > max_sequence_length:
            values = values[:max_sequence_length]
        if len(values) > 0:
            result[batch_index, :len(values)] = values
    return result


def truncate_batch_values(batch_values: list, max_sequence_length: int) -> list:
    return [
        row[:max_sequence_length]
        for row in batch_values
    ]


def iter_stateless_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int) -> Iterable[Tuple[int, int]]:
    for sequence_index, sequence_length in enumerate(sequence_lengths):
        sequence_offset = 0
        while sequence_offset < sequence_length:
            yield sequence_index, sequence_offset
            if not window_stride:
                break
            sequence_offset += window_stride


def get_stateless_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int) -> List[Tuple[int, int]]:
    return list(iter_stateless_window_indices_and_offset(
        sequence_lengths, window_stride
    ))


def iter_batch_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int,
        batch_size: int) -> Iterable[List[Tuple[int, int]]]:
    if len(sequence_lengths) < batch_size:
        batch_size = len(sequence_lengths)
    next_sequence_indices = list(range(len(sequence_lengths)))
    batch_sequence_indices = next_sequence_indices[:batch_size]
    next_sequence_indices = next_sequence_indices[batch_size:]
    batch_offsets = [0] * batch_size
    batch_complete = [False] * batch_size
    yield list(zip(batch_sequence_indices, batch_offsets))
    while True:
        for batch_item_index in range(batch_size):
            current_sequence_length = sequence_lengths[batch_sequence_indices[batch_item_index]]
            batch_offsets[batch_item_index] += window_stride
            if batch_offsets[batch_item_index] >= current_sequence_length:
                # we already reached the end of the sequence
                if not next_sequence_indices:
                    batch_complete[batch_item_index] = True
                    continue
                batch_offsets[batch_item_index] = 0
                batch_sequence_indices[batch_item_index] = next_sequence_indices.pop(0)
        if all(batch_complete):
            return
        yield list(zip(batch_sequence_indices, batch_offsets))


def get_batch_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int,
        batch_size: int) -> List[List[Tuple[int, int]]]:
    return list(iter_batch_window_indices_and_offset(
        sequence_lengths=sequence_lengths,
        window_stride=window_stride,
        batch_size=batch_size
    ))


def get_chunk_at_offset(sequence: list, offset: int, max_sequence_length: int = None):
    if not max_sequence_length:
        return sequence[offset:]
    return sequence[offset:min(len(sequence), offset + max_sequence_length)]


def take_with_offset(
        sequences: list,
        indices_and_offset: List[Tuple[int, int]],
        max_sequence_length: int = None) -> list:
    return [
        get_chunk_at_offset(sequences[index], offset, max_sequence_length)
        for index, offset in indices_and_offset
    ]


def to_batch_embedding_vector(
        batch_tokens: List[List[str]],
        embeddings: Embeddings,
        max_length: int = 300,
        **kwargs):
    batch_x = np.zeros((len(batch_tokens), max_length, embeddings.embed_size), dtype='float32')
    for i, tokens in enumerate(batch_tokens):
        batch_x[i] = to_vector_single(tokens, embeddings, max_length, **kwargs)
    return batch_x


def to_dummy_batch_embedding_vector(
        batch_tokens: List[List[str]],
        max_length: int = 300):
    return np.zeros((len(batch_tokens), max_length, 0), dtype='float32')


def get_batch_tokens_with_additional_token_features(
        batch_tokens: List[List[str]],
        batch_features: List[List[List[str]]],
        additional_token_feature_indices: List[int]) -> List[List[List[str]]]:
    return [batch_tokens] + [
        [
            [
                token_features[additional_token_feature_index]
                for token_features in doc_features
            ]
            for doc_features in batch_features
        ]
        for additional_token_feature_index in additional_token_feature_indices
    ]


def get_tokens_from_text_features(
        token_features: List[str],
        text_feature_indices: List[int]) -> List[str]:
    return tokenizeAndFilterSimple(' '.join([
        token_features[text_feature_index]
        for text_feature_index in text_feature_indices
    ]).replace(NBSP, ' '))


def get_batch_tokens_from_text_features(
        batch_features: List[List[List[str]]],
        text_feature_indices: List[int]) -> List[List[List[str]]]:
    LOGGER.debug('text_feature_indices: %s', text_feature_indices)
    LOGGER.debug('batch_features: %s', batch_features)
    word_tokens_by_doc_sequence_token = [
        [
            get_tokens_from_text_features(
                token_features,
                text_feature_indices
            )
            for token_features in doc_features
        ]
        for doc_features in batch_features
    ]
    # for better or worse, the expected output is first indexed by the word token index
    # that is why we are converting it to that form, first finding the maximum number
    # of word tokens
    LOGGER.debug('word_tokens_by_doc_sequence_token: %s', word_tokens_by_doc_sequence_token)
    max_word_token_count = max(
        len(word_tokens)
        for doc_sequence in word_tokens_by_doc_sequence_token
        for word_tokens in doc_sequence
    )
    LOGGER.debug('max_word_token_count: %s', max_word_token_count)
    word_token_by_word_token_index = [
        [
            [
                (
                    word_tokens[word_token_index]
                    if word_token_index < len(word_tokens)
                    else PAD
                )
                for word_tokens in word_tokens_by_token
            ]
            for word_tokens_by_token in word_tokens_by_doc_sequence_token
        ]
        for word_token_index in range(max_word_token_count)
    ]
    return word_token_by_word_token_index


def get_all_batch_tokens(
        batch_tokens: List[List[str]],
        batch_features: List[List[List[str]]],
        additional_token_feature_indices: List[int],
        text_feature_indices: List[int]) -> List[List[List[str]]]:
    if additional_token_feature_indices and text_feature_indices:
        raise ValueError('both, additional token and text features, not supported')
    if additional_token_feature_indices:
        return get_batch_tokens_with_additional_token_features(
            batch_tokens=batch_tokens,
            batch_features=batch_features,
            additional_token_feature_indices=additional_token_feature_indices
        )
    if text_feature_indices:
        return get_batch_tokens_from_text_features(
            batch_features=batch_features,
            text_feature_indices=text_feature_indices
        )
    return [batch_tokens]


def iter_batch_text_from_tokens_with_additional_token_features(
        batch_tokens: Iterable[List[str]],
        batch_features: Iterable[List[List[str]]],
        additional_token_feature_indices: List[int]) -> List[List[List[str]]]:
    if not additional_token_feature_indices:
        return batch_tokens
    return ([
        [
            ' '.join([token] + [
                token_features[additional_token_feature_index]
                for additional_token_feature_index in additional_token_feature_indices
            ])
            for token, token_features in zip(doc_tokens, doc_features)
        ]
        for doc_tokens, doc_features in zip(batch_tokens, batch_features)
    ])


def iter_batch_text_from_text_features(
        batch_features: Iterable[List[List[str]]],
        text_feature_indices: List[int]) -> Iterable[List[str]]:
    LOGGER.debug('text_feature_indices: %s', text_feature_indices)
    LOGGER.debug('batch_features: %s', batch_features)
    return (
        [
            ' '.join(get_tokens_from_text_features(
                token_features,
                text_feature_indices
            ))
            for token_features in doc_features
        ]
        for doc_features in batch_features
    )


def iter_batch_text_list(
        batch_tokens: List[List[str]],
        batch_features: List[List[List[str]]],
        additional_token_feature_indices: List[int],
        text_feature_indices: List[int]) -> Iterable[List[str]]:
    if additional_token_feature_indices and text_feature_indices:
        raise ValueError('both, additional token and text features, not supported')
    if additional_token_feature_indices:
        return iter_batch_text_from_tokens_with_additional_token_features(
            batch_tokens=batch_tokens,
            batch_features=batch_features,
            additional_token_feature_indices=additional_token_feature_indices
        )
    if text_feature_indices:
        return iter_batch_text_from_text_features(
            batch_features=batch_features,
            text_feature_indices=text_feature_indices
        )
    return batch_tokens


def concatenate_all_batch_tokens(
        all_batch_tokens: List[List[List[str]]]) -> List[List[str]]:
    if len(all_batch_tokens) == 1:
        return all_batch_tokens[0]
    return [
        [
            ' '.join(all_tokens)
            for all_tokens in zip(*all_doc_tokens)
        ]
        for all_doc_tokens in zip(*all_batch_tokens)
    ]


def get_embeddings_tokens_for_concatenation(
        all_batch_tokens: List[List[List[str]]],  # token column x doc x sequence x token
        concatenated_embeddings_token_count: int,
        pad_token: str = PAD) -> List[List[List[str]]]:
    default_batch_tokens = [[pad_token] * len(all_batch_tokens[0][0])] * len(all_batch_tokens[0])
    batch_tokens_list = [
        (batch_tokens or default_batch_tokens)
        for batch_tokens, _ in zip_longest(
            all_batch_tokens[:concatenated_embeddings_token_count],
            range(concatenated_embeddings_token_count)
        )
    ]
    LOGGER.debug('batch_tokens_list: %s', batch_tokens_list)
    return batch_tokens_list


def to_concatenated_batch_vector(
        to_batch_vector_fn: callable,
        all_batch_tokens: List[List[List[str]]],  # token column x doc x sequence x token
        *args,
        concatenated_embeddings_token_count: int,
        **kwargs):
    batch_tokens_list = get_embeddings_tokens_for_concatenation(
        all_batch_tokens,
        concatenated_embeddings_token_count=concatenated_embeddings_token_count
    )
    batch_vector_list = [
        to_batch_vector_fn(
            batch_tokens,
            *args,
            **kwargs
        )
        for batch_tokens in batch_tokens_list
    ]
    concatenated_batch_vector = np.concatenate(
        batch_vector_list,
        axis=-1
    )
    return concatenated_batch_vector


def to_batch_casing(
        batch_tokens: List[List[str]],
        max_length: int = 300):
    batch_a = np.zeros((len(batch_tokens), max_length), dtype='float32')
    for i, tokens in enumerate(batch_tokens):
        batch_a[i] = to_casing_single(tokens, max_length)
    return batch_a


def get_concatenated_embeddings_token_count(
        concatenated_embeddings_token_count: int = None,
        additional_token_feature_indices: List[int] = None) -> int:
    return (
        concatenated_embeddings_token_count
        or (1 + len(additional_token_feature_indices or []))
    )


# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self,
            x: List[List[str]],
            y: List[List[str]],
            batch_size: int = 24,
            preprocessor: Preprocessor = None,
            input_window_stride: int = None,
            stateful: bool = True,
            char_embed_size: int = 25,
            use_word_embeddings: bool = None,
            embeddings: Embeddings = None,
            max_sequence_length: int = None,
            tokenize: bool = False,
            shuffle: bool = True,
            features: List[List[List[str]]] = None,
            additional_token_feature_indices: List[int] = None,
            text_feature_indices: List[int] = None,
            concatenated_embeddings_token_count: int = None,
            name: str = None):
        'Initialization'
        if use_word_embeddings is None:
            use_word_embeddings = embeddings is not None
        self.x = x
        self.y = y
        # features here are optional additional features provided
        # in the case of GROBID input for instance
        self.input_window_stride = input_window_stride
        self.features = features
        self.additional_token_feature_indices = additional_token_feature_indices
        self.text_feature_indices = text_feature_indices
        self.concatenated_embeddings_token_count = get_concatenated_embeddings_token_count(
            concatenated_embeddings_token_count=concatenated_embeddings_token_count,
            additional_token_feature_indices=additional_token_feature_indices
        )
        LOGGER.debug(
            'concatenated_embeddings_token_count: %s', self.concatenated_embeddings_token_count
        )
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.use_word_embeddings = use_word_embeddings
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        if preprocessor.return_features and self.features is None:
            raise ValueError('features required')
        if additional_token_feature_indices and features is None:
            raise ValueError('features required for additional token values')
        self.batch_window_indices_and_offset = None
        self.window_indices_and_offset = None
        self.name = name
        if self.shuffle:
            # do we need to shuffle here?, the input was already shuffled
            self._shuffle_dataset()
        if not stateful and self.input_window_stride:
            self.window_indices_and_offset = self.generate_stateless_window_indices_and_offset()
        elif stateful and self.input_window_stride:
            self.batch_window_indices_and_offset = self.generate_batch_window_indices_and_offset()
        else:
            self.log_window_config()
        if self.shuffle and self.input_window_stride and stateful:
            LOGGER.info('not shuffling between epochs as number of batch windows could change')
            self.shuffle = False

    def get_sample_count(self) -> int:
        if self.window_indices_and_offset:
            return len(self.window_indices_and_offset)
        if self.batch_window_indices_and_offset:
            return len(self.batch_window_indices_and_offset) * self.batch_size
        return len(self.x)

    def get_batch_count(self) -> int:
        return int((self.get_sample_count() + self.batch_size - 1) / self.batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        return self.get_batch_count()

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        return self.__data_generation(index)

    def _shuffle_dataset(self):
        if self.window_indices_and_offset:
            np.random.shuffle(self.window_indices_and_offset)
            return
        arrays_to_shuffle = [self.x]
        if self.y is not None:
            arrays_to_shuffle.append(self.y)
        if self.features is not None:
            arrays_to_shuffle.append(self.features)
        shuffle_arrays(arrays_to_shuffle)

    def get_sequence_lengths(self) -> List[int]:
        return [len(item) for item in self.x]

    def log_window_config(self):
        LOGGER.info(
            ' '.join([
                'max sequence length: %s, input window stride: %s'
                ' (%d samples -> %d batches) (name=%s)'
            ]),
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            self.get_batch_count(),
            self.name
        )

    def generate_stateless_window_indices_and_offset(self):
        window_indices_and_offset = get_stateless_window_indices_and_offset(
            sequence_lengths=self.get_sequence_lengths(),
            window_stride=self.input_window_stride
        )
        LOGGER.info(
            'max sequence length: %s, input window stride: %s (%d samples -> %s windows) (name=%s)',
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            len(window_indices_and_offset),
            self.name
        )
        return window_indices_and_offset

    def generate_batch_window_indices_and_offset(self):
        batch_window_indices_and_offset = get_batch_window_indices_and_offset(
            sequence_lengths=self.get_sequence_lengths(),
            window_stride=self.input_window_stride,
            batch_size=self.batch_size
        )
        LOGGER.info(
            ' '.join([
                'max sequence length: %s, input window stride: %s'
                ' (%d samples -> %d batches) (name=%s)'
            ]),
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            len(batch_window_indices_and_offset),
            self.name
        )
        return batch_window_indices_and_offset

    def on_epoch_end(self):
        # shuffle dataset at each epoch
        if self.shuffle:
            self._shuffle_dataset()

    def get_batch_window_indices_and_offsets(self, batch_index: int) -> List[Tuple[int, int]]:
        if self.batch_window_indices_and_offset:
            return self.batch_window_indices_and_offset[batch_index]
        elif self.window_indices_and_offset:
            return self.window_indices_and_offset[
                batch_index * self.batch_size:(batch_index + 1) * self.batch_size
            ]
        else:
            indices = list(range(
                batch_index * self.batch_size,
                min(len(self.x), (batch_index + 1) * self.batch_size)
            ))
            return [(index, 0) for index in indices]

    def __data_generation(self, index):
        return self.get_window_batch_data(
            self.get_batch_window_indices_and_offsets(index)
        )

    def to_batch_embedding_vector(
            self,
            batch_tokens: List[List[str]],
            max_length: int) -> np.array:
        if not self.use_word_embeddings:
            return to_dummy_batch_embedding_vector(batch_tokens, max_length)
        elif self.embeddings.use_ELMo:
            return to_vector_simple_with_elmo(batch_tokens, self.embeddings, max_length)
        elif self.embeddings.use_BERT:
            return to_vector_simple_with_bert(batch_tokens, self.embeddings, max_length)
        else:
            return to_batch_embedding_vector(batch_tokens, self.embeddings, max_length)

    def to_concatenated_batch_vector(
            self,
            all_batch_tokens: List[List[str]],
            max_length: int) -> np.array:
        return to_concatenated_batch_vector(
            self.to_batch_embedding_vector,
            all_batch_tokens,
            max_length,
            concatenated_embeddings_token_count=self.concatenated_embeddings_token_count
        )

    def get_window_batch_data(  # pylint: disable=too-many-statements
            self,
            window_indices_and_offsets: List[Tuple[int, int]]):
        'Generates data containing batch_size samples'

        # restrict data to index window
        # Note: can't apply max_sequence_length here because we may tokenize
        sub_x = take_with_offset(self.x, window_indices_and_offsets)

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

        batch_y = None

        sub_f = None
        if (
                self.preprocessor.return_features
                or self.additional_token_feature_indices
                or self.text_feature_indices
        ):
            sub_f = take_with_offset(
                self.features,
                window_indices_and_offsets,
                max_sequence_length=max_length_x
            )

        all_batch_tokens = get_all_batch_tokens(
            x_tokenized,
            batch_features=sub_f,
            additional_token_feature_indices=self.additional_token_feature_indices,
            text_feature_indices=self.text_feature_indices
        )
        LOGGER.debug('all_batch_tokens: %s', all_batch_tokens)

        batch_x = self.to_concatenated_batch_vector(
            all_batch_tokens,
            max_length_x
        )

        concatenated_batch_tokens = concatenate_all_batch_tokens(all_batch_tokens)

        if self.preprocessor.return_casing:
            batch_a = to_batch_casing(x_tokenized, max_length_x)

        batch_y = None
        # store tag embeddings
        if self.y is not None:
            batch_y = take_with_offset(self.y, window_indices_and_offsets)
            max_length_y = max((len(y_row) for y_row in batch_y))
            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                max_length_y = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                batch_y = truncate_batch_values(batch_y, self.max_sequence_length)

            batches, batch_y = self.preprocessor.transform(
                concatenated_batch_tokens, batch_y, extend=extend
            )
        else:
            batches = self.preprocessor.transform(
                concatenated_batch_tokens, extend=extend
            )

        batch_c = np.asarray(batches[0])

        batch_l = batches[1]

        inputs = []
        inputs.append(batch_x)
        inputs.append(batch_c)
        if self.preprocessor.return_casing:
            inputs.append(batch_a)
        if self.preprocessor.return_features:
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

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug('inputs shapes: %s', [
                np.asarray(x).shape for x in inputs
            ])

        return inputs, batch_y
