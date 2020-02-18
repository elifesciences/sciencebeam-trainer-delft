import logging
from functools import partial
from typing import List, Iterable, Set, Union

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from delft.sequenceLabelling.preprocess import PAD
from delft.sequenceLabelling.preprocess import WordPreprocessor as _WordPreprocessor

from sciencebeam_trainer_delft.sequence_labelling.features_preprocess import (
    FeaturesPreprocessor as FeaturesIndicesInputPreprocessor
)


LOGGER = logging.getLogger(__name__)


def to_dict(value_list_batch: List[list], feature_indices: Set[int] = None):
    return [
        {
            index: value
            for index, value in enumerate(value_list)
            if not feature_indices or index in feature_indices
        }
        for value_list in value_list_batch
    ]


class WordPreprocessor(_WordPreprocessor):
    def transform(self, X, y=None, extend=False):
        """
        transforms input into sequence
        the optional boolean `extend` indicates that we need to avoid sequence
        of length 1 alone in a batch
        (which would cause an error with tf)

        Args:
            X: list of list of word tokens
            y: list of list of tags

        Returns:
            numpy array: sentences with char sequences, and optionally length,
            casing and custom features
            numpy array: sequence of tags
        """
        chars = []
        lengths = []
        for sent in X:
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                if self.use_char_feature:
                    char_ids.append(self.get_char_ids(w))
                    if extend:
                        char_ids.append([])

            if self.use_char_feature:
                chars.append(char_ids)

        if y is not None:
            pad_index = self.vocab_tag[PAD]
            LOGGER.debug('vocab_tag: %s', self.vocab_tag)
            y = [[self.vocab_tag.get(t, pad_index) for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(chars, y)
        else:
            sents = [chars]

        # optional additional information
        # lengths
        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices: Iterable[int] = None):
        self.features_indices = feature_indices
        self.features_map_to_index = None
        feature_indices_set = None
        if feature_indices:
            feature_indices_set = set(feature_indices)
        to_dict_fn = partial(to_dict, feature_indices=feature_indices_set)
        self.pipeline = Pipeline(steps=[
            ('to_dict', FunctionTransformer(to_dict_fn, validate=False)),
            ('vectorize', DictVectorizer(sparse=False))
        ])

    def fit(self, X):
        flattened_features = [
            word_features
            for sentence_features in X
            for word_features in sentence_features
        ]
        LOGGER.debug('flattened_features: %s', flattened_features)
        self.pipeline.fit(flattened_features)
        return self

    def transform(self, X):
        LOGGER.debug('transform, X: %s', X)
        return np.asarray([
            self.pipeline.transform(sentence_features)
            for sentence_features in X
        ])


T_FeaturesPreprocessor = Union[FeaturesPreprocessor, FeaturesIndicesInputPreprocessor]


class Preprocessor(WordPreprocessor):
    def __init__(self, *args, feature_preprocessor: T_FeaturesPreprocessor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_preprocessor = feature_preprocessor

    def fit_features(self, features_batch):
        return self.feature_preprocessor.fit(features_batch)

    def transform_features(self, features_batch, **kwargs):
        return self.feature_preprocessor.transform(features_batch, **kwargs)
