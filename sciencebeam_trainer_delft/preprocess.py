import logging
from functools import partial
from typing import List, Iterable, Set

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from delft.sequenceLabelling.preprocess import WordPreprocessor


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


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices: Iterable[int] = None):
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


class Preprocessor(WordPreprocessor):
    def __init__(self, *args, feature_preprocessor: FeaturesPreprocessor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_preprocessor = feature_preprocessor

    def fit_features(self, features_batch):
        return self.feature_preprocessor.fit(features_batch)

    def transform_features(self, features_batch):
        return self.feature_preprocessor.transform(features_batch)
