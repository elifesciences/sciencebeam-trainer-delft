import logging
from functools import partial
from typing import List, Iterable, Set, Union

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from delft.sequenceLabelling.preprocess import (
    FeaturesPreprocessor as DelftFeaturesPreprocessor,
    WordPreprocessor as _WordPreprocessor
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
    pass


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices: Iterable[int] = None):
        self.features_indices = feature_indices
        self.features_map_to_index = None
        self.pipeline = FeaturesPreprocessor._create_pipeline(
            feature_indices=feature_indices
        )

    @staticmethod
    def _create_pipeline(feature_indices: Iterable[int] = None):
        feature_indices_set = None
        if feature_indices:
            feature_indices_set = set(feature_indices)
        to_dict_fn = partial(to_dict, feature_indices=feature_indices_set)
        return Pipeline(steps=[
            ('to_dict', FunctionTransformer(to_dict_fn, validate=False)),
            ('vectorize', DictVectorizer(sparse=False))
        ])

    @property
    def vectorizer(self) -> DictVectorizer:
        return self.pipeline.steps[-1][1]

    def __getstate__(self):
        return {
            'features_indices': self.features_indices,
            'vectorizer.feature_names': self.vectorizer.feature_names_,
            'vectorizer.vocabulary': self.vectorizer.vocabulary_
        }

    def __setstate__(self, state):
        try:
            if 'pipeline' in state:
                # original pickle
                return super().__setstate__(state)
            self.features_indices = state['features_indices']
            self.pipeline = FeaturesPreprocessor._create_pipeline(
                feature_indices=self.features_indices
            )
            self.vectorizer.feature_names_ = state['vectorizer.feature_names']
            self.vectorizer.vocabulary_ = state['vectorizer.vocabulary']
        except KeyError as exc:
            raise KeyError('%r: found %s' % (exc, state.keys())) from exc
        return self

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


T_FeaturesPreprocessor = Union[FeaturesPreprocessor, DelftFeaturesPreprocessor]


class Preprocessor(WordPreprocessor):
    def __init__(self, *args, feature_preprocessor: T_FeaturesPreprocessor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_preprocessor = feature_preprocessor

    def fit_features(self, features_batch):
        return self.feature_preprocessor.fit(features_batch)

    def transform_features(self, features_batch, **kwargs):  # pylint: disable=arguments-differ
        return self.feature_preprocessor.transform(features_batch, **kwargs)
