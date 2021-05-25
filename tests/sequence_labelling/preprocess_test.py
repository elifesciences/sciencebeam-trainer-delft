import logging

import numpy as np
from sklearn.base import TransformerMixin

from sciencebeam_trainer_delft.utils.typing import T_GetSetStateProtocol
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    WordPreprocessor,
    FeaturesPreprocessor
)


LOGGER = logging.getLogger(__name__)


FEATURE_VALUE_1 = 'feature1'
FEATURE_VALUE_2 = 'feature2'
FEATURE_VALUE_3 = 'feature3'
FEATURE_VALUE_4 = 'feature4'


class TestWordPreprocessor:
    def test_should_be_able_to_instantiate_with_default_values(self):
        WordPreprocessor()

    def test_should_fit_empty_dataset(self):
        preprocessor = WordPreprocessor()
        preprocessor.fit([], [])

    def test_should_fit_single_word_dataset(self):
        preprocessor = WordPreprocessor()
        X = [['Word1']]
        y = [['label1']]
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        LOGGER.debug('vocab_char: %s', preprocessor.vocab_char)
        LOGGER.debug('vocab_case: %s', preprocessor.vocab_case)
        LOGGER.debug('vocab_tag: %s', preprocessor.vocab_tag)
        LOGGER.debug('X_transformed: %s', X_transformed)
        LOGGER.debug('y_transformed: %s', y_transformed)
        for c in 'Word1':
            assert c in preprocessor.vocab_char
        for case in {'numeric', 'allLower', 'allUpper', 'initialUpper'}:
            assert case in preprocessor.vocab_case
        assert 'label1' in preprocessor.vocab_tag
        assert len(X_transformed) == 1
        assert len(y_transformed) == 1

    def test_should_be_able_to_inverse_transform_label(self):
        preprocessor = WordPreprocessor()
        X = [['Word1']]
        y = [['label1']]
        _, y_transformed = preprocessor.fit_transform(X, y)
        y_inverse = preprocessor.inverse_transform(y_transformed[0])
        assert y_inverse == y[0]

    def test_should_transform_unseen_label(self):
        preprocessor = WordPreprocessor(return_lengths=False, padding=False)
        X_train = [['Word1']]
        y_train = [['label1']]
        X_test = [['Word1', 'Word1']]
        y_test = [['label1', 'label2']]
        preprocessor.fit(X_train, y_train)
        _, y_transformed = preprocessor.transform(X_test, y_test)
        assert y_transformed == [[1, 0]]


def _to_dense(a: np.array):
    try:
        return a.todense()
    except AttributeError:
        return a


def all_close(a: np.array, b: np.array):
    return np.allclose(_to_dense(a), _to_dense(b))


def _get_state_and_restore(obj: T_GetSetStateProtocol) -> T_GetSetStateProtocol:
    state = obj.__getstate__()
    LOGGER.debug('state: %s', state)
    obj = type(obj)()
    obj.__setstate__(state)
    new_state = obj.__getstate__()
    LOGGER.debug('new_state: %s', new_state)
    assert new_state == state
    return obj


def _fit_transform_with_state_restore_check(transformer: TransformerMixin, X, **kwargs):
    transformed = transformer.fit_transform(X, **kwargs)
    LOGGER.debug('transformed: %s', transformed)
    LOGGER.debug('transformed.shape: %s', transformed.shape)
    restored_transformer = _get_state_and_restore(transformer)
    restored_transformed = restored_transformer.transform(X)
    LOGGER.debug('restored_transformed: %s', restored_transformed)
    assert restored_transformed.tolist() == transformed.tolist()
    return transformed


class TestFeaturesPreprocessor:
    def test_should_be_able_to_instantiate_with_default_values(self):
        FeaturesPreprocessor()

    def test_should_fit_empty_dataset(self):
        preprocessor = FeaturesPreprocessor()
        preprocessor.fit([])

    def test_should_fit_single_value_feature(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1]]]
        features_transformed = preprocessor.fit_transform(features_batch)
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.shape == (1, 1, 1)
        assert all_close(features_transformed, [[[1]]])

    def test_should_fit_single_multiple_value_features(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1], [FEATURE_VALUE_2]]]
        features_transformed = preprocessor.fit_transform(features_batch)
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.shape == (1, 2, 2)
        assert all_close(features_transformed, [[[1, 0], [0, 1]]])

    def test_should_fit_multiple_single_value_features(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1, FEATURE_VALUE_2]]]
        features_transformed = preprocessor.fit_transform(features_batch)
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.shape == (1, 1, 2)
        assert all_close(features_transformed, [[[1, 1]]])

    def test_should_transform_unseen_to_zero(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1]]]
        preprocessor.fit(features_batch)
        features_transformed = preprocessor.transform([[FEATURE_VALUE_2]])
        assert all_close(features_transformed, [[[0]]])

    def test_should_select_features(self):
        preprocessor = FeaturesPreprocessor(features_indices=[1])
        features_batch = [[
            [FEATURE_VALUE_1, FEATURE_VALUE_2],
            [FEATURE_VALUE_1, FEATURE_VALUE_3],
            [FEATURE_VALUE_1, FEATURE_VALUE_4]
        ]]
        features_transformed = preprocessor.fit_transform(features_batch)
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.shape == (1, 3, 3)
        assert all_close(features_transformed, [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    def test_should_fit_and_scale_single_continuous_value_feature(self):
        preprocessor = FeaturesPreprocessor(
            continuous_features_indices=[0]
        )
        features_batch = [[['0'], ['100']]]
        features_transformed = _fit_transform_with_state_restore_check(
            preprocessor, features_batch
        )
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.tolist() == [[[0.0], [1.0]]]

    def test_should_append_continuous_and_discreet_features(self):
        preprocessor = FeaturesPreprocessor(
            features_indices=[1],
            continuous_features_indices=[0]
        )
        features_batch = [[['0', FEATURE_VALUE_1], ['100', FEATURE_VALUE_2]]]
        features_transformed = _fit_transform_with_state_restore_check(
            preprocessor, features_batch
        )
        LOGGER.debug('features_transformed: %s', features_transformed)
        LOGGER.debug('features_transformed.shape: %s', features_transformed.shape)
        assert features_transformed.tolist() == [[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]
