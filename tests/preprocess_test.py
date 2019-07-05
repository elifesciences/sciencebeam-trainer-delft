import logging

import numpy as np

from sciencebeam_trainer_delft.preprocess import WordPreprocessor, FeaturesPreprocessor


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


def _to_dense(a: np.array):
    try:
        return a.todense()
    except AttributeError:
        return a


def all_close(a: np.array, b: np.array):
    return np.allclose(_to_dense(a), _to_dense(b))


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
        preprocessor = FeaturesPreprocessor(feature_indices=[1])
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
