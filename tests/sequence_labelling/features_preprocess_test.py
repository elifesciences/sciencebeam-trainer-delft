import logging

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.features_preprocess import (
    FeaturesPreprocessor
)

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests
# from
# https://github.com/kermitt2/delft/blob/d2f8390ac01779cab959f57aa6e1a8f1d2723505/
# delft/sequenceLabelling/preprocess.py

LOGGER = logging.getLogger(__name__)

FEATURE_VALUE_1 = 'feature1'
FEATURE_VALUE_2 = 'feature2'
FEATURE_VALUE_3 = 'feature3'
FEATURE_VALUE_4 = 'feature4'


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
        features_transformed, features_length = preprocessor.fit_transform(features_batch)
        assert features_length == 1
        assert all_close(features_transformed, [[[1]]])

    def test_should_fit_single_multiple_value_features(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1], [FEATURE_VALUE_2]]]
        features_transformed, features_length = preprocessor.fit_transform(features_batch)
        assert features_length == 1
        assert len(features_transformed[0]) == 2
        assert np.array_equal(features_transformed, np.asarray([[[1], [2]]]))

    def test_should_fit_multiple_single_value_features(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1, FEATURE_VALUE_2]]]
        features_transformed, features_length = preprocessor.fit_transform(features_batch)
        assert features_length == 2
        assert all_close(features_transformed, [[[1, 1]]])

    def test_should_transform_unseen_to_zero(self):
        preprocessor = FeaturesPreprocessor()
        features_batch = [[[FEATURE_VALUE_1]]]
        preprocessor.fit(features_batch)
        features_transformed, features_length = preprocessor.transform([[[FEATURE_VALUE_2]]])
        assert features_length == 1
        assert all_close(features_transformed, [[[0]]])

    def test_should_select_features(self):
        preprocessor = FeaturesPreprocessor(features_indices=[1])
        features_batch = [[
            [FEATURE_VALUE_1, FEATURE_VALUE_2],
            [FEATURE_VALUE_1, FEATURE_VALUE_3],
            [FEATURE_VALUE_1, FEATURE_VALUE_4]
        ]]
        features_transformed, features_length = preprocessor.fit_transform(features_batch)
        assert features_length == 1
        assert all_close(features_transformed, [[[1], [2], [3]]])
