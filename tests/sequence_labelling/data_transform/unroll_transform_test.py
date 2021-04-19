from sciencebeam_trainer_delft.sequence_labelling.dataset_transform.unroll_transform import (
    UnrollingTextFeatureDatasetTransformer
)


class TestUnrollingTextFeatureDatasetTransformer:
    def test_should_unroll_and_inverse_transform_x_y_and_features(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(2)
        x = [['token1', 'token2']]
        y = [['label1', 'label2']]
        features = [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22']
        ]]
        transformed_x, transformed_y, transformed_features = data_transformer.fit_transform(
            x, y, features
        )
        assert transformed_x == [['word11', 'word12', 'word21', 'word22']]
        assert transformed_y == [['label1', 'label1', 'label2', 'label2']]
        assert transformed_features == [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22'],
            [0, 1, 'word21 word22']
        ]]
        inverse_transformed_x, inverse_transformed_y, inverse_transformed_features = (
            data_transformer.inverse_transform(transformed_x, transformed_y, transformed_features)
        )
        assert inverse_transformed_x == x
        assert inverse_transformed_y == y
        assert inverse_transformed_features == features

    def test_should_unroll_x_and_features_and_inverse_transform_y(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(2)
        x = [['token1', 'token2']]
        y = [['label1', 'label2']]
        features = [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22']
        ]]
        transformed_x, transformed_features = data_transformer.fit_transform_x_and_features(
            x, features
        )
        assert transformed_x == [['word11', 'word12', 'word21', 'word22']]
        assert transformed_features == [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22'],
            [0, 1, 'word21 word22']
        ]]
        inverse_transformed_y = data_transformer.inverse_transform_y(
            [['label1', 'label1', 'label2', 'label2']]
        )
        assert inverse_transformed_y == y

    def test_should_unroll_x_and_features_and_inverse_transform_truncated_y(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(2)
        x = [['token1', 'token2']]
        features = [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22']
        ]]
        data_transformer.fit_transform_x_and_features(
            x, features
        )
        inverse_transformed_y = data_transformer.inverse_transform_y(
            [['label1', 'label1']]
        )
        assert inverse_transformed_y == [['label1']]

    def test_should_unroll_x_y_and_features_and_inverse_transform_y_with_ib_prefix(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(2)
        x = [['token1', 'token2']]
        y = [['B-label1', 'B-label2']]
        features = [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22']
        ]]
        transformed_x, transformed_y, transformed_features = data_transformer.fit_transform(
            x, y, features
        )
        assert transformed_x == [['word11', 'word12', 'word21', 'word22']]
        assert transformed_y == [['B-label1', 'I-label1', 'B-label2', 'I-label2']]
        assert transformed_features == [[
            [0, 1, 'word11 word12'],
            [0, 1, 'word11 word12'],
            [0, 1, 'word21 word22'],
            [0, 1, 'word21 word22']
        ]]
        inverse_transformed_y = data_transformer.inverse_transform_y(transformed_y)
        assert inverse_transformed_y == y

    def test_should_select_major_class_name(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(2)
        x = [['token1', 'token2']]
        features = [[
            [0, 1, 'word11 word12 word13'],
            [0, 1, 'word21 word22 word23']
        ]]
        data_transformer.fit_transform_x_and_features(
            x, features
        )
        inverse_transformed_y = data_transformer.inverse_transform_y(
            [['O', 'B-label1', 'I-label1', 'B-label2', 'I-label2', 'I-label2']]
        )
        assert inverse_transformed_y == [['B-label1', 'B-label2']]
