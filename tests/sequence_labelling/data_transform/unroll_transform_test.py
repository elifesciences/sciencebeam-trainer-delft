import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform.unroll_transform import (
    LineStatus,
    get_line_status,
    UnrollingTextFeatureDatasetTransformer
)


class TestGetLineStatus:
    def test_should_return_linestart_if_first_token(self):
        assert get_line_status(0, 10) == LineStatus.LINESTART

    def test_should_return_lineend_if_last_token(self):
        assert get_line_status(9, 10) == LineStatus.LINEEND

    def test_should_return_linein_if_not_first_or_last_token(self):
        token_indices = list(range(10))
        expected_line_status = (
            [LineStatus.LINESTART] + [LineStatus.LINEIN] * 8 + [LineStatus.LINEEND]
        )
        actual_line_status = [get_line_status(i, 10) for i in token_indices]
        assert actual_line_status == expected_line_status


class TestUnrollingTextFeatureDatasetTransformer:
    def test_should_unroll_and_inverse_transform_x_y_and_features_using_list(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(
            2,
            used_features_indices=[3]
        )
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
            [0, 1, 'word11 word12', LineStatus.LINESTART],
            [0, 1, 'word11 word12', LineStatus.LINEEND],
            [0, 1, 'word21 word22', LineStatus.LINESTART],
            [0, 1, 'word21 word22', LineStatus.LINEEND]
        ]]
        inverse_transformed_x, inverse_transformed_y, inverse_transformed_features = (
            data_transformer.inverse_transform(transformed_x, transformed_y, transformed_features)
        )
        assert inverse_transformed_x == x
        assert inverse_transformed_y == y
        assert inverse_transformed_features == features

    def test_should_unroll_and_inverse_transform_x_y_and_features_using_ndarray(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(
            2,
            used_features_indices=[1, 2, 3]
        )
        x = np.asarray([['token1', 'token2']], dtype='object')
        y = np.asarray([['label1', 'label2']], dtype='object')
        features = np.asarray([[
            ['0', '1', 'word11 word12'],
            ['0', '1', 'word21 word22']
        ]], dtype='object')
        transformed_x, transformed_y, transformed_features = data_transformer.fit_transform(
            x, y, features
        )
        assert transformed_x.tolist() == [['word11', 'word12', 'word21', 'word22']]
        assert transformed_y.tolist() == [['label1', 'label1', 'label2', 'label2']]
        assert transformed_features.tolist() == [[
            ['0', '1', 'word11 word12', LineStatus.LINESTART],
            ['0', '1', 'word11 word12', LineStatus.LINEEND],
            ['0', '1', 'word21 word22', LineStatus.LINESTART],
            ['0', '1', 'word21 word22', LineStatus.LINEEND]
        ]]
        inverse_transformed_x, inverse_transformed_y, inverse_transformed_features = (
            data_transformer.inverse_transform(transformed_x, transformed_y, transformed_features)
        )
        assert inverse_transformed_x.tolist() == x.tolist()
        assert inverse_transformed_y.tolist() == y.tolist()
        assert inverse_transformed_features.tolist() == features.tolist()

    def test_should_unroll_and_inv_transform_x_y_and_features_using_ndarray_no_line_status(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(
            2,
            used_features_indices=[1, 2]
        )
        x = np.asarray([['token1', 'token2']], dtype='object')
        y = np.asarray([['label1', 'label2']], dtype='object')
        features = np.asarray([[
            ['0', '1', 'word11 word12'],
            ['0', '1', 'word21 word22']
        ]], dtype='object')
        transformed_x, transformed_y, transformed_features = data_transformer.fit_transform(
            x, y, features
        )
        assert transformed_x.tolist() == [['word11', 'word12', 'word21', 'word22']]
        assert transformed_y.tolist() == [['label1', 'label1', 'label2', 'label2']]
        assert transformed_features.tolist() == [[
            ['0', '1', 'word11 word12'],
            ['0', '1', 'word11 word12'],
            ['0', '1', 'word21 word22'],
            ['0', '1', 'word21 word22']
        ]]
        inverse_transformed_x, inverse_transformed_y, inverse_transformed_features = (
            data_transformer.inverse_transform(transformed_x, transformed_y, transformed_features)
        )
        assert inverse_transformed_x.tolist() == x.tolist()
        assert inverse_transformed_y.tolist() == y.tolist()
        assert inverse_transformed_features.tolist() == features.tolist()

    def test_should_unroll_x_and_features_and_inverse_transform_y(self):
        data_transformer = UnrollingTextFeatureDatasetTransformer(
            2,
            used_features_indices=[3]
        )
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
            [0, 1, 'word11 word12', LineStatus.LINESTART],
            [0, 1, 'word11 word12', LineStatus.LINEEND],
            [0, 1, 'word21 word22', LineStatus.LINESTART],
            [0, 1, 'word21 word22', LineStatus.LINEEND]
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
        data_transformer = UnrollingTextFeatureDatasetTransformer(
            2,
            used_features_indices=[3]
        )
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
            [0, 1, 'word11 word12', LineStatus.LINESTART],
            [0, 1, 'word11 word12', LineStatus.LINEEND],
            [0, 1, 'word21 word22', LineStatus.LINESTART],
            [0, 1, 'word21 word22', LineStatus.LINEEND]
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

    def test_should_select_beginning_class_name_only_if_tag_changes(self):
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
            [['B-label1', 'I-label1', 'O', 'B-label1', 'O', 'B-label1']]
        )
        assert inverse_transformed_y == [['B-label1', 'I-label1']]
