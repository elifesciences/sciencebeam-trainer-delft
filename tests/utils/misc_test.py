import pytest

from sciencebeam_trainer_delft.utils.misc import (
    parse_comma_separated_str,
    parse_number_ranges,
    parse_dict,
    str_to_bool
)


class TestParseCommaSeparatedStr:
    def test_should_return_empty_array_for_empty_expr(self):
        assert parse_comma_separated_str('') == []

    def test_should_parse_single_value(self):
        assert parse_comma_separated_str('value1') == ['value1']

    def test_should_parse_comma_separated_values(self):
        assert parse_comma_separated_str('value1,value2,value3') == [
            'value1', 'value2', 'value3'
        ]

    def test_should_ignore_spaces(self):
        assert parse_comma_separated_str(' value1 , value2 , value3 ') == [
            'value1', 'value2', 'value3'
        ]


class TestParseNumberRanges:
    def test_should_return_empty_array_for_empty_expr(self):
        assert parse_number_ranges('') == []

    def test_should_parse_single_number(self):
        assert parse_number_ranges('1') == [1]

    def test_should_parse_comma_separated_numbers(self):
        assert parse_number_ranges('1,2,3') == [1, 2, 3]

    def test_should_parse_single_number_range(self):
        assert parse_number_ranges('1-3') == [1, 2, 3]

    def test_should_parse_multiple_number_ranges(self):
        assert parse_number_ranges('1-3,5-6') == [1, 2, 3, 5, 6]

    def test_should_ignore_spaces(self):
        assert parse_number_ranges(' 1 - 3 , 5 - 6 ') == [1, 2, 3, 5, 6]


class TestParseDict:
    def test_should_return_empty_dict_for_empty_expr(self):
        assert parse_dict('') == {}

    def test_should_parse_single_key_value_pair(self):
        assert parse_dict('key1=value1') == {'key1': 'value1'}

    def test_should_allow_equals_sign_in_value(self):
        assert parse_dict('key1=value=1') == {'key1': 'value=1'}

    def test_should_parse_multiple_key_value_pair(self):
        assert parse_dict(
            'key1=value1|key2=value2', delimiter='|'
        ) == {'key1': 'value1', 'key2': 'value2'}

    def test_should_ignore_spaces(self):
        assert parse_dict(
            ' key1 = value1 | key2 = value2 ', delimiter='|'
        ) == {'key1': 'value1', 'key2': 'value2'}


class TestStrToBool:
    def test_should_return_none_if_value_is_empty(self):
        assert str_to_bool('') is None

    @pytest.mark.parametrize("value", ["true", "True", "T", "Yes", "1"])
    def test_should_return_true_for_true_values(self, value: str):
        assert str_to_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "F", "No", "0"])
    def test_should_return_false_for_false_values(self, value: str):
        assert str_to_bool(value) is False
