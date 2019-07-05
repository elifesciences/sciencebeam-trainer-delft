from sciencebeam_trainer_delft.utils import is_external_location, parse_number_ranges


class TestIsExternalLocation:
    def test_should_return_false_for_name(self):
        assert not is_external_location('name')

    def test_should_return_true_for_url(self):
        assert is_external_location('http://name')


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
