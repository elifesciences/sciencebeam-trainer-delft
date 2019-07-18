from sciencebeam_trainer_delft.utils.io import is_external_location


class TestIsExternalLocation:
    def test_should_return_false_for_name(self):
        assert not is_external_location('name')

    def test_should_return_true_for_url(self):
        assert is_external_location('http://name')
