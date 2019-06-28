import pytest

from sciencebeam_trainer_delft.grobid_trainer import (
    parse_args
)

class TestGrobidTrainer:
    class TestParseArgs:
        def test_should_require_arguments(self):
            with pytest.raises(SystemExit):
                parse_args([])
