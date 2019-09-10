import json
from pathlib import Path
from unittest.mock import patch

import pytest

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.debug import (
    TagDebugReporter
)

FILE_1_PREFIX = 'file1'

JSON_FILE_1 = FILE_1_PREFIX + '.json'
DATA_FILE_1 = FILE_1_PREFIX + '.data'

TEXTS_1 = np.array([
    ['token1', 'token2']
])

FEATURES_1 = np.array([
    [['feat1.1', 'feat1.2'], ['feat2.1', 'feat2.2']]
])

ANNOTATIONS_1 = [
    [['token1', 'tag1'], ['token2', 'tag2']]
]

DATA_LINES_1 = [
    'token1 feat1.1 feat1.2 tag1',
    'token2 feat2.1 feat2.2 tag2'
]

FLAT_TEXT_1 = 'token1 token2'

MODEL_1 = 'model1'


@pytest.fixture(name='tag_debug_reporter')
def _tag_debug_reporter(temp_dir: Path):
    return TagDebugReporter(str(temp_dir))


class TestTagDebugReporter:
    def test_should_create_file(
            self,
            temp_dir: Path,
            tag_debug_reporter: TagDebugReporter):
        with patch.object(tag_debug_reporter, 'get_base_output_name') as mock:
            mock.return_value = str(temp_dir.joinpath(FILE_1_PREFIX))
            tag_debug_reporter.report_tag_results(
                texts=TEXTS_1,
                features=FEATURES_1,
                annotations=ANNOTATIONS_1,
                model_name=MODEL_1
            )
            json_output_file = temp_dir.joinpath(JSON_FILE_1)
            output_dict = json.loads(json_output_file.read_text())
            assert output_dict['texts'] == TEXTS_1.tolist()
            assert output_dict['features'] == FEATURES_1.tolist()
            assert output_dict['annotations'] == ANNOTATIONS_1
            assert output_dict['flat_text'] == FLAT_TEXT_1

            data_output_file = temp_dir.joinpath(DATA_FILE_1)
            assert data_output_file.read_text().splitlines() == DATA_LINES_1
