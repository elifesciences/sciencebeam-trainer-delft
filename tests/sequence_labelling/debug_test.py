import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sciencebeam_trainer_delft.sequence_labelling.debug import (
    TagDebugReporter
)

from .tag_formatter_test import (
    TEXTS_1,
    FEATURES_1,
    ANNOTATIONS_1,
    DATA_LINES_1,
    FLAT_TEXT_1,
    XML_1,
    MODEL_1
)


FILE_1_PREFIX = 'file1'

JSON_FILE_1 = FILE_1_PREFIX + '-1.json'
DATA_FILE_1 = FILE_1_PREFIX + '-1.data'
TXT_FILE_1 = FILE_1_PREFIX + '-1.txt'
XML_FILE_1 = FILE_1_PREFIX + '-1.xml'


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

            data_output_file = temp_dir.joinpath(DATA_FILE_1)
            assert data_output_file.read_text().splitlines() == DATA_LINES_1

            text_output_file = temp_dir.joinpath(TXT_FILE_1)
            assert text_output_file.read_text() == FLAT_TEXT_1

            xml_output_file = temp_dir.joinpath(XML_FILE_1)
            assert xml_output_file.read_text() == XML_1
