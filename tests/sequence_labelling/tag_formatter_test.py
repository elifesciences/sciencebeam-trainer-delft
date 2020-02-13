
import json
import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    format_tag_result
)


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

MODEL_1 = 'model1'


class TestFormatTagResult:
    def test_should_handle_numpy_arrays_in_text(self):
        tag_result = {
            'texts': [{'text': np.array(['token1', 'token2', 'token3'])}]
        }
        result = json.loads(format_tag_result(
            tag_result, output_format=TagOutputFormats.JSON
        ))
        assert result['texts'][0]['text'] == ['token1', 'token2', 'token3']

    def test_should_format_tag_list_result_as_json(self):
        result = json.loads(format_tag_result(
            tag_result=ANNOTATIONS_1,
            output_format=TagOutputFormats.JSON,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        ))
        assert result['model'] == MODEL_1
        assert result['texts'] == TEXTS_1.tolist()
        assert result['features'] == FEATURES_1.tolist()
        assert result['annotations'] == ANNOTATIONS_1
