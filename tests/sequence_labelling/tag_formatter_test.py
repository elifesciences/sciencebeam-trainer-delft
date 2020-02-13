
import json
import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    format_tag_result
)


class TestFormatTagResult:
    def test_should_handle_numpy_arrays_in_text(self):
        tag_result = {
            'texts': [{'text': np.array(['token1', 'token2', 'token3'])}]
        }
        result = json.loads(format_tag_result(
            tag_result, output_format=TagOutputFormats.JSON
        ))
        assert result['texts'][0]['text'] == ['token1', 'token2', 'token3']
