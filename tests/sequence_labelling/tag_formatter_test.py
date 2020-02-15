
import json
import logging
import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    get_xml_tag_for_annotation_label,
    format_tag_result
)


LOGGER = logging.getLogger(__name__)


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

XML_1 = '\n'.join([
    '<xml>',
    '  <p>',
    '    <tag1>token1</tag1>',
    '    <tag2>token2</tag2>',
    '  </p>',
    '</xml>'
])

MODEL_1 = 'model1'


class TestGetXmlTagForAnnotationLabel:
    def test_should_remove_greater_less_signs(self):
        assert get_xml_tag_for_annotation_label('<tag>') == 'tag'

    def test_should_remove_prefix(self):
        assert get_xml_tag_for_annotation_label('B-<tag>') == 'tag'


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

    def test_should_format_tag_list_result_as_data(self):
        result = format_tag_result(
            tag_result=ANNOTATIONS_1,
            output_format=TagOutputFormats.DATA,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result.splitlines() == DATA_LINES_1

    def test_should_format_tag_list_result_as_text(self):
        result = format_tag_result(
            tag_result=ANNOTATIONS_1,
            output_format=TagOutputFormats.TEXT,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result == FLAT_TEXT_1

    def test_should_format_tag_list_result_as_xml(self):
        result = format_tag_result(
            tag_result=ANNOTATIONS_1,
            output_format=TagOutputFormats.XML,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result == XML_1

    def test_should_format_tag_list_result_as_xml_and_combined_tags(self):
        result = format_tag_result(
            tag_result=[[['token1', 'tag1'], ['token2', 'tag1'], ['token3', 'tag2']]],
            output_format=TagOutputFormats.XML
        )
        assert result.splitlines() == [
            '<xml>',
            '  <p>',
            '    <tag1>token1 token2</tag1>',
            '    <tag2>token3</tag2>',
            '  </p>',
            '</xml>'
        ]

    def test_should_format_tag_list_result_as_xml_diff_and_combined_tags(self):
        result = format_tag_result(
            tag_result=[[['token1', 'tag1'], ['token2', 'tag1'], ['token3', 'tag2']]],
            expected_tag_result=[[['token1', 'tag1'], ['token2', 'tag1'], ['token3', 'tag3']]],
            output_format=TagOutputFormats.XML_DIFF
        )
        LOGGER.debug('result:\n%s', result)
        assert result.splitlines() == [
            '  <xml>',
            '    <p>',
            '      <tag1>token1 token2</tag1>',
            '-     <tag3>token3</tag3>',
            '?         ^            ^',
            '+     <tag2>token3</tag2>',
            '?         ^            ^',
            '    </p>',
            '  </xml>'
        ]
