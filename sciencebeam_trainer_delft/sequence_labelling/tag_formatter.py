import json
import difflib
import logging
from xml.sax.saxutils import escape as xml_escape
from typing import Union, Iterable, List, Tuple

import numpy as np

from delft.sequenceLabelling.evaluation import get_entities


LOGGER = logging.getLogger(__name__)


class TagOutputFormats:
    JSON = 'json'
    DATA = 'data'
    DATA_DIFF = 'data_diff'
    TEXT = 'text'
    XML = 'xml'
    XML_DIFF = 'xml_diff'


TAG_OUTPUT_FORMATS = [
    TagOutputFormats.JSON,
    TagOutputFormats.DATA,
    TagOutputFormats.DATA_DIFF,
    TagOutputFormats.TEXT,
    TagOutputFormats.XML,
    TagOutputFormats.XML_DIFF,
]


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_tag_result(texts: List[List[str]], labels: List[List[str]]):
    return [
        list(zip(doc_texts, doc_labels))
        for doc_texts, doc_labels in zip(texts, labels)
    ]


def format_json_tag_result_as_json(tag_result: dict) -> str:
    return json.dumps(tag_result, indent=2, cls=CustomJsonEncoder)


def format_list_tag_result_as_json(
        tag_result: List[List[Tuple[str, str]]],
        texts: np.array = None,
        features: np.array = None,
        model_name: str = None) -> str:
    output_props = {
        'model': model_name,
        'texts': np.array(texts).tolist(),
        'features': np.array(features).tolist() if features is not None else None,
        'annotations': tag_result
    }
    return json.dumps(output_props, indent=2, cls=CustomJsonEncoder)


def to_data_lines(
        features: np.array,
        annotations: List[List[Tuple[str, str]]]) -> List[str]:
    return [
        ' '.join([token_annoation[0]] + list(token_features) + [token_annoation[1]])
        for line_annotations, line_features in zip(annotations, features.tolist())
        for token_annoation, token_features in zip(line_annotations, line_features)
    ]


def format_list_tag_result_as_data(
        tag_result: List[List[Tuple[str, str]]],
        texts: np.array = None,  # pylint: disable=unused-argument
        features: np.array = None,
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    assert features is not None
    return '\n'.join(to_data_lines(
        features=features,
        annotations=tag_result
    ))


def simple_diff(a, b, fromfile='', tofile='', lineterm='\n'):
    assert len(a) == len(b)
    if fromfile:
        yield f'--- {fromfile}{lineterm}'
    if tofile:
        yield f'+++ {tofile}{lineterm}'
    line_count = len(a)
    is_diff_list = [
        value_1 != value_2
        for value_1, value_2 in zip(a, b)
    ]
    LOGGER.debug('is_diff_list: %s', is_diff_list)
    diff_count = sum(is_diff_list)
    removed_with_prefix = f'-{diff_count}' if diff_count else '-0'
    added_with_prefix = f'+{diff_count}' if diff_count else '+0'
    yield f'@@ {removed_with_prefix},{line_count} {added_with_prefix},{line_count} @@{lineterm}'
    for is_diff, value_1, value_2 in zip(is_diff_list, a, b):
        if is_diff:
            yield f'-{value_1}'
            yield f'+{value_2}'
        else:
            yield f' {value_1}'


def split_lines_with_line_feed(text: str, line_feed: str = '\n') -> List[str]:
    # Note: similar to .splitlines(keepends=True), but always adds the line feed
    return [
        line + line_feed
        for line in text.splitlines()
    ]


def format_list_tag_result_as_data_diff(
        tag_result: List[List[Tuple[str, str]]],
        expected_tag_result: List[Tuple[str, str]] = None,
        texts: np.array = None,  # pylint: disable=unused-argument
        features: np.array = None,
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    assert expected_tag_result
    actual_data = format_list_tag_result_as_data(
        tag_result, texts=texts, features=features
    )
    expected_data = format_list_tag_result_as_data(
        expected_tag_result, texts=texts, features=features
    )
    LOGGER.debug('actual_data: %r', actual_data)
    LOGGER.debug('expected_data: %r', expected_data)
    return ''.join(simple_diff(
        split_lines_with_line_feed(expected_data),
        split_lines_with_line_feed(actual_data),
        fromfile='expected.data',
        tofile='actual.data'
    ))


def to_flat_text(texts: np.array) -> str:
    return '\n'.join([
        ' '.join(line_tokens)
        for line_tokens in texts
    ])


def format_list_tag_result_as_text(
        tag_result: List[List[Tuple[str, str]]],  # pylint: disable=unused-argument
        texts: np.array = None,
        features: np.array = None,  # pylint: disable=unused-argument
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    assert texts is not None
    return to_flat_text(texts=texts)


def get_xml_tag_for_annotation_label(annotation_label: str) -> str:
    return annotation_label.replace('<', '').replace('>', '').split('-', maxsplit=1)[-1]


def iter_add_untagged_token_spans(
        entity_chunks: List[Tuple[str, int, int]],
        token_count: int,
        untagged_chunk_type: str = None) -> List[Tuple[str, int, int]]:
    prev_chunk_end_excl = 0
    for chunk_type, chunk_start, chunk_end in entity_chunks:
        if chunk_start > prev_chunk_end_excl:
            yield untagged_chunk_type, prev_chunk_end_excl, (chunk_start - 1)
        yield chunk_type, chunk_start, chunk_end
        prev_chunk_end_excl = chunk_end + 1
    if token_count > prev_chunk_end_excl:
        yield untagged_chunk_type, prev_chunk_end_excl, (token_count - 1)


def iter_doc_annotations_xml_text(
        doc_annotations: List[Tuple[str, str]]) -> Iterable[str]:
    LOGGER.debug('doc_annotations: %s', doc_annotations)
    text_tokens = [token_text for token_text, _ in doc_annotations]
    token_labels = [token_label for _, token_label in doc_annotations]
    entity_chunks = list(iter_add_untagged_token_spans(
        get_entities(token_labels),
        len(token_labels)
    ))
    LOGGER.debug('text_tokens: %s', text_tokens)
    LOGGER.debug('token_labels: %s', token_labels)
    LOGGER.debug('entity_chunks: %s', entity_chunks)
    return '\n'.join((
        (
            '    <{tag}>{text}</{tag}>'.format(
                tag=get_xml_tag_for_annotation_label(chunk_type),
                text=xml_escape(' '.join(text_tokens[chunk_start:chunk_end + 1]))
            )
            if chunk_type
            else
            '    {text}'.format(
                text=xml_escape(' '.join(text_tokens[chunk_start:chunk_end + 1]))
            )
        )
        for chunk_type, chunk_start, chunk_end in entity_chunks
    )) + '\n'


def iter_annotations_xml_text(
        annotations: List[List[Tuple[str, str]]]) -> Iterable[str]:
    for doc_index, doc_annotations in enumerate(annotations):
        if doc_index > 0:
            yield '\n\n'
        yield '  <p>\n'
        yield from iter_doc_annotations_xml_text(doc_annotations)
        yield '  </p>\n'


def format_list_tag_result_as_xml(
        tag_result: List[List[Tuple[str, str]]],
        texts: np.array = None,  # pylint: disable=unused-argument
        features: np.array = None,  # pylint: disable=unused-argument
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    return '<xml>\n%s</xml>' % ''.join(iter_annotations_xml_text(
        annotations=tag_result
    ))


def format_list_tag_result_as_xml_diff(
        tag_result: List[List[Tuple[str, str]]],
        expected_tag_result: List[Tuple[str, str]] = None,
        texts: np.array = None,  # pylint: disable=unused-argument
        features: np.array = None,  # pylint: disable=unused-argument
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    assert expected_tag_result
    actual_xml = format_list_tag_result_as_xml(tag_result)
    expected_xml = format_list_tag_result_as_xml(expected_tag_result)
    return ''.join(difflib.ndiff(
        expected_xml.splitlines(keepends=True),
        actual_xml.splitlines(keepends=True)
    ))


def format_list_tag_result(
        *args,
        output_format: str,
        expected_tag_result: List[Tuple[str, str]] = None,
        **kwargs) -> str:
    if output_format == TagOutputFormats.JSON:
        return format_list_tag_result_as_json(*args, **kwargs)
    if output_format == TagOutputFormats.DATA:
        return format_list_tag_result_as_data(*args, **kwargs)
    if output_format == TagOutputFormats.DATA_DIFF:
        return format_list_tag_result_as_data_diff(
            *args,
            expected_tag_result=expected_tag_result,
            **kwargs
        )
    if output_format == TagOutputFormats.TEXT:
        return format_list_tag_result_as_text(*args, **kwargs)
    if output_format == TagOutputFormats.XML:
        return format_list_tag_result_as_xml(*args, **kwargs)
    if output_format == TagOutputFormats.XML_DIFF:
        return format_list_tag_result_as_xml_diff(
            *args,
            expected_tag_result=expected_tag_result,
            **kwargs
        )
    raise ValueError('unrecognised output format: %s' % output_format)


def format_tag_result(
        tag_result: Union[dict, list],
        output_format: str,
        expected_tag_result: List[Tuple[str, str]] = None,
        texts: np.array = None,
        features: np.array = None,
        model_name: str = None) -> str:
    if isinstance(tag_result, dict):
        assert output_format == TagOutputFormats.JSON
        return format_json_tag_result_as_json(tag_result)
    return format_list_tag_result(
        tag_result,
        output_format=output_format,
        expected_tag_result=expected_tag_result,
        texts=texts,
        features=features,
        model_name=model_name
    )
