import json
from typing import Union, List, Tuple

import numpy as np


class TagOutputFormats:
    JSON = 'json'
    DATA = 'data'
    TEXT = 'text'
    XML = 'xml'


TAG_OUTPUT_FORMATS = [
    TagOutputFormats.JSON,
    TagOutputFormats.DATA,
    TagOutputFormats.TEXT,
    TagOutputFormats.XML
]


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


def iter_annotations_xml_text(
        annotations: List[List[Tuple[str, str]]]) -> List[str]:
    current_xml_tag = None
    for doc_index, doc_annotations in enumerate(annotations):
        if doc_index > 0:
            yield '\n\n'
        yield '<p>'
        for token_index, token_annoation in enumerate(doc_annotations):
            token_text, token_label = token_annoation[:2]
            xml_tag = get_xml_tag_for_annotation_label(token_label)
            should_open_tag = (token_label.startswith('B-') or xml_tag != current_xml_tag)
            if should_open_tag and current_xml_tag:
                yield '</%s>' % current_xml_tag
            if token_index > 0:
                yield ' '
            if should_open_tag:
                yield '<%s>' % xml_tag
                current_xml_tag = xml_tag
            yield token_text
        if current_xml_tag:
            yield '</%s>' % current_xml_tag
            current_xml_tag = None
        yield '</p>'


def format_list_tag_result_as_xml(
        tag_result: List[List[Tuple[str, str]]],
        texts: np.array = None,  # pylint: disable=unused-argument
        features: np.array = None,  # pylint: disable=unused-argument
        model_name: str = None) -> str:  # pylint: disable=unused-argument
    return '<xml>%s</xml>' % ''.join(iter_annotations_xml_text(
        annotations=tag_result
    ))


def format_list_tag_result(
        *args,
        output_format: str,
        **kwargs) -> str:
    if output_format == TagOutputFormats.JSON:
        return format_list_tag_result_as_json(*args, **kwargs)
    if output_format == TagOutputFormats.DATA:
        return format_list_tag_result_as_data(*args, **kwargs)
    if output_format == TagOutputFormats.TEXT:
        return format_list_tag_result_as_text(*args, **kwargs)
    if output_format == TagOutputFormats.XML:
        return format_list_tag_result_as_xml(*args, **kwargs)
    raise ValueError('unrecognised output format: %s' % output_format)


def format_tag_result(
        tag_result: Union[dict, list],
        output_format: str,
        texts: np.array = None,
        features: np.array = None,
        model_name: str = None) -> str:
    if isinstance(tag_result, dict):
        assert output_format == TagOutputFormats.JSON
        return format_json_tag_result_as_json(tag_result)
    return format_list_tag_result(
        tag_result,
        output_format=output_format,
        texts=texts,
        features=features,
        model_name=model_name
    )
