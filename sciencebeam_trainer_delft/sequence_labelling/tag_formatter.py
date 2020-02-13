import json
from typing import Union

import numpy as np


class TagOutputFormats:
    JSON = 'json'
    LIST = 'list'


TAG_OUTPUT_FORMATS = [
    TagOutputFormats.JSON,
    TagOutputFormats.LIST
]


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def format_json_tag_result(tag_result: dict) -> str:
    return json.dumps(tag_result, indent=2, cls=CustomJsonEncoder)


def format_tag_result(
        tag_result: Union[dict, list],
        output_format: str = TagOutputFormats.JSON) -> str:
    assert output_format == TagOutputFormats.JSON
    if isinstance(tag_result, dict):
        return format_json_tag_result(tag_result)
    return str(tag_result)
