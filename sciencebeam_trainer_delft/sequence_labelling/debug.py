import os
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np


LOGGER = logging.getLogger(__name__)


SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT = "SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT"


def to_data_lines(
        features: np.array,
        annotations) -> List[str]:
    return [
        ' '.join([token_annoation[0]] + list(token_features) + [token_annoation[1]])
        for line_annotations, line_features in zip(annotations, features.tolist())
        for token_annoation, token_features in zip(line_annotations, line_features)
    ]


def to_flat_text(texts: np.array) -> str:
    return '\n'.join([
        ' '.join(line_tokens)
        for line_tokens in texts
    ])


class TagDebugReporter:
    def __init__(self, output_directory: str):
        self.output_directory = output_directory

    def get_base_output_name(self, model_name: str) -> str:
        return os.path.join(self.output_directory, 'sciencebeam-delft-%s-%s' % (
            round(time.time()),
            model_name
        ))

    def report_tag_results(
            self,
            texts: np.array,
            features: np.array,
            annotations,
            model_name: str):
        output_file = self.get_base_output_name(model_name=model_name) + '.json'
        LOGGER.info('tagger, output_file: %s', output_file)
        output_props = {
            'texts': np.array(texts).tolist(),
            'features': np.array(features).tolist() if features is not None else None,
            'annotations': annotations,
            'flat_text': to_flat_text(texts)
        }
        with open(output_file, 'w', encoding='utf-8') as fp:
            json.dump(output_props, fp, indent=4)
        if features is not None:
            data_output_file = self.get_base_output_name(model_name=model_name) + '.data'
            Path(data_output_file).write_text('\n'.join(to_data_lines(
                features=features,
                annotations=annotations
            )), encoding='utf-8')


def get_tag_debug_reporter_if_enabled() -> TagDebugReporter:
    output_directory = os.environ.get(SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT)
    if not output_directory:
        return None
    return TagDebugReporter(output_directory)
