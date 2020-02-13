import os
import logging
import time
from pathlib import Path

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    format_tag_result
)


LOGGER = logging.getLogger(__name__)


SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT = "SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT"


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
        filename_prefix = self.get_base_output_name(model_name=model_name)
        output_file = filename_prefix + '.json'
        LOGGER.info('tagger, output_file: %s', output_file)

        format_tag_result_kwargs = dict(
            tag_result=annotations,
            texts=texts,
            features=features,
            model_name=model_name
        )

        formatted_text = format_tag_result(
            output_format=TagOutputFormats.TEXT,
            **format_tag_result_kwargs
        )
        Path(filename_prefix + '.txt').write_text(formatted_text, encoding='utf-8')

        formatted_json = format_tag_result(
            output_format=TagOutputFormats.JSON,
            **format_tag_result_kwargs
        )
        Path(output_file).write_text(formatted_json, encoding='utf-8')

        formatted_xml = format_tag_result(
            output_format=TagOutputFormats.XML,
            **format_tag_result_kwargs
        )
        Path(filename_prefix + '.xml').write_text(formatted_xml, encoding='utf-8')

        if features is not None:
            formatted_data = format_tag_result(
                output_format=TagOutputFormats.DATA,
                **format_tag_result_kwargs
            )
            Path(filename_prefix + '.data').write_text(formatted_data, encoding='utf-8')


def get_tag_debug_reporter_if_enabled() -> TagDebugReporter:
    output_directory = os.environ.get(SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT)
    if not output_directory:
        return None
    return TagDebugReporter(output_directory)
