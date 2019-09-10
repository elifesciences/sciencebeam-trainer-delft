import os
import json
import logging
import time

import numpy as np


LOGGER = logging.getLogger(__name__)


SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT = "SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT"


class TagDebugReporter:
    def __init__(self, output_directory: str):
        self.output_directory = output_directory

    def report_tag_results(
            self,
            texts: np.array,
            features: np.array,
            annotations,
            model_name: str):
        output_file = os.path.join(self.output_directory, 'sciencebeam-delft-%s-%s.data' % (
            model_name,
            round(time.time())
        ))
        LOGGER.info('tagger, output_file: %s', output_file)
        with open(output_file, 'w') as fp:
            json.dump(
                {
                    'texts': np.array(texts).tolist(),
                    'features': np.array(features).tolist() if features is not None else None,
                    'annotations': annotations
                },
                fp,
                indent=4
            )


def get_tag_debug_reporter_if_enabled() -> TagDebugReporter:
    output_directory = os.environ.get(SCIENCEBEAM_DELFT_TAGGING_DEBUG_OUT)
    if not output_directory:
        return None
    return TagDebugReporter(output_directory)
