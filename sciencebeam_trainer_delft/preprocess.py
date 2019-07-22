# this module provides backwards compatibility for pickle files (do not use otherwise)

import warnings

# pylint: disable=unused-import
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (  # noqa
    WordPreprocessor,
    Preprocessor,
    FeaturesPreprocessor,
    to_dict
)

warnings.warn('%s is deprecated' % __name__, category=DeprecationWarning)
