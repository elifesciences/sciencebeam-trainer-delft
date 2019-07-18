import os
import sys

from sciencebeam_trainer_delft.utils.warnings import hide_warnings


def _hide_warnings_if_disabled():
    # Note: usualy PYTHONWARNINGS is reflected in sys.warnoptions
    #   but when run via JEP, sys.warnoptions appear to be empty
    if 'ignore' in sys.warnoptions or os.getenv('PYTHONWARNINGS') == 'ignore':
        # respect no warning
        # see https://github.com/scikit-learn/scikit-learn/issues/2531
        hide_warnings()
