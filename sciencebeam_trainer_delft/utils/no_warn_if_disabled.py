import os
import sys

from sciencebeam_trainer_delft.utils.warnings import hide_warnings


def _is_warnings_if_disabled():
    # Note: usualy PYTHONWARNINGS is reflected in sys.warnoptions
    #   but when run via JEP, sys.warnoptions appear to be empty
    if 'ignore' in sys.warnoptions or os.getenv('PYTHONWARNINGS') == 'ignore':
        return True
    if not sys.warnoptions and not os.getenv('PYTHONWARNINGS'):
        # ignore warnings by default
        return True
    return False


def _hide_warnings_if_disabled():
    if _is_warnings_if_disabled():
        # respect no warning
        # see https://github.com/scikit-learn/scikit-learn/issues/2531
        hide_warnings()


_hide_warnings_if_disabled()
