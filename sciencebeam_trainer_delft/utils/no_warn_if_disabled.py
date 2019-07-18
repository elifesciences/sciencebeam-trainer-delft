import os
import sys
import warnings


def _hide_warnings_if_disabled():
    # Note: usualy PYTHONWARNINGS is reflected in sys.warnoptions
    #   but when run via JEP, sys.warnoptions appear to be empty
    if 'ignore' in sys.warnoptions or os.getenv('PYTHONWARNINGS') == 'ignore':
        # respect no warning
        # see https://github.com/scikit-learn/scikit-learn/issues/2531

        def no_warn(*_, **__):
            pass

        warnings.warn = no_warn

        try:
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except (ImportError, AttributeError):
            pass


_hide_warnings_if_disabled()
