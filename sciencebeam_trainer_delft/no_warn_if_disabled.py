import sys
import warnings


def _hide_warnings_if_disabled():
    if 'ignore' in sys.warnoptions:
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
