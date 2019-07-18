import warnings


def hide_warnings():
    def no_warn(*_, **__):
        pass

    warnings.warn = no_warn

    try:
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except (ImportError, AttributeError):
        pass
