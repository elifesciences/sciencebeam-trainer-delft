import warnings
import logging


def hide_warnings():
    def no_warn(*_, **__):
        pass

    warnings.simplefilter("ignore")
    warnings.warn = no_warn
    logging.getLogger('tensorflow').setLevel('ERROR')
