from __future__ import absolute_import

import logging


def configure_logging(level='INFO', secondary_level='WARN'):
    logging.basicConfig(level=secondary_level)
    logging.getLogger('delft').setLevel(level)
    logging.getLogger('sciencebeam_trainer_delft').setLevel(level)


def reset_logging(**kwargs):
    logging.root.handlers = []
    configure_logging(**kwargs)
