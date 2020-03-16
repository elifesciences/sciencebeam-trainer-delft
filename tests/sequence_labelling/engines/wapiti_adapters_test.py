import logging
from io import StringIO

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    write_wapiti_train_data
)


LOGGER = logging.getLogger(__name__)


class TestWriteWapitiTrainData:
    def test_should_write_single_document(self):
        buffer = StringIO()
        write_wapiti_train_data(
            buffer,
            x=[['token1', 'token2']],
            y=[['b-<label1>', 'b-<label2>']],
            features=[[['f1.1', 'f1.2'], ['f2.1', 'f2.2']]]
        )
        LOGGER.debug('buffer:\n%s', buffer.getvalue())
        assert buffer.getvalue().splitlines() == [
            'token1\tf1.1\tf1.2\tb-<label1>',
            'token2\tf2.1\tf2.2\tb-<label2>',
            '',
            ''
        ]

    def test_should_write_multiple_documents(self):
        buffer = StringIO()
        write_wapiti_train_data(
            buffer,
            x=[
                ['token1.1', 'token1.2'],
                ['token2.1', 'token2.2']
            ],
            y=[
                ['b-<label1.1>', 'b-<label1.2>'],
                ['b-<label2.1>', 'b-<label2.2>']
            ],
            features=[
                [['f1.1.1', 'f1.1.2'], ['f1.2.1', 'f1.2.2']],
                [['f2.1.1', 'f2.1.2'], ['f2.2.1', 'f2.2.2']]
            ]
        )
        LOGGER.debug('buffer:\n%s', buffer.getvalue())
        assert buffer.getvalue().splitlines() == [
            'token1.1\tf1.1.1\tf1.1.2\tb-<label1.1>',
            'token1.2\tf1.2.1\tf1.2.2\tb-<label1.2>',
            '',
            '',
            'token2.1\tf2.1.1\tf2.1.2\tb-<label2.1>',
            'token2.2\tf2.2.1\tf2.2.2\tb-<label2.2>',
            '',
            ''
        ]
