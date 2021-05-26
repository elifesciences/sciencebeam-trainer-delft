import logging
from pathlib import Path

import pytest

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import (
    WapitiWrapper
)


LOGGER = logging.getLogger(__name__)


@pytest.mark.slow
class TestWapitiWrapper:
    def test_should_pass_if_binary_is_available(self):
        WapitiWrapper().check_available()

    def test_should_fail_if_binary_is_not_available(self):
        with pytest.raises(Exception):
            WapitiWrapper('does-not-exist').check_available()

    def test_should_train_model(self, temp_dir: Path):
        wapiti = WapitiWrapper()
        template_path = temp_dir.joinpath('template')
        template_path.write_text('U00:%x[-4,0]')
        data_path = temp_dir.joinpath('train.data')
        data_path.write_text('\n'.join([
            'Token1 <label>',
            'Token2 <label>'
        ]))
        output_model_path = temp_dir.joinpath('model.wapiti')
        wapiti.train(
            template_path=str(template_path),
            data_path=str(data_path),
            output_model_path=str(output_model_path)
        )
        assert output_model_path.exists()

        wapiti_model = wapiti.load_model(str(output_model_path))

        label_data_path = temp_dir.joinpath('label.data')
        label_data_path.write_text('\n'.join([
            'Token1',
            'Token2'
        ]))
        labelled_features = wapiti_model.label_features([
            ['Token1'],
            ['Token2']
        ])
        LOGGER.debug('labelled_features: %s', labelled_features)
        assert labelled_features == [
            ['Token1', '<label>'],
            ['Token2', '<label>']
        ]
