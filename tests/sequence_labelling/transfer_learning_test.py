import argparse
from typing import List

from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    TransferLearningConfig,
    add_transfer_learning_arguments,
    get_transfer_learning_config_for_parsed_args
)


MODEL_PATH_1 = '/path/to/model1'

LAYER_1 = 'layer_1'
LAYER_2 = 'layer_2'

FIELD_1 = 'field1'
FIELD_2 = 'field2'


def parse_args_as_config(argv: List[str]) -> TransferLearningConfig:
    parser = argparse.ArgumentParser()
    add_transfer_learning_arguments(parser)
    args = parser.parse_args(argv)
    return get_transfer_learning_config_for_parsed_args(args)


class TestParseTransferLearningArgumentsAsConfig:
    def test_should_accept_no_parameters(self):
        parse_args_as_config([])

    def test_should_parse_model_path_with_layers(self):
        config = parse_args_as_config([
            f'--transfer-source-model-path={MODEL_PATH_1}',
            f'--transfer-layers={LAYER_1},{LAYER_2}',
            f'--transfer-preprocessor-fields={FIELD_1},{FIELD_2}'
        ])
        assert config.source_model_path == MODEL_PATH_1
        assert config.layers == [LAYER_1, LAYER_2]
        assert config.preprocessor_fields == [FIELD_1, FIELD_2]
