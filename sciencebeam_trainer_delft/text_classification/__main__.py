import argparse
import logging
from typing import List

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
import sciencebeam_trainer_delft.utils.no_keras_backend_message  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

from sciencebeam_trainer_delft.utils.cli import (
    SubCommand,
    SubCommandProcessor
)

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_trainer_delft.text_classification.config import (
    ModelConfig,
    TrainingConfig
)
from sciencebeam_trainer_delft.text_classification.cli_utils import (
    load_input_data,
    load_label_data,
    train,
    predict,
    evaluate
)


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = 'data/models/textClassification/toxic'
DEFAULT_EMBEDDNGS_NAME = 'glove.6B.50d'

DEFAULT_TRAIN_INPUT_PATHS = [
    "https://github.com/kermitt2/delft/raw/v0.2.3/data/textClassification/toxic/train.csv"
]
DEFAULT_TEST_INPUT_PATHS = [
    "https://github.com/kermitt2/delft/raw/v0.2.3/data/textClassification/toxic/test.csv"
]
DEFAULT_TEST_LABEL_INPUT_PATHS = [
    "https://github.com/kermitt2/delft/raw/v0.2.3/data/textClassification/toxic/test_labels.csv"
]


def add_common_arguments(
        parser: argparse.ArgumentParser):
    parser.add_argument("--quiet", action="store_true", help="Only log errors")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDNGS_NAME
    )


def add_train_arguments(
        parser: argparse.ArgumentParser):
    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        "--train-input",
        nargs='+',
        default=[DEFAULT_TRAIN_INPUT_PATHS],
        action='append',
        help="provided training file"
    )
    train_group.add_argument(
        "--train-input-limit",
        type=int,
        help=(
            "limit the number of training samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    train_group.add_argument(
        "--architecture",
        default='bidLstm',
        help="The desired architecture"
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=100,
        help="max epoch to train to"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size"
    )


def add_predict_arguments(
        parser: argparse.ArgumentParser):
    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument(
        "--predict-input",
        nargs='+',
        default=[DEFAULT_TEST_INPUT_PATHS],
        action='append',
        help="provided predict file"
    )
    eval_group.add_argument(
        "--predict-input-limit",
        type=int,
        help=(
            "limit the number of predict samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )


def add_eval_arguments(
        parser: argparse.ArgumentParser):
    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument(
        "--eval-input",
        nargs='+',
        default=[DEFAULT_TEST_INPUT_PATHS],
        action='append',
        help="provided evaluation file"
    )
    eval_group.add_argument(
        "--eval-label-input",
        nargs='+',
        default=[DEFAULT_TEST_LABEL_INPUT_PATHS],
        action='append',
        help="provided separate evaluation label file"
    )
    eval_group.add_argument(
        "--eval-input-limit",
        type=int,
        help=(
            "limit the number of evaluation samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )



def _flatten_input_paths(input_paths_list: List[List[str]]) -> List[str]:
    if not input_paths_list:
        return input_paths_list
    return [input_path for input_paths in input_paths_list for input_path in input_paths]


class SubCommandNames:
    TRAIN = 'train'
    PREDICT = 'predict'
    EVAL = 'eval'


class TrainSubCommand(SubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)

    def run(self, args: argparse.Namespace):
        LOGGER.info('train')
        download_manager = DownloadManager()
        train_input_paths = _flatten_input_paths(args.train_input)
        train_input_texts, train_input_labels, list_classes = load_input_data(
            train_input_paths,
            download_manager=download_manager,
            limit=args.train_input_limit
        )
        LOGGER.info('list_classes: %s', list_classes)
        train(
            model_config=ModelConfig(
                embeddings_name=args.embeddings,
                model_type=args.architecture,
                list_classes=list_classes
            ),
            training_config=TrainingConfig(
                batch_size=args.batch_size,
                max_epoch=args.max_epoch
            ),
            train_input_texts=train_input_texts,
            train_input_labels=train_input_labels,
            model_path=args.model_path
        )


class EvalSubCommand(SubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_eval_arguments(parser)

    def run(self, args: argparse.Namespace):
        LOGGER.info('eval')
        download_manager = DownloadManager()
        eval_input_paths = _flatten_input_paths(args.eval_input)
        eval_label_input_paths = _flatten_input_paths(args.eval_label_input)
        eval_input_texts, eval_input_labels, list_classes = load_input_data(
            eval_input_paths,
            download_manager=download_manager,
            limit=args.eval_input_limit
        )
        if eval_label_input_paths:
            eval_input_labels, _ = load_label_data(
                eval_label_input_paths,
                download_manager=download_manager,
                limit=args.eval_input_limit
            )
        LOGGER.info('list_classes: %s', list_classes)
        result = evaluate(
            eval_input_texts=eval_input_texts,
            eval_input_labels=eval_input_labels,
            model_path=args.model_path
        )
        print(result.text_formatted_report)


class PredictSubCommand(SubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_predict_arguments(parser)

    def run(self, args: argparse.Namespace):
        LOGGER.info('train')
        download_manager = DownloadManager()
        predict_input_paths = _flatten_input_paths(args.predict_input)
        predict_input_texts, _, list_classes = load_input_data(
            predict_input_paths,
            download_manager=download_manager,
            limit=args.predict_input_limit
        )
        LOGGER.info('list_classes: %s', list_classes)
        result = predict(
            eval_input_texts=predict_input_texts,
            model_path=args.model_path
        )
        print(result)


SUB_COMMANDS = [
    TrainSubCommand(
        SubCommandNames.TRAIN,
        'Train the model using the provided input(s)'
    ),
    EvalSubCommand(
        SubCommandNames.EVAL,
        'Evaluate the model using the provided input(s)'
    ),
    PredictSubCommand(
        SubCommandNames.PREDICT,
        'Predict the model using the provided input(s)'
    ),
]


def create_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )


def get_subcommand_processor():
    return SubCommandProcessor(SUB_COMMANDS, command_dest='command')


def parse_args(argv: List[str] = None, subcommand_processor: SubCommandProcessor = None):
    parser = create_parser()
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='command')

    subcommand_processor.add_sub_command_parsers(parser)

    args = parser.parse_args(argv)
    return args


def run(args: argparse.Namespace, subcommand_processor: SubCommandProcessor = None):
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')
    try:
        subcommand_processor.run(args)
    except BaseException as e:
        LOGGER.error('uncaught exception: %s', e, exc_info=1)
        raise


def main(argv: List[str] = None):
    subcommand_processor = get_subcommand_processor()
    args = parse_args(argv, subcommand_processor=subcommand_processor)
    if args.quiet:
        logging.root.setLevel('ERROR')
    elif args.debug:
        for name in [__name__, 'sciencebeam_trainer_delft', 'delft']:
            logging.getLogger(name).setLevel('DEBUG')
    run(args, subcommand_processor=subcommand_processor)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
