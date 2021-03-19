# mostly copied from https://github.com/kermitt2/delft/blob/master/grobidTagger.py
import logging
import argparse
from abc import abstractmethod
from typing import List

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
import sciencebeam_trainer_delft.utils.no_keras_backend_message  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import keras.backend as K

from sciencebeam_trainer_delft.utils.misc import parse_number_ranges
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.cloud_support import patch_cloud_support
from sciencebeam_trainer_delft.utils.tf import get_tf_info
from sciencebeam_trainer_delft.utils.io import (
    copy_file,
    auto_uploading_output_file
)
from sciencebeam_trainer_delft.utils.logging import (
    tee_stdout_and_stderr_lines_to,
    tee_logging_lines_to
)

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    add_train_notification_arguments,
    get_train_notification_manager
)

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_default_batch_size,
    get_default_stateful
)
from sciencebeam_trainer_delft.sequence_labelling.models import get_model_names, patch_get_model

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import (
    DEFAULT_STOP_EPSILON_VALUE,
    DEFAULT_STOP_WINDOW_SIZE
)
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_install import (
    install_wapiti_and_get_path_or_none
)

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TAG_OUTPUT_FORMATS
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    EvaluationOutputFormats,
    EVALUATION_OUTPUT_FORMATS
)

from sciencebeam_trainer_delft.utils.cli import (
    SubCommand,
    SubCommandProcessor
)

from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    set_random_seeds,
    train,
    wapiti_train,
    train_eval,
    wapiti_train_eval,
    eval_model,
    wapiti_eval_model,
    tag_input,
    wapiti_tag_input,
    print_input_info
)


LOGGER = logging.getLogger(__name__)


WAPITI_MODEL_NAME = 'wapiti'

GROBID_MODEL_NAMES = [
    'affiliation-address', 'citation', 'date', 'figure', 'fulltext', 'header',
    'name', 'name-citation', 'name-header', 'patent', 'reference-segmenter',
    'segmentation', 'software', 'table'
]


class Tasks:
    TRAIN = 'train'
    TRAIN_EVAL = 'train_eval'
    EVAL = 'eval'
    TAG = 'tag'
    WAPITI_TRAIN = 'wapiti_train'
    WAPITI_TRAIN_EVAL = 'wapiti_train_eval'
    WAPITI_EVAL = 'wapiti_eval'
    WAPITI_TAG = 'wapiti_tag'
    INPUT_INFO = 'input_info'


DEFAULT_RANDOM_SEED = 42

DEFAULT_TAG_OUTPUT_FORMAT = TagOutputFormats.XML


def add_common_arguments(
        parser: argparse.ArgumentParser,
        max_sequence_length_default: int = None):
    input_group = parser.add_argument_group('input')
    input_group.add_argument(
        "--input",
        nargs='+',
        action='append',
        help="provided training file"
    )
    input_group.add_argument(
        "--shuffle-input",
        action="store_true",
        help="Shuffle the input before splitting"
    )
    input_group.add_argument(
        "--limit",
        type=int,
        help=(
            "limit the number of training samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Set the random seed for reproducibility"
    )
    parser.add_argument(
        "--batch-size", type=int, default=get_default_batch_size(),
        help="batch size"
    )
    parser.add_argument(
        "--max-sequence-length", type=int,
        default=max_sequence_length_default,
        help="maximum sequence length"
    )
    parser.add_argument(
        "--no-use-lmdb", action="store_true",
        help="Do not use LMDB embedding cache (load embeddings into memory instead)"
    )

    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")

    parser.add_argument("--quiet", action="store_true", help="Only log errors")

    parser.add_argument(
        "--save-input-to-and-exit",
        help=(
            "If set, saves the input to the specified path and exits."
            " This can be useful to retrain the model outside GROBID."
        )
    )

    parser.add_argument(
        "--log-file",
        help=(
            "If set, saves the output to the specified log file."
            " This may also be a file in a bucket, in which case it will be uploaded at the end."
            " Add the .gz extension if you wish to compress the file."
        )
    )

    parser.add_argument("--job-dir", help="job dir (only used when running via ai platform)")


def add_model_path_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--model-path", **kwargs)


def add_fold_count_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--fold-count", type=int, default=1, **kwargs)


def add_eval_output_format_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-format",
        choices=EVALUATION_OUTPUT_FORMATS,
        default=EvaluationOutputFormats.TEXT
    )


def add_eval_first_entity_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-first-entity",
        action="store_true",
        help=''.join([
            'If set, additional evaluates the first entity (e.g. first_<author>).'
        ])
    )


def add_eval_output_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-path",
        help='If specified, saves the evaluation to the specified path in the JSON format'
    )


def add_eval_output_arguments(parser: argparse.ArgumentParser):
    add_eval_output_format_argument(parser)
    add_eval_first_entity_argument(parser)
    add_eval_output_path_argument(parser)


def add_eval_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-input",
        nargs='+',
        action='append',
        help=' '.join([
            "Evaluation data at the end of training. If not specified,",
            "it will use a slice of the training data"
        ])
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help=' '.join([
            "Limit the number of documents to use for evaluation.",
            "This is mostly for testing to make evaluation faster."
        ])
    )


def add_dl_eval_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-max-sequence-length",
        type=int,
        help=' '.join([
            "Maximum sequence length to use for evaluation.",
            "If not specified, no limit will be applied."
        ])
    )
    parser.add_argument(
        "--eval-input-window-stride",
        type=int,
        help="Should be equal or less than eval max sequence length"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help=' '.join([
            "The batch size to be used for evaluation.",
            "If not specified, the training batch size is used.",
            "This may be useful to evaluate on longer sequences",
            "that could require a smaller batch size."
        ])
    )


def add_tag_output_format_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--tag-output-format",
        default=DEFAULT_TAG_OUTPUT_FORMAT,
        choices=TAG_OUTPUT_FORMATS,
        help="output format for tag results",
        **kwargs
    )


def add_output_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--output", help="directory where to save a trained model", **kwargs)


def add_max_epoch_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--max-epoch", type=int, default=10,
        help="max epoch to train to",
        **kwargs
    )


def add_stateful_argument(parser: argparse.ArgumentParser, **kwargs):
    default_value = get_default_stateful()
    parser.add_argument(
        "--stateful",
        dest="stateful",
        default=default_value,
        action="store_true",
        help="Make RNNs stateful (required for truncated BPTT)",
        **kwargs
    )
    parser.add_argument(
        "--no-stateful",
        dest="stateful",
        default=default_value,
        action="store_false",
        help="Disable statefulness (default)",
        **kwargs
    )


def add_input_window_stride_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--input-window-stride",
        type=int,
        help="Should be equal or less than max sequence length",
        **kwargs
    )


def add_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--architecture", default='BidLSTM_CRF',
        choices=get_model_names(),
        help="type of model architecture to be used"
    )
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")

    parser.add_argument(
        "--max-char-length",
        type=int,
        default=30,
        help="The maximum number of chars used by the model"
    )

    parser.add_argument(
        "--additional-token-feature-indices",
        type=parse_number_ranges,
        help="".join([
            "Additional feature values that should be used as tokens.",
            " e.g. 0 or 0-3.",
            " If blank, no additional token features will be used."
        ])
    )

    parser.add_argument(
        "--text-feature-indices",
        type=parse_number_ranges,
        help="".join([
            "Feature values that should be treated as text input.",
            " e.g. 0 or 0-3.",
            " If blank, no additext features will be used.",
            " Cannot be used together with --additional-token-feature-indices.",
            " Text features will get tokenized."
            " If word embeddings are used, then the number of tokens will depend on",
            " --concatenated-embeddings-token-count.",
            " Tokens from text features replace regular tokens from the training data."
        ])
    )

    parser.add_argument(
        "--concatenated-embeddings-token-count",
        type=int,
        help="".join([
            "The number of tokens to concatenate as word embeddings.",
            " If not specified, it concatenate the main token with any",
            " --additional-token-feature-indices (if any).",
            " This option is mainly useful in combination with --text-feature-indices.",
            " It has no effect, if no word embeddings are used."
        ])
    )

    features_group = parser.add_argument_group('features')
    features_group.add_argument("--use-features", action="store_true", help="Use features")
    features_group.add_argument(
        "--features-indices", "--feature-indices",
        type=parse_number_ranges,
        help="The feature indices to use. e.g. 7-10. If blank, all of the features will be used."
    )
    features_group.add_argument(
        "--features-embedding-size", "--feature-embedding-size",
        type=int,
        help="size of feature embedding, use 0 to disable embedding"
    )
    features_group.add_argument(
        "--use-features-indices-input", action="store_true",
        help="Use features indices values (should be inferred from the model)"
    )
    features_group.add_argument(
        "--features-lstm-units", type=int,
        help="Number of LSTM units used by the features"
    )

    add_stateful_argument(parser)
    add_input_window_stride_argument(parser)

    output_group = parser.add_argument_group('output')
    add_output_argument(output_group)
    output_group.add_argument("--checkpoint", help="directory where to save a checkpoint model")

    parser.add_argument(
        "--embedding", default="glove-6B-50d",
        help="name of word embedding"
    )
    parser.add_argument(
        "--preload-embedding",
        help=" ".join([
            "Name or URL to embedding to preload.",
            "This can be useful in combination with resuming model training."
        ])
    )
    features_group.add_argument(
        "--no-embedding",
        dest="use_word_embeddings",
        default=True,
        action="store_false",
        help="Disable the use of word embedding"
    )
    parser.add_argument(
        "--char-embedding-size", type=int, default=25,
        help="size of char embedding"
    )
    parser.add_argument(
        "--char-lstm-units", type=int, default=25,
        help="number of list units for chars"
    )
    parser.add_argument(
        "--word-lstm-units", type=int, default=100,
        help="number of lstm units for words"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="main dropout"
    )
    parser.add_argument(
        "--recurrent-dropout", type=float, default=0.5,
        help="recurrent dropout"
    )
    add_max_epoch_argument(parser)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="how many epochs to continue training after the f1 score hasn't improved"
    )
    parser.add_argument(
        "--resume-train-model-path",
        help="path to the model training should be resumed from (e.g. path to checkpoint)"
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Sets the initial epoch for model training."
    )
    add_train_notification_arguments(parser)


def add_wapiti_train_arguments(parser: argparse.ArgumentParser):
    add_output_argument(parser)
    add_max_epoch_argument(parser)
    parser.add_argument("--wapiti-template", required=True)
    parser.add_argument(
        "--wapiti-gzip",
        action="store_true",
        help="whether to gzip wapiti models before saving"
    )
    parser.add_argument(
        "--wapiti-stop-epsilon-value",
        default=DEFAULT_STOP_EPSILON_VALUE
    )
    parser.add_argument(
        "--wapiti-stop-window-size",
        type=int,
        default=DEFAULT_STOP_WINDOW_SIZE
    )
    add_train_notification_arguments(parser)


def get_wapiti_train_args(args: argparse.Namespace) -> dict:
    return dict(
        stop_epsilon_value=args.wapiti_stop_epsilon_value,
        stop_window_size=args.wapiti_stop_window_size
    )


def add_wapiti_install_arguments(parser: argparse.ArgumentParser):
    example_url = "https://github.com/kermitt2/Wapiti/archive/master.tar.gz"
    parser.add_argument(
        "--wapiti-install-source",
        help="source file to install wapiti from, e.g. %s" % example_url
    )


def add_all_non_positional_arguments(parser: argparse.ArgumentParser):
    add_common_arguments(parser)
    add_train_arguments(parser)


def add_model_positional_argument(parser: argparse.ArgumentParser):
    parser.add_argument("model", nargs='?', choices=GROBID_MODEL_NAMES)


def _flatten_input_paths(input_paths_list: List[List[str]]) -> List[str]:
    if not input_paths_list:
        return input_paths_list
    return [input_path for input_paths in input_paths_list for input_path in input_paths]


def process_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input = _flatten_input_paths(args.input)
    try:
        args.eval_input = _flatten_input_paths(args.eval_input)
    except AttributeError:
        pass


def create_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )


def save_input_to(input_paths: List[str], output_path: str):
    assert len(input_paths) == 1, "exactly one input path expected (got: %s)" % input_paths
    input_path = input_paths[0]
    LOGGER.info('saving input (%s) to: %s', input_path, output_path)
    copy_file(input_path, output_path)


def get_eval_input_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_input_paths=args.eval_input,
        eval_limit=args.eval_limit,
    )


def get_eval_output_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_output_format=args.eval_output_format,
        eval_first_entity=args.eval_first_entity,
        eval_output_path=args.eval_output_path
    )


def get_dl_eval_model_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_max_sequence_length=args.eval_max_sequence_length,
        eval_input_window_stride=args.eval_input_window_stride,
        eval_batch_size=args.eval_batch_size
    )


class GrobidTrainerSubCommand(SubCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_manager = None
        self.embedding_manager = None

    @abstractmethod
    def do_run(self, args: argparse.Namespace):
        pass

    def preload_and_validate_embedding(
            self,
            embedding_name: str,
            use_word_embeddings: bool = True) -> str:
        if not use_word_embeddings:
            return None
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        self.embedding_manager.validate_embedding(embedding_name)
        return embedding_name

    def get_common_args(self, args: argparse.Namespace) -> dict:
        return dict(
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            shuffle_input=args.shuffle_input,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            multiprocessing=args.multiprocessing,
            embedding_manager=self.embedding_manager,
            download_manager=self.download_manager
        )

    def get_train_args(self, args: argparse.Namespace) -> dict:
        return dict(
            architecture=args.architecture,
            use_ELMo=args.use_ELMo,
            output_path=args.output,
            log_dir=args.checkpoint,
            char_emb_size=args.char_embedding_size,
            char_lstm_units=args.char_lstm_units,
            word_lstm_units=args.word_lstm_units,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            max_epoch=args.max_epoch,
            use_features=args.use_features,
            features_indices=args.features_indices,
            features_embedding_size=args.features_embedding_size,
            patience=args.early_stopping_patience,
            config_props=dict(
                max_char_length=args.max_char_length,
                additional_token_feature_indices=args.additional_token_feature_indices,
                text_feature_indices=args.text_feature_indices,
                concatenated_embeddings_token_count=args.concatenated_embeddings_token_count,
                use_word_embeddings=args.use_word_embeddings,
                use_features_indices_input=args.use_features_indices_input,
                features_lstm_units=args.features_lstm_units,
                stateful=args.stateful
            ),
            training_props=dict(
                initial_epoch=args.initial_epoch,
                input_window_stride=args.input_window_stride
            ),
            resume_train_model_path=args.resume_train_model_path,
            train_notification_manager=get_train_notification_manager(args),
            **self.get_common_args(args)
        )

    def run(self, args: argparse.Namespace):
        if args.save_input_to_and_exit:
            save_input_to(args.input, args.save_input_to_and_exit)
            return

        self.download_manager = DownloadManager()
        self.embedding_manager = EmbeddingManager(
            download_manager=self.download_manager
        )
        if args.no_use_lmdb:
            self.embedding_manager.disable_embedding_lmdb_cache()

        set_random_seeds(args.random_seed)

        self.do_run(args)

        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()


class TrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        if args.preload_embedding:
            self.preload_and_validate_embedding(
                args.preload_embedding,
                use_word_embeddings=True
            )
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings and not args.resume_train_model_path
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train(
            embeddings_name=embedding_name,
            **self.get_train_args(args)
        )


class WapitiTrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train(
            model=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args),
            train_notification_manager=get_train_notification_manager(args)
        )


class TrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')
        add_fold_count_argument(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)
        add_dl_eval_model_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        if args.preload_embedding:
            self.preload_and_validate_embedding(
                args.preload_embedding,
                use_word_embeddings=True
            )
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings and not args.resume_train_model_path
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train_eval(
            fold_count=args.fold_count,
            embeddings_name=embedding_name,
            eval_output_args=get_eval_output_args(args),
            **get_eval_input_args(args),
            **get_dl_eval_model_args(args),
            **self.get_train_args(args)
        )


class WapitiTrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train_eval(
            model=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            eval_input_paths=args.eval_input,
            eval_limit=args.eval_limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args),
            train_notification_manager=get_train_notification_manager(args),
            eval_output_args=get_eval_output_args(args)
        )


class EvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        parser.add_argument(
            "--use-eval-train-test-split",
            action="store_true",
            help=" ".join([
                "If enabled, split the input when running 'eval'",
                "(in the same way it is split for 'train_eval')"
            ])
        )
        add_eval_output_arguments(parser)
        add_stateful_argument(parser)
        add_dl_eval_model_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        eval_model(
            model_path=args.model_path,
            split_input=args.use_eval_train_test_split,
            eval_output_args=get_eval_output_args(args),
            stateful=args.stateful,
            **get_dl_eval_model_args(args),
            **self.get_common_args(args)
        )


class WapitiEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_eval_model(
            model_path=args.model_path,
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            eval_output_args=get_eval_output_args(args),
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class TagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_stateful_argument(parser)
        add_input_window_stride_argument(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)

    def do_run(self, args: argparse.Namespace):
        tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            stateful=args.stateful,
            input_window_stride=args.input_window_stride,
            **self.get_common_args(args)
        )


class WapitiTagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class InputInfoSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        print_input_info(
            model=args.model,
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager
        )


SUB_COMMANDS = [
    TrainSubCommand(
        Tasks.TRAIN,
        'Train the model using the provided input(s)'
    ),
    TrainEvalSubCommand(
        Tasks.TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    EvalSubCommand(
        Tasks.EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    TagSubCommand(
        Tasks.TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    WapitiTrainSubCommand(
        Tasks.WAPITI_TRAIN,
        'Train the model using the provided input(s)'
    ),
    WapitiTrainEvalSubCommand(
        Tasks.WAPITI_TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    WapitiEvalSubCommand(
        Tasks.WAPITI_EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    WapitiTagSubCommand(
        Tasks.WAPITI_TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    InputInfoSubCommand(
        Tasks.INPUT_INFO,
        'Display input summary information relating to the passed in input(s)'
    )
]


def get_subcommand_processor():
    return SubCommandProcessor(SUB_COMMANDS, command_dest='action')


def parse_args(argv: List[str] = None, subcommand_processor: SubCommandProcessor = None):
    parser = create_parser()
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')

    add_model_positional_argument(parser)
    subcommand_processor.add_sub_command_parsers(parser)

    args = parser.parse_args(argv)
    process_args(args)
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
    if args.log_file:
        with auto_uploading_output_file(args.log_file, mode='w') as log_fp:
            try:
                with tee_stdout_and_stderr_lines_to(log_fp.write, append_line_feed=True):
                    with tee_logging_lines_to(log_fp.write, append_line_feed=True):
                        run(args, subcommand_processor=subcommand_processor)
            finally:
                logging.shutdown()
    else:
        run(args, subcommand_processor=subcommand_processor)


def main_setup():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()


if __name__ == "__main__":
    main_setup()
    main()
