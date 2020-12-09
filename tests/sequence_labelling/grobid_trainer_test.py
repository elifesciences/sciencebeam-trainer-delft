import gzip
import json
import logging
import os
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from unittest.mock import call, patch, MagicMock
from typing import List

import pytest

import numpy as np

from delft.utilities.Embeddings import Embeddings

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    EnvironmentVariables,
    get_model_directory
)
import sciencebeam_trainer_delft.sequence_labelling.grobid_trainer as grobid_trainer_module
from sciencebeam_trainer_delft.sequence_labelling.grobid_trainer import (
    set_random_seeds,
    parse_args,
    load_data_and_labels,
    train,
    train_eval,
    tag_input,
    wapiti_train,
    wapiti_train_eval,
    wapiti_eval_model,
    wapiti_tag_input,
    main
)

from ..embedding.test_data import TEST_DATA_PATH
from ..test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)

EMBEDDING_NAME_1 = 'embedding1'

EMBEDDING_1 = {
    "name": EMBEDDING_NAME_1,
    "path": os.path.join(TEST_DATA_PATH, 'embedding1.txt'),
    "type": "glove",
    "format": "vec",
    "lang": "en",
    "item": "word"
}

MODEL_NAME_1 = 'model1'

INPUT_PATH_1 = '/path/to/dataset1'
INPUT_PATH_2 = '/path/to/dataset2'

GROBID_HEADER_MODEL_URL = (
    'https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/'
    'delft-grobid-header-biorxiv-no-word-embedding-2020-05-05.tar.gz'
)

GROBID_HEADER_TEST_DATA_URL = (
    'https://github.com/elifesciences/sciencebeam-datasets/releases/download/'
    'grobid-0.6.1/delft-grobid-0.6.1-header.test.gz'
)

GROBID_HEADER_TEST_DATA_TITLE_1 = (
    'Projections : A Preliminary Performance Tool for Charm'
)

FEATURE_INDICES_1 = [9, 10, 11]

FEATURES_EMBEDDING_SIZE_1 = 13


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path(temp_dir: Path):
    return temp_dir.joinpath('embedding-registry.json')


@pytest.fixture(name='embedding_registry', autouse=True)
def _embedding_registry(embedding_registry_path: Path):
    embedding_registry_path.write_text(json.dumps({
        'embedding-lmdb-path': None,
        'embeddings': [EMBEDDING_1],
        'embeddings-contextualized': []
    }, indent=4), encoding='utf-8')


@pytest.fixture(autouse=True)
def _embedding_class(embedding_registry_path: Path):
    embedding_class_with_defaults = partial(Embeddings, path=str(embedding_registry_path))
    target = 'delft.sequenceLabelling.wrapper.Embeddings'
    with patch(target, new=embedding_class_with_defaults) as mock:
        yield mock


@pytest.fixture(name='get_default_training_data_mock')
def _get_default_training_data_mock():
    with patch.object(grobid_trainer_module, 'get_default_training_data') as mock:
        yield mock


@pytest.fixture(name='load_data_and_labels_crf_file_mock')
def _load_data_and_labels_crf_file_mock():
    with patch.object(grobid_trainer_module, 'load_data_and_labels_crf_file') as mock:
        mock.return_value = (MagicMock(), MagicMock(), MagicMock())
        yield mock


def _mock_shuffle_array(a: np.array) -> np.array:
    # reverse array
    return a[::-1]


def _mock_shuffle_arrays(arrays: List[np.array], **_) -> np.array:
    return [_mock_shuffle_array(a) for a in arrays]


@pytest.fixture(name='shuffle_arrays_mock')
def _shuffle_arrays_mock():
    with patch.object(grobid_trainer_module, 'shuffle_arrays') as mock:
        mock.side_effect = _mock_shuffle_arrays
        yield mock


@pytest.fixture(name='download_manager_mock')
def _download_manager_mock():
    mock = MagicMock(name='download_manager_mock')
    mock.download_if_url.side_effect = str
    return mock


@pytest.fixture(name='model_base_path')
def _model_base_path(temp_dir: Path):
    p = temp_dir.joinpath('models')
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(name='default_args')
def _default_args(
        sample_train_file: str,
        model_base_path: Path,
        embedding_registry_path: Path,
        download_manager_mock: MagicMock):
    # download_manager = MagicMock(name='download_manager')
    # download_manager.download_if_url.return_value = str(sample_train_file)
    # download_manager.download_if_url.side_effect = lambda file_url: str(file_url)
    return dict(
        model='header',
        embeddings_name=EMBEDDING_NAME_1,
        input_paths=[sample_train_file],
        download_manager=download_manager_mock,
        output_path=str(model_base_path),
        architecture='CustomBidLSTM_CRF',
        word_lstm_units=11,
        features_indices=list(range(7, 1 + 10)),
        embedding_registry_path=str(embedding_registry_path)
    )


@pytest.fixture(name='default_model_directory')
def _default_model_directory(default_args: dict):
    return get_model_directory(
        model_name=default_args['model'],
        dir_path=default_args['output_path']
    )


def load_model_config(model_path: str) -> ModelConfig:
    LOGGER.debug('model_path: %s', model_path)
    return ModelLoader().load_model_config_from_directory(model_path)


class TestGrobidTrainer:
    class TestSetRandomSeeds:
        def test_should_not_fail(self):
            set_random_seeds(123)

    class TestParseArgs:
        def test_should_require_arguments(self):
            with pytest.raises(SystemExit):
                parse_args([])

        def test_should_allow_multiple_input_files_via_single_input_param(self):
            opt = parse_args([
                'header',
                'train',
                '--input', '/path/to/dataset1', '/path/to/dataset2'
            ])
            assert opt.input == ['/path/to/dataset1', '/path/to/dataset2']

        def test_should_allow_multiple_input_files_via_multiple_input_params(self):
            opt = parse_args([
                'header',
                'train',
                '--input', INPUT_PATH_1,
                '--input', INPUT_PATH_2
            ])
            assert opt.input == [INPUT_PATH_1, INPUT_PATH_2]

        def test_should_use_stateful_env_variable_true_by_default(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'true'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is True

        def test_should_use_stateful_env_variable_false_by_default(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'false'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is False

        def test_should_fallback_to_none_statefulness(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = ''
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is None

        def test_should_allow_to_set_stateful(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'false'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2,
                '--stateful'
            ])
            assert opt.stateful is True

        def test_should_allow_to_unset_stateful(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'true'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2,
                '--no-stateful'
            ])
            assert opt.stateful is False

    @pytest.mark.usefixtures(
        'get_default_training_data_mock', 'load_data_and_labels_crf_file_mock'
    )
    class TestLoadDataAndLabels:
        def test_should_load_using_default_dataset(
                self,
                get_default_training_data_mock: MagicMock,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
            get_default_training_data_mock.return_value = '/tmp/dummy/training/data'
            load_data_and_labels(
                MODEL_NAME_1,
                [],
                download_manager=download_manager_mock
            )
            get_default_training_data_mock.assert_called_with(MODEL_NAME_1)
            download_manager_mock.download_if_url.assert_called_with(
                get_default_training_data_mock.return_value
            )
            load_data_and_labels_crf_file_mock.assert_called_with(
                get_default_training_data_mock.return_value,
                limit=None
            )

        def test_should_load_single_input_without_limit(
                self,
                download_manager_mock: MagicMock):
            load_data_and_labels(
                MODEL_NAME_1,
                [INPUT_PATH_1],
                download_manager=download_manager_mock
            )
            download_manager_mock.download_if_url.assert_called_with(
                INPUT_PATH_1
            )

        def test_should_load_single_input_with_limit(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
            load_data_and_labels(
                MODEL_NAME_1,
                [INPUT_PATH_1],
                limit=123,
                download_manager=download_manager_mock
            )
            load_data_and_labels_crf_file_mock.assert_called_with(
                INPUT_PATH_1,
                limit=123
            )

        def test_should_load_multiple_inputs_with_limit(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
            download_manager_mock.download_if_url.side_effect = lambda input_path: input_path
            load_data_and_labels_crf_file_mock.side_effect = [
                (np.array([['x1']]), np.array([['y1']]), np.array([[['f1']]])),
                (np.array([['x2']]), np.array([['y2']]), np.array([[['f2']]]))
            ]
            x, y, f = load_data_and_labels(
                MODEL_NAME_1,
                [INPUT_PATH_1, INPUT_PATH_2],
                limit=123,
                download_manager=download_manager_mock
            )
            assert (x == np.array([['x1'], ['x2']])).all()
            assert (y == np.array([['y1'], ['y2']])).all()
            assert (f == np.array([[['f1']], [['f2']]])).all()
            load_data_and_labels_crf_file_mock.assert_has_calls(
                [call(INPUT_PATH_1, limit=123), call(INPUT_PATH_2, limit=123)]
            )

        def test_should_shuffle_single_source_input_data(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                shuffle_arrays_mock: MagicMock,
                download_manager_mock: MagicMock):
            download_manager_mock.download_if_url.side_effect = lambda input_path: input_path
            x_unshuffled = np.array([['x1'], ['x2']])
            y_unshuffled = np.array([['y1'], ['y2']])
            f_unshuffled = np.array([[['f1']], [['f2']]])
            load_data_and_labels_crf_file_mock.side_effect = [
                (x_unshuffled, y_unshuffled, f_unshuffled)
            ]
            x, y, f = load_data_and_labels(
                MODEL_NAME_1,
                [INPUT_PATH_1],
                limit=123,
                shuffle_input=True,
                random_seed=424,
                download_manager=download_manager_mock
            )
            shuffle_arrays_mock.assert_called()
            assert shuffle_arrays_mock.call_args[1] == {'random_seed': 424}
            input_arrays = shuffle_arrays_mock.call_args[0][0]
            assert (input_arrays[0] == x_unshuffled).all()
            assert (input_arrays[1] == y_unshuffled).all()
            assert (input_arrays[2] == f_unshuffled).all()
            assert (x == _mock_shuffle_arrays(x)).all()
            assert (y == _mock_shuffle_arrays(y)).all()
            assert (f == _mock_shuffle_arrays(f)).all()

    @pytest.mark.slow
    @pytest.mark.very_slow
    class TestEndToEnd:
        @log_on_exception
        def test_should_be_able_to_train_without_features(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=False,
                **default_args
            )
            model_config = load_model_config(default_model_directory)
            assert not model_config.use_features
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_features_without_feature_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=True,
                **{
                    **default_args,
                    'features_indices': FEATURE_INDICES_1,
                    'features_embedding_size': None
                }
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.use_features
            assert model_config.features_indices == FEATURE_INDICES_1
            assert model_config.features_embedding_size is None
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_features_and_features_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=True,
                **{
                    **default_args,
                    'features_indices': FEATURE_INDICES_1,
                    'features_embedding_size': FEATURES_EMBEDDING_SIZE_1
                }
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.use_features
            assert model_config.features_indices == FEATURE_INDICES_1
            assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_without_word_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                **{
                    **default_args,
                    'embeddings_name': None,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'use_word_embeddings': False
                    }
                }
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.embeddings_name is None
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_additional_token_feature_indices(
                self, default_args: dict, default_model_directory: str):
            train(
                **{
                    **default_args,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'max_char_length': 60,
                        'additional_token_feature_indices': [0]
                    }
                }
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.max_char_length == 60
            assert model_config.additional_token_feature_indices == [0]
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_text_feature_indices(
                self, default_args: dict, default_model_directory: str):
            train(
                **{
                    **default_args,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'max_char_length': 60,
                        'text_feature_indices': [0],
                        'concatenated_embeddings_token_count': 2
                    }
                }
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.max_char_length == 60
            assert model_config.text_feature_indices == [0]
            assert model_config.concatenated_embeddings_token_count == 2
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_BidLSTM_CRF_FEATURES(
                self, default_args: dict, default_model_directory: str):
            train_args = {
                **default_args,
                'architecture': 'BidLSTM_CRF_FEATURES',
                'features_embedding_size': 4,
                'config_props': {
                    'features_lstm_units': 4
                }
            }
            train(
                **train_args
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.model_type == 'BidLSTM_CRF_FEATURES'
            assert model_config.features_embedding_size == 4
            assert model_config.features_lstm_units == 4
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_CustomBidLSTM_CRF_FEATURES(
                self, default_args: dict, default_model_directory: str):
            train_args = {
                **default_args,
                'architecture': 'CustomBidLSTM_CRF_FEATURES',
                'features_embedding_size': 4,
                'config_props': {
                    'features_lstm_units': 4
                }
            }
            train(
                **train_args
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.model_type == 'CustomBidLSTM_CRF_FEATURES'
            assert model_config.features_embedding_size == 4
            assert model_config.features_lstm_units == 4
            tag_input(
                model=default_args['model'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train(
                model=default_args['model'],
                template_path=template_path,
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager'],
                gzip_enabled=False
            )
            wapiti_eval_model(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        @log_on_exception
        def test_should_be_able_to_train_eval_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train_eval(
                model=default_args['model'],
                template_path=template_path,
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        @log_on_exception
        def test_should_be_able_to_train_and_gzip_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train(
                model=default_args['model'],
                template_path=template_path,
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager'],
                gzip_enabled=True
            )
            wapiti_eval_model(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        def test_should_be_able_to_train_eval(self, default_args: dict):
            train_eval(
                **default_args
            )

        def test_should_be_able_to_train_eval_nfold(self, default_args: dict):
            train_eval(
                fold_count=2,
                **default_args
            )

    @pytest.mark.slow
    class TestEndToEndMain:
        @log_on_exception
        def test_should_be_able_capture_train_input_data(
                self, temp_dir: Path):
            input_path = temp_dir.joinpath('input.train')
            input_path.write_text('some training data')

            output_path = temp_dir.joinpath('captured-input.train')

            main([
                'header',
                'train',
                '--input=%s' % input_path,
                '--save-input-to-and-exit=%s' % output_path
            ])

            assert output_path.read_text() == 'some training data'

        @log_on_exception
        def _test_should_be_able_capture_train_input_data_gzipped(
                self, temp_dir: Path):
            input_path = temp_dir.joinpath('input.train')
            input_path.write_text('some training data')

            output_path = temp_dir.joinpath('captured-input.train.gz')

            main([
                'header',
                'train',
                '--input=%s' % input_path,
                '--save-input-to-and-exit=%s' % output_path
            ])

            with gzip.open(str(output_path), mode='rb') as fp:
                assert fp.read() == 'some training data'

        @log_on_exception
        def test_should_be_able_tag_using_existing_grobid_model(
                self, capsys):
            main([
                'tag',
                '--input=%s' % GROBID_HEADER_TEST_DATA_URL,
                '--model-path=%s' % GROBID_HEADER_MODEL_URL,
                '--limit=1',
                '--tag-output-format=xml'
            ])
            captured = capsys.readouterr()
            output_text = captured.out
            LOGGER.debug('output_text: %r', output_text)
            assert output_text
            root = ET.fromstring(output_text)
            title = ' '.join(node.text for node in root.findall('.//title'))
            assert title == GROBID_HEADER_TEST_DATA_TITLE_1

        @log_on_exception
        def test_should_be_able_eval_using_existing_grobid_model(
                self, temp_dir: Path):
            eval_output_path = temp_dir / 'eval.json'
            main([
                'eval',
                '--input=%s' % GROBID_HEADER_TEST_DATA_URL,
                '--model-path=%s' % GROBID_HEADER_MODEL_URL,
                '--limit=100',
                '--eval-output-format=json',
                '--eval-output-path=%s' % eval_output_path
            ])
            eval_data = json.loads(eval_output_path.read_text())
            assert eval_data['scores']['<title>']['f1'] >= 0.5
