import json
import logging
import os
from functools import partial
from pathlib import Path
from unittest.mock import call, patch, MagicMock
from typing import List

import pytest

import numpy as np

from delft.utilities.Embeddings import Embeddings

from sciencebeam_trainer_delft.sequence_labelling.wrapper import get_model_directory
import sciencebeam_trainer_delft.sequence_labelling.grobid_trainer as grobid_trainer_module
from sciencebeam_trainer_delft.sequence_labelling.grobid_trainer import (
    parse_args,
    load_data_and_labels,
    train,
    train_eval,
    tag_input
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
        embedding_registry_path: Path):
    download_manager = MagicMock(name='download_manager')
    download_manager.download_if_url.return_value = str(sample_train_file)
    return dict(
        model='header',
        embeddings_name=EMBEDDING_NAME_1,
        input_paths=[sample_train_file],
        download_manager=download_manager,
        output_path=str(model_base_path),
        architecture='CustomBidLSTM_CRF',
        word_lstm_units=11,
        feature_indices=list(range(7, 1 + 10)),
        embedding_registry_path=str(embedding_registry_path)
    )


@pytest.fixture(name='default_model_directory')
def _default_model_directory(default_args: dict):
    return get_model_directory(
        model_name=default_args['model'],
        dir_path=default_args['output_path']
    )


class TestGrobidTrainer:
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

    @pytest.mark.usefixtures(
        'get_default_training_data_mock', 'load_data_and_labels_crf_file_mock'
    )
    class TestLoadDataAndLabels:
        def test_should_load_using_default_dataset(
                self,
                get_default_training_data_mock: MagicMock,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
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
                download_manager_mock.download_if_url.return_value,
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
                download_manager_mock.download_if_url.return_value,
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
    class TestEndToEnd:
        @log_on_exception
        def test_should_be_able_to_train_without_features(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=False,
                **default_args
            )
            tag_input(
                use_features=False,
                model_path=default_model_directory,
                **default_args
            )

        @log_on_exception
        def test_should_be_able_to_train_with_features(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=True,
                **default_args
            )
            tag_input(
                use_features=True,
                model_path=default_model_directory,
                **default_args
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
