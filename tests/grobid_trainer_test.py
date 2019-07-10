import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from delft.utilities.Embeddings import Embeddings

from sciencebeam_trainer_delft.grobid_trainer import (
    parse_args,
    train,
    train_eval
)

from .test_data import TEST_DATA_PATH


EMBEDDING_NAME_1 = 'embedding1'

EMBEDDING_1 = {
    "name": EMBEDDING_NAME_1,
    "path": os.path.join(TEST_DATA_PATH, 'embedding1.txt'),
    "type": "glove",
    "format": "vec",
    "lang": "en",
    "item": "word"
}


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


@pytest.fixture(name='default_args')
def _default_args(sample_train_file: str):
    download_manager = MagicMock(name='download_manager')
    download_manager.download_if_url.return_value = str(sample_train_file)
    return dict(
        model='header',
        embeddings_name=EMBEDDING_NAME_1,
        input_path=sample_train_file,
        download_manager=download_manager
    )


class TestGrobidTrainer:
    class TestParseArgs:
        def test_should_require_arguments(self):
            with pytest.raises(SystemExit):
                parse_args([])

    @pytest.mark.slow
    class TestEndToEnd:
        def test_should_be_able_to_train_without_features(self, default_args: dict):
            train(
                use_features=False,
                **default_args
            )

        def test_should_be_able_to_train_with_features(self, default_args: dict):
            train(
                use_features=True,
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
