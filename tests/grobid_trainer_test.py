import json
import os
from functools import partial
from unittest.mock import patch

import pytest
from py._path.local import LocalPath

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
def _embedding_registry_path(tmpdir: LocalPath):
    return tmpdir.join('embedding-registry.json')


@pytest.fixture(name='embedding_registry', autouse=True)
def _embedding_registry(embedding_registry_path: LocalPath):
    embedding_registry_path.write_text(json.dumps({
        'embedding-lmdb-path': None,
        'embeddings': [EMBEDDING_1],
        'embeddings-contextualized': []
    }, indent=4), encoding='utf-8')


@pytest.fixture(autouse=True)
def _embedding_class(embedding_registry_path: str):
    embedding_class_with_defaults = partial(Embeddings, path=embedding_registry_path)
    target = 'delft.sequenceLabelling.wrapper.Embeddings'
    with patch(target, new=embedding_class_with_defaults) as mock:
        yield mock


class TestGrobidTrainer:
    class TestParseArgs:
        def test_should_require_arguments(self):
            with pytest.raises(SystemExit):
                parse_args([])

    @pytest.mark.slow
    class TestEndToEnd:
        def test_should_be_able_to_train(self, sample_train_file: str):
            train(
                model='header',
                embeddings_name=EMBEDDING_NAME_1,
                input_path=sample_train_file
            )

        def test_should_be_able_to_train_eval(self, sample_train_file: str):
            train_eval(
                model='header',
                embeddings_name=EMBEDDING_NAME_1,
                input_path=sample_train_file
            )

        def test_should_be_able_to_train_eval_nfold(self, sample_train_file: str):
            train_eval(
                model='header',
                embeddings_name=EMBEDDING_NAME_1,
                input_path=sample_train_file,
                fold_count=2
            )
