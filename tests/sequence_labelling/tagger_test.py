import logging
from unittest.mock import MagicMock
from typing import Dict, List

import pytest

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    PAD,
    WordPreprocessor
)

from sciencebeam_trainer_delft.sequence_labelling.tagger import (
    Tagger
)


LOGGER = logging.getLogger(__name__)


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'
ALL_TOKENS = [TOKEN_1, TOKEN_2, TOKEN_3]

TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'
ALL_TAGS = [TAG_1, TAG_2, TAG_3]

DEFAULT_TAG_BY_TOKEN_MAP = dict(zip([''] + ALL_TOKENS, [PAD] + ALL_TAGS))


@pytest.fixture(name='model_mock')
def _model_mock():
    return MagicMock(name='model')


@pytest.fixture(name='model_config')
def _model_config():
    return ModelConfig(batch_size=1)


def get_preprocessor(
        tokens: List[str],
        tags: List[str],
        **kwargs) -> WordPreprocessor:
    preprocessor = WordPreprocessor(**kwargs)
    preprocessor.fit([tokens], [tags])
    return preprocessor


@pytest.fixture(name='preprocessor')
def _preprocessor() -> WordPreprocessor:
    return get_preprocessor(tokens=ALL_TOKENS, tags=ALL_TAGS)


@pytest.fixture(name='preprocessor_mock')
def _preprocessor_mock():
    preprocessor_mock = MagicMock(name='preprocessor')
    preprocessor_mock.return_features = False
    return preprocessor_mock


def inverse_char_transform(
        char_indices: List[List[List[int]]],
        char_by_index_map: Dict[int, str]):
    LOGGER.debug('char_by_index_map: %s', char_by_index_map)
    return [
        [
            ''.join([
                char_by_index_map[char_index]
                for char_index in seq
                if char_index
            ])
            for seq in doc
        ]
        for doc in char_indices
    ]


def get_prediction_by_tag(vocab_tag: Dict[str, int]):
    tag_count = len(vocab_tag)
    return {
        tag: [
            1 if i == tag_index else 0
            for i in range(tag_count)
        ]
        for tag, tag_index in vocab_tag.items()
    }


def get_predict_on_batch_by_token_fn(
        tag_by_token_map: Dict[str, List[float]],
        preprocessor: WordPreprocessor,
        batch_size: int = None):
    vocab_tag = preprocessor.vocab_tag
    LOGGER.debug('vocab_tag=%s', vocab_tag)
    char_by_index_map = {i: c for c, i in preprocessor.vocab_char.items()}
    prediction_by_tag = get_prediction_by_tag(vocab_tag)
    LOGGER.debug('prediction_by_tag=%s', prediction_by_tag)

    def predict_on_batch(inputs):
        char_inputs = inputs[1]
        inverse_char_inputs = inverse_char_transform(char_inputs, char_by_index_map)
        if batch_size:
            assert len(char_inputs) == batch_size, (
                "expected batch size: %d, but was %d" % (batch_size, len(char_inputs))
            )
        LOGGER.debug('inverse_char_inputs: %s', inverse_char_inputs)
        return np.asarray([
            [
                prediction_by_tag[tag_by_token_map[token]]
                for token in doc
            ]
            for doc in inverse_char_inputs
        ])
    return predict_on_batch


class TestTagger:
    def test_should_truncate_if_not_stateful(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = False
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2
        )
        model_mock.predict_on_batch.side_effect = get_predict_on_batch_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2)]
        ]

    def test_should_not_truncate_if_stateful(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = True
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2
        )
        model_mock.predict_on_batch.side_effect = get_predict_on_batch_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_not_truncate_without_max_sequence_length(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = False
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=None
        )
        model_mock.predict_on_batch.side_effect = get_predict_on_batch_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_tag_tokenized_texts_with_varying_lengths_and_stateful_sliding_windows(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = True
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2
        )
        model_mock.predict_on_batch.side_effect = get_predict_on_batch_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3],
                [TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)],
            [(TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_tag_tokenized_texts_with_exact_batch_size(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = True
        model_config.batch_size = 2
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2
        )
        model_mock.predict_on_batch.side_effect = get_predict_on_batch_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3],
                [TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)],
            [(TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]
