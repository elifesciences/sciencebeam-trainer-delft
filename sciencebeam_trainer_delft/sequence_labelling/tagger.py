import datetime
import logging
from typing import Iterable, List, Tuple, Union

import numpy as np

from delft.utilities.Embeddings import Embeddings
from delft.utilities.Tokenizer import tokenizeAndFilter

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor
from sciencebeam_trainer_delft.sequence_labelling.data_generator import DataGenerator


LOGGER = logging.getLogger(__name__)


def iter_batch_window_indices_and_offsets(
        data_generator: DataGenerator) -> Iterable[List[Tuple[int, int]]]:
    return (
        data_generator.get_batch_window_indices_and_offsets(batch_index)
        for batch_index in range(len(data_generator))
    )


def predict_texts_with_sliding_window_if_enabled(
        texts: List[Union[str, List[str]]],
        model_config: ModelConfig,
        preprocessor: Preprocessor,
        max_sequence_length: int,
        model,
        embeddings: Embeddings = None,
        features: List[List[List[str]]] = None):
    should_tokenize = (
        len(texts) > 0  # pylint: disable=len-as-condition
        and isinstance(texts[0], str)
    )

    if not should_tokenize and max_sequence_length:
        max_actual_sequence_length = max(len(text) for text in texts)
        if max_actual_sequence_length <= max_sequence_length:
            LOGGER.info(
                'all text sequences below max sequence length: %d <= %d',
                max_actual_sequence_length, max_sequence_length
            )
        elif model_config.stateful:
            LOGGER.info(
                'some text sequences exceed max sequence length (using sliding windows): %d > %d',
                max_actual_sequence_length, max_sequence_length
            )
        else:
            LOGGER.info(
                (
                    'some text sequences exceed max sequence length'
                    ' (truncate, model is not stateful): %d > %d'
                ),
                max_actual_sequence_length, max_sequence_length
            )

    predict_generator = DataGenerator(
        texts,
        None,
        batch_size=model_config.batch_size,
        preprocessor=preprocessor,
        additional_token_feature_indices=model_config.additional_token_feature_indices,
        text_feature_indices=model_config.text_feature_indices,
        concatenated_embeddings_token_count=(
            model_config.concatenated_embeddings_token_count
        ),
        char_embed_size=model_config.char_embedding_size,
        is_deprecated_padded_batch_text_list_enabled=(
            model_config.is_deprecated_padded_batch_text_list_enabled
        ),
        max_sequence_length=max_sequence_length,
        input_window_stride=(
            max_sequence_length if model_config.stateful
            else None
        ),
        stateful=model_config.stateful,
        embeddings=embeddings,
        tokenize=should_tokenize,
        shuffle=False,
        features=features,
        name='predict_generator'
    )

    prediction_list_list = [[] for _ in texts]
    batch_window_indices_and_offsets_iterable = iter_batch_window_indices_and_offsets(
        predict_generator
    )
    for batch_window_indices_and_offsets in batch_window_indices_and_offsets_iterable:
        LOGGER.debug(
            'predict batch_window_indices_and_offsets: %s',
            batch_window_indices_and_offsets
        )
        generator_output = predict_generator.get_window_batch_data(
            batch_window_indices_and_offsets
        )
        LOGGER.debug('predict on batch: %d', len(batch_window_indices_and_offsets))
        batch_predictions = model.predict_on_batch(generator_output[0])
        LOGGER.debug('preds.shape: %s', batch_predictions.shape)
        for window_indices_and_offsets, seq_predictions in zip(
            batch_window_indices_and_offsets, batch_predictions
        ):
            text_index, text_offset = window_indices_and_offsets
            LOGGER.debug(
                'prediction_list_list[%d]: %s',
                text_index,
                prediction_list_list[text_index]
            )
            current_offset = sum((len(a) for a in prediction_list_list[text_index]))
            assert (
                current_offset == text_offset
            ), "expected %d to be %d" % (
                current_offset, text_offset
            )
            prediction_list_list[text_index].append(seq_predictions)

    preds_concatenated_list = [
        np.concatenate(
            prediction_list,
            axis=0
        )
        for prediction_list in prediction_list_list
    ]
    LOGGER.debug('preds_concatenated_list: %s', preds_concatenated_list)
    return preds_concatenated_list


class Tagger:
    def __init__(
            self,
            model,
            model_config,
            embeddings=None,
            preprocessor=None,
            max_sequence_length: int = None):
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings
        self.max_sequence_length = max_sequence_length

    def tag(self, texts, output_format, features=None):
        assert isinstance(texts, list)

        if output_format == 'json':
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model.config.model_name,
                "texts": []
            }
        else:
            list_of_tags = []

        preds_concatenated_list = predict_texts_with_sliding_window_if_enabled(
            texts=texts,
            features=features,
            model=self.model,
            model_config=self.model_config,
            preprocessor=self.preprocessor,
            max_sequence_length=self.max_sequence_length,
            embeddings=self.embeddings
        )
        for i, pred_item in enumerate(preds_concatenated_list):
            LOGGER.debug('pred_item.shape: %s', pred_item.shape)

            pred = [pred_item]
            text = texts[i]

            if isinstance(text, str):
                tokens, offsets = tokenizeAndFilter(text)
            else:
                # it is a list of string, so a string already tokenized
                # note that in this case, offset are not present and json output is impossible
                tokens = text
                offsets = []

            tags = self._get_tags(pred)
            LOGGER.debug('tags: %s', tags)
            prob = self._get_prob(pred)

            if output_format == 'json':
                piece = {}
                piece["text"] = text
                piece["entities"] = self._build_json_response(
                    tokens, tags, prob, offsets
                )["entities"]
                res["texts"].append(piece)
            else:
                the_tags = list(zip(tokens, tags))
                list_of_tags.append(the_tags)

        if output_format == 'json':
            return res
        else:
            return list_of_tags

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_json_response(self, tokens, tags, prob, offsets):
        res = {
            "entities": []
        }
        chunks = get_entities_with_offsets(tags, offsets)
        LOGGER.debug('entity chunks: %s', chunks)
        for chunk_type, chunk_start, chunk_end, pos_start, pos_end in chunks:
            # TODO: get the original string rather than regenerating it from tokens
            entity = {
                "text": ' '.join(tokens[chunk_start: chunk_end]),
                "class": chunk_type,
                "score": float(np.average(prob[chunk_start:chunk_end])),
                "beginOffset": pos_start,
                "endOffset": pos_end
            }
            res["entities"].append(entity)

        return res


def get_entities_with_offsets(seq, offsets):
    """
    Gets entities from sequence

    Args:
        seq (list): sequence of labels.
        offsets (list of integer pair): sequence of offset position

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> offsets = [(0,10), (11, 15), (16, 29), (30, 41)]
        >>> print(get_entities(seq))
        [('PER', 0, 2, 0, 15), ('LOC', 3, 4, 30, 41)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    max_length = min(len(seq)-1, len(offsets))
    while i < max_length:
        if seq[i].startswith('B'):
            # if we are at the end of the offsets, we can stop immediatly
            j = max_length
            if i+2 != max_length:
                for j in range(i+1, max_length):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
            start_pos = offsets[i][0]
            end_pos = offsets[j-1][1]-1
            chunks.append((types[i], i, j, start_pos, end_pos))
            i = j
        else:
            i += 1
    return chunks
