import logging
from collections import Counter
from itertools import zip_longest
from typing import List, Optional, Tuple

import numpy as np

from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform import (
    DatasetTransformer
)

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array_Or_List,
    T_Batch_Token_Array_Or_List,
    T_Batch_Token_Type_Var,
    T_Optional_Batch_Features_Type_Var,
    T_Optional_Batch_Label_Type_Var,
    T_Optional_Batch_Token_Type_Var
)


LOGGER = logging.getLogger(__name__)


NBSP = '\u00A0'


class LineStatus:
    # replicate line status used in GROBID
    LINESTART = 'LINESTART'
    LINEIN = 'LINEIN'
    LINEEND = 'LINEEND'


def strip_tag_prefix(tag: Optional[str]) -> str:
    if tag and (tag.startswith('B-') or tag.startswith('I-')):
        return tag[2:]
    return tag or ''


def get_next_transform_token_y(token_y: str) -> str:
    if token_y and token_y.startswith('B-'):
        return 'I-' + token_y[2:]
    return token_y


def inverse_transform_token_y(unrolled_token_y: List[str], previous_tag: Optional[str]) -> str:
    tags_with_stripped_prefix = [strip_tag_prefix(tag) for tag in unrolled_token_y]
    tag_counts = Counter(tags_with_stripped_prefix)
    top_tag = tag_counts.most_common(1)[0][0]
    LOGGER.debug('tag_counts: %s, top_tag=%r', tag_counts, top_tag)
    if f'B-{top_tag}' in unrolled_token_y or f'I-{top_tag}' in unrolled_token_y:
        if top_tag != strip_tag_prefix(previous_tag):
            return f'B-{top_tag}'
        return f'I-{top_tag}'
    return top_tag


def get_line_status(
    token_index: int,
    line_length: int
):
    if token_index == 0:
        return LineStatus.LINESTART
    if token_index == line_length - 1:
        return LineStatus.LINEEND
    return LineStatus.LINEIN


def get_transformed_features(
    token_features: List[str],
    unrolled_token_index: int,
    unrolled_tokens_length: int,
    line_status_enabled: bool = True
):
    if not line_status_enabled:
        return token_features
    return list(token_features) + [get_line_status(unrolled_token_index, unrolled_tokens_length)]


class UnrollingTextFeatureDatasetTransformer(DatasetTransformer):
    def __init__(
        self,
        unroll_text_feature_index: int,
        used_features_indices: Optional[List[int]] = None
    ):
        # Note: used_features_indices is used to determine, whether to add the line status
        #   (i.e. no need to add it if it is not used)
        self.unroll_text_feature_index = unroll_text_feature_index
        self.used_features_indices = used_features_indices
        self._saved_x: Optional[T_Batch_Token_Array_Or_List] = None
        self._saved_features: Optional[T_Batch_Features_Array_Or_List] = None
        self._unrolled_token_lengths: Optional[List[List[int]]] = None

    def tokenize(self, text: str) -> List[str]:
        return tokenizeAndFilterSimple(text.replace(NBSP, ' '))

    def fit_transform(
        self,
        x: T_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        assert features is not None
        x_transformed = []
        _y_transformed = []
        features_transformed = []
        line_status_enabled: Optional[bool] = None
        unrolled_token_lengths = []
        for y_doc, features_doc in zip_longest(
            y if y is not None else [],  # type: ignore
            features,  # type: ignore
            fillvalue=[]
        ):
            x_doc_transformed = []
            y_doc_transformed = []
            features_doc_transformed = []
            unrolled_token_lengths_doc = []
            for features_row, y_row in zip_longest(features_doc, y_doc, fillvalue=None):
                text = features_row[self.unroll_text_feature_index]
                if line_status_enabled is None:
                    line_status_enabled = (
                        self.used_features_indices is not None
                        and len(features_row) in self.used_features_indices
                    )
                tokens = self.tokenize(text)
                assert tokens
                assert y is None or y_row is not None
                tokens_length = len(tokens)
                for unrolled_token_index, token in enumerate(tokens):
                    x_doc_transformed.append(token)
                    y_doc_transformed.append(y_row)
                    features_doc_transformed.append(
                        get_transformed_features(
                            features_row,
                            unrolled_token_index=unrolled_token_index,
                            unrolled_tokens_length=tokens_length,
                            line_status_enabled=line_status_enabled
                        )
                    )
                    y_row = get_next_transform_token_y(y_row)
                unrolled_token_lengths_doc.append(tokens_length)
            x_transformed.append(x_doc_transformed)
            _y_transformed.append(y_doc_transformed)
            features_transformed.append(features_doc_transformed)
            unrolled_token_lengths.append(unrolled_token_lengths_doc)
        LOGGER.debug('x_transformed: %s', x_transformed)
        LOGGER.debug('y_transformed: %s', _y_transformed)
        LOGGER.debug('features_transformed: %s', features_transformed)
        y_transformed = _y_transformed if y is not None else None
        self._saved_x = x
        self._saved_features = features
        self._unrolled_token_lengths = unrolled_token_lengths
        # Note: convert back to ndarray of object to match input type
        if isinstance(x, np.ndarray):
            x_transformed = np.asarray(x_transformed, dtype='object')  # type: ignore
        if isinstance(y, np.ndarray):
            y_transformed = np.asarray(y_transformed, dtype='object')  # type: ignore
        if isinstance(features, np.ndarray):
            features_transformed = np.asarray(features_transformed, dtype='object')  # type: ignore
        return x_transformed, y_transformed, features_transformed  # type: ignore

    def inverse_transform(
        self,
        x: T_Optional_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Optional_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        if x is not None:
            x = self._saved_x  # type: ignore
        if features is not None:
            features = self._saved_features  # type: ignore
        inverse_transformed_y = None
        if y is not None:
            inverse_transformed_y = []
            assert self._saved_x is not None
            assert self._saved_features is not None
            assert self._unrolled_token_lengths is not None
            for x_doc, features_doc, y_doc, unrolled_token_lengths_doc in zip(  # type: ignore
                self._saved_x, self._saved_features, y, self._unrolled_token_lengths  # type: ignore
            ):
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug('unrolled_token_lengths_doc: %s', unrolled_token_lengths_doc)
                    LOGGER.debug('y_doc: %s', y_doc)
                    LOGGER.debug('xy_doc: %s', list(zip(x_doc, y_doc)))
                index = 0
                inverse_transformed_y_doc = []
                previous_tag = None
                for x_token, features_token, unrolled_token_length in zip(
                    x_doc, features_doc, unrolled_token_lengths_doc
                ):
                    if index >= len(y_doc):
                        # y_doc may be truncated using max sequence length
                        break
                    y_tokens = y_doc[index:index + unrolled_token_length]
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        tokens = self.tokenize(features_token[self.unroll_text_feature_index])
                        LOGGER.debug(
                            'inverse transforming: indices=[%d:%d], x=%r, f=%r, tokens_y=%r',
                            index, index + unrolled_token_length,
                            x_token, features_token, list(zip_longest(tokens, y_tokens))
                        )
                    y_token = inverse_transform_token_y(y_tokens, previous_tag=previous_tag)
                    previous_tag = y_token
                    inverse_transformed_y_doc.append(y_token)
                    index += unrolled_token_length
                inverse_transformed_y.append(inverse_transformed_y_doc)
        if isinstance(y, np.ndarray):
            # convert to ndarray of object to match input type
            inverse_transformed_y_array = np.asarray(inverse_transformed_y, dtype='object')
            return x, inverse_transformed_y_array, features  # type: ignore
        return x, inverse_transformed_y, features  # type: ignore
