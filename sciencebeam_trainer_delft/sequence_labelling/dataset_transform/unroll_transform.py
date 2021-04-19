import logging
from collections import Counter
from itertools import zip_longest
from typing import List, Optional

from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform import (
    DatasetTransformer
)

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Tokens,
    T_Batch_Features,
    T_Batch_Labels
)


LOGGER = logging.getLogger(__name__)


NBSP = '\u00A0'


def strip_tag_prefix(tag: str) -> str:
    if tag.startswith('B-') or tag.startswith('I-'):
        return tag[2:]
    return tag


def get_next_transform_token_y(token_y: str) -> str:
    if token_y and token_y.startswith('B-'):
        return 'I-' + token_y[2:]
    return token_y


def inverse_transform_token_y(unrolled_token_y: List[str]) -> str:
    tags_with_stripped_prefix = [strip_tag_prefix(tag) for tag in unrolled_token_y]
    tag_counts = Counter(tags_with_stripped_prefix)
    top_tag = tag_counts.most_common(1)[0][0]
    LOGGER.debug('tag_counts: %s, top_tag=%r', tag_counts, top_tag)
    for prefix in ['B-', 'I-']:
        top_tag_with_prefix = prefix + top_tag
        if top_tag_with_prefix in unrolled_token_y:
            return top_tag_with_prefix
    return top_tag


class UnrollingTextFeatureDatasetTransformer(DatasetTransformer):
    def __init__(self, unroll_text_feature_index: int):
        self.unroll_text_feature_index = unroll_text_feature_index
        self._saved_x: Optional[T_Batch_Tokens] = None
        self._saved_features: Optional[T_Batch_Features] = None
        self._unrolled_token_lengths: Optional[List[List[int]]] = None

    def tokenize(self, text: str) -> List[str]:
        return tokenizeAndFilterSimple(text.replace(NBSP, ' '))

    def fit_transform(
        self,
        x: T_Batch_Tokens,
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ):
        assert features is not None
        x_transformed = []
        y_transformed = []
        features_transformed = []
        unrolled_token_lengths = []
        for y_doc, features_doc in zip_longest(
            y if y is not None else [],
            features,
            fillvalue=[]
        ):
            x_doc_transformed = []
            y_doc_transformed = []
            features_doc_transformed = []
            unrolled_token_lengths_doc = []
            for features_row, y_row in zip_longest(features_doc, y_doc, fillvalue=None):
                text = features_row[self.unroll_text_feature_index]
                tokens = self.tokenize(text)
                for token in tokens:
                    x_doc_transformed.append(token)
                    y_doc_transformed.append(y_row)
                    features_doc_transformed.append(features_row)
                    y_row = get_next_transform_token_y(y_row)
                unrolled_token_lengths_doc.append(len(tokens))
            x_transformed.append(x_doc_transformed)
            y_transformed.append(y_doc_transformed)
            features_transformed.append(features_doc_transformed)
            unrolled_token_lengths.append(unrolled_token_lengths_doc)
        LOGGER.debug('x_transformed: %s', x_transformed)
        LOGGER.debug('y_transformed: %s', y_transformed)
        LOGGER.debug('features_transformed: %s', features_transformed)
        if y is None:
            y_transformed = None
        self._saved_x = x
        self._saved_features = features
        self._unrolled_token_lengths = unrolled_token_lengths
        return x_transformed, y_transformed, features_transformed

    def inverse_transform(
        self,
        x: Optional[T_Batch_Tokens],
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ):
        if x is not None:
            x = self._saved_x
        if features is not None:
            features = self._saved_features
        inverse_transformed_y = None
        if y is not None:
            inverse_transformed_y = []
            for y_doc, unrolled_token_lengths_doc in zip(y, self._unrolled_token_lengths):
                LOGGER.debug('unrolled_token_lengths_doc: %s', unrolled_token_lengths_doc)
                LOGGER.debug('y_doc: %s', y_doc)
                index = 0
                inverse_transformed_y_doc = []
                for unrolled_token_length in unrolled_token_lengths_doc:
                    if index >= len(y_doc):
                        # y_doc may be truncated using max sequence length
                        break
                    inverse_transformed_y_doc.append(
                        inverse_transform_token_y(y_doc[index:index + unrolled_token_length])
                    )
                    index += unrolled_token_length
                inverse_transformed_y.append(inverse_transformed_y_doc)
        return x, inverse_transformed_y, features
