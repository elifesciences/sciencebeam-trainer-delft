from collections import Counter, defaultdict, OrderedDict
from typing import Dict, Iterable, List

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Token_Array
)


def iter_flat_batch_tokens(batch_tokens: T_Batch_Token_Array):
    return (
        token
        for doc_tokens in batch_tokens
        for token in doc_tokens
    )


def iter_flat_features(features: T_Batch_Features_Array):
    return (
        features_vector
        for features_doc in features
        for features_vector in features_doc
    )


def get_quantiles(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values))
    return OrderedDict([
        ('q.00', np.quantile(arr, 0)),
        ('q.25', np.quantile(arr, 0.25)),
        ('q.50', np.quantile(arr, 0.50)),
        ('q.75', np.quantile(arr, 0.75)),
        ('q1.0', np.quantile(arr, 1))
    ])


def get_quantiles_feature_value_length_by_index(features: np.ndarray):
    return dict(enumerate(map(
        lambda feature_values: get_quantiles(map(len, feature_values)),
        zip(*iter_flat_features(features))
    )))


def get_feature_value_counts_by_index(features: np.ndarray, max_feature_values: int = 1000):
    feature_value_counts_by_index: Dict[int, Counter] = defaultdict(Counter)
    for feature_vector in iter_flat_features(features):
        for index, value in enumerate(feature_vector):
            feature_value_counts = feature_value_counts_by_index[index]
            if (
                    len(feature_value_counts) >= max_feature_values
                    and value not in feature_value_counts):
                continue
            feature_value_counts[value] += 1
    return feature_value_counts_by_index


def get_feature_counts(features: np.ndarray):
    feature_value_counts_by_index = get_feature_value_counts_by_index(features)
    return OrderedDict([
        (index, len(feature_value_counts_by_index[index]))
        for index in sorted(feature_value_counts_by_index.keys())
    ])


def get_suggested_feature_indices(feature_counts: Dict[int, int], threshold: int = 12):
    return [
        index
        for index in sorted(feature_counts.keys())
        if feature_counts[index] <= threshold
    ]


def format_dict(d: dict) -> str:
    return '{' + ', '.join([
        '%s: %s' % (
            repr(key),
            format_dict(value) if isinstance(value, dict) else repr(value)
        )
        for key, value in d.items()
    ]) + '}'


def iter_index_groups(indices: List[int]) -> Iterable[List[int]]:
    group: List[int] = []
    for index in indices:
        if not group or group[-1] + 1 == index:
            group.append(index)
            continue
        yield group
        group = []
    if group:
        yield group


def iter_formatted_index_groups(indices: List[int]) -> Iterable[str]:
    for group in iter_index_groups(indices):
        if len(group) == 1:
            yield str(group[0])
            continue
        yield '%s-%s' % (group[0], group[-1])


def format_indices(indices: List[int]) -> str:
    return ','.join(list(iter_formatted_index_groups(indices)))
