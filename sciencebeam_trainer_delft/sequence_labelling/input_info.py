from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List

import numpy as np


def iter_flat_features(features: np.array):
    return (
        features_vector
        for features_doc in features
        for features_vector in features_doc
    )


def get_feature_value_counts_by_index(features: np.array, max_feature_values: int = 1000):
    feature_value_counts_by_index = defaultdict(Counter)
    for feature_vector in iter_flat_features(features):
        for index, value in enumerate(feature_vector):
            feature_value_counts = feature_value_counts_by_index[index]
            if (
                    len(feature_value_counts) >= max_feature_values
                    and value not in feature_value_counts):
                continue
            feature_value_counts[value] += 1
    return feature_value_counts_by_index


def get_feature_counts(features: np.array):
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
    return ', '.join([
        '%s: %s' % (key, value)
        for key, value in d.items()
    ])


def iter_index_groups(indices: List[int]) -> str:
    group = []
    for index in indices:
        if not group or group[-1] + 1 == index:
            group.append(index)
            continue
        yield group
        group = []
    if group:
        yield group


def iter_formatted_index_groups(indices: List[int]) -> str:
    group = []
    for group in iter_index_groups(indices):
        if len(group) == 1:
            yield str(group[0])
            continue
        yield '%s-%s' % (group[0], group[-1])


def format_indices(indices: List[int]) -> str:
    return ','.join(list(iter_formatted_index_groups(indices)))
