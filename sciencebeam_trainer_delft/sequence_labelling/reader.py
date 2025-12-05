import logging
import re
from itertools import islice
from typing import Iterable, List, Tuple

import numpy as np

from delft.sequenceLabelling.reader import _translate_tags_grobid_to_IOB


LOGGER = logging.getLogger(__name__)


# partially copied from delft/sequenceLabelling/reader.py

def iter_load_data_and_labels_crf_lines(
    lines: Iterable[str]
) -> Iterable[Tuple[List[str], List[str], List[List[str]]]]:
    tokens: List[str] = []
    tags: List[str] = []
    features: List[List[str]] = []
    for line in lines:
        line = line.strip()
        LOGGER.debug('line: %s', line)
        if not line:
            if tokens:
                yield tokens, tags, features
                tokens, tags, features = [], [], []
        else:
            pieces = re.split(' |\t', line)
            token = pieces[0]
            tag = pieces[len(pieces)-1]
            localFeatures = pieces[1:len(pieces)-1]
            tokens.append(token)
            tags.append(_translate_tags_grobid_to_IOB(tag))
            features.append(localFeatures)
    if tokens:
        yield tokens, tags, features


def iter_load_data_crf_lines(
    lines: Iterable[str]
) -> Iterable[Tuple[List[str], List[List[str]]]]:
    tokens: List[str] = []
    features: List[List[str]] = []
    for line in lines:
        line = line.strip()
        LOGGER.debug('line: %s', line)
        if not line:
            if tokens:
                yield tokens, features
                tokens, features = [], []
        else:
            pieces = re.split(' |\t', line)
            token = pieces[0]
            localFeatures = pieces[1:]
            tokens.append(token)
            features.append(localFeatures)
    if tokens:
        yield tokens, features


def load_data_and_labels_crf_lines(
    lines: Iterable[str],
    limit: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data, features and label from a CRF matrix string
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n label_0
    token_1 f1_0 f1_1 ... f1_n label_1
    ...
    token_m fm_0 fm_1 ... fm_n label_m

    field separator can be either space or tab

    Returns:
        tuple(numpy array, numpy array, numpy array): tokens, labels, features

    """
    sents = []
    labels = []
    featureSets = []
    documents = iter_load_data_and_labels_crf_lines(lines)
    if limit:
        LOGGER.info('limiting training data to: %s', limit)
        documents = islice(documents, limit)
    for tokens, tags, features in documents:
        sents.append(tokens)
        labels.append(tags)
        featureSets.append(features)
    # specifying dtype object can significantly reduce the memory consumption
    # e.g. for features it could be 20 MB instead of 1 GB
    return (
        np.asarray(sents, dtype='object'),
        np.asarray(labels, dtype='object'),
        np.asarray(featureSets, dtype='object')
    )


def load_data_crf_lines(
    lines: Iterable[str],
    limit: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data, features (no label!) from a CRF matrix file
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n
    token_1 f1_0 f1_1 ... f1_n
    ...
    token_m fm_0 fm_1 ... fm_n

    field separator can be either space or tab

    Returns:
        tuple(numpy array, numpy array): tokens, features

    """
    sents = []
    featureSets = []
    documents = iter_load_data_crf_lines(lines)
    if limit:
        LOGGER.info('limiting training data to: %s', limit)
        documents = islice(documents, limit)
    for tokens, features in documents:
        sents.append(tokens)
        featureSets.append(features)
    # specifying dtype object can significantly reduce the memory consumption
    # e.g. for features it could be 20 MB instead of 1 GB
    return (
        np.asarray(sents, dtype='object'),
        np.asarray(featureSets, dtype='object')
    )


def load_data_and_labels_crf_file(
    filepath: str,
    limit: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with open(filepath, 'r', encoding='utf-8') as fp:
            return load_data_and_labels_crf_lines(fp, limit=limit)
    except Exception as exc:
        raise RuntimeError('failed to read file %r' % filepath) from exc


def load_data_crf_string(
    crf_string: str,
    limit: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    return load_data_crf_lines(crf_string.splitlines(), limit=limit)
