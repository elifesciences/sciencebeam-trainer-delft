from typing import Tuple, List

import pandas as pd
import numpy as np


# mostly copied from:
# https://github.com/kermitt2/delft/blob/v0.2.3/delft/textClassification/reader.py


def load_texts_and_classes_pandas(
        filepath: str,
        limit: int = None,
        **kwargs) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    Load texts and classes from a file in csv format using pandas dataframe:

    id      text    class_0     ... class_n
    id_0    text_0  class_00    ... class_n0
    id_1    text_1  class_01    ... class_n1
    ...
    id_m    text_m  class_0m    ... class_nm

    It should support any CSV file format.

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """

    df = pd.read_csv(filepath, nrows=limit, **kwargs)
    df.iloc[:, 1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 1])

    classes = df.iloc[:, 2:]
    classes_list = classes.values.tolist()
    classes_label_names = list(classes.columns.values)

    return np.asarray(texts_list), np.asarray(classes_list), classes_label_names
