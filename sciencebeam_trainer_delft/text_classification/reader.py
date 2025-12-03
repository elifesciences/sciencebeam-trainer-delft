from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

from sciencebeam_trainer_delft.text_classification.typing import (
    T_Batch_Text_Array,
    T_Batch_Text_Classes_Array
)
from sciencebeam_trainer_delft.utils.io import auto_uploading_output_file


# mostly copied from:
# https://github.com/kermitt2/delft/blob/v0.2.3/delft/textClassification/reader.py


def get_filepath_csv_separator(filepath: str):
    if filepath.endswith('.tsv') or filepath.endswith('.tsv.gz'):
        return '\t'
    return ','


def load_data_frame(
    filepath: str,
    limit: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    sep = get_filepath_csv_separator(filepath)
    return pd.read_csv(filepath, nrows=limit, sep=sep, **kwargs)


def save_data_frame(
        df: pd.DataFrame,
        filepath: str,
        index: bool = False,
        **kwargs) -> pd.DataFrame:
    sep = get_filepath_csv_separator(filepath)
    with auto_uploading_output_file(filepath, mode='w') as fp:
        return df.to_csv(fp, sep=sep, index=index, **kwargs)


def get_texts_and_classes_from_data_frame(
    df: pd.DataFrame
) -> Tuple[T_Batch_Text_Array, T_Batch_Text_Classes_Array, List[str]]:
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
    df = df.copy()
    df.iloc[:, 1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 1])

    classes = df.iloc[:, 2:]
    classes_list = classes.values.tolist()
    classes_label_names = list(classes.columns.values)

    return np.asarray(texts_list), np.asarray(classes_list), classes_label_names


def load_texts_and_classes_pandas(
    filepath: str,
    limit: int = None,
    **kwargs
) -> Tuple[T_Batch_Text_Array, T_Batch_Text_Classes_Array, List[str]]:
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
    return get_texts_and_classes_from_data_frame(
        load_data_frame(filepath, limit=limit, **kwargs)
    )


def load_classes_pandas(
    filepath: str,
    limit: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, List[str]]:
    """
    Load texts and classes from a file in csv format using pandas dataframe:

    id      class_0     ... class_n
    id_0    class_00    ... class_n0
    id_1    class_01    ... class_n1
    ...
    id_m    class_0m    ... class_nm

    It should support any CSV file format.

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """

    df = load_data_frame(filepath, limit=limit, **kwargs)

    classes = df.iloc[:, 1:]
    classes_list = classes.values.tolist()
    classes_label_names = list(classes.columns.values)

    return np.asarray(classes_list), classes_label_names
