from typing import List, Union

import numpy as np


T_Batch_Tokens = Union[List[List[str]], np.ndarray]
T_Batch_Features = Union[List[List[List[str]]], np.ndarray]
T_Batch_Labels = Union[List[List[str]], np.ndarray]
