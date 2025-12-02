from typing import List, Union

import numpy as np


T_Batch_Tokens = List[List[str]]
T_Batch_Features = Union[np.ndarray, List[List[List[str]]]]
T_Batch_Labels = List[List[str]]
