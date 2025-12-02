from typing import List, Union

import numpy as np
import numpy.typing as npt


T_Batch_Tokens = List[List[str]]
T_Batch_Features = npt.NDArray[np.float32]  # shape: (batch, seq_len, feat_dim)
T_Batch_Labels = List[List[str]]
