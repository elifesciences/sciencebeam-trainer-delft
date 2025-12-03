from typing import Sequence

import numpy as np
import numpy.typing as npt


T_Batch_Tokens = npt.NDArray[np.object_]       # shape: (batch, seq_len)
T_Batch_Labels = npt.NDArray[np.object_]       # shape: (batch, seq_len)
T_Batch_Features = npt.NDArray[np.float32]  # shape: (batch, seq_len, feat_dim)

T_Document_Label_List = Sequence[str]  # shape: (seq_len)
T_Batch_Label_List = Sequence[T_Document_Label_List]  # shape: (batch, seq_len)
