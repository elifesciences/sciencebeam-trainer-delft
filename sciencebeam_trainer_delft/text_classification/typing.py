import numpy as np
import numpy.typing as npt


T_Batch_Text_Array = npt.NDArray[np.object_]       # shape: (batch)
T_Batch_Text_Classes_Array = npt.NDArray[np.object_]       # shape: (batch, num_classes)
