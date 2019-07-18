from typing import List

import numpy as np


def concatenate_or_none(arrays: List[np.array], **kwargs) -> np.array:
    if arrays[0] is None:
        return None
    return np.concatenate(arrays, **kwargs)
