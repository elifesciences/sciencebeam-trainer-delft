from typing import List, Optional, Sequence

import numpy as np


def concatenate_or_none(
    arrays: Sequence[Optional[np.ndarray]],
    **kwargs
) -> Optional[np.ndarray]:
    if arrays[0] is None:
        return None
    return np.concatenate(arrays, **kwargs)  # type: ignore


# https://stackoverflow.com/a/51526109/8676953
def shuffle_arrays(arrays: List[np.ndarray], random_seed: int = None):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    random_seed : Seed value if not None, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    if random_seed is None:
        random_seed = np.random.randint(0, 2**(32 - 1) - 1)

    for arr in arrays:
        rstate = np.random.RandomState(random_seed)  # pylint: disable=no-member
        rstate.shuffle(arr)
