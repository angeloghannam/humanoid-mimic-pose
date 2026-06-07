import numpy as np
from math import acos
from numpy.typing import NDArray


def angle_between(vec1: NDArray, vec2: NDArray) -> float:
    """
    Args:
        vec1 (NDArray)
        vec2 (NDArray)

    Raises:
        ValueError: Vector shape mismatch

    Returns:
        float: Angle between vec1 and vec2 in radians.
    """
    if vec1.shape != vec2.shape:
        raise ValueError(
            f"vector shapes do not match, vector 1 shape: {vec1.shape}, vector 2 shape: {vec2.shape}")

    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    return acos(cos_angle)
