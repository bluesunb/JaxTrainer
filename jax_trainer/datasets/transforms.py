from functools import partial
from typing import Any, Union

import numpy as np


def image_to_numpy(img: Any):
    """
    Convert an image to a numpy array.

    Args:
        img: Image to convert.

    Returns:
        np.ndarray: Converted image.
    """
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def normalize_transform(mean: Union[np.ndarray, float] = 0.0, std: Union[np.ndarray, float] = 1.0):
    """
    Normalize the input image.

    Args:
        mean: Mean value for normalization.
        std: Standard deviation value for normalization.

    Returns:
        Callable: Normalization function.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return partial(normalize, mean=mean, std=std)


def normalize(x: np.ndarray, mean: np.ndarray = 0.0, std: np.ndarray = 1.0):
    """
    Normalize the input image.

    Args:
        x:      Image to normalize.
        mean:   Mean value for normalization.
        std:    Standard deviation value for normalization.

    Returns:
        np.ndarray: Normalized image.
    """
    return (x - mean.astype(x.dtype)) / std.astype(x.dtype)
