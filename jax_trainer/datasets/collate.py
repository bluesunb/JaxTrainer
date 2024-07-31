from functools import partial
from typing import Any, NamedTuple, Sequence, Union, Type

import numpy as np


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Collate function for numpy arrays.

    Args:
        batch:  A batch of numpy arrays. Can be a list of ndarrays or nested list of ndarrays.

    Returns:
        Batch of data as a single ndarray or a list of ndarrays.
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        # batchify the list of single data pairs (e.g. [(x1, y1), (x2, y2), ...])
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def batch_collate(tuple_class: Type[NamedTuple], batch: Sequence[Any]) -> NamedTuple:
    """
    Collate function for NamedTuple classes.
    
    Args:
        tuple_class:    NamedTuple class the batch is collated into.
        batch:          A batch of data samples.

    Returns:
        NamedTuple instance with batch data.
    """
    size = batch[0].shape[0]
    return tuple_class(size, *batch)


def numpy_batch_collate(tuple_class: Type[NamedTuple], batch: Sequence[Any]) -> NamedTuple:
    """
    Collate the batch into a np.ndarray and then into a NamedTuple.
    
    Args:
        tuple_class:    NamedTuple class the batch is collated into.
        batch:          A batch of data samples.

    Returns:
        NamedTuple instance with batch data.
    """
    return batch_collate(tuple_class, numpy_collate(batch))


def build_batch_collate(tuple_class: Type[NamedTuple]):
    """
    Build a batch collate function for a NamedTuple class.

    Args:
        tuple_class:    NamedTuple class the batch is collated into.

    Returns:
        Collate function for the NamedTuple class.
    """
    return partial(batch_collate, tuple_class)
