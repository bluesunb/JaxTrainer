from typing import Iterable, Optional, SupportsIndex

import jax.numpy as jp
import numpy as np
import torch.utils.data as data
from flax.struct import dataclass
from ml_collections import ConfigDict

Dataset = data.Dataset | SupportsIndex
DataLoader = data.DataLoader | Iterable
ArrayLike = jp.ndarray | np.ndarray


@dataclass
class DatasetModule:
    """
    Dataset module that wraps a PyTorch Dataset.
    """
    config: ConfigDict
    train: Optional[Dataset]
    valid: Optional[Dataset]
    test: Optional[Dataset]
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    matadata: Optional[dict] = None


@dataclass
class Batch:
    size: int

    def __getitem__(self, key):
        vals = {}
        if isinstance(key, int):
            vals["size"] = 1
        for k, v in self.__dict__.items():
            if k == "size":
                continue
            if isinstance(v, Iterable):
                vals[k] = v[key]
                if "size" not in vals:
                    vals["size"] = vals[k].shape[0] if isinstance(vals[k], ArrayLike) else len(vals[k])
            else:
                vals[k] = v
        return self.__class__(**vals)


@dataclass
class SupervisedBatch(Batch):
    inputs: np.ndarray
    targets: np.ndarray
