from typing import Iterable, Optional, SupportsIndex

import jax.numpy as jp
import numpy as np
import torch.utils.data as data
from flax.struct import dataclass, field
from ml_collections import ConfigDict

Dataset = data.Dataset | SupportsIndex
DataLoader = data.DataLoader | Iterable
ArrayLike = jp.ndarray | np.ndarray


@dataclass
class DatasetModule:
    """
    Dataset module that wraps a PyTorch Dataset.

    Args:
        config:         Configuration for the dataset module.
        train_set:      Training dataset.
        valid_set:      Validation dataset.
        test_set:       Testing dataset.
        train_loader:   Training data loader.
        valid_loader:   Validation data loader.
        test_loader:    Testing data loader.
        matadata:       Metadata for the dataset module.
    """
    config: ConfigDict
    train_set: Optional[Dataset]
    valid_set: Optional[Dataset]
    test_set: Optional[Dataset]
    train_loader: Optional[DataLoader]
    valid_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    matadata: Optional[dict] = None


@dataclass
class Batch:
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
    
    def __len__(self):
        raise NotImplementedError


@dataclass
class SupervisedBatch(Batch):
    inputs: np.ndarray
    targets: np.ndarray

    def __len__(self):
        return self.inputs.shape[0]