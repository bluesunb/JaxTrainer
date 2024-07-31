from typing import Callable, Sequence, Union

import numpy as numpy
import PIL
import torch
import torch.utils.data as data
from ml_collections import ConfigDict

from jax_trainer.datasets.collate import numpy_collate


def build_dataloaders(
    *datasets: Sequence[data.Dataset],
    train: Union[bool, Sequence[bool]] = True,
    collate_fn: Callable = numpy_collate,
    config: ConfigDict = ConfigDict(),
):
    """
    Creates dataloaders for JAX for a given datasets.

    Args:
        datasets:   Datasets to create dataloaders for.
        train:      Indicates which datasets for training.
        collate_fn: Function to collate the data.
        config:     Configuration for the dataloaders.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train] * len(datasets)

    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    seed = config.get("seed", 42)

    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=is_train and (num_workers > 0),
            generator=torch.Generator().manual_seed(seed),
        )
        loaders.append(loader)

    return loaders
