from ml_collections import ConfigDict

from jax_trainer.datasets.data_struct import DatasetModule
from jax_trainer.utils import resolve_import


def build_dataset_module(dataset_config: ConfigDict) -> DatasetModule:
    """
    Builds the dataset module from the given configuration.

    Args:
        dataset_config: Configuration dictionary for the dataset.

    Returns:
        DatasetModule: The dataset module.
    """
    constructor = resolve_import(dataset_config.constructor)
    module = constructor(dataset_config)
    return module
