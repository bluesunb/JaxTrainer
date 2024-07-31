import os
from typing import Dict, Tuple

import jax.tree_util as jtr
from ml_collections import ConfigDict
from flax.traverse_util import flatten_dict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from jax_trainer.utils import class_to_name


def flatten_configdict(cfg: ConfigDict, sep: str = ".") -> Dict:
    return flatten_dict(cfg.to_dict(), sep=sep)


def get_logging_dir(logger_config: ConfigDict, full_config: ConfigDict) -> Tuple[str, str]:
    """
    Returns the logging directory and its version.

    Args:
        logger_config: Configuration for the logger.
        full_config:   Full configuration.

    Returns:
        (Tuple[str, str]): logging directory and version.
    """
    log_dir = logger_config.get("log_dir", None)
    if log_dir == "None":
        log_dir = None

    # If log_dir is not provided, create a log_dir based on the model name
    if not log_dir:
        base_log_dir = logger_config.get("base_log_dir", "checkpoints/")
        model_name = logger_config.get("model_log_dir", None)

        # Make default model name from the model class
        if model_name is None:
            model_name = full_config.model.name
            if not isinstance(model_name, str):
                model_name = model_name.__name__
            model_name = model_name.split(".")[-1]

        log_dir = os.path.join(base_log_dir, model_name)
        version = None
        if logger_config.get("logger_name", None) is not None and logger_config.logger_name != "":
            log_dir = os.path.join(log_dir, logger_config.logger_name)
            version = ""

    else:
        version = ""

    return log_dir, version


def build_tool_logger(logger_config: ConfigDict, full_config: ConfigDict) -> TensorBoardLogger | WandbLogger:
    """
    Builds the logger tool, either TensorBoard or Wandb for now.

    Args:
        logger_config:  Configuration for the logger.
        full_config:    Full configuration.

    Returns:
        (TensorBoardLogger | WandbLogger): Logger tool.
    """
    log_dir, version = get_logging_dir(logger_config, full_config)
    logger_type = logger_config.get("tool", "TensorBoard").lower()

    if logger_type == "tensorboard":
        logger = TensorBoardLogger(save_dir=log_dir, version=version, name="")
        hparams = flatten_configdict(full_config)
        hparams = jtr.tree_map(class_to_name, hparams)  # Convert classes to strings
        logger.log_hyperparams(hparams)

    elif logger_type == "wandb":
        logger = WandbLogger(
            name=logger_config.get("project_name", None),
            save_dir=log_dir,
            version=version,
            config=full_config,
        )

    else:
        raise NotImplementedError(f"Logger type {logger_type} is not implemented.")

    return logger
