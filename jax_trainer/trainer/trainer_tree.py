import json
import os
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import flax
import flax.struct
import jax
import jax.numpy as jp
import optax
import yaml
from absl import logging
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

from ml_collections import ConfigDict, FrozenConfigDict
from tabulate import tabulate as py_tabulate
from tqdm.auto import tqdm

from jax_trainer import callbacks
from jax_trainer.callbacks import BaseCallback, ModelCheckpoint
from jax_trainer.datasets import Batch, DatasetModule
from jax_trainer.logger import (
    HostMetrics,
    ImmutableMetrics,
    LogFreq,
    LogMetricMode,
    LogStage,
    Logger,
    load_pytree,
    save_pytree,
    update_metrics
)
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.utils import class_to_name, resolve_import
from jax_trainer.trainer.train_state import TrainState

nonpytree_field = partial(flax.struct.field, pytree_node=False)     # consider to be leaf

class Trainer(flax.struct.PyTreeNode):
    trainer_config: ConfigDict
    model_config: ConfigDict
    optimizer_config: ConfigDict
    sample_input: Batch
    
    state: TrainState = None
    logger: Logger = nonpytree_field()
    callbacks: List[BaseCallback] = nonpytree_field()
    train_step_callbacks: List[BaseCallback] = nonpytree_field()

    @classmethod
    def create(
        cls,
        trainer_config: ConfigDict,
        model_config: ConfigDict,
        optimizer_config: ConfigDict,
        sample_input: Batch,
    ):
        
        def build_model(self, model_config: ConfigDict):
            model_class = resolve_import(model_config.name)
            hparams = model_config.get('hparams', {})
            return model_class(hparams)
        
        def build_logger(self, logger_config: ConfigDict):
            full_config = ConfigDict(
                {"trainer": trainer_config, 
                 "model": model_config, 
                 "optimizer": optimizer_config}
            )
            logger_class = resolve_import(logger_config.get('name', Logger))
            return logger_class(logger_config, full_config)
        
        def build_callbacks(self):
            callbacks = []
            train_step_callbacks = []
            callback_configs = trainer_config.get('callbacks', ConfigDict())

            for name in callback_configs:
                logging.info(f"Initializing callback: {name}")
                callback_cfg = callback_configs[name]
                if callback_cfg.get("name", None) is not None:
                    callback_class = resolve_import(callback_cfg.name)
                elif hasattr(callbacks, name):
                    callback_class = getattr(callbacks, name)
                else:
                    raise ValueError(f"Callback {name} not found.")
                
                callback = callback_class(config=callback_cfg, trainer=self, data_module=None)