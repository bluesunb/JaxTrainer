from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jp
import optax
import yaml
from absl import logging
from flax import linen as nn
from flax.core import FrozenDict
from flax.struct import PyTreeNode, field
from ml_collections import ConfigDict

from jax_trainer.callbacks import BaseCallback, TrainingCallback
from jax_trainer.datasets import DatasetModule
from jax_trainer.logger import Logger, save_pytree
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.trainer.train_state import Params, TrainState
from jax_trainer.utils import class_to_name, resolve_import

nonpytree_node = partial(field, pytree_node=False)


class TrainerBase(PyTreeNode):
    """
    Trainer class that controls the
    - training loop
    - evaluation loop
    - testing loop
    - metrics update
    - logging class (Logger)
    - callbacks (checkpointing, monitoring, etc.)
    """
    trainer_config: ConfigDict = nonpytree_node()
    model_config: ConfigDict = nonpytree_node()
    optimizer_config: ConfigDict = nonpytree_node()
    data_module: DatasetModule = nonpytree_node()
    sample_input: jp.ndarray | Dict[str, jp.ndarray]
    global_step: int = nonpytree_node(default=0)

    state: TrainState = field(default=None)
    model: Any | nn.Module = nonpytree_node(default=None)
    logger: Any | Logger = nonpytree_node(default=None)
    callbacks: List[BaseCallback] = nonpytree_node(default_factory=list)
    train_step_callbacks: List[TrainingCallback] = nonpytree_node(default_factory=list)

    train_step: callable = nonpytree_node(default=None)
    eval_step: callable = nonpytree_node(default=None)

    @classmethod
    def create(
        cls,
        trainer_config: ConfigDict,
        model_config: ConfigDict,
        optimizer_config: ConfigDict,
        data_module: DatasetModule,
        sample_input: jp.ndarray | Dict[str, jp.ndarray],
    ) -> "TrainerBase":
        """
        Args:
            trainer_config:     Configuration for the trainer. (config.trainer)
            model_config:       Configuration for the model. (config.model)
            optimizer_config:   Configuration for the gradient transformations (include lr scheduler). (config.optimizer)
            data_module:        Data module for the dataset.
            sample_input:       Sample input for the model.
        """
        trainer = cls(
            trainer_config=trainer_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            data_module=data_module,
            sample_input=sample_input,
        )
        trainer = trainer.replace(
            model=trainer.init_model(),
            logger=trainer.init_logger(),
            state=trainer.init_state(),
        )
        callback, train_step_callbacks = trainer.init_callbacks()
        trainer = trainer.replace(
            callbacks=callback, train_step_callbacks=train_step_callbacks
        )

        train_step, eval_step = trainer._pmap_functions()
        trainer = trainer.replace(
            train_step=train_step,
            eval_step=eval_step,
        )
        return trainer

    def init_model(self) -> nn.Module:
        """
        Instantiate the model from the configuration.

        Returns:
            The model instance.
        """
        model_class = resolve_import(self.model_config._class)
        hparams = self.model_config.get("hparams", {})
        return model_class(**hparams)

    def init_logger(self) -> Logger:
        """
        Instantiate the logger from the configuration.

        Returns:
            The logger instance.
        """
        logger_config = self.trainer_config.get("logger", ConfigDict())
        logger_class = resolve_import(logger_config.get("_class", Logger))
        return logger_class(logger_config, self.full_config)

    def init_callbacks(self) -> Tuple[List[BaseCallback], List[TrainingCallback]]:
        """
        Initialize and categorize the callbacks from the configuration.
        We categorize the callbacks into those that are called on every training steps (`train_step_callbacks`)
        and those that are called on every epoch (`callbacks`).

        Returns:
            Tuple of callbacks and train step callbacks.
        """
        callbacks = []
        train_step_callbacks = []
        callback_config = self.trainer_config.get("callbacks", ConfigDict())

        for name in callback_config:
            logging.info(f"Initializing callback: {name}")
            cfg = callback_config[name]
            if "." not in name:
                name = f"jax_trainer.callbacks.{name}"

            callback_class = resolve_import(name)
            callback = callback_class(config=cfg, trainer=self, data_module=None)

            callbacks.append(callback)
            if isinstance(callback, TrainingCallback):
                train_step_callbacks.append(callback)

        return callbacks, train_step_callbacks

    def init_optimizer(self, num_epochs: int, num_train_step_per_epoch: int) -> optax.GradientTransformation:
        """
        Initializes the optimizer based on the provided configuration.

        Uses the specified optimizer builder class to create the optimizer
        with the given number of epochs and training steps per epoch.

        Args:
            num_epochs:                 The number of epochs for training.
            num_train_step_per_epoch:   The number of training steps per epoch.

        Returns:
            The optimizer instance.
        """
        optim_builder_class = self.optimizer_config.get("builder", OptimizerBuilder)
        optim_builder_class = resolve_import(optim_builder_class)
        builder: OptimizerBuilder = optim_builder_class(self.optimizer_config)
        tx = builder.build_optimizer(num_epochs, num_train_step_per_epoch)
        return tx

    def init_model_params(self, rng: jax.Array) -> Params:
        """
        Initialize the model parameters.

        Args:
            rng: Base random number generator.

        Returns:
            The model parameters.
        """
        rngs = self.get_model_rngs(rng)
        sample_input = self.batch_to_input(self.sample_input)
        variables = self.model.init(rngs, sample_input, train=True)
        return variables

    def init_state(self, rng: Optional[jax.Array] = None) -> TrainState:
        """
        Create the train state (`train_state.TrainState`) for the model.

        Args:
            rng: Base random number generator.

        Returns:
            The train state instance.
        """
        rng = rng or jax.random.key(self.trainer_config.seed)
        rng, init_rng = jax.random.split(rng)
        variables = self.init_model_params(init_rng)
        if isinstance(variables, FrozenDict):
            extra_variables, params = variables.pop("params")
        else:
            params = variables.pop("params")
            extra_variables = variables

        tx = self.init_optimizer(
            num_epochs=self.trainer_config.train_epochs, 
            num_train_step_per_epoch=len(self.data_module.train_loader)
        )
        return TrainState.create(
            model_def=self.model,
            params=params,
            tx=tx,
            extra_variables=extra_variables,
            rng=rng,
        )

    def start_logger(self):
        """Initialize the log_dir & save the initial states of the model."""
        logger_config = self.trainer_config.get("logger", ConfigDict())
        log_dir = Path(self.log_dir)
        logging.info(f"Logging to {log_dir}")

        (log_dir / "metrics").mkdir(parents=True, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=str(log_dir), program_name="absl_logging")
        logging.set_verbosity(logger_config.get("verbosity", logging.INFO))
        logging.set_stderrthreshold(logger_config.get("stderrthreshold", "warning"))

        # log the full config
        if not (log_dir / "config.yaml").exists():
            config_dict = self.full_config.to_dict()
            config_dict = jax.tree.map(class_to_name, config_dict)
            with open(log_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f)

        # log the sample input
        if not (log_dir / "sample_input.pkl").exists():
            save_pytree(self.sample_input, log_dir / "sample_input.pkl")

        # log the model summary
        if self.trainer_config.get("tabulate_model", True):
            table = self.tabulate_model(self.sample_input)
            logging.info(f"Model summary:\n{table}")
            with open(log_dir / "model_summary.txt", "w") as f:
                f.write(table)

        if self.trainer_config.get("tabulate_params", True):
            table = self.tabulate_params()
            logging.info(f"Parameter summary:\n{table}")
            with open(log_dir / "param_summary.txt", "w") as f:
                f.write(table)

    def create_train_function(self, **kwawgs) -> Any:
        """
        Create the training function for the model.
        
        Returns:
            The training function.
        """
        raise NotImplementedError("create_train_function method must be implemented.")

    def create_eval_function(self, **kwargs) -> Any:
        """
        Create the evaluation function for the model.
        
        Returns:
            The evaluation function.
        """
        raise NotImplementedError("create_eval_function method must be implemented.")

    def _pmap_functions(self) -> Tuple[Callable, Callable]:
        """
        Create the pmap functions for training and evaluation.
        
        Returns:
            Tuple of pmap training and evaluation functions.
        """
        raise NotImplementedError("_pmap_functions method must be implemented.")

    @property
    def log_dir(self) -> str:
        return self.logger.log_dir
