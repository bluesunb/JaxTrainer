import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax
import yaml
from absl import logging
from flax.core import FrozenDict
from flax.struct import dataclass, field
from flax.jax_utils import unreplicate
from ml_collections import ConfigDict
from tabulate import tabulate as py_tabulate
from tqdm import tqdm

from jax_trainer.callbacks import BaseCallback, TrainingCallback
from jax_trainer.datasets import Batch, DatasetModule
from jax_trainer.logger import Logger, reduce_array_to_scalar, save_pytree
from jax_trainer.metrics import MultiMetric
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.trainer.train_state import Params, TrainState
from jax_trainer.trainer.utils import loss_fn_return_check, replicate_pjit
from jax_trainer.utils import class_to_name, resolve_import

nonpytree_node = partial(field, pytree_node=False)


@partial(dataclass, frozen=False)
class Trainer:
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
    ) -> "Trainer":
        """
        Args:
            trainer_config:     Configuration for the trainer. (config.trainer)
            model_config:       Configuration for the model. (config.model)
            optimizer_config:   Configuration for the gradient transformations (include lr scheduler). (config.optimizer)
            data_module:        Data module for the dataset.
            sample_input:       Sample input for the model.
        """
        trainer: Trainer = cls(
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

        tx = self.init_optimizer(self.trainer_config.train_epochs, len(self.data_module.train_loader))
        return TrainState.create(
            model_def=self.model,
            params=params,
            tx=tx,
            extra_variables=extra_variables,
            rng=rng,
        )

    def init_train_metrics(self) -> MultiMetric:
        """
        Initialize the training metrics.

        Returns:
            The training metrics instance.
        """
        return MultiMetric.create()

    def init_eval_metrics(self) -> MultiMetric:
        """
        Initialize the evaluation metrics.

        Returns:
            The evaluation metrics instance.
        """
        return MultiMetric.create()

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

    def train_model(self):
        # >>> Start training
        self.global_step = 0
        self.start_debug_ctx()

        # pre-training callbacks & checks
        self.start_training_ctx()
        self.on_training_start()
        self.verify_eval_step(self.data_module.valid_loader)

        # >>> Training loop
        total_eval_metrics = {"eval": defaultdict(list), "test": dict()}
        train_metrics = self.init_train_metrics()

        num_epochs = self.trainer_config.train_epochs
        for epoch in self.get_tracker(range(1, num_epochs + 1), desc="Epoch", position=0):
            self.on_training_epoch_start(epoch)
            train_metrics, epoch_info = self.train_epoch(epoch, train_metrics)
            epoch_info = self.maybe_reduce(epoch_info)
            self.on_training_epoch_end(epoch_info, epoch)

            # >>> Validation loop
            valid_freq_epoch = self.trainer_config.valid_freq_epoch
            if valid_freq_epoch > 0 and epoch % valid_freq_epoch == 0:
                self.on_validation_epoch_start(epoch)
                _, eval_info = self.evaluate_model(self.data_module.valid_loader, stage="valid", epoch=epoch)
                eval_info = self.maybe_reduce(eval_info)

                # update total eval metrics
                for k, v in eval_info.items():
                    total_eval_metrics["eval"][k].append(v)

                total_info = {
                    **{f"train/{k}": v for k, v in epoch_info.items()},
                    **{f"valid/{k}": v for k, v in eval_info.items()},
                }
                self.on_validation_epoch_end(total_info, epoch)

        # >>> Post-training
        self.on_training_end()
        self.end_training_ctx()

        # >>> Testing loop
        if self.data_module.test_loader is not None and self.trainer_config.get("restore_best", True):
            self.load_model(raise_error=True)
            self.on_test_epoch_start(num_epochs)
            _, test_info = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=num_epochs)
            test_info = self.maybe_reduce(test_info)
            self.on_test_epoch_end(test_info, num_epochs)
            total_eval_metrics["test"].update(test_info)  # update total eval metrics

        # >>> Finalize logger & callbacks
        self.logger.finalize("success")
        for callback in self.callbacks:
            callback.finalize("success")

        # >>> End training
        self.end_debug_ctx()
        return total_eval_metrics

    def train_epoch(
        self,
        epoch: int,
        metrics: MultiMetric,
    ) -> Tuple[MultiMetric, MultiMetric]:
        assert self.training, "Training must be enabled to train"

        for batch in self.get_tracker(self.data_module.train_loader, desc="Training", position=1):
            # >>> Single training step
            if self.global_step == 0:  # Initial compilation
                logging.info("Compiling train_step ...")
                st = time.time()
                self.state, metrics = self.train_step(self.state, batch, metrics)
                logging.info(f"Train step compiled in {time.time() - st:.2f} sec")
            else:
                with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                    self.state, metrics = self.train_step(self.state, batch, metrics)

            # >>> Callbacks
            if len(self.train_step_callbacks):  # to avoid unnecessary computation of metrics
                step_metrics = self.maybe_reduce(metrics).compute()
                for callback in self.train_step_callbacks:
                    callback._on_training_step(step_metrics, epoch, self.global_step)

            # >>> Logging
            if self.global_step % self.trainer_config.get("log_freq_step", 1) == 0:
                step_metrics = self.maybe_reduce(metrics).compute()  # to avoid unnecessary computation of metrics
                self.logger.log_metrics(step_metrics, step=self.global_step, prefix="train")

            # >>> Step end & step training info update
            self.global_step += 1

        # >>> Epoch end & epoch training info update
        epoch_metrics = metrics.compute()
        return metrics, epoch_metrics

    def evaluate_model(
        self,
        dataloader: Iterator,
        stage: Literal["valid", "test"],
        epoch: int,
    ) -> tuple[MultiMetric, dict[str, jp.ndarray]]:
        # >>> Prepare evaluation
        step_count = 0
        metrics = self.init_eval_metrics()

        # >>> Evaluation over 1 epoch
        for batch in self.get_tracker(dataloader, desc=stage.capitalize(), position=1):
            metrics = self.eval_step(self.state, batch, metrics)
            step_count += 1

        # warning if no data found
        if step_count == 0:
            logging.warning(f"No data found for {stage} evaluation")

        # >>> Epoch end & epoch evaluation info update
        eval_metrics = metrics.compute()
        return metrics, eval_metrics

    def test_model(self, apply_callbacks: bool = False, epoch: int = 0) -> MultiMetric:
        # >>> Testing loop
        _, test_metrics = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=epoch)
        # >>> Callbacks
        test_metrics = self.maybe_reduce(test_metrics)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch)
        return test_metrics

    def verify_eval_step(self, val_loader: Iterator):
        print("Verifying eval step ...")
        batch = next(iter(val_loader))
        eval_metrics = self.init_eval_metrics()
        st = time.time()
        logging.info("Testing & compiling eval_step ...")
        _ = self.eval_step(self.state, batch, eval_metrics)
        logging.info(f"Eval step compiled in {time.time() - st:.2f} sec")

    def create_train_function(self, axis_name: str = "batch") -> Any:
        def train_step(
            state: TrainState,
            batch: Batch,
            metrics: MultiMetric,
        ) -> Tuple[TrainState, MultiMetric]:
            rng, step_rng = jax.random.split(state.rng)
            loss_fn = partial(self.loss_function, state=state, batch=batch, rng=step_rng, train=True )
            (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            updates, metric_updates = out

            metric_updates.update(
                loss=jax.lax.pmean(loss, axis_name=axis_name),
                grads_norm=optax.global_norm(grads),
                params_norm=optax.global_norm(state.params),
            )
            metrics = metrics.update(**metric_updates)
            grads = jax.lax.pmean(grads, axis_name)
            state = state.apply_gradients(grads=grads, extra_variables=updates, rng=rng)
            return state, metrics

        return train_step

    def create_eval_function(self, axis_name: str = "batch") -> Any:
        def eval_step(
            state: TrainState,
            batch: Batch,
            metrics: MultiMetric,
        ) -> MultiMetric:
            eval_rng = jax.random.key(self.trainer_config.get("eval_seed", 0))
            loss, (_, metric_updates) = self.loss_function(params=state.params, state=state, batch=batch, rng=eval_rng, train=False)
            metric_updates.update(loss=loss)
            metrics = metrics.update(**metric_updates)
            return metrics

        return eval_step

    def _pmap_functions(self):
        train_step = self.create_train_function()
        eval_step = self.create_eval_function()

        pmap = self.trainer_config.get("pmap", True)
        kwargs = {
            "pmap": pmap,
            "pad_static_argnums": (0, 2),
            "pad_static_argnames": ("state", "metrics"),
        }
        if pmap:
            kwargs["axis_name"] = "batch"

        donate_argnums = ((0, 2) if self.trainer_config.get("donate_state", True) else (2,))
        train_step = replicate_pjit(train_step, donate_argnums=donate_argnums, **kwargs)
        eval_step = replicate_pjit(eval_step, donate_argnums=(2,), **kwargs)
        return train_step, eval_step

    @loss_fn_return_check
    def loss_function(
        self,
        params: Any,
        *,
        state: TrainState,
        batch: Batch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Dict[str, jp.ndarray] | None, Dict[str, Any]]]:
        """
        Loss function for the model. Return must be output of model and mutable extra variables.

        Args:
            state:  The current state of the training.
            batch:  The batch of data.
            rng:    The random number generator.
            train:  Whether the using the model with training mode.

        Returns:
            Tuple of loss and a tuple of updates and metrics
            - (0): The loss value.
            - (1-0): Mutable param updates.
            - (1-1): Metrics update values
        """
        # FIXME: It doesn't reflect member changes of `self`
        # (See: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree)
        raise NotImplementedError("loss_function not implemented")

    def load_model(self, epoch: int = -1, raise_error: bool = True):
        logging.info(f"Loading model from epoch {epoch}")
        for callback in self.callbacks:
            if hasattr(callback, "load_model"):
                state_dict = callback.load_model(epoch)
                self.restore(state_dict)
                return
        else:
            if raise_error:
                raise ValueError("No callback found to load model")
            logging.warning("No callback found to load model")

    def restore(self, state_dict: Dict[str, Any]):
        logging.info(f"Restoring model from state_dict with keys: {state_dict.keys()}")
        state_dict.pop("metrics")
        state_dict.pop("metadata")
        self.state = unreplicate(self.state).replace(**state_dict)

    def start_training_ctx(self):
        """Start the context for training."""
        self.training = True

    def end_training_ctx(self):
        """Start the context for training."""
        self.training = False

    def start_debug_ctx(self):
        """Start the context for debugging."""
        debug = self.trainer_config.debug
        jax.config.update("jax_debug_nans", debug)
        if debug:
            logging.info("Debugging enabled, jit disabled.")
            self._ctx = chex.fake_pmap_and_jit()
            self._ctx.start()

    def end_debug_ctx(self):
        """End the context for debugging."""
        if hasattr(self, "_ctx"):
            self._ctx.stop()
            delattr(self, "_ctx")

    def tabulate_model(self, sample_input: Batch) -> str:
        """
        Summarize the model and tabulate it.

        Args:
            sample_input: A sample input to the model to initialization.
        """
        rngs = self.get_model_rngs(jax.random.PRNGKey(0))
        sample_input = self.batch_to_input(sample_input)
        return self.model.tabulate(
            rngs,
            sample_input,
            train=True,
            console_kwargs={"force_terminal": False, "width": 300}
        )

    def tabulate_params(self) -> str:
        """
        Summarize the parameters of the model and tabulate it.
        """
        params = flax.traverse_util.flatten_dict(self.state.params, sep=".")
        summary = {
            "Name": list(params.keys()),
            "Shape": jax.tree.map(jp.shape, params),
            "Count": jax.tree.map(np.prod, jax.tree.map(jp.shape, params)),
            "Dtype": jax.tree.map(lambda x: x.dtype, params),
            "Mean": jax.tree.map(lambda x: jp.mean(x).item(), params),
            "Std": jax.tree.map(lambda x: jp.std(x).item(), params),
            "Min": jax.tree.map(lambda x: jp.min(x).item() if x.size > 0 else 0, params),
            "Max": jax.tree.map(lambda x: jp.max(x).item() if x.size > 0 else 0, params),
        }

        return py_tabulate(summary, headers="keys", tablefmt="pretty")

    def batch_to_input(self, batch: Batch) -> Any:
        """
        Create the appropriate input for the model from the given batch instance.

        Returns:
            The input to the model. It can be a single input or a dictionary of inputs or ect.
        """
        raise NotImplementedError

    def get_model_rngs(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Returns a rngs dictionary for the model.

        Args:
            rng: Tue rng to split.

        Returns:
            Dictionary of rngs.
        """
        return {"params": rng}

    def get_tracker(self, iterator: Iterator, **kwargs):
        """Get the iterator with tqdm progress bar if enabled."""
        if self.trainer_config.get("pbar", True):
            return tqdm(iterator, **kwargs)
        return iterator

    def on_training_start(self):
        logging.info("Training started")
        for callback in self.callbacks:
            callback._on_training_start()

    def on_training_end(self):
        logging.info("Training ended")
        for callback in self.callbacks:
            callback._on_training_end()

    def on_training_epoch_start(self, epoch: int):
        logging.info(f"Training epoch {epoch} started")
        for callback in self.callbacks:
            callback._on_training_epoch_start(epoch)

    def on_training_epoch_end(self, train_metrics: Dict[str, Any], epoch: int):
        logging.info(f"Training epoch {epoch} ended")
        for callback in self.callbacks:
            callback._on_training_epoch_end(train_metrics, epoch)

    def on_validation_epoch_start(self, epoch: int):
        logging.info(f"Validation epoch {epoch} started")
        for callback in self.callbacks:
            callback._on_validation_epoch_start(epoch)

    def on_validation_epoch_end(self, total_metrics: Dict[str, Any], epoch: int):
        """`total_metric`: {'train/...': train_epoch_info, 'valid/...': valid_epoch_info}"""
        logging.info(f"Validation epoch {epoch} ended")
        for callback in self.callbacks:
            callback._on_validation_epoch_end(total_metrics, epoch)

    def on_test_epoch_start(self, epoch: int):
        logging.info(f"Test epoch {epoch} started")
        for callback in self.callbacks:
            callback._on_test_epoch_start(epoch)

    def on_test_epoch_end(self, test_metrics: Dict[str, Any], epoch: int):
        logging.info(f"Test epoch {epoch} ended")
        for callback in self.callbacks:
            callback._on_test_epoch_end(test_metrics, epoch)

    def maybe_unreplicate(self, value: Any) -> Any:
        """Unreplicate the maybe replicated value."""
        if self.trainer_config.get("pmap", True):
            return reduce_array_to_scalar(unreplicate(value))
        return reduce_array_to_scalar(value)

    def maybe_reduce(self, value: Any) -> Any:
        """Reduce the maybe replicated value."""
        return jax.tree.map(reduce_array_to_scalar, self.maybe_unreplicate(value))

    @property
    def full_config(self):
        return ConfigDict({
            "trainer": self.trainer_config,
            "model": self.model_config,
            "optimizer": self.optimizer_config,
        })

    @property
    def log_dir(self) -> str:
        return self.logger.log_dir
