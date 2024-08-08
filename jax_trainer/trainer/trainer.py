import time
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple

import chex
import flax
import flax.linen
import jax
import jax.numpy as jp
import numpy as np
import optax
import yaml
from absl import logging
from flax.core import FrozenDict
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate, unreplicate, pad_shard_unpad as flax_pad_shard_unpad

from ml_collections import ConfigDict
from tabulate import tabulate as py_tabulate
from tqdm.auto import tqdm

from jax import ShapeDtypeStruct
from jax_trainer.callbacks import BaseCallback, TrainingCallback
from jax_trainer.datasets import Batch
from jax_trainer.logger import (
    HostMetrics,
    ImmutableMetrics,
    LogFreq,
    LogMetricMode,
    Logger,
    save_pytree,
    update_metrics
)
from jax_trainer.datasets import DatasetModule
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.trainer.train_state import TrainState, Params
from jax_trainer.trainer.utils import loss_fn_return_check, pad_shard_unpad
from jax_trainer.utils import class_to_name, resolve_import

def fake_pmap_jit_wrapper(func, debug: bool = False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not debug:
            return func(*args, **kwargs)    
        with chex.fake_pmap_and_jit():
            return func(*args, **kwargs)
    return wrapper


# noinspection PyPackageRequirements
class TrainerModule:
    def __init__(
        self,
        trainer_config: ConfigDict,
        model_config: ConfigDict,
        optimizer_config: ConfigDict,
        data_module: DatasetModule,
        sample_input: Batch,
    ):
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.sample_input = sample_input
        self.data_module = data_module

        self.trainer_config.check_valid_freq_epoch = self.trainer_config.get("check_valid_freq_epoch", 1)

        self.model = self.init_model()
        self.logger = self.init_logger()
        self.state = self.init_state()
        self.callbacks, self.train_step_callbacks = self.init_callbacks()

        self._pmap_functions()
        self.training = False
        self.global_step = 0

        # self.train_model = fake_pmap_jit_wrapper(self.train_model, self.trainer_config.debug)

    def init_model(self) -> flax.linen.Module:
        model_class = resolve_import(self.model_config._class)
        hparams = self.model_config.get("hparams", {})
        return model_class(**hparams)

    def init_logger(self) -> Logger:
        logger_config = self.trainer_config.get("logger", ConfigDict())
        full_config = ConfigDict(
            {"trainer": self.trainer_config, "model": self.model_config, "optimizer": self.optimizer_config})
        logger_class = resolve_import(logger_config.get("name", Logger))
        return logger_class(logger_config, full_config)

    def init_callbacks(self) -> Tuple[List[BaseCallback], List[TrainingCallback]]:
        callbacks = []
        train_step_callbacks = []
        callback_configs = self.trainer_config.get("callbacks", ConfigDict())

        for name in callback_configs:
            logging.info(f"Initializing callback: {name}")
            callback_cfg = callback_configs[name]
            if not '.' in name:
                name = f"jax_trainer.callbacks.{name}"
            callback_class = resolve_import(name)
            # if callback_cfg.get("name") is not None:
            #     callback_class = resolve_import(callback_cfg.name)
            # else:
            #     callback_class = resolve_import(f"jax_trainer.callbacks.{name}")

            callback = callback_class(config=callback_cfg, trainer=self, data_module=None)
            callbacks.append(callback)
            if isinstance(callback, TrainingCallback):
                train_step_callbacks.append(callback)

        return callbacks, train_step_callbacks

    def init_model_params(self, rng: jax.Array) -> Params:
        rngs = self.get_model_rngs(rng)
        sample_input = self.batch_to_input(self.sample_input)
        variables = self.model.init(rngs, sample_input, train=True)
        return variables

    def init_optimizer(self, num_epochs: int, num_train_steps_per_epoch: int) -> optax.GradientTransformation:
        optim_builder_class = self.optimizer_config.get("builder", OptimizerBuilder)
        optim_builder_class = resolve_import(optim_builder_class)
        builder: OptimizerBuilder = optim_builder_class(self.optimizer_config)
        tx = builder.build_optimizer(num_epochs, num_train_steps_per_epoch)
        return tx

    def init_state(self, rng: Optional[jax.Array] = None) -> TrainState:
        rng = rng or jax.random.PRNGKey(self.trainer_config.seed)
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
            rng=rng
        )

    def init_train_metrics(self, batch: Optional[Batch] = None) -> FrozenDict:
        batch = batch or self.sample_input
        train_metric_shapes = getattr(self, "train_metric_shapes", None)
        if train_metric_shapes is None:
            # FIXME: since `eval_shape` jitting the given function, `chex.fake_pmap_and_jit` context at the funcion giving is not working.
            _, train_metric_shapes = jax.eval_shape(self.train_step, state=self.state, batch=batch, metrics=None)
            self.train_metric_shapes = train_metric_shapes
        
        # init_metrics = jax.tree_map(lambda x: jp.atleast_2d(jp.zeros_like(x)), train_metric_shapes)
        init_metrics = jax.tree.map(lambda x: jp.zeros_like(x).reshape(-1, *x.shape), train_metric_shapes)
        return init_metrics

    def init_eval_metrics(self, batch: Optional[Batch] = None) -> FrozenDict:
        batch = batch or self.sample_input
        eval_metric_shapes = getattr(self, "eval_metric_shapes", None)
        if eval_metric_shapes is None:
            # eval_metric_shapes = jax.tree.map(
            #     lambda x: ShapeDtypeStruct(jp.shape(x)), 
            #     self.eval_step(state=self.state, batch=batch, metrics=None)
            # )
            # FIXME: since `eval_shape` jitting the given function, `chex.fake_pmap_and_jit` context at the funcion giving is not working.
            eval_metric_shapes = jax.eval_shape(self.eval_step, state=self.state, batch=batch, metrics=None)
            self.eval_metric_shapes = eval_metric_shapes

        init_metrics = jax.tree.map(lambda x: jp.zeros_like(x).reshape(-1, *x.shape), eval_metric_shapes)
        return init_metrics

    def start_logger(self):
        """
        Initialize the log_dir & save the initial states of the model.
        """
        logger_config = self.trainer_config.get("logger", ConfigDict())
        full_config = ConfigDict(
            {"trainer": self.trainer_config, "model": self.model_config, "optimizer": self.optimizer_config})

        log_dir = Path(self.log_dir)
        self.trainer_config.logger.log_dir = str(log_dir)
        logging.info(f"Logging to {log_dir}")

        (log_dir / "metrics").mkdir(parents=True, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=str(log_dir), program_name="absl_logging")
        logging.set_verbosity(logger_config.get("verbosity", logging.INFO))
        logging.set_stderrthreshold(logger_config.get("stderrthreshold", "warning"))

        if not (log_dir / "config.yaml").exists():
            config_dict = full_config.to_dict()
            config_dict = jax.tree.map(class_to_name, config_dict)
            with open(log_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f)

        if not (log_dir / "sample_input.pkl").exists():
            save_pytree(self.sample_input, log_dir / "sample_input.pkl")

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

    def start_debug_ctx(self):
        debug = self.trainer_config.debug
        jax.config.update('jax_debug_nans', debug)
        jax.config.update('jax_log_compiles', debug)
        if debug:
            logging.info("Debugging enabled, jit disabled")
            self._ctx = chex.fake_pmap_and_jit()
    
    def end_debug_ctx(self):
        if hasattr(self, "_ctx"):
            self._ctx.stop()

    def start_training_ctx(self):
        """
        Prepare for GPU parallel training.
        """
        self.training = True
        self.state = replicate(self.state)

    def end_training_ctx(self):
        """
        Clean up after training.
        """
        self.training = False
        self.state = unreplicate(self.state)

    def train_model(self):
        self.global_step = 0
        self.start_debug_ctx()
        self.start_training_ctx()
        
        self.on_training_start()
        self.verify_eval_step(self.data_module.valid_loader)

        total_eval_metrics = {}  # {int: eval_metrics, "test"(Optional): test_metrics}
        train_metrics = None

        num_epochs = self.trainer_config.train_epochs
        for epoch in self.get_tracker(range(1, num_epochs), desc="Epochs"):
            self.on_training_epoch_start(epoch)
            train_metrics, epoch_metrics = self.train_epoch(epoch, train_metrics)
            self.on_training_epoch_end(train_metrics, epoch)

            check_valid_freq_epoch = self.trainer_config.check_valid_freq_epoch
            if check_valid_freq_epoch > 0 and epoch % check_valid_freq_epoch == 0:
                self.on_validation_epoch_start(epoch)
                eval_metrics = self.evaluate_model(self.data_module.valid_loader, stage="valid", epoch=epoch)
                total_eval_metrics[epoch] = eval_metrics
                self.on_validation_epoch_end(eval_metrics, epoch)

        self.on_training_end()

        self.end_training_ctx()
        if self.data_module.test_loader is not None:
            if self.trainer_config.get("restore_best", True):
                self.load_model(raise_error=True)
                self.on_test_epoch_start(num_epochs)
                test_metrics = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=num_epochs)
                self.on_test_epoch_end(test_metrics, num_epochs)
                total_eval_metrics["test"] = test_metrics

        self.logger.finalize("success")
        for callback in self.callbacks:
            callback.finalize("success")
        
        self.end_debug_ctx()
        return total_eval_metrics

    def train_epoch(
        self, 
        epoch: int, 
        train_metrics: Optional[ImmutableMetrics] = None
    ) -> Tuple[Dict[str, HostMetrics], ...]:
        """
        Train an epoch of the model.

        Args:
            epoch: The epoch number.
            train_metrics: Running metrics for the training epoch.

        Returns:
            (0): Updated running metrics for the training epoch.
            (1): Aggregated metrics for this epoch.
        """

        assert self.training, "Training must be started before training an epoch"
        self.logger.start_epoch(epoch, stage="train")

        for batch in self.get_tracker(self.data_module.train_loader, desc="Training", leave=False):
            if train_metrics is None:
                train_metrics = self.init_train_metrics(batch)

            if self.global_step == 0:
                # pre-compile the train step to start training
                logging.info("Compiling train_step ...")
                st = time.time()
                self.state, train_metrics = self.train_step(self.state, batch, train_metrics)
                logging.info(f"Train step compiled in {time.time() - st:.2f} seconds")
            else:
                # Execute the train step with profiling
                with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                    self.state, train_metrics = self.train_step(self.state, batch, train_metrics)

            for callback in self.train_step_callbacks:
                callback.on_training_step(self.state, batch, train_metrics)

            train_metrics = self.logger.log_step(train_metrics)
            self.global_step += 1

        train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics)
        return train_metrics, epoch_metrics

    def evaluate_model(self, dataloader: Iterator, stage: Literal["valid", "test"], epoch: int) -> HostMetrics:
        self.logger.start_epoch(epoch, stage=stage)
        eval_metrics = self.init_eval_metrics()
        step_count = 0

        for batch in self.get_tracker(dataloader, desc=stage.capitalize(), leave=False):
            eval_metrics = self.eval_step(self.state, batch, eval_metrics)
            step_count += 1

        if step_count == 0:
            logging.warning(f"No data for {stage} evaluation")

        _, epoch_metrics = self.logger.end_epoch(eval_metrics)
        return epoch_metrics

    def test_model(self, apply_callbacks: bool = False, epoch: int = 0) -> HostMetrics:
        test_metrics = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=epoch)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch)
        return test_metrics

    def create_train_function(self, axis_name: str = "batch") -> Any:
        def train_step(
            state: TrainState,
            batch: Batch,
            metrics: Optional[ImmutableMetrics] = None
        ) -> Tuple[TrainState, ImmutableMetrics]:
            rng, step_rng = jax.random.split(state.rng)
            loss_fn = partial(self.loss_function, state=state, batch=batch, rng=step_rng, train=True)
            out, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, (updates, step_metrics) = out

            step_metrics["loss"] = loss
            grads = jax.lax.pmean(grads, axis_name)
            state = state.apply_gradients(grads=grads, extra_variables=updates, rng=rng)

            if self.trainer_config.get("log_grad_norm", False):
                grad_norm = optax.global_norm(grads)
                step_metrics["optimizer/grad_global_norm"] = {
                    "value": grad_norm,
                    "log_freq": LogFreq.STEP
                }
                step_metrics["optimizer/grad_global_norm_max"] = {
                    "value": jp.max(grad_norm),
                    "log_freq": LogFreq.STEP,
                    "mode": LogMetricMode.MAX
                }
                params_norm = optax.global_norm(state.params)
                step_metrics["optimizer/params_global_norm"] = {
                    "value": params_norm,
                    "log_freq": LogFreq.STEP
                }

            # pmean is not applied because there are mixed metrics that should be applied with pmean and not.
            metrics = update_metrics(metrics, step_metrics, train=True, batch_size=len(batch))
            return state, metrics

        return train_step

    def create_eval_function(self):
        def eval_step(
            state: TrainState,
            batch: Batch,
            metrics: Optional[ImmutableMetrics] = None
        ) -> ImmutableMetrics:
            eval_rng = jax.random.PRNGKey(self.trainer_config.get("eval_seed", 0))
            loss, (_, step_metrics) = self.loss_function(state.params, state, batch, rng=eval_rng, train=False)
            step_metrics["loss"] = loss
            metrics = update_metrics(metrics, step_metrics, train=False, batch_size=len(batch))
            return metrics

        return eval_step

    def _pmap_functions(self):
        train_step = self.create_train_function()
        eval_step = self.create_eval_function()

        # if self.trainer_config.get("debug", False):
        #     logging.info("Skipping pmap for debugging")
        #     self.train_step = train_step
        #     self.eval_step = eval_step
        # else:
        donate_argnums = (0, 2) if self.trainer_config.get("donate_state", True) else (2,)
        p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=donate_argnums)
        p_eval_step = jax.pmap(eval_step, axis_name="batch", donate_argnums=(2,))

        p_train_step = pad_shard_unpad(
            p_train_step, static_argnums=(0, 2), static_argnames=('state', 'metrics'), static_returns=True)
        p_eval_step = flax_pad_shard_unpad(
            p_eval_step, static_argnums=(0, 2), static_argnames=('state', 'metrics'), static_return=True)

        self.train_step = p_train_step
        self.eval_step = p_eval_step

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
        self.state = self.state.replace(**state_dict)

    def tabulate_model(self, sample_input: Batch) -> str:
        """
        Summarize the model and tabulate it.

        Args:
            sample_input: A sample input to the model to initialization.
        """
        rngs = self.get_model_rngs(jax.random.PRNGKey(0))
        sample_input = self.batch_to_input(sample_input)
        return self.model.tabulate(rngs, sample_input, train=True,
                                   console_kwargs={"force_terminal": False, "width": 300})

    def tabulate_params(self) -> str:
        """
        Summarize the parameters of the model and tabulate it.
        """
        params = flax.traverse_util.flatten_dict(self.state.params, sep='.')
        summary = {"Name": list(params.keys()),
                   "Shape": jax.tree.map(jp.shape, params),
                   "Count": jax.tree.map(np.prod, jax.tree.map(jp.shape, params)),
                   "Dtype": jax.tree.map(lambda x: x.dtype, params),
                   "Mean": jax.tree.map(lambda x: jp.mean(x).item(), params),
                   "Std": jax.tree.map(lambda x: jp.std(x).item(), params),
                   "Min": jax.tree.map(lambda x: jp.min(x).item() if x.size > 0 else 0, params),
                   "Max": jax.tree.map(lambda x: jp.max(x).item() if x.size > 0 else 0, params)}

        return py_tabulate(summary, headers="keys", tablefmt="pretty")

    @loss_fn_return_check
    def loss_function(
        self,
        params: Any,
        state: TrainState,
        batch: Batch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Dict[str, jp.ndarray] | None, Dict[str, Any]]]:
        """
        Loss function for the model. Return must be output of model and mutable extra variables.
        """
        raise NotImplementedError("loss_function not implemented")

    def get_model_rngs(self, rng: jax.Array) -> Dict[str, jax.random.PRNGKey]:
        """
        Returns a rngs dictionary for the model.
        
        Args:
            rng: The rng to split.
        
        Returns:
            Dictionary of rngs.
        """
        return {"params": rng}

    def batch_to_input(self, batch: Batch) -> Any:
        raise NotImplementedError

    def get_tracker(self, iterator: Iterator, **kwargs):
        if self.trainer_config.get("pbar", True):
            return tqdm(iterator, **kwargs)
        return iterator

    def on_training_start(self):
        logging.info("Training started")
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        logging.info("Training ended")
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self, epoch: int):
        logging.info(f"Training epoch {epoch} started")
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch)

    def on_training_epoch_end(self, train_metrics: Dict[str, Any], epoch: int):
        logging.info(f"Training epoch {epoch} ended")
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch)

    def on_validation_epoch_start(self, epoch: int):
        logging.info(f"Validation epoch {epoch} started")
        for callback in self.callbacks:
            callback.on_validation_epoch_start(epoch)

    def on_validation_epoch_end(self, eval_metrics: Dict[str, Any], epoch: int):
        logging.info(f"Validation epoch {epoch} ended")
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch)

    def on_test_epoch_start(self, epoch: int):
        logging.info(f"Test epoch {epoch} started")
        for callback in self.callbacks:
            callback.on_test_epoch_start(epoch)

    def on_test_epoch_end(self, test_metrics: Dict[str, Any], epoch: int):
        logging.info(f"Test epoch {epoch} ended")
        for callback in self.callbacks:
            callback.on_test_epoch_end(test_metrics, epoch)

    def verify_eval_step(self, valid_loader: Iterator):
        print("Verifying eval step ...")
        batch = next(iter(valid_loader))
        eval_metrics = self.init_eval_metrics(batch)
        st = time.time()
        logging.info("Testing & compiling eval step ...")
        _ = self.eval_step(state=self.state, batch=batch, metrics=eval_metrics)
        logging.info(f"Eval step compiled in {time.time() - st:.2f} seconds")

    @property
    def log_dir(self) -> str:
        return self.trainer_config.logger.log_dir
