import time
from functools import partial, wraps
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
from flax.jax_utils import replicate, unreplicate, pad_shard_unpad

from ml_collections import ConfigDict
from tabulate import tabulate as py_tabulate
from tqdm import tqdm

from jax_trainer.datasets import Batch, DatasetModule
from jax_trainer.callbacks import BaseCallback, TrainingCallback
from jax_trainer.logger import Logger, save_pytree
from jax_trainer.metrics import (
    Accuracy,
    Average,
    MultiMetric,
    Timeit,
    Welford, Metric
)
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.trainer.train_state import Params, TrainState
from jax_trainer.trainer.utils import loss_fn_return_check, replicate_pjit
from jax_trainer.utils import class_to_name, resolve_import


class Trainer:
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
        self.data_module = data_module
        self.sample_input = sample_input

        self.valid_freq_epoch = self.trainer_config.get("valid_freq_epoch", 1)
        self.training = False
        self.global_step = 0

        self.model = self.init_model()
        self.logger = self.init_logger()
        self.state = self.init_state()
        self.callbacks, self.train_step_callbacks = self.init_callbacks()
        self._pmap_functions()

    def init_model(self) -> nn.Module:
        model_class = resolve_import(self.model_config._class)
        hparams = self.model_config.get("hparams", {})
        return model_class(**hparams)
    
    def init_logger(self) -> Logger:
        logger_config = self.trainer_config.get("logger", ConfigDict())
        logger_class = resolve_import(logger_config.get("_class", Logger))
        return logger_class(logger_config, self.full_config)
    
    def init_callbacks(self) -> Tuple[List[BaseCallback], List[TrainingCallback]]:
        callbacks = []
        train_step_callbacks = []
        callback_config = self.trainer_config.get("callbacks", ConfigDict())

        for name in callback_config:
            logging.info(f"Initializing callback: {name}")
            cfg = callback_config[name]
            if not '.' in name:
                name = f"jax_trainer.callbacks.{name}"
            
            callback_class = resolve_import(name)
            callback = callback_class(config=cfg, trainer=self, data_module=None)

            callbacks.append(callback)
            if isinstance(callback, TrainingCallback):
                train_step_callbacks.append(callback)

        return callbacks, train_step_callbacks
    
    def init_optimizer(self, num_epochs: int, num_train_step_per_epoch: int) -> optax.GradientTransformation:
        optim_builder_class = self.optimizer_config.get("builder", OptimizerBuilder)
        optim_builder_class = resolve_import(optim_builder_class)
        builder: OptimizerBuilder = optim_builder_class(self.optimizer_config)
        tx = builder.build_optimizer(num_epochs, num_train_step_per_epoch)
        return tx
    
    def init_model_params(self, rng: jax.Array) -> Params:
        rngs = self.get_model_rngs(rng)
        sample_input = self.batch_to_input(self.sample_input)
        variables = self.model.init(rngs, sample_input, train=True)
        return variables
    
    def init_state(self, rng: Optional[jax.Array] = None) -> TrainState:
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
            rng=rng
        )
    
    def init_train_metrics(self) -> MultiMetric:
        return MultiMetric.create()
    
    def init_eval_metrics(self) -> MultiMetric:
        return MultiMetric.create()

    def start_logger(self):
        """Initialize the log_dir & save the initial states of the model."""
        logger_config = self.trainer_config.get("logger", ConfigDict())
        log_dir = Path(self.log_dir)
        self.trainer_config.logger.log_dir = str(log_dir)
        logging.log(f"Logging to {log_dir}")

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
        self.global_step = 0
        self.start_debug_ctx()
        self.start_training_ctx()
        
        self.on_training_start()
        self.verify_eval_step(self.data_module.valid_loader)
        
        total_eval_metrics = {}
        train_metrics = self.init_train_metrics()
        
        num_epochs = self.trainer_config.train_epochs
        for epoch in self.get_tracker(range(1, num_epochs + 1), desc="Epoch"):
            self.on_training_epoch_start(epoch)
            train_metrics, epoch_info = self.train_epoch(epoch, train_metrics)
            self.on_training_epoch_end(train_metrics, epoch)
            
            valid_freq_epoch = self.trainer_config.valid_freq_epoch
            if valid_freq_epoch > 0 and epoch % valid_freq_epoch == 0:
                self.on_validation_epoch_start(epoch)
                _, eval_info = self.evaluate_model(self.data_module.valid_loader, stage="valid", epoch=epoch)
                total_eval_metrics[epoch] = eval_info
                self.on_validation_epoch_end(eval_info, epoch)
                
        self.on_training_end()
        self.end_training_ctx()
        
        if self.data_module.test_loader is not None and self.trainer_config.get("restore_best", True):
            self.load_model(raise_error=True)
            self.on_test_epoch_start(num_epochs)
            _, test_info = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=num_epochs)
            self.on_test_epoch_end(test_info, num_epochs)
            total_eval_metrics["test"] = test_info

        self.logger.finalize("success")
        for callback in self.callbacks:
            callback.finalize("success")

        self.end_debug_ctx()
        return total_eval_metrics
                
    def train_epoch(
        self,
        epoch: int,
        metrics: MultiMetric,
    ) -> Tuple[MultiMetric, MultiMetric]:
        assert self.training, "Training must be enabled to train"
        
        for batch in self.get_tracker(self.data_module.train_loader, desc="Training"):
            if self.global_step == 0:
                logging.info("Compiling train_step ...")
                st = time.time()
                self.state, metrics = self.train_step(self.state, batch, metrics)
                logging.info(f"Train step compiled in {time.time() - st:.2f} sec")
            else:
                with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                    self.state, metrics = self.train_step(self.state, batch, metrics)
                    
            for callback in self.train_step_callbacks:
                callback.on_training_step(self.state, batch, metrics)
        
            self.global_step += 1
        
        epoch_metrics = metrics.compute()
        return metrics, epoch_metrics
    
    def evaluate_model(
        self, 
        dataloader: Iterator, 
        stage: Literal["valid", "test"],
        epoch: int,
    ) -> tuple[MultiMetric, dict[str, jp.ndarray]]:
        metrics = self.init_eval_metrics()
        step_count = 0
        
        for batch in self.get_tracker(dataloader, desc=stage.capitalize()):
            metrics = self.eval_step(self.state, batch, metrics)
            step_count += 1
        
        if step_count == 0:
            logging.warning(f"No data found for {stage} evaluation")
            
        eval_metrics = metrics.compute()
        return metrics, eval_metrics
    
    def test_model(self, apply_callbacks: bool = False, epoch: int = 0) -> MultiMetric:
        _, test_metrics = self.evaluate_model(self.data_module.test_loader, stage="test", epoch=epoch)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch)
        return test_metrics
    
    def verify_eval_step(self, val_loader: Iterator):
        print("Verifying eval step ...")
        batch = next(iter(val_loader))
        eval_metrics = self.init_eval_metrics()
        st = time.time()
        logging.info("Testing & compiling eval_step ...")
        _ = self.eval_step(state=self.state, batch=batch, metrics=eval_metrics)
        logging.info(f"Eval step compiled in {time.time() - st:.2f} sec")
                
    def create_train_function(self, axis_name: str = "batch") -> Any:
        def train_step(
            state: TrainState,
            batch: Batch,
            metrics: MultiMetric,
        ) -> Tuple[TrainState, MultiMetric]:
            rng, step_rng = jax.random.split(state.rng)
            loss_fn = partial(self.loss_function, state=state, batch=batch, rng=step_rng, train=True)
            grads, out = jax.grad(loss_fn, has_aux=True)(state.params)
            updates, metric_updates = out

            metric_updates.update(
                grads_norm=optax.global_norm(grads), 
                params_norm=optax.global_norm(state.params)
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
            _, (_, metric_updates) = self.loss_function(
                params=state.params, state=state, batch=batch, rng=eval_rng, train=False)
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
            "donate_argnums": (0, 2) if self.trainer_config.get("donate_state", True) else (2,),
        }
        if pmap:
            kwargs['axis_name'] = "batch"
        
        self.train_step = replicate_pjit(train_step, **kwargs)
        self.eval_step = replicate_pjit(eval_step, **kwargs)

    @loss_fn_return_check
    def loss_function(
        self, 
        params: Any,
        *,
        state: TrainState,
        batch: Batch, 
        rng: jax.Array, 
        train: bool = True
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
        self.state = self.state.replace(**state_dict)
        
    def start_training_ctx(self):
        """Prepare for GPU parallel training."""
        self.training = True
        
    def end_training_ctx(self):
        """Clean for GPU parallel training."""
        self.training = False
            
    def start_debug_ctx(self):
        """
        Create a context manager for debugging and start it.
        """
        debug = self.trainer_config.debug
        jax.config.update("jax_debug_nans", debug)
        if debug:
            logging.info("Debugging enabled, jit disabled.")
            self._ctx = chex.fake_pmap_and_jit()
            self._ctx.start()
    
    def end_debug_ctx(self):
        """
        End the debugging context manager.
        """
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

    def batch_to_input(self, batch: Batch) -> Any:
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

    @property
    def full_config(self):
        return ConfigDict({
            "trainer": self.trainer_config,
            "model": self.model_config,
            "optimizer": self.optimizer_config,
        })
        
    @property
    def log_dir(self) -> str:
        return self.trainer_config.logger.log_dir