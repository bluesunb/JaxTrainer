from functools import partial
from numbers import Number
from typing import Any, Callable, List, Tuple

import optax
from ml_collections import ConfigDict

from jax_trainer.utils import resolve_import_from_str


class OptimizerBuilder:
    """
    Builder(Factory) class to create optimizer instances from configuration.
    """

    def __init__(self, optimizer_config: ConfigDict):
        self.optimizer_config = optimizer_config

    def build_optimizer(self, num_epochs: int = 0, num_train_steps_per_epoch: int = 0):
        opt_class = self.build_optimizer_function()
        schedule = self.build_scheduler(num_epochs, num_train_steps_per_epoch)
        pre_trans, post_trans = self.build_grad_transformations()

        @optax.inject_hyperparams
        def tx_factory(schedule):
            return optax.chain(*pre_trans, opt_class(schedule), *post_trans)

        return tx_factory(schedule)

    def build_optimizer_function(self) -> Callable[..., optax.GradientTransformation]:
        """
        Build an optax optimizer function from the configuration.

        Returns:
            optax.GradientTransformation: Optimizer function.
        """
        optim_config = self.optimizer_config.optim
        opt_class = optim_config._class
        opt_params = optim_config.get("params", {})

        if opt_class == "adam":
            opt_func = partial(optax.adam, **opt_params)
        elif opt_class == "adamw":
            opt_func = partial(optax.adamw, **opt_params)
        elif opt_class == "sgd":
            opt_func = partial(optax.sgd, **opt_params)
        else:
            opt_func = self.configure_optimizer(opt_class)
        return opt_func

    def configure_optimizer(self, opt_name: str, opt_params: dict, **kwargs) -> Callable:
        """
        Build custom optimizer function from a string.

        Args:
            opt_name:   Name of the optimizer function.
            opt_params: Keyword arguments for the optimizer.

        Returns:
            Callable: optimizer function.
        """
        try:
            opt_class_def = resolve_import_from_str(opt_name)
            return partial(opt_class_def, **opt_params, **kwargs)
        except ImportError:
            raise ImportError(f"Optimizer {opt_name} not found.")

    def build_scheduler(self, num_epochs: int, num_train_steps_per_epoch: int = 0) -> Callable:
        """
        Build a learning rate scheduler from the configuration.

        Args:
            num_epochs:                 Number of epochs.
            num_train_steps_per_epoch:  Number of training steps per epoch.
        
        Returns:
            Callable: Learning rate scheduler.
        """
        lr = float(self.optimizer_config.lr)
        scheduler_config = self.optimizer_config.get("scheduler", ConfigDict())
        scheduler_name = scheduler_config.get("name", "constant")
        scheduler_params = scheduler_config.get("params", {})

        decay_steps = scheduler_params.get("decay_steps", num_epochs * num_train_steps_per_epoch)
        assert decay_steps > 0, "decay_steps must be > 0."

        if scheduler_name == "constant":
            scheduler = optax.constant_schedule(lr)
        elif scheduler_name == "cosine_decay":
            scheduler = optax.cosine_decay_schedule(lr, decay_steps=decay_steps, **scheduler_params)
        elif scheduler_name == "warmup_cosine_decay":
            scheduler = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=lr, decay_steps=decay_steps, **scheduler_params)
        else:
            scheduler = self.configure_scheduler(scheduler_name, scheduler_params, decay_steps=decay_steps)

        return scheduler

    def configure_scheduler(self,
                            scheduler_name: str,
                            scheduler_params: dict,
                            num_epochs: int = 0,
                            num_train_steps_per_epoch: int = 0,
                            **kwargs) -> Callable:
        """
        Build custom learning rate scheduler from a string.

        Args:
            scheduler_name:             Name of the scheduler function.
            scheduler_params:           Parameters for the scheduler.
            num_epochs:                 Number of epochs.
            num_train_steps_per_epoch:  Number of training steps per epoch.
        Returns:
            Callable: scheduler function.
        """
        try:
            scheduler_def = resolve_import_from_str(scheduler_name)
            scheduler_params.update(**kwargs)
            return scheduler_def(**scheduler_params, **kwargs)
        except ImportError:
            raise ImportError(f"Scheduler {scheduler_name} not found.")

    def build_grad_transformations(self) -> Tuple[List[optax.GradientTransformation], ...]:
        """
        Build gradient transformations for the optimizer.
        Returns two transforms, first for applying before the optimizer and second for after.

        Returns:
            Pre- and post-optimizer transforms.
        """
        transform_config = self.optimizer_config.get("transforms", ConfigDict())
        pre_transforms = transform_config.get("pre", {})
        post_transforms = transform_config.get("post", {})
        transforms = {"pre": [], "post": []}

        for name, val_or_param in pre_transforms.items():
            transform_fn = resolve_import_from_str(name)
            if isinstance(val_or_param, ConfigDict):
                transforms["pre"].append(transform_fn(**val_or_param))
            else:
                transforms["pre"].append(transform_fn(val_or_param))

        for name, val_or_param in post_transforms.items():
            transform_fn = resolve_import_from_str(name)
            if isinstance(val_or_param, ConfigDict):
                transforms["post"].append(transform_fn(**val_or_param))
            else:
                transforms["post"].append(transform_fn(val_or_param))

        return transforms["pre"], transforms["post"]
