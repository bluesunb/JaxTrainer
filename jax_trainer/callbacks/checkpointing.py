from numbers import Number
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import orbax.checkpoint as ckpt
from absl import logging
from flax.training import orbax_utils
from ml_collections import ConfigDict

from jax_trainer.callbacks.callback import BaseCallback
from jax_trainer.utils import class_to_name
from jax_trainer.datasets import DatasetModule


class ModelCheckpoint(BaseCallback):
    """
    Callback to save model parameters & mutable variables to the logging directory.
    """

    def __init__(self, config: ConfigDict, trainer: Any, data_module: DatasetModule):
        super().__init__(config, trainer, data_module)
        assert self.config.get("monitor", None) is not None, "Not specified model monitor metric."

        options = ckpt.CheckpointManagerOptions(
            max_to_keep=self.config.get("save_top_k", 1),
            best_fn=lambda models: models[self.config.monitor],
            best_mode=self.config.get("mode", "min"),
            cleanup_tmp_directories=True,
            create=True
        )

        self.metadata = {
            "trainer": self.trainer.trainer_config.to_dict(),
            "model": self.trainer.model_config.to_dict(),
            "optimizer": self.trainer.optimizer_config.to_dict()
        }
        self.metadata = jax.tree.map(class_to_name, self.metadata)

        item_handlers = {
            "params": ckpt.StandardCheckpointHandler(),
            "metadata": ckpt.JsonCheckpointHandler(),
        }

        if self.trainer.state.extra_variables is not None:
            item_handlers["extra_variables"] = ckpt.StandardCheckpointHandler()
        if self.config.get("save_opt_state", False):
            item_handlers["opt_state"] = ckpt.StandardCheckpointHandler()

        self.manager = ckpt.CheckpointManager(
            directory=Path(self.log_dir).absolute() / "checkpoints/",
            item_names=tuple(item_handlers.keys()),
            item_handlers=item_handlers,
            options=options
        )

    def on_validation_epoch_end(self, val_metrics, epoch_idx: int):
        self.save_model(val_metrics, epoch_idx)

    def save_model(self, eval_metrics: Dict[str, Any], epoch_idx: int):
        logging.info(f"Saving model checkpoint at epoch: {epoch_idx} with metrics: {eval_metrics}")
        assert self.config.monitor in eval_metrics, \
            f"Monitor metric {self.config.monitor} not found in metrics, but has: {'. '.join(eval_metrics.keys())}"

        state = self.trainer.maybe_unreplicate(self.trainer.state)
        save_items = {
            "params": ckpt.args.StandardSave(state.params),
            "metadata": ckpt.args.JsonSave(self.metadata)
        }

        if self.trainer.state.extra_variables is not None:
            save_items["extra_variables"] = ckpt.args.StandardSave(state.extra_variables)
        if self.config.get("save_opt_state", False):
            save_items["opt_state"] = ckpt.args.StandardSave(state.opt_state)

        save_items = ckpt.args.Composite(**save_items)
        eval_metrics = {k: v for k, v in eval_metrics.items() if isinstance(v, (Number, str, bool))}
        self.manager.save(epoch_idx, args=save_items, metrics=eval_metrics)

    def load_model(self, epoch_idx: int = -1) -> Dict[str, Any]:
        logging.info(f"Loading model at epoch: {epoch_idx}")
        if epoch_idx < 0:
            epoch_idx = self.manager.best_step()

        state_dict = self.manager.restore(epoch_idx)
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        return state_dict

    def finalize(self, status: Optional[str] = None):
        logging.info("Finalizing ModelCheckpoint callback.")
        self.manager.wait_until_finished()
        self.manager.close()