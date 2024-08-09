import os

import jax
import numpy as np
import optax
from absl import logging

from jax_trainer.callbacks.callback import BaseCallback


class LearningRateMonitor(BaseCallback):
    def on_training_epoch_start(self, epoch_idx: int):
        if epoch_idx == 1:
            self._log_lr(epoch_idx - 1)

    def on_training_epoch_end(self, train_metrics, epoch_idx: int):
        self._log_lr(epoch_idx)

    def _log_lr(self, epoch_idx: int):
        last_lr = self.trainer.state.opt_state.hyperparams["schedule"].mean()
        self.trainer.logger.log_scalar("optimizer/lr", last_lr, epoch_idx)