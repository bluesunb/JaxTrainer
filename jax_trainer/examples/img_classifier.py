from typing import Any, Tuple, Dict

import jax
import jax.numpy as jp
import optax
import flax
import flax.linen as nn

from jax_trainer.datasets import SupervisedBatch, Batch
from jax_trainer.logger import LogFreq, LogMetricMode, LogStage
from jax_trainer.trainer.train_state import TrainState
from jax_trainer.trainer.trainer import TrainerModule


class ImgClassifierTrainer(TrainerModule):
    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.inputs

    def loss_function(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Dict[str, jp.ndarray] | None, Dict[str, Any]]]:

        rngs = self.get_model_rngs(rng)
        inputs = self.batch_to_input(batch)
        logits, update = state(inputs, params=params, train=train, rngs=rngs, mutable=["batch_stats"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch.targets).mean()

        preds = jp.argmax(logits, axis=-1)
        acc = (preds == batch.targets).mean()
        confusion_matrix = jp.histogram2d(batch.targets, preds, bins=(logits.shape[-1], logits.shape[-1]))[0]
        metrics = {
            "acc": acc,
            "acc_std": {
                "value": acc,
                "mode": LogMetricMode.STD,
                "log_stage": LogStage.EVAL,
            },
            "acc_max": {
                "value": acc,
                "mode": LogMetricMode.MAX,
                "log_stage": LogStage.TRAIN,
                "log_freq": LogFreq.EPOCH
            },
            "conf_matrix": {
                "value": confusion_matrix,
                "mode": LogMetricMode.SUM,
                "log_stage": LogStage.EVAL,
            }
        }

        return loss, (update, metrics)
