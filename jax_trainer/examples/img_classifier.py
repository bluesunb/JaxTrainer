from typing import Any, Dict, Tuple

import jax.numpy as jp
import optax
from jax.numpy import ndarray

from jax_trainer.datasets import Batch, SupervisedBatch
from jax_trainer.metrics import Average, Lambda, Max, MultiMetric, Sum, Welford
from jax_trainer.trainer.train_state import TrainState

# from jax_trainer.trainer.trainer import TrainerModule
from jax_trainer.trainer.trainer import Trainer


class ImgClassifierTrainer(Trainer):
    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.inputs

    def loss_function(
        self,
        params: Any,
        *,
        state: TrainState,
        batch: Batch,
        rng: ndarray,
        train: bool = True,
    ) -> Tuple[Any | Tuple[Dict[str, ndarray] | None | Dict[str, Any]]]:
        rngs = self.get_model_rngs(rng)
        inputs = self.batch_to_input(batch)
        logits, update = state(inputs, params=params, train=train, rngs=rngs, mutable=["batch_stats"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch.targets).mean()

        preds = jp.argmax(logits, axis=-1)
        acc = (preds == batch.targets).mean()
        confusion_matrix = jp.histogram2d(batch.targets, preds, bins=(logits.shape[-1], logits.shape[-1]))[0]
        loss_info = {"acc": acc, "conf_matrix": confusion_matrix}

        return loss, (update, loss_info)

    def init_train_metrics(self) -> MultiMetric:
        num_classes = self.model_config.hparams.num_classes
        return MultiMetric.create(
            loss=Average("loss"),
            acc=Lambda("acc"),
            acc_stat=Welford("acc"),
            acc_max=Max("acc"),
            conf_matrix=Sum("conf_matrix", reduce=False, total=jp.zeros((num_classes, num_classes))),
            # conf_matrix=Sum('conf_matrix', reduce=False),
            grad_norm=Lambda("grads_norm"),
            grad_norm_max=Max("grads_norm"),
            param_norm=Lambda("params_norm"),
        )

    def init_eval_metrics(self) -> MultiMetric:
        return MultiMetric.create(
            loss=Average("loss"),
            acc=Lambda("acc"),
            acc_stat=Welford("acc"),
            acc_max=Max("acc"),
            conf_matrix=Sum("conf_matrix", reduce=False),
        )
