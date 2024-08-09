import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging

from jax_trainer.callbacks.callback import BaseCallback


class ConfusionMatrixCallback(BaseCallback):
    def on_validation_epoch_end(self, eval_metrics, epoch_idx: int):
        return self._visualize_confusion_matrix(eval_metrics, epoch_idx)

    def on_test_epoch_end(self, test_metrics, epoch_idx: int):
        return self._visualize_confusion_matrix(test_metrics, epoch_idx)

    def _visualize_confusion_matrix(self, metrics, epoch_idx):
        conf_keys = list(filter(lambda x: x.endswith("conf_matrix"), metrics.keys()))
        assert len(conf_keys) > 0, f"No confusion matrix found in metrics. We got: {metrics.keys()}"
        conf_key = conf_keys[0]
        conf_matrix = metrics[conf_key]

        if self.config.get("normalize", False):
            conf_matrix = conf_matrix / conf_matrix.sum(axis=-1, keepdims=True)
            repr_fmt = self.config.get("repr_fmt", ".2%")
        else:
            repr_fmt = self.config.get("repr_fmt", "d")

        fig, ax = plt.subplots(figsize=self.config.get("figsize", (8, 8)), dpi=self.config.get("dpi", 100))

        sns.heatmap(conf_matrix, annot=True, fmt=repr_fmt, cmap=self.config.get("cmap", "Blues"), ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(np.arange(conf_matrix.shape[0]) + 0.5)
        ax.set_yticks(np.arange(conf_matrix.shape[1]) + 0.5)
        
        if self.data_module is not None and hasattr(self.data_module.train_set, "class_names"):
            ax.set_xticklabels(self.data_module.train_set.class_names, rotation=45)
            ax.set_yticklabels(self.data_module.train_set.class_names, rotation=0)
        else:
            ax.set_xticklabels(range(conf_matrix.shape[0]))
            ax.set_yticklabels(range(conf_matrix.shape[1]))
        
        fig.tight_layout()
        self.trainer.logger.log_figure("confusion_matrix", fig, epoch_idx)