from typing import Any, Optional, Literal

from ml_collections import ConfigDict

from jax_trainer.datasets import DatasetModule


class BaseCallback:
    def __init__(self, config: ConfigDict, trainer: Any, data_module: Optional[DatasetModule] = None):
        """
        Base class for all callbacks.

        Args:
            config:     Configuration for the callback.
            trainer:    Trainer to attach the callback to.
            data_module: DatasetModule to attach the callback
        """
        self.config = config
        self.trainer = trainer
        self.data_module = data_module

        self.log_dir = self.trainer.log_dir
        self.freq_epoch = config.get("freq_epoch", 1)

    def on_training_start(self):
        """Called when the training starts."""
        pass

    def on_training_end(self):
        """Called when the training ends."""
        pass

    def __on_training_epoch_start(self, epoch_idx: int):
        """
        Called when a training epoch starts.

        Args:
            epoch_idx: Index of the epoch.
        """
        if epoch_idx % self.freq_epoch == 0:
            self.on_training_epoch_start(epoch_idx)

    def on_training_epoch_start(self, epoch_idx: int):
        """
        Called when a training epoch starts. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the epoch.
        """
        pass

    def __on_training_epoch_end(self, train_metrics, epoch_idx: int):
        """
        Called when a training epoch ends.

        Args:
            train_metrics:  Metrics for the training epoch.
            epoch_idx:      Index of the epoch.
        """
        if epoch_idx % self.freq_epoch == 0:
            self.on_training_epoch_end(train_metrics, epoch_idx)

    def on_training_epoch_end(self, train_metrics, epoch_idx: int):
        """
        Called when a training epoch ends. To be implemented by subclasses.

        Args:
            train_metrics:  Metrics for the training epoch.
            epoch_idx:      Index of the epoch.
        """
        pass

    def __on_validation_epoch_start(self, epoch_idx: int):
        """
        Called when a validation epoch starts.

        Args:
            epoch_idx: Index of the epoch.
        """
        if epoch_idx % self.freq_epoch == 0:
            self.on_validation_epoch_start(epoch_idx)

    def on_validation_epoch_start(self, epoch_idx: int):
        """
        Called when a validation epoch starts. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the epoch.
        """
        pass

    def __on_validation_epoch_end(self, eval_metrics, epoch_idx: int):
        """
        Called when a validation epoch ends.

        Args:
            eval_metrics:   Metrics for the validation epoch.
            epoch_idx:      Index of the epoch.
        """
        if epoch_idx % self.freq_epoch == 0:
            self.on_validation_epoch_end(eval_metrics, epoch_idx)

    def on_validation_epoch_end(self, eval_metrics, epoch_idx: int):
        """
        Called when a validation epoch ends. To be implemented by subclasses.

        Args:
            eval_metrics:   Metrics for the validation epoch.
            epoch_idx:      Index of the epoch.
        """
        pass

    def on_test_epoch_start(self, epoch_idx: int):
        """Called when a test epoch starts."""
        pass

    def on_test_epoch_end(self, test_metrics, epoch_idx: int):
        """
        Called when a test epoch ends.

        Args:
            test_metrics:   Metrics for the test epoch.
            epoch_idx:      Index of the epoch.
        """
        pass

    def finalize(self, status: Optional[Literal["success", "failed"]] = None):
        """
        Called when the training ends.

        Args:
            status: Status of the training end.
        """
        pass


class TrainingCallback(BaseCallback):
    def on_training_step(self, train_metrics, epoch_idx: int, step_idx: int):
        pass
