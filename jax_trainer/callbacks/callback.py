from typing import Any, Optional

from ml_collections import ConfigDict

from jax_trainer.datasets import DatasetModule


class BaseCallback:
    def __init__(self, config: ConfigDict, trainer: Any, datamodule: Optional[DatasetModule] = None):
        """
        Base class for all callbacks.

        Args:
            config:     Configuration for the callback.
            trainer:    Trainer to attach the callback to.
            datamodule: DatasetModule to attach the callback
        """
        self.config = config
        self.trainer = trainer
        self.datamodule = datamodule
        self.every_n_epochs = config.get("every_n_epochs", 1)

    def on_training_start(self):
        """Called when the training starts."""
        pass

    def on_training_end(self):
        """Called when the training ends."""
        pass

    
    def _on_training_epoch_start(self, epoch_idx: int):
        """
        Called when a training epoch starts.

        Args:
            epoch_idx: Index of the epoch.
        """
        if epoch_idx % self.every_n_epochs == 0:
            self.on_training_epoch_start(epoch_idx)

    def on_training_epoch_start(self, epoch_idx: int):
        """
        Called when a training epoch starts. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the epoch.
        """
        pass

    def _on_training_epoch_end(self, train_metrics, epoch_idx: int):
        """
        Called when a training epoch ends.

        Args:
            train_metrics:  Metrics for the training epoch.
            epoch_idx:      Index of the epoch.
        """
        if epoch_idx % self.every_n_epochs == 0:
            self.on_training_epoch_end(train_metrics, epoch_idx)

    def on_training_epoch_end(self, train_metrics, epoch_idx: int):
        """
        Called when a training epoch ends. To be implemented by subclasses.

        Args:
            train_metrics:  Metrics for the training epoch.
            epoch_idx:      Index of the epoch.
        """
        pass

    def _on_validation_epoch_start(self, epoch_idx: int):
        """
        Called when a validation epoch starts.

        Args:
            epoch_idx: Index of the epoch.
        """
        if epoch_idx % self.every_n_epochs == 0:
            self.on_validation_epoch_start(epoch_idx)
    
    def on_validation_epoch_start(self, epoch_idx: int):
        """
        Called when a validation epoch starts. To be implemented by subclasses.

        Args:
            epoch_idx: Index of the epoch.
        """
        pass

    def _on_validation_epoch_end(self, val_metrics, epoch_idx: int):
        """
        Called when a validation epoch ends.

        Args:
            val_metrics:    Metrics for the validation epoch.
            epoch_idx:      Index of the epoch.
        """
        if epoch_idx % self.every_n_epochs == 0:
            self.on_validation_epoch_end(val_metrics, epoch_idx)

    def on_validation_epoch_end(self, val_metrics, epoch_idx: int):
        """
        Called when a validation epoch ends. To be implemented by subclasses.

        Args:
            val_metrics:    Metrics for the validation epoch.
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

    def finalize(self, status: Optional[str] = None):
        """
        Called when the training ends.

        Args:
            status: Status of the training end.
        """
        pass