import json
import os
import time
from collections import defaultdict
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union, Literal

import jax
import jax.numpy as jp
import jax.tree_util as jtr
import matplotlib.pyplot as plt
import numpy as np
import torch

from absl import logging
from flax.core import FrozenDict
from ml_collections import ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from jax_trainer.logger.enums import LogFreq, LogMetricMode
from jax_trainer.logger.metrics import Metrics, HostMetrics, get_metrics
from jax_trainer.logger.utils import build_tool_logger

ArrayLike = Union[jp.ndarray, np.ndarray]


def _get_new_metrics_dict() -> Dict[str, Dict[str, float | jp.ndarray | LogMetricMode]]:
    """
    Returns defaultdict for metrics like:
    {
        'KEY': {
            'value': float | jp.ndarray,
            'mode': LogMetricMode,
        }
    }

    Returns:
        Dict[str, Dict[str, float | jp.ndarray | LogMetricMode]]: New metrics dictionary.
    """
    return defaultdict(lambda: {"value": 0.0, "mode": "mean"})


def reduce_array_to_scalar(array: jp.ndarray | np.ndarray) -> float | ArrayLike:
    if isinstance(array, ArrayLike) and array.size == 1:
        return array.item()
    return array


class Logger:
    """
    Logger class to log metrics, images, etc. to TensorBoard or Wandb (or more in the future).
    """

    def __init__(self, config: ConfigDict, full_config: ConfigDict):
        self.config = config
        self.full_config = full_config

        self.logger = build_tool_logger(config, full_config)
        self.logging_stage = "train"
        self.log_steps_every = config.get("log_steps_every", 50)

        self.step_metrics = _get_new_metrics_dict()
        self.step_count = 0
        self.total_step_counter = 0

        self.epoch_metrics = _get_new_metrics_dict()
        self.epoch_idx = 0
        self.epoch_element_count = 0
        self.epoch_step_count = 0
        self.epoch_log_prefix = ""
        self.epoch_start_time = None

    def log_metrics(self, metrics: HostMetrics, step: int, postfix: str = "") -> None:
        """
        Log metrics to the logger using the logger tool.

        Args:
            metrics:    Metrics to log.
            step:       Step number.
            postfix:    Postfix to append to the log key.
        """
        metrics_to_log = {}
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, (jp.ndarray, np.ndarray)):
                if metric_value.size == 1:
                    metric_value = metric_value.item()
                else:  # value is a non-scalar array
                    continue

            save_key = f"{metric_key}_{postfix}" if postfix else metric_key
            metrics_to_log[save_key] = metric_value

        if len(metrics_to_log) > 0:
            self.logger.log_metrics(metrics_to_log, step)

    def log_scalar(self, metric_key: str, metric_value: Union[float, jp.ndarray], step: int, postfix: str = "") -> None:
        """
        Log scalar to the logger using the logger tool.

        Args:
            metric_key:     Metric key.
            metric_value:   Metric value.
            step:           Step number.
            postfix:        Postfix to append to the log key.
        """
        self.log_metrics({metric_key: metric_value}, step, postfix)

    def log_image(
        self,
        key: str,
        image: ArrayLike,
        step: int = None,
        postfix: str = "",
        logging_stage: Optional[str] = None
    ):
        """
        Logs image to the logger using the logger tool.

        Args:
            key:            Name of the image.
            image:          Image to log.
            step:           Step number.
            postfix:        Postfix to append to the log key.
            logging_stage:  Logging stage. Default is None.
        """
        step = step or self.total_step_counter
        logging_stage = logging_stage or self.logging_stage

        if isinstance(image, jp.ndarray):
            image = jax.device_get(image)

        log_key = f"{logging_stage}/{key}{postfix}"
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                tag=log_key,
                img_tensor=image,
                global_step=step,
                dataformats="HWC"
            )
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=log_key, image=image, step=step)
        else:
            raise ValueError(f"Unsupported logger type: {type(self.logger)}")

    def log_figure(
        self,
        key: str,
        figure: plt.Figure,
        step: int = None,
        postfix: str = "",
        logging_stage: Optional[str] = None
    ):
        """
        Logs a matplotlib figure to the logger using the logger tool.

        Args:
            key:            Name of the figure.
            figure:         Figure to log.
            step:           Step number.
            postfix:        Postfix to append to the log key.
            logging_stage:  Logging stage. Default is None.
        """
        step = step or self.total_step_counter
        logging_stage = logging_stage or self.logging_stage

        log_key = f"{logging_stage}/{key}{postfix}"
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(tag=log_key, figure=figure, global_step=step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({log_key: figure}, step=step)
        else:
            raise ValueError(f"Unsupported logger type: {type(self.logger)}")

    def log_embedding(
        self,
        key: str,
        encodings: np.ndarray,
        step: int = None,
        metadata: Optional[Any] = None,
        images: Optional[np.ndarray] = None,
        postfix: str = "",
        logging_stage: Optional[str] = None
    ):
        """
        Logs embeddings to the logger using the logger tool.

        Args:
            key:            Name of the embedding.
            encodings:      Encodings to log.
            step:           Step number.
            metadata:       Metadata for the embeddings.
            images:         Images for the embeddings.
            postfix:        Postfix to append to the log key.
            logging_stage:  Logging stage. Default is None.
        """
        step = step or self.total_step_counter
        logging_stage = logging_stage or self.logging_stage

        log_key = f"{logging_stage}/{key}{postfix}"
        if isinstance(self.logger, TensorBoardLogger):
            images = np.transpose(images, (0, 3, 1, 2))  # NHWC -> NCHW
            images = torch.from_numpy(images)
            self.logger.experiment.add_embedding(
                tag=log_key,
                mat=encodings,
                mateadata=metadata,
                label_img=images,
                global_step=step
            )
        elif isinstance(self.logger, WandbLogger):
            logging.warning("Wandb does not support embedding logging.")
        else:
            raise ValueError(f"Unsupported logger type: {type(self.logger)}")

    def finalize(self, status: Literal["success", "failed"]):
        """
        Close the logger and finalize the logging

        Args:
            status: The status of the run. (e.g. "success", "failed")
        """
        self.logger.finalize(status)

    def log_step(self, metrics: Metrics) -> Metrics:
        """
        Log metrics that are calculated for each step.

        It calculates the metrics up to current steps and logs them if the log_steps_every is reached.

        Args:
            metrics:   Running Metrics for the step.

        Returns:
        """
        self.epoch_step_count += 1

        if self.logging_stage == "train" and self.log_steps_every > 0:
            self.step_count += 1
            self.total_step_counter += 1
            if self.step_count >= self.log_steps_every:
                if self.step_count > self.log_steps_every:
                    logging.warning(f"Logging frequency is too high. "
                                    f"Step count: {self.step_count}, "
                                    f"log_steps_every: {self.log_steps_every}")

                metrics, step_metrics = get_metrics(metrics, log_freq=LogFreq.STEP, reset_metrics=True)
                final_step_metrics = self._finalize_metrics(step_metrics)
                self.log_metrics(final_step_metrics, step=self.total_step_counter, postfix="step")
                self._reset_step_metrics()

        return metrics

    def start_epoch(self, epoch: int, stage: Literal["train", "valid", "test"] = "train"):
        """
        Start epoch logging by reset epoch metrics and related fields.

        Args:
            epoch:  The index of the epoch.
            stage:  Logging stage. Default is "train".
        """
        assert stage in ["train", "valid", "test"], f"Invalid stage {stage}."
        self.logging_stage = stage
        self.epoch_idx = epoch
        self._reset_epoch_metrics()

    def end_epoch(self, metrics: Metrics, save_metrics: bool = False) -> Tuple[Metrics, HostMetrics]:
        """
        Ends the current metrics and finalize(aggregate) it with the epoch metrics.

        Args:
            metrics:        The metrics to log.
            save_metrics:   Whether to save the metrics to a file or not.

        Returns:
            Tuple[Metrics, HostMetrics]: The running metrics and the finalized epoch metrics
        """
        self.log_epoch_scalar("time", time.time() - self.epoch_start_time)
        metrics, epoch_metrics = get_metrics(metrics, log_freq=LogFreq.EPOCH, reset_metrics=True)
        epoch_metrics.update(self.epoch_metrics)
        final_epoch_metrics = self._finalize_metrics(epoch_metrics)

        self.log_metrics(
            final_epoch_metrics,
            step=self.epoch_idx,
            postfix="epoch" if self.logging_stage == "train" else self.logging_stage
        )

        if save_metrics:
            self.save_metrics(f"{self.logging_stage}_epoch_{self.epoch_idx:04d}", final_epoch_metrics)

        if (
            self.logging_stage == "train"
            and self.log_steps_every > 0
            and self.epoch_step_count < self.log_steps_every
        ):
            # Log the metrics for the remaining steps if the epoch has fewer steps than the logging frequency.
            logging.info(
                f"Training epoch has fewer steps ({self.epoch_step_count}) than logging frequency ({self.log_steps_every}).")
            metrics, _ = get_metrics(metrics, log_freq=LogFreq.STEP, reset_metrics=True)
            self._reset_step_metrics()

        self._reset_epoch_metrics()
        return metrics, final_epoch_metrics

    def save_metrics(self, filename: str, metrics: Metrics):
        """
        Save the metrics to a file in JSON.

        Args:
            filename:   The filename to save the metrics.
            metrics:    The metrics to save.
        """
        metrics = {k: metrics[k] for k in metrics if isinstance(metrics[k], (Number, str, bool))}
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def log_epoch_scalar(self, key: str, value: Union[Number, jp.ndarray]):
        """Just set the value of the key in the epoch metrics."""
        self.epoch_metrics[key] = value

    def _reset_step_metrics(self):
        """Resets the step related metrics field."""
        self.step_count = 0

    def _reset_epoch_metrics(self):
        """Resets the epoch related metrics field."""
        self.epoch_metrics = {}
        self.epoch_step_count = 0
        self.epoch_start_time = time.time()

    def _finalize_metrics(self, metrics: HostMetrics) -> HostMetrics:
        """
        Append the logging stage to the key and reduce the array to make logable finalized metrics.

        Args:
            metrics: The metrics to finalize.

        Returns:
            HostMetrics: The finalized metrics.
        """
        final_metrics = {f'{self.logging_stage}/{key}' if "/" not in key else key: value for key, value in
                         metrics.items()}
        final_metrics = jax.tree.map(reduce_array_to_scalar, final_metrics)
        return final_metrics

    @property
    def log_dir(self):
        return self.logger.log_dir
