import json
import os
import time
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union, Literal

import jax
import jax.numpy as jp
from flax.traverse_util import flatten_dict, unflatten_dict
import matplotlib.pyplot as plt
import numpy as np
import torch

from absl import logging
from ml_collections import ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from jax_trainer.metrics.metrics_base import Metric, MultiMetric
from jax_trainer.logger.utils import build_tool_logger

Array = Union[jp.ndarray, np.ndarray]


def reduce_array_to_scalar(array: Array) -> Number | Array:
    """Reduce an array to a scalar value if the array has only one element."""
    if isinstance(array, Array) and array.size == 1:
        return array.item()
    return array


class Logger:
    def __init__(self, config: ConfigDict, full_config: ConfigDict):
        self.config = config
        self.full_config = full_config
        self.logger = build_tool_logger(config, full_config)
        self.logging_stage = "train"

    def infer_type(self, metrics: Dict[str, Any]) -> Literal["scalar", "heatmap", "image", "unknown"]:
        """
        Infer the representation type of the metrics. 
        The type can be one of the following: `scalar`, `heatmap`, `image`, `unknown`.
        - `scalar`:     A scalar value.
        - `heatmap`:    2d array representing a heatmap.
        - `image`:      3d array representing an image.
        - `unknown`:    Unknown(undefined) type.
        
        Args:
            metrics:    Dictionary of metrics.
            
        Returns:
            Dictionary of metric types.
        """
        infer_use_name = self.config.get("infer_use_name", False)
        metrics = flatten_dict(metrics, sep='#')
        
        def check_type(k: str, v: Any):
            if isinstance(v, Number):
                return "scalar"
            elif isinstance(v, Array):
                if v.size == 1:
                    return "scalar"
                elif v.ndim == 2:
                    return "heatmap" if not k.endswith("img") or (infer_use_name and k.endswith("map")) else "image"
                elif v.ndim == 3:
                    return "image"
            return "unknown"
        
        metrics_type = {k: check_type(k, v) for k, v in metrics.items()}
        return unflatten_dict(metrics_type, sep='#')
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = 0, prefix: str = ""):
        metrics = flatten_dict(metrics, sep='_')
        metrics_type = self.infer_type(metrics)
        metrics_to_log = {k: reduce_array_to_scalar(v) for k, v in metrics.items() if metrics_type[k] != "unknown"}
        for k, v in metrics_to_log.items():
            savekey = f"{prefix}/{k}" if prefix else k
            if metrics_type[k] == "image":
                self.log_image(key=savekey, image=v, step=step)
            elif metrics_type[k] == "heatmap":
                # FIXME: This is a temporary solution to log heatmap
                self.log_figure(key=savekey, figure=plt.pcolormesh(v).figure, step=step)
            else:
                self.logger.log_metrics({savekey: v}, step=step)

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
        image: Array,
        step: int = None,
        postfix: str = "",
        logging_stage: Literal["train", "val", "test"] = "train"
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
        log_postfix: str = "",
        logging_stage: Literal["train", "val", "test"] = "train"
    ):
        """
        Logs pyplot figure to the logger using the logger tool.
        
        Args:
            key:            Name of the figure.
            figure:         Figure to log.
            step:           Step number.
            log_postfix:    Postfix to append to the log key.
            logging_stage:  Logging mode. Default is None.
        """
        if isinstance(self.logger, TensorBoardLogger):
            TensorBoardLogger.experiment
            self.logger.experiment.add_figure(
                tag=f"{logging_stage}/{key}{log_postfix}", figure=figure, global_step=step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({f"{logging_stage}/{key}{log_postfix}": figure}, step=step)
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

    @property
    def log_dir(self) -> str:
        """
        Get the log directory of the logger.

        Returns:
            Log directory.
        """
        return self.logger.log_dir