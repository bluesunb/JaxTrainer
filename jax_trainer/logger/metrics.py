from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import numpy as np
from numbers import Number
from flax.core import FrozenDict, freeze, unfreeze

from jax_trainer.logger.enums import LogMetricMode, LogStage, LogFreq

# Immutable metrics for compilation
ImmutableMetricElement = FrozenDict[str, Number | jax.Array | LogMetricMode | LogFreq | LogStage]
ImmutableMetrics = FrozenDict[str, ImmutableMetricElement]

# Mutable metrics for updating/editing
MutableMetricElement = Dict[str, Number | jax.Array | LogMetricMode | LogFreq | LogStage]
MutableMetrics = Dict[str, MutableMetricElement]

# Metrics logged per step
StepMetrics = Dict[str, Number | jax.Array | MutableMetrics]

# Combined metrics
# Union of ImmutableMetricElement and MutableMetricElement, Note that array components are must be jax.Array
MetricElement = ImmutableMetricElement | MutableMetricElement
# Union of ImmutableMetrics and MutableMetrics, Note that array components are must be jax.Array
Metrics = ImmutableMetrics | MutableMetrics

# Metrics on host (for logging, outside of calculation loop)
HostMetricsElement = Number | np.ndarray
HostMetrics = Dict[str, HostMetricsElement]


def update_metrics(
    global_metrics: Metrics | None,
    step_metrics: StepMetrics,
    train: bool,
    batch_size: int | jax.Array,
) -> ImmutableMetrics:
    """
    Update value of metrics using new step metrics.

    Metrics has the following structure:
    {
        "MetricKey1": {
            "value": (Number | jax.Array),      # Value of the metric.
            "count": (Number | jax.Array),      # Counting for number of metric calculation (to calculate avg or std).
            "mode": (Optional[LogMetricMode]),  # Determines how to update the metric.
            "log_freq": (Optional[LogFreq]),    # Determines when to log the metric.
        }
        {...}
    }

    Args:
        global_metrics: Global metrics to update. If None, a new dictionary is created.
        step_metrics:   Metrics to update global metrics with.
        train:          Whether metrics are logged during training or evaluation.
        batch_size:     Batch size of the step.

    Returns:
        Updated global metrics.
    """

    global_metrics = global_metrics or {}

    if isinstance(global_metrics, FrozenDict):
        global_metrics = unfreeze(global_metrics)

    for key in step_metrics:
        metric_in = step_metrics[key]
        if not isinstance(metric_in, dict):
            metric_in = {"value": metric_in}

        value = metric_in["value"]
        mode = metric_in.get("mode", LogMetricMode.MEAN)
        log_freq = metric_in.get("log_freq", LogFreq.ANY)
        log_stage = metric_in.get("log_stage", LogStage.ANY)
        count = metric_in.get("count", None)  # Number of samples in the batch

        if (log_stage == LogStage.TRAIN and not train) or (log_stage not in (LogStage.TRAIN, LogStage.ANY) and train):
            # if log_stage doesn't make sense for the current stage, skip
            continue

        postfix = []
        if train:
            if log_freq in (LogFreq.ANY, LogFreq.STEP):
                postfix.append((LogFreq.STEP, "step"))  # Add step logging condition
            if log_freq in (LogFreq.ANY, LogFreq.EPOCH):
                postfix.append((LogFreq.EPOCH, "epoch"))  # Add epoch logging condition
        else:
            # Fixed logging frequency for evaluation when not training
            postfix.append((LogFreq.EPOCH, "epoch"))

        for freq, freq_name in postfix:
            key_name = f"{key}_{freq_name}" if freq_name else key
            global_metrics = _update_single_metrics(
                global_metrics=global_metrics,
                key=key_name,
                value=value,
                mode=mode,
                log_freq=freq,
                log_stage=log_stage,
                count=count,
                batch_size=batch_size,
            )

    global_metrics = freeze(global_metrics)
    return global_metrics


def _update_single_metrics(
    global_metrics: MutableMetrics,
    key: str,
    value: Any,
    mode: LogMetricMode,
    log_freq: LogFreq,
    log_stage: LogStage,
    count: Any,
    batch_size: int | jax.Array,
) -> MutableMetrics:
    """
    Update values of a single metric for given key and log settings.

    Args:
        global_metrics: Global metrics to update.
        key:            Key of the metric to update.
        value:          Value of the metric to update.
        mode:           Metric mode.
        log_freq:       Logging frequency.
        log_stage:      Logging stage.
        count:          Count of elements in the metrics to update.
        batch_size:     Batch size of the step.

    Returns:
        (MutableMetrics): Updated global metrics.
    """
    metrics_dict = global_metrics.get(key, {"value": 0.0, "count": 0})
    metrics_dict["mode"] = mode
    metrics_dict["log_freq"] = log_freq
    metrics_dict["log_stage"] = log_stage

    # count update
    if count is None:
        if mode == LogMetricMode.MEAN:
            count = batch_size
            value = value * batch_size
        else:
            count = 1
    metrics_dict["count"] += count

    # value update according to mode
    if mode == LogMetricMode.MEAN:
        metrics_dict["value"] += value
    elif mode == LogMetricMode.SUM:
        metrics_dict["value"] += value
    elif mode == LogMetricMode.SINGLE:
        metrics_dict["value"] = value
    elif mode == LogMetricMode.MAX:
        metrics_dict["value"] = jp.maximum(metrics_dict["value"], value)
    elif mode == LogMetricMode.MIN:
        metrics_dict["value"] = jp.minimum(metrics_dict["value"], value)
    elif mode == LogMetricMode.STD:
        metrics_dict["value"] += value
        if "value2" not in metrics_dict:
            assert key not in global_metrics, (  # assert key is logged first time
                f"For metric(mode: {mode}) {key}, "
                "the second moment (mean square) of the metric must be provided "
                "if the metric is already logged."
            )
            metrics_dict["value2"] = 0.0
        metrics_dict["value2"] += value ** 2
    else:
        raise ValueError(f"Invalid metric mode: {mode}")

    global_metrics[key] = metrics_dict
    return global_metrics


def get_metrics(
    global_metrics: Metrics,
    log_freq: LogFreq = LogFreq.ANY,
    reset_metrics: bool = True,
) -> Tuple[ImmutableMetrics, HostMetrics]:
    """
    Get metrics from global metrics that match the log frequency and aggregate them into host metrics.
    If `reset_metrics` is True, the metrics are reset to zero.
    

    Args:
        global_metrics: Global metrics to calculate metrics to log from.
        log_freq:       Logging frequency of the metrics to log.
        reset_metrics:  Whether to reset the metrics after logging.

    Returns:
        (0): (ImmutableMetrics): Updated global metrics.
        (1): (HostMetrics):      Aggregated metrics to log which are matched with the log frequency.
    """
    if isinstance(global_metrics, FrozenDict) and reset_metrics:
        global_metrics = unfreeze(global_metrics)
    host_metrics = jax.device_get(global_metrics)
    metrics = {}

    for key in host_metrics:
        if log_freq == LogFreq.ANY or log_freq == host_metrics[key]["log_freq"]:
            host_key = key.rsplit("_", 1)[0]  # Remove postfix indicating logging stage (train/test)
            value = host_metrics[key]["value"]
            count = host_metrics[key]["count"]

            if host_metrics[key]["mode"] == LogMetricMode.MEAN:
                value = value / count
            elif host_metrics[key]["mode"] == LogMetricMode.STD:
                value = value / count
                value2 = host_metrics[key]["value2"] / count
                value = np.sqrt(value2 - value ** 2)
            metrics[host_key] = value

            if reset_metrics:
                # reset global metrics to zero
                global_metrics[key]["value"] = jp.zeros_like(global_metrics[key]["value"])
                global_metrics[key]["count"] = jp.zeros_like(global_metrics[key]["count"])

    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)

    return global_metrics, metrics
