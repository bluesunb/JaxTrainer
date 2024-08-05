from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import numpy as np
from flax.training.common_utils import shard
from flax.core import FrozenDict, freeze, unfreeze
from flax.struct import dataclass, field
from numbers import Number

from jax_trainer.logger.enums import LogMetricMode, LogStage, LogFreq

nonpytree_node = partial(field, pytree_node=False)


@dataclass
class MetricElement:
    value: jax.Array
    mode: LogMetricMode = nonpytree_node(default=LogMetricMode.MEAN)
    log_stage: LogStage = nonpytree_node(default=LogStage.ANY)
    log_freq: LogFreq = nonpytree_node(default=LogFreq.ANY)


def tmp(x: jp.ndarray, metric: MetricElement = None):
    x = 2 * x
    metric.replace(value=metric.value + 1)
    return x, metric


if __name__ == "__main__":
    value = jp.arange(12).reshape(3, 4)
    metric = MetricElement(value=value)
    print(metric)

    tmp = jax.pmap(tmp, donate_argnums=(1,))
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (3, 4))
    x = shard(x)
    metric = shard(metric)
    print(x.shape)
    print(jax.tree.map(jp.shape, metric))
    out = tmp(x, metric)
    print(out)