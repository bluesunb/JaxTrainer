from dataclasses import make_dataclass
from functools import partial
from typing import Any, Tuple, Union

import jax
import jax.numpy as jp
import numpy as np
from flax.training.common_utils import shard
from flax.core import FrozenDict, freeze, unfreeze
from flax.struct import dataclass, field, PyTreeNode
from numbers import Number

# from jax_trainer.logger.enums import LogMetricMode, LogStage, LogFreq

nonpytree_node = partial(field, pytree_node=False)

def array_factory(value: Any):
    def _factory():
        return jp.array(value)
    return _factory


@dataclass
class Metric:
    def reset(self) -> 'Metric':
        raise NotImplementedError
    
    def update(self, value: Any) -> 'Metric':
        raise NotImplementedError
    
    def compute(self) -> Any:
        raise NotImplementedError


@dataclass
class Average(Metric):
    argname: str = nonpytree_node(default='values')
    total: jp.float32 = field(default_factory=array_factory(0))
    count: jp.float32 = field(default_factory=array_factory(0))

    def reset(self):
        return self.replace(
            total=jp.array(0, dtype=jp.float32),
            count=jp.array(0, dtype=jp.float32),
        )
    
    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: Union[int, float, jax.Array] = kwargs[self.argname]
        total = self.total + (
            values if isinstance(values, (int, float)) else values.sum()
        )
        count = self.count + (1 if isinstance(values, (int, float)) else values.size)
        return self.replace(total=total, count=count)
    
    def compute(self):
        return self.total / self.count
    

@dataclass
class Statistics:
    mean: jp.float32
    standard_error_of_mean: jp.float32
    standard_deviation: jp.float32


@dataclass
class Welford(Metric):
    argname: str = nonpytree_node(default='values')
    count: jp.float32 = field(default_factory=array_factory(0))
    mean: jp.float32 = field(default_factory=array_factory(0))
    m2: jp.float32 = field(default_factory=array_factory(0))

    def reset(self):
        return self.replace(
            count=jp.array(0, dtype=jp.float32),
            mean=jp.array(0, dtype=jp.float32),
            m2=jp.array(0, dtype=jp.float32),
        )

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: Union[int, float, jax.Array] = kwargs[self.argname]
        count = 1 if isinstance(values, (int, float)) else values.size
        original_count = self.count
        new_count = original_count + count
        delta = (
            values if isinstance(values, (int, float)) else values.mean()
        ) - self.mean
        new_mean = self.mean + delta * count / new_count
        m2 = 0.0 if isinstance(values, (int, float)) else values.var() * count
        new_m2 = self.m2 + m2 + delta ** 2 * count * original_count / new_count
        return self.replace(count=new_count, mean=new_mean, m2=new_m2)
    
    def compute(self):
        variance = self.m2 / self.count
        standard_deviation = variance ** 0.5
        sem = standard_deviation / (self.count ** 0.5)
        return Statistics(
            mean=self.mean,
            standard_error_of_mean=sem,
            standard_deviation=standard_deviation,
        )
    

@dataclass
class Accuracy(Average):
    def update(self, *, logits: jax.Array, labels: jax.Array, **_):
        if logits.ndim != labels.ndim + 1 or labels.dtype != jp.int32:
            raise ValueError(
                f'Expected labels.dtype==jnp.int32 and logits.ndim={logits.ndim}=='
                f'labels.ndim+1={labels.ndim + 1}'
            )
        return super().update(values=(logits.argmax(axis=-1) == labels))


@dataclass
class __MultiMetric(Metric):
    def reset(self):
        metrics = {}
        for metric_name in getattr(self, '_metric_names'):
            metrics[metric_name] = getattr(self, metric_name).reset()
        return self.replace(**metrics)
    
    def update(self, **updates):
        metrics = {}
        for metric_name in getattr(self, '_metric_names'):
            metrics[metric_name] = getattr(self, metric_name).update(**updates)
        return self.replace(**metrics)
    
    def compute(self) -> dict[str, Metric]:
        return {
            f'{metric_name}': getattr(self, metric_name).compute()
            for metric_name in getattr(self, '_metric_names')
        }
    

class MultiMetric(PyTreeNode):
    _metric_names: Tuple[str] = field(pytree_node=False, default_factory=tuple)
    metrics: FrozenDict[str, Metric] = field(default_factory=FrozenDict)

    @classmethod
    def create(cls, **metrics):
        metric_names = tuple(metrics.keys())
        metrics = freeze({name: metric for name, metric in metrics.items()})
        return cls(_metric_names=metric_names, metrics=freeze(metrics))
    
    def reset(self):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].reset()
        return self.replace(metrics=metrics)
    
    def update(self, **updates):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].update(**updates)
        return self.replace(metrics=metrics)
    
    def compute(self) -> dict[str, Metric]:
        return {
            f'{metric_name}': self.metrics[metric_name].compute()
            for metric_name in self._metric_names
        }


if __name__ == "__main__":
    avg = Average(argname='loss')
    acc = Accuracy()
    # multi_metric = make_multi_metric(loss=avg, accuracy=acc)
    multi_metric = MultiMetric.create(loss=avg, accuracy=acc)
    print(multi_metric)

    print(jax.tree.map(jp.shape, multi_metric))
    print()