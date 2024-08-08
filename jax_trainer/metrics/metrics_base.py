from dataclasses import make_dataclass
from functools import partial
from typing import Any, Sequence, Tuple, Union

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
            total=jp.zeros_like(self.total, dtype=jp.float32),
            count=jp.zeros_like(self.count, dtype=jp.float32),
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
            count=jp.zeros_like(self.count, dtype=jp.float32),
            mean=jp.zeros_like(self.mean, dtype=jp.float32),
            m2=jp.zeros_like(self.m2, dtype=jp.float32),
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
    

class MultiMetric(PyTreeNode):
    _metric_names: Tuple[str] = field(pytree_node=False, default_factory=tuple)
    metrics: FrozenDict[str, Metric] = field(default_factory=FrozenDict)

    @classmethod
    def create(cls, **metrics):
        metric_names = tuple(metrics.keys())
        metrics = freeze({name: metric.reset() for name, metric in metrics.items()})
        return cls(_metric_names=metric_names, metrics=freeze(metrics))
        
    def reset(self):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].reset()
        return self.replace(metrics=freeze(metrics))
    
    def update(self, **updates):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].update(**updates)
        return self.replace(metrics=freeze(metrics))
    
    def compute(self, keys: Sequence[str] = None) -> dict[str, Any]:
        keys = self._metric_names if keys is None else keys
        return {
            f'{metric_name}': self.metrics[metric_name].compute()
            for metric_name in self._metric_names
            if metric_name in keys
        }
        
    def merge(self, other: 'MultiMetric'):
        metrics = {**self.metrics, **other.metrics}
        return self.replace(metrics=metrics)


if __name__ == "__main__":
    logits = jax.random.normal(jax.random.key(0), (5, 2))
    logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    labels = jp.array([1, 1, 0, 1, 0])
    labels2 = jp.array([0, 1, 1, 1, 1])

    batch_loss = jp.array([1, 2, 3, 4])
    batch_loss2 = jp.array([3, 2, 1, 0])

    metrics = MultiMetric.create(acc=Accuracy(), loss=Welford('loss'))
    print(metrics.compute())
    #{'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
    metrics = metrics.update(logits=logits, labels=labels, loss=batch_loss)
    print(metrics.compute())
    #{'accuracy': Array(0.6, dtype=float32), 'loss': Array(2.5, dtype=float32)}
    metrics = metrics.update(logits=logits2, labels=labels2, loss=batch_loss2)
    print(metrics.compute())
    # {'accuracy': Array(0.7, dtype=float32), 'loss': Array(2., dtype=float32)}
    metrics = metrics.reset()
    print(metrics.compute())
    # {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}

    from flax.nnx import metrics
    metrics = metrics.MultiMetric(
        acc=metrics.Accuracy(),
        loss=metrics.Welford('loss'),
    )
    print('========')
    metrics.update(logits=logits, labels=labels, loss=batch_loss)
    print(metrics.compute())

    metrics.update(logits=logits2, labels=labels2, loss=batch_loss2)
    print(metrics.compute())