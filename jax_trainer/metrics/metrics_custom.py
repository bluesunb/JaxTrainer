import time
import jax.numpy as jp
from flax.struct import dataclass, field
from jax_trainer.metrics.metrics_base import Metric, nonpytree_node, array_factory


@dataclass
class Max(Metric):
    argname: str = nonpytree_node(default='values')
    max_value: jp.float32 = field(default_factory=array_factory(jp.finfo(jp.float32).min))

    def reset(self):
        return self.replace(max_value=jp.full_like(self.max_value, jp.finfo(jp.float32).min))
    
    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(max_value=jp.maximum(self.max_value, values))
    
    def compute(self):
        return self.max_value
    

@dataclass
class Min(Metric):
    argname: str = nonpytree_node(default='values')
    min_value: jp.float32 = field(default_factory=array_factory(jp.finfo(jp.float32).max))

    def reset(self):
        return self.replace(min_value=jp.finfo(jp.float32).max)
    
    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(min_value=jp.minimum(self.min_value, values))
    
    def compute(self):
        return self.min_value


@dataclass
class Sum(Metric):
    argname: str = nonpytree_node(default='values')
    total: jp.float32 = field(default_factory=array_factory(0))
    reduce: bool = nonpytree_node(default=False)
    
    def reset(self):
        return self.replace(total=jp.zeros_like(self.total, dtype=jp.float32))
    
    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(total=self.total + (values if not self.reduce else values.sum()))
    
    def compute(self):
        return self.total
    

@dataclass
class Lambda(Metric):
    argname: str = nonpytree_node(default='values')
    update_fn: callable = nonpytree_node(default=lambda v, x: x)
    comput_fn: callable = nonpytree_node(default=lambda v: v)
    values: jp.float32 = field(default_factory=array_factory(0))
    
    def reset(self):
        return self.replace(values=jp.zeros_like(self.values, dtype=jp.float32))

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(values=self.update_fn(self.values, values))
    
    def compute(self):
        return self.values


@dataclass
class Timeit(Metric):
    elapsed: float = field(default=0.0)

    def reset(self) -> Metric:
        return self.replace(elapsed=time.time())
    
    def update(self, **kwargs):
        return self
    
    def compute(self):
        return time.time() - self.start