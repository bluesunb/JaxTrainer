from functools import wraps
from typing import Iterable

import jax
import jax.numpy as jp
import numpy as np

SEP = {
    "log_stage": "/",
    "log_freq": "_",
    "params": ".",
    "module": "."
}


def loss_fn_return_check(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        assert len(result) == 2 and len(result[1]) == 2, \
            "loss function must return with format (loss, (updates, metrics))"
    return wrapper


def pad_shard_unpad(wrapped_fn, static_argnums=(0,), static_argnames=(), static_returns=(0,)):
    """
    -- See details in `flax.jax_utils.pad_shard_unpad`.
    Wraps a function with padding, sharding, then un-sharding, un-padding.
    It extends `static_returns` argument of `flax.jax_utils.pad_shard_unpad` to support multiple return values.

    Args:
        wrapped_fn:         The function to wrap.
        static_argnums:     The positional arguments that should be considered static.
        static_argnames:    The keyword arguments that should be considered static.
        static_returns:     If assigned, the positional return values won't be un-sharded and un-padded.

    Returns:
        A wrapped function.
    """
    def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kwargs):
        nd = jax.local_device_count()
        batch_sizes = set()
        
        for i, arg in enumerate(args):
            if i not in static_argnums:
                batch_sizes |= {t.shape[0] for t in jax.tree.leaves(arg)}
        for k, v in kwargs.items():
            if k not in static_argnames:
                batch_sizes |= {t.shape[0] for t in jax.tree_leaves(v)}

        assert len(batch_sizes) == 1, f"Inconsistent batch sizes: {batch_sizes}"
        bs = batch_sizes.pop()

        def pad(x):
            _, *shape = x.shape
            device_batch, rest = divmod(bs, nd)
        
            if rest:
                x = np.concatenate([x, np.zeros((nd - rest, *shape), dtype=x.dtype)], axis=0)
                device_batch += 1
            if min_device_batch and device_batch < min_device_batch:
                x = np.concatenate([x, np.zeros((nd * (min_device_batch - device_batch), *shape), dtype=x.dtype)])
                device_batch = min_device_batch
            
            return x.reshape(nd, device_batch, *shape)
        
        def maybe_pad(tree, actually_pad=True):
            if not actually_pad:
                return tree
            return jax.tree.map(pad, tree)
        
        args = [maybe_pad(arg, i not in static_argnums) for i, arg in enumerate(args)]
        kwargs = {k: maybe_pad(v, k not in static_argnames) for k, v in kwargs.items()}
        out = wrapped_fn(*args, **kwargs)

        def unpad(x):
            return jax.device_get(x).reshape([np.prod(x.shape[:2]), *x.shape[2:]])[:bs]
        
        if isinstance(out, tuple):
            return tuple(o if i in static_returns else jax.tree.map(unpad, o) for i, o in enumerate(out))
        
        return out if len(static_returns) else jax.tree.map(unpad, out)
    
    return pad_shard_unpad_wrapper