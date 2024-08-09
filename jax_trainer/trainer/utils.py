from functools import wraps
from typing import Iterable, Sequence

import jax
import jax.numpy as jp
import numpy as np
from flax.jax_utils import replicate, unreplicate, pad_shard_unpad

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


def replicate_pjit(
    fn, 
    pmap: bool = False,
    pad_static_argnums: Sequence[int] = (),
    pad_static_argnames: Sequence[str] = (),
    static_return: bool = True,
    **pjit_kwargs
):
    """
    Wrapping a function with `pmap` & `pad_shard_unpad` or `jit`.
    If pmaped, we replicate the input arguments to comply with the pmap API.

    Args:
        fn:                     The function to pmap.
        pmap:                   Whether to pmap the function.
        pad_static_argnums:     The positional arguments that should be considered static. (See `flax.jax_utils.pad_shard_unpad`)
        pad_static_argnames:    The keyword arguments that should be considered static. (See `flax.jax_utils.pad_shard_unpad`)
        static_return:          whether not to un-shard, and un-pad the return value
        pjit_kwargs:            The keyword arguments to pass to `jax.pjit`.

    Returns:
        The replicated function if `jax.local_device_count() > 1`, otherwise the original function.
    """
    static_argnums = pjit_kwargs.get("static_argnums", ())
    static_argnames = pjit_kwargs.get("static_argnames", ())
    
    if pmap:
        p_fn = jax.pmap(fn, **pjit_kwargs)
        p_fn = pad_shard_unpad(
            p_fn, 
            static_argnums=pad_static_argnums, 
            static_argnames=pad_static_argnames, 
            static_return=static_return
        )
    else:
        p_fn = jax.jit(fn, **pjit_kwargs)
        
    def nonzero_rank(v):
        ranks = jax.tree.leaves(jax.tree.map(jp.ndim, v))
        return not len(ranks) or min(ranks) > 0
    
    def wrapper(*args, **kwargs):
        if pmap:
            args = tuple(
                arg if i not in static_argnums and nonzero_rank(arg) else replicate(arg)
                for i, arg in enumerate(args)
            )
            kwargs = {
                k: v if k not in static_argnames and nonzero_rank(v) else replicate(v)
                for k, v in kwargs.items()
            }
                    
        return p_fn(*args, **kwargs)
    
    return wrapper