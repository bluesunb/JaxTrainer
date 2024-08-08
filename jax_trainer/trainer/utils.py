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


# def pad_shard_unpad(wrapped_fn, static_argnums=(0,), static_argnames=(), static_returns=(0,)):
#     """
#     -- See details in `flax.jax_utils.pad_shard_unpad`.
#     Wraps a function with padding, sharding, then un-sharding, un-padding.
#     It extends `static_returns` argument of `flax.jax_utils.pad_shard_unpad` to support multiple return values.

#     Args:
#         wrapped_fn:         The function to wrap.
#         static_argnums:     The positional arguments that should be considered static.
#         static_argnames:    The keyword arguments that should be considered static.
#         static_returns:     If assigned, the positional return values won't be un-sharded and un-padded.

#     Returns:
#         A wrapped function.
#     """
#     def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kwargs):
#         nd = jax.local_device_count()
#         batch_sizes = set()
        
#         for i, arg in enumerate(args):
#             if i not in static_argnums:
#                 batch_sizes |= {t.shape[0] for t in jax.tree.leaves(arg)}
#         for k, v in kwargs.items():
#             if k not in static_argnames:
#                 batch_sizes |= {t.shape[0] for t in jax.tree_leaves(v)}

#         assert len(batch_sizes) == 1, f"Inconsistent batch sizes: {batch_sizes}"
#         bs = batch_sizes.pop()

#         def pad(x):
#             _, *shape = x.shape
#             device_batch, rest = divmod(bs, nd)
        
#             if rest:
#                 x = np.concatenate([x, np.zeros((nd - rest, *shape), dtype=x.dtype)], axis=0)
#                 device_batch += 1
#             if min_device_batch and device_batch < min_device_batch:
#                 x = np.concatenate([x, np.zeros((nd * (min_device_batch - device_batch), *shape), dtype=x.dtype)])
#                 device_batch = min_device_batch
            
#             return x.reshape(nd, device_batch, *shape)
        
#         def maybe_pad(tree, actually_pad=True):
#             if not actually_pad:
#                 return tree
#             return jax.tree.map(pad, tree)
        
#         args = [maybe_pad(arg, i not in static_argnums) for i, arg in enumerate(args)]
#         kwargs = {k: maybe_pad(v, k not in static_argnames) for k, v in kwargs.items()}
#         out = wrapped_fn(*args, **kwargs)

#         def unpad(x):
#             return jax.device_get(x).reshape([np.prod(x.shape[:2]), *x.shape[2:]])[:bs]
        
#         if isinstance(out, tuple):
#             return tuple(o if i in static_returns else jax.tree.map(unpad, o) for i, o in enumerate(out))
        
#         return out if len(static_returns) else jax.tree.map(unpad, out)
    
#     return pad_shard_unpad_wrapper

# def maybe_pmap(fn, pad_static_argnums=(0,), pad_static_argnames=(), force=False, **pmap_kwargs):
#     """
#     Pmap the function if `jax.local_device_count() > 1`.
#     If pmaped, we replicate the input arguments to comply with the pmap API.

#     Args:
#         fn:                     The function to pmap.
#         pad_static_argnums:     The positional arguments that should be considered static. (See `flax.jax_utils.pad_shard_unpad`)
#         pad_static_argnames:    The keyword arguments that should be considered static. (See `flax.jax_utils.pad_shard_unpad`)
#         force:                  Whether to force pmap the function even if there is only one device.
#         pmap_kwargs:            The keyword arguments to pass to `jax.pmap`.

#     Returns:
#         The pmaped function if `jax.local_device_count() > 1`, otherwise the original function.
#     """
#     def wrapper(*args, **kwargs):
#         if jax.local_device_count() > 1 or (jax.local_device_count() == 1 and force):
#             static_argnums = pmap_kwargs.get("static_argnums", ())
#             static_argnames = pmap_kwargs.get("static_argnames", ())
#             args = tuple(
#                 arg 
#                 if i not in static_argnums and min(jax.tree.leaves(jax.tree.map(jp.ndim, arg))) < 1
#                 else replicate(arg)
#                 for i, arg in enumerate(args)
#             )

#             kwargs = {
#                 k: v
#                 if k not in static_argnames and min(jax.tree.leaves(jax.tree.map(jp.ndim, v))) < 1
#                 else replicate(v)
#                 for k, v in kwargs.items()
#             }

#             p_fn = jax.pmap(fn, **pmap_kwargs)
#             p_fn = pad_shard_unpad(fn, static_argnums=pad_static_argnums, static_argnames=pad_static_argnames)
#         else:
#             p_fn = jax.jit(fn, **pmap_kwargs)(*args, **kwargs)
    
#         return p_fn(*args, **kwargs)
#     return wrapper


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
    
    def wrapper(*args, **kwargs):
        if pmap:
            # args = [
            #     arg
            #     if i in static_argnums or min(jax.tree.leaves(jax.tree.map(jp.ndim, arg))) > 0
            #     else replicate(arg)
            #     for i, arg in enumerate(args)
            # ]
            # kwargs = {
            #     k: v
            #     if k in static_argnames or min(jax.tree.leaves(jax.tree.map(jp.ndim, v))) > 0
            #     else replicate(v)
            #     for k, v in kwargs.items()
            # }
            
            args_ = []
            kwargs_ = {}
            for i, arg in enumerate(args):
                if i not in static_argnums:
                    ranks = jax.tree.leaves(jax.tree.map(jp.ndim, arg))
                    if len(ranks) and min(ranks) == 0:
                        arg = replicate(arg)
                
                args_.append(arg)
            
            for k, v in kwargs.items():
                if k not in static_argnames:
                    ranks = jax.tree.leaves(jax.tree.map(jp.ndim, v))
                    if len(ranks) and min(ranks) == 0:
                        v = replicate(v)
                        
                kwargs_[k] = v
                    
        return p_fn(*args_, **kwargs_)
    
    return wrapper