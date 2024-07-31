import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jp
import jax.tree_util as jtr
import numpy as np
from absl import logging


@dataclass
class ArraySpec:
    shape: Tuple[int, ...]
    dtype: Any
    device: Any
    value: Any = 0


def array_to_spec(arr: jp.ndarray) -> ArraySpec:
    return ArraySpec(
        shape=arr.shape,
        dtype=arr.dtype,
        device=str(arr.devices()),
        value=arr.reshape(-1)[0].item()
    )


def np_array_to_spec(arr: np.ndarray) -> ArraySpec:
    return ArraySpec(
        shape=arr.shape,
        dtype=arr.dtype,
        device="numpy",
        value=arr.reshape(-1)[0]
    )


def spec_to_array(spec: ArraySpec) -> jp.ndarray:
    device = spec.device
    if device == "numpy":
        return np.full(spec.shape, spec.value, dtype=spec.dtype)
    else:
        array = jp.full(spec.shape, spec.value, dtype=spec.dtype)
        if isinstance(device, str):
            if ':' in device:
                backend_name, device_id = device.split(':')
                device_id = int(device_id)
            else:
                backend_name, device_id = device, 0

            try:
                device = jax.devices(backend_name)[device_id]
            except Exception as e:
                logging.warning(f"Failed to get device {device} with error: {e}")
                logging.warning(f"Using CPU instead.")

        array = jax.device_put(array, device=device)
        return array


def convert_to_array_spec(data: Any) -> Any:
    if isinstance(data, jp.ndarray):
        return array_to_spec(data)
    elif isinstance(data, np.ndarray):
        return np_array_to_spec(data)
    else:
        logging.warning(f"Data type {type(data)} is not supported for conversion to ArraySpec.")
        return data


def convert_from_array_spec(data: Any) -> Any:
    if isinstance(data, ArraySpec):
        return spec_to_array(data)
    else:
        logging.warning(f"Data type {type(data)} is not supported for conversion from ArraySpec.")
        return data


def save_pytree(pytree: Any, path: str | Path):
    pytree = jtr.tree_map(convert_to_array_spec, pytree)
    with open(path, "wb") as f:
        pickle.dump(pytree, f)


def load_pytree(path: str | Path) -> Any:
    with open(path, "rb") as f:
        pytree = pickle.load(f)
    return jtr.tree_map(convert_from_array_spec, pytree)
