from typing import Any, Dict, Optional

import jax
import orbax.checkpoint as ckpt
from absl import logging
from flax.training import orbax_utils

from jax_trainer.callbacks.callback import BaseCallback
from jax_trainer.utils import class_to_name


class ModelCheckpoint(BaseCallback):
    """
    Callback to save model parameters & mutable variables to the logging directory.
    """
    def __init__(self, config)