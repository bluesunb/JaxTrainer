import jax
import jax.numpy as jp
import flax.linen as nn


class SimpleNetwork(nn.Module):
    hidden_size: int
    num_classes: int
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        while x.shape[1] > 4:
            x = nn.Conv(self.hidden_size, (3, 3), strides=2)(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.gelu(x)
            x = nn.Conv(self.hidden_size, (3, 3))(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.gelu(x)
            
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_classes)(x)
        x = nn.log_softmax(x)
        return x