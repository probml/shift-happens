import jax.numpy as jnp

from typing import Any, Callable
from flax import linen as nn

ModuleDef = Any


class LeNet5(nn.Module):
  num_classes: int
  act: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = True):
    """Network inspired by LeNet-5."""
    x = self.act(nn.Conv(features=6, kernel_size=(5, 5), padding="SAME", dtype=self.dtype)(x))
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = self.act(nn.Conv(features=16, kernel_size=(5, 5), padding="VALID", dtype=self.dtype)(x))
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = x.reshape((x.shape[0], -1))
    x = self.act(nn.Dense(features=120, dtype=self.dtype)(x))
    x = self.act(nn.Dense(features=84, dtype=self.dtype)(x))
    x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)
    return x
