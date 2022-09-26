import jax.numpy as jnp

from typing import Any, Sequence
from flax import linen as nn

ModuleDef = Any


class MLP(nn.Module):
    layer_dims: Sequence[int]
    num_classes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((x.shape[0], -1))
        for layer_dim in self.layer_dims:
            x = nn.Dense(features=layer_dim, dtype=self.dtype)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x
