import flax.linen as nn
from typing import Callable
import jax.numpy as jnp

class LeNet5(nn.Module):
  num_classes : int
  activation  : Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  @nn.compact
  def __call__(self, x):
    """Aleyna's network inspired by LeNet-5."""
    x = x if len(x.shape) > 1 else x[None, :]
    x = x.reshape((x.shape[0], 28, 28, 1))
    x = self.activation(nn.Conv(features=6, kernel_size=(5,5), padding="SAME")(x))
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = self.activation(nn.Conv(features=16, kernel_size=(5,5), padding="VALID")(x))
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = x.reshape((x.shape[0], -1))
    x = self.activation(nn.Dense(features=120)(x))
    x = self.activation(nn.Dense(features=84)(x))
    x = nn.Dense(features=self.num_classes)(x)
    x = nn.log_softmax(x)
    return x


class MLPDataV1(nn.Module):
    num_outputs: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(800)(x))
        x = nn.relu(nn.Dense(500)(x))
        x = nn.Dense(self.num_outputs)(x)
        x = nn.log_softmax(x)
        return x


class MLPWeightsV1(nn.Module):
    num_outputs: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(self.num_outputs)(x)
        return x

