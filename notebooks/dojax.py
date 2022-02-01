# Library for domain shift in Jax
import jax
import numpy as np
import jax.numpy as jnp


def rotation_matrix(angle):
    """
    Create a rotation matrix that rotates the
    space 'angle'-radians.
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return R


def make_mse_func(model, x_batched, y_batched):
  def mse(params):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
      pred = model.apply(params, x)
      residual = pred - y
      return residual @ residual / 2.0
    # We vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
  return jax.jit(mse)

