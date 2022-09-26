import jax.numpy as jnp
from jax import vmap, ops, lax
from jax.scipy.stats import multivariate_normal

from typing import Any, Tuple

Array = Any


def init(inputs: Array, labels: Array):
    _, *input_shape = inputs.shape
    num_classes = jnp.unique(labels).size

    input_size = jnp.product(jnp.array(input_shape))

    mean = jnp.zeros((num_classes, input_size))
    scale = jnp.repeat(jnp.eye(input_size)[None, ...], repeats=num_classes, axis=0)
    counts = jnp.zeros((num_classes,))
    state = (mean, scale, counts)

    def init_step(state, carry):
        input, cls = carry
        state = update(state, input.reshape((1, -1)), cls)
        return state, None

    state, _ = lax.scan(init_step, state, (inputs, labels))

    return state


def predict(state: Tuple[Array, Array, Array], inputs: Array, prior: Array):
    mean, scale, _ = state

    def cond_prob(input):
        return multivariate_normal.logpdf(input.reshape((1, -1)),
                                          mean=mean,
                                          cov=scale)

    logits = vmap(cond_prob)(inputs)
    return logits + prior


def update(state: Tuple[Array, Array, Array], inputs: Array, cls: int):
    mean, scale, counts = state
    n = counts[cls] + len(inputs)

    prev_sum = mean[cls] * counts[cls]
    cur_sum = inputs.reshape((len(inputs), -1)).sum(axis=0)
    running_avg = (prev_sum + cur_sum) / n

    mean = ops.index_update(mean, jnp.index_exp[cls], running_avg)
    state = (mean, scale, counts)
    return state
