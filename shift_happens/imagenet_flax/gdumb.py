import jax.numpy as jnp
from jax import ops, random

from collections import defaultdict


def init_sampler():
  counts = defaultdict(int)
  dataset = (jnp.empty([]), jnp.empty([]))
  return counts, dataset


def sample(state, carry, memory_size=200):
  counts, dataset = state
  key, x, y = carry
  n = len(counts.keys())
  inputs, labels = dataset
  nperclass = memory_size / (n if n > 0 else 1)

  if n == 0:
    counts[y.item()] = counts[y.item()] + 1
    return counts, (x[None, ...], y)

  if y.item() not in counts or counts[y.item()] < nperclass:
    if len(inputs) >= memory_size:
      c_max = max(counts, key=counts.get)
      sample_key, key = random.split(key)
      indices = jnp.argwhere(labels == c_max)
      row = random.choice(sample_key, indices, shape=(1,))
      inputs = ops.index_update(inputs, jnp.index_exp[row], x)
      labels = ops.index_update(labels, jnp.index_exp[row], y)
      counts[c_max] -= 1
    else:
      inputs = jnp.vstack([inputs, x[None, ...]])
      labels = jnp.append(labels, y)

    counts[y.item()] += 1

  dataset = (inputs, labels)

  return counts, dataset
