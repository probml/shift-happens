from jax._src.random import split
import jax.numpy as jnp
from jax import lax, ops, random

from collections import defaultdict


def init_sampler(x , y, num_classes):
  dataset = x, y
  counts = jnp.bincount(y, minlength=num_classes)
  return counts, dataset


def sample(key, X, Y, counts, datasets, memory_size=200):  
    inputs, labels = datasets
    n = jnp.sum(jnp.where(counts>0, 1, 0))
    nperclass = memory_size // n

    def true_fun(key, inputs, labels, counts, x, y):
        c_max = jnp.argwhere(counts == jnp.max(counts))
        sample_key, key = random.split(key)
        indices = jnp.argwhere(labels == c_max)
        row = random.choice(sample_key, indices, shape=(1, ))
        inputs = ops.index_update(inputs, jnp.index_exp[row], x)
        labels = ops.index_update(labels, jnp.index_exp[row], y)
        counts = jnp.where(jnp.arange(len(counts)) == c_max, counts-1, counts)
        counts = jnp.where(jnp.arange(len(counts)) == y, counts+1, counts)
        return inputs, labels, counts
        
    def false_fun(key, inputs, labels, counts, x, y):
        return inputs, labels, counts


    def scan_fun(state, carry):
            inputs, labels, counts = state
            key, x,  y = carry

            inputs, labels, counts = lax.cond(counts[y] == 0 or counts[y] < nperclass, true_fun, false_fun, 
                                                operands=(key, inputs, labels, x, y))
            return (inputs, labels, counts), None

    if len(datasets) < memory_size:
        for x, y in zip(X, Y):
            if counts[y] == 0 or counts[y] < nperclass:
                inputs = jnp.vstack([inputs, x[None, ...]])
                labels = jnp.append(labels, y)
                counts = jnp.where(jnp.arange(len(counts)) ==y, counts+1, counts)

    else:
        scan_key, key = split(key)
        keys = split(scan_key, len(X))
        (inputs, labels, counts), _ = lax.scan(scan_fun, (inputs, labels, counts), (keys, X, Y))

    return counts, (inputs, labels)




import jax.numpy as jnp
from jax import vmap
from jax import random
import jax.dlpack

from imax import transforms

import tensorflow as tf
import tensorflow_datasets as tfds

from functools import partial
from typing import Any

Array = Any

def tf_to_jax(arr):
  return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))

class Environment:
  def prepare_data(self, key : Any, dataset_name : str):
    
    ds_builder = tfds.builder(dataset_name)
    ds_builder.download_and_prepare()
    ds_train = ds_builder.as_dataset(split="train", as_supervised=True)
    self.test_data = ds_builder.as_dataset(split="test", as_supervised=True)

    self.num_classes = ds_builder.info.features['label'].num_classes
    self.current_class = random.choice(key, jnp.arange(self.num_classes), shape=(1,))
    self.sets = [ds_train.filter(lambda x, y: y == i) for i in range(self.num_classes)]
    self.counts = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    # [st.reduce(np.int64(0), lambda x, _ : x + 1).numpy() for st in self.sets]
    
  
  def __init__(self, dataset_name='mnist', class_trans_mat=None, rot_mat=None, seed=0, batch_size=64):
    self.class_trans_mat = class_trans_mat
    self.rot = 0
    self.rot_mat = rot_mat
    self.seed= seed
    self.batch_size = batch_size
    
    tf.random.set_seed(seed)
    key = random.PRNGKey(seed)
    self.prepare_data(key, dataset_name)
   

  def get_data(self, key):
    class_key, rot_key, key = random.split(key, 3)
    self.current_class = random.choice(class_key, jnp.arange(self.num_classes), p=self.class_trans_mat[self.current_class].squeeze())
    batch = self.get_batch(self.current_class)
    self.rot = random.choice(rot_key, jnp.arange(len(self.rot_mat)), p=self.rot_mat[self.rot].squeeze())
    batch['image'] = self.apply(batch['image'], self.rot)
    return batch


  def apply(self, x : Array, degrees : float):
    rad = jnp.radians(degrees)
    # Creates transformation matrix
    transform = transforms.rotate(rad=rad)
    apply_transform = partial(transforms.apply_transform, transform=transform, mask_value=jnp.array([0, 0, 0]))
    return vmap(apply_transform)(x)

  def get_batch(self, c: int):
    dataset = self.sets[c]
    nexamples = self.counts[c]
    for images, labels in dataset.shuffle(nexamples).batch(self.batch_size):
      images = tf_to_jax(images)
      labels = tf_to_jax(labels)
      break
    
    *_, nchannels = images.shape
    if nchannels == 1:
      images = jnp.repeat(images, axis=-1, repeats=3)
    return {'image': images, 'label': labels}

  def warmup(self, num_pulls : int):
    warmup_classes = jnp.arange(self.num_classes)
    warmup_classes = jnp.repeat(warmup_classes, num_pulls).reshape(self.num_classes, -1)
    classes = warmup_classes.reshape(-1, order="F").astype(jnp.int32)
    num_warmup_classes, *_ = classes.shape
    inputs, labels = [], []
      
    for c in classes:
      batch = self.get_batch(c)
      inputs.append(batch['image'][None, ...])
      labels.append(batch['label'])
    
    images = jnp.vstack(inputs)
    return images.reshape((-1, *images.shape[-3:])), jnp.concatenate(labels)