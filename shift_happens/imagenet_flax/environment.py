import jax
import jax.numpy as jnp
from jax import vmap, tree_map
from jax import random
from imax import transforms

import tensorflow_datasets as tfds
import numpy as np

from functools import partial
from typing import Any

Array = Any


class Environment:
    def prepare_data(self, dataset_name: str):
        ds_builder = tfds.builder(dataset_name)
        ds_builder.download_and_prepare()
        ds_train = ds_builder.as_dataset(split="train")
        self.test_data = ds_builder.as_dataset(split="test").repeat().batch(self.batch_size).as_numpy_iterator()

        self.num_classes = ds_builder.info.features['label'].num_classes
        self.current_class = np.random.choice(np.arange(self.num_classes), size=(1,))
        self.sets = [ds_train.filter(lambda x: x['label'] == i) for i in range(self.num_classes)]
        self.counts = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
        # [st.reduce(np.int64(0), lambda x, _ : x + 1).numpy() for st in self.sets]

    def __init__(self, dataset_name='mnist', class_trans_mat=None, rot_mat=None, seed=0, batch_size=64):
        self.class_trans_mat = class_trans_mat
        self.rot = 0
        self.rot_mat = rot_mat
        self.seed = seed
        self.batch_size = batch_size
        self.key = random.PRNGKey(seed)
        self.prepare_data(dataset_name)

    def get_data(self):
        class_key, rot_key, self.key = random.split(self.key, 3)
        self.current_class = random.choice(class_key, jnp.arange(self.num_classes),
                                           p=self.class_trans_mat[self.current_class].squeeze())
        batch = self.get_train_batch(self.current_class)
        self.rot = random.choice(rot_key, jnp.arange(len(self.rot_mat)), p=self.rot_mat[self.rot].squeeze())
        batch['image'] = self.apply(batch['image'], self.rot)
        return {'image': batch['image'], 'label': batch['label']}

    def apply(self, x: Array, degrees: float):
        rad = jnp.radians(degrees)
        # Creates transformation matrix
        transform = transforms.rotate(rad=rad)
        apply_transform = partial(transforms.apply_transform, transform=transform, mask_value=jnp.array([0, 0, 0]))
        return vmap(apply_transform)(x)

    def get_test_data(self, device_count=1, nchannels=3):
        rotations = jnp.argwhere(self.rot_mat.sum(axis=0) > 0).flatten().tolist()

        batch = next(self.test_data)
        input, label = jnp.array(batch['image']), jnp.array(batch['label'])
        if input.shape[-1] != nchannels:
            input = jnp.repeat(input, axis=-1, repeats=nchannels)

        label = label[None, ...]
        for degrees in rotations:
            if degrees:
                input = jnp.vstack([input, self.apply(input, degrees)])
                label = jnp.vstack([label, label])
        label = label.squeeze()
        if device_count > 1:
            input = input.reshape((device_count, -1, *input.shape[-3:]))
            label = label.reshape((device_count, -1))

        return {'image': input, 'label': label}

    def get_train_batch(self, c: int, seed: int = 0):
        dataset = self.sets[c]
        nexamples = self.counts[c]
        batch = tree_map(jnp.array,
                         next(dataset.shuffle(nexamples, seed=seed).batch(self.batch_size).as_numpy_iterator()))
        images, labels = batch['image'], batch['label']
        *_, nchannels = images.shape
        if nchannels == 1:
            images = jnp.repeat(images, axis=-1, repeats=3)
        return {'image': images, 'label': labels}

    def warmup(self, num_pulls: int):
        warmup_classes = jnp.arange(self.num_classes)
        warmup_classes = jnp.repeat(warmup_classes, num_pulls).reshape(self.num_classes, -1)
        classes = warmup_classes.reshape(-1, order="F").astype(jnp.int32)
        num_warmup_classes, *_ = classes.shape
        seeds = jnp.arange(len(classes))
        inputs, labels = [], []

        for c, seed in zip(classes, seeds):
            batch = self.get_train_batch(c, seed)
            inputs.append(batch['image'])
            labels.append(batch['label'])
        return jnp.vstack(inputs), jnp.concatenate(labels)
