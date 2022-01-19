# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions

import jax
import jax.numpy as jnp
from jax import lax
from jax import random

import flax
from flax import jax_utils
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state

import optax

import tensorflow as tf
import tensorflow_datasets as tfds

import ml_collections

import numpy as np

# Local imports
import gdumb
from environment import Environment
import agents.models as models


def create_model(model_cls, num_classes, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def init_trans_mat_and_rot_mat(key, num_classes: int, max_degree: int = 359, rotation_indices: list = [0, 180]):
    trans_mat = jax.nn.softmax(jax.random.normal(key, shape=(num_classes, num_classes)))
    rot_mat = np.zeros((max_degree + 1, max_degree + 1))
    for row in rotation_indices:
        for col in rotation_indices:
            rot_mat[row, col] = 1 / len(rotation_indices)
    rot_mat = jnp.array(rot_mat)
    return trans_mat, rot_mat


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    if "batch_stats" in variables:
        return variables['params'], variables['batch_stats']
    return variables['params'], {"mean": jnp.array([])}


def cross_entropy_loss(logits, labels, num_classes):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes):
    print(logits.shape, labels.shape)
    loss = cross_entropy_loss(logits, labels, num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def create_learning_rate_fn(
        num_steps: int,
        init_learning_rate: float):
    """Create learning rate schedule."""

    def schedule(step):
        t = step / num_steps
        return 0.5 * init_learning_rate * (1 + jnp.cos(t * jnp.pi))

    return schedule


def train_step(state, batch, learning_rate_fn, num_classes, weight_decay=0.0001):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'], num_classes)
        weight_penalty_params = jax.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x ** 2)
                         for x in weight_penalty_params
                         if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, (new_model_state, logits)

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')

    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'], num_classes)
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch, num_classes):
    params = state.params
    variables = {'params': params, 'batch_stats': state.batch_stats}

    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'], num_classes)


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform

    if config.half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_size, model)

    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=config.momentum_decay,
        nesterov=True,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=dynamic_scale)
    return state


def get_input_dtype(half_precision, platform):
    if half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32
    return input_dtype


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      Final TrainState.
    """

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)

    rng = random.PRNGKey(config.seed)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()

    platform = jax.local_devices()[0].platform
    input_dtype = get_input_dtype(config.half_precision, platform)

    dataset_builder = tfds.builder(config.dataset)

    num_classes = dataset_builder.info.features['label'].num_classes
    train, test = "train", "validation"

    if config.dataset == "cifar10" or config.dataset == "mnist":
        test = "test"

    train_freq = config.train_freq

    init_key, rng = random.split(rng)
    trans_mat, rot_mat = init_trans_mat_and_rot_mat(init_key, num_classes)
    environment = Environment(config.dataset, trans_mat, rot_mat, batch_size=local_batch_size)

    steps_per_epoch = (
            dataset_builder.info.splits[train].num_examples // config.batch_size
    )

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits[
            test].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    model_cls = getattr(models, config.model)
    model = create_model(
        model_cls=model_cls, num_classes=num_classes, half_precision=config.half_precision)

    learning_rate_fn = create_learning_rate_fn(
        num_steps, config.learning_rate)

    if config.image_size == -1:
        image_size = dataset_builder.info.features['image'].shape[0]
    else:
        image_size = config.image_size

    init_key, rng = random.split(rng)
    state = create_train_state(init_key, config, model, image_size, learning_rate_fn)
    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                          num_classes=num_classes,
                          weight_decay=config.weight_decay), axis_name='batch')

    p_eval_step = jax.pmap(functools.partial(eval_step, num_classes=num_classes), axis_name='batch')

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]

    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    local_device_count = jax.local_device_count()

    for step in range(step_offset, num_steps):
        batch = environment.get_data()
        if step == step_offset:
            counts, datasets = gdumb.init_sampler()

        if train_freq != 1:
            scan_key, rng = random.split(rng)
            keys = random.split(scan_key, local_batch_size)
            images, labels = batch['image'], batch['label'].flatten()
            images = images.reshape((local_batch_size, *images.shape[-3:]))
            for key, image, label in zip(keys, images, labels):
                counts, datasets = gdumb.sample((counts, datasets), (key, image, label))

            n = (len(datasets[0]) // local_device_count) * local_device_count
            batch = {'image': datasets[0][:n].reshape((local_device_count, -1, *images.shape[-3:])),
                     'label': datasets[1][:n].reshape((local_device_count, -1))}

        if (step + 1) % train_freq == 0:
            state, metrics = p_train_step(state, batch)
            for h in hooks:
                h(step)
            if step == step_offset:
                logging.info('Initial compilation completed.')

            if config.get('log_every_steps'):
                train_metrics.append(metrics)
                if (step + 1) % config.log_every_steps == 0:
                    train_metrics = common_utils.get_metrics(train_metrics)
                    summary = {
                        f'{train}_{k}': v
                        for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                    }
                    summary['steps_per_second'] = config.log_every_steps / (
                            time.time() - train_metrics_last_t)
                    writer.write_scalars(step + 1, summary)
                    train_metrics = []
                    train_metrics_last_t = time.time()

            if (step + 1) % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                eval_metrics = []

                # sync batch statistics across replicas
                state = sync_batch_stats(state)
                for _ in range(steps_per_eval):
                    eval_batch = environment.get_test_data(device_count=local_device_count)
                    metrics = p_eval_step(state, eval_batch)
                    eval_metrics.append(metrics)

                eval_metrics = common_utils.get_metrics(eval_metrics)
                summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
                logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                             epoch, summary['loss'], summary['accuracy'] * 100)
                writer.write_scalars(
                    step + 1, {f'eval_{key}': val for key, val in summary.items()})
                writer.flush()

            if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
                state = sync_batch_stats(state)
                save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    random.normal(random.PRNGKey(0), ()).block_until_ready()

    return state
