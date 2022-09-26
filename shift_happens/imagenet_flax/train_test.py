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

"""Tests for flax.examples.imagenet.train."""
import tempfile

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import random

import tensorflow as tf

# Local imports.
import experiments.continual_learning.agents.models as models
import train
from configs import default as default_lib

jax.config.update('jax_disable_most_optimizations', True)


class TrainTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        # Make sure tf does not allocate gpu memory.
        tf.config.experimental.set_visible_devices([], 'GPU')

    def test_create_model(self):
        """Tests creating model."""
        num_classes = 10
        model = train.create_model(model_cls=models._ResNet1, num_classes=num_classes,
                                   half_precision=False)  # pylint: disable=protected-access
        params, batch_stats = train.initialized(random.PRNGKey(0), 32, model)
        variables = {'params': params, 'batch_stats': batch_stats}
        x = random.normal(random.PRNGKey(1), (8, 32, 32, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (8, num_classes))

    def test_train_and_evaluate(self):
        """Tests training and evaluation loop using mocked data."""
        # Create a temporary directory where tensorboard metrics are written.
        workdir = tempfile.mkdtemp()

        # Define training configuration
        config = default_lib.get_config()
        config.model = '_ResNet1'
        config.batch_size = 8
        config.num_epochs = 1
        config.num_train_steps = 1
        config.steps_per_eval = 1
        train.train_and_evaluate(workdir=workdir, config=config)

    def test_log_likelihood(self):
        n, num_classes = 10, 10
        logits = jax.nn.log_softmax(jnp.ones((n, num_classes)))
        labels = jnp.arange(num_classes)
        loss = train.log_likelihood(logits, labels, num_classes)
        self.assertLess(loss, 0)
        self.assertEqual(loss, n * logits[0][0])


if __name__ == '__main__':
    absltest.main()
