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

"""Tests for flax.examples.imagenet.models."""

from absl.testing import absltest

import jax
from jax import numpy as jnp

import experiments.continual_learning.agents.models as models

jax.config.update('jax_disable_most_optimizations', True)


class ResNetV1Test(absltest.TestCase):
    """Test cases for ResNet v1 model definition."""

    def test_resnet_v1_model(self):
        """Tests ResNet V1 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = models.ResNet50(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(
            rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # Resnet50 model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   BottleneckResNetBlock in stages: [3, 4, 6, 3] = 16
        #   Followed by a Dense layer = 1
        self.assertLen(variables['params'], 19)

    def test_lenet5_model(self):
        """Tests LeNeT5 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        num_classes = 10
        model_def = models.LeNet(num_classes=num_classes, dtype=jnp.float32)
        variables = model_def.init(
            rng, jnp.ones((8, 32, 32, 3), jnp.float32))

        self.assertLen(variables, 1)
        # LeNet5 model will create parameters for the following layers:
        #   2 Conv + 3 Dense = 2
        self.assertLen(variables['params'], 5)
        # The output of the last layer of LeNet5 will be equal to the number
        # of classes. In this case, it is 10
        self.assertLen(variables['params']['Dense_2']['bias'], num_classes)
        self.assertEqual(variables['params']['Dense_2']['kernel'].shape[-1], num_classes)

    def test_mlp_model(self):
        """Tests LeNeT5 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        num_classes = 10
        layer_dims = [256, 256]
        model_def = models.MLP(layer_dims=layer_dims, num_classes=num_classes, dtype=jnp.float32)
        variables = model_def.init(
            rng, jnp.ones((8, 32, 32, 3), jnp.float32))

        self.assertLen(variables, 1)
        # MLP model will create parameters for the following layers:
        # 2 Dense Layer + 1 Output Layer = 3
        self.assertLen(variables['params'], len(layer_dims) + 1)

        layer_dims += [num_classes]
        for layer_idx, layer_dim in enumerate(layer_dims):
            self.assertLen(variables['params'][f'Dense_{layer_idx}']['bias'], layer_dim)
            self.assertEqual(variables['params'][f'Dense_{layer_idx}']['kernel'].shape[-1], layer_dim)


if __name__ == '__main__':
    absltest.main()
