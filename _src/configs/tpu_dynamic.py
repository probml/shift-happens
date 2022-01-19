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

"""Hyperparameter configuration to run the example on TPUs."""

import ml_collections


def get_config(model='ResNet50',
               dataset='imagenet2012:5.*.*',
               optimizer='Adam',
               optimizer_params=None,

               num_devices=8,
               cache=True,
               half_precision=True,
               num_classes=1000,
               image_size=224,
               crop_padding=32):
    """Get the hyperparameter configuration to train on TPUs."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = model
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = dataset

    # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
    # train on a larger pod slice.
    config.num_devices = num_devices

    config.cache = cache
    config.half_precision = half_precision
    # list of optimizer configs dicts
    config.optimizers = optimizer
    config.optimizers_params = [get_optimizer_config(optimizer_params, opt_num_devices=num_devices)]
    config.num_classes = num_classes
    config.image_size = image_size
    config.crop_padding = crop_padding
    return config


def get_optimizer_config(config=None, learning_rate=0.1,
                         warmup_epochs=5,
                         momentum=0.9,
                         num_epochs=100,
                         log_every_steps=100,
                         num_train_steps=-1,
                         steps_per_eval=-1,
                         batch_size=-1,
                         opt_num_devices=8):
    if config is None:
        config = ml_collections.ConfigDict()
        config.learning_rate = learning_rate
        config.warmup_epochs = warmup_epochs
        config.momentum = momentum
        config.opt_num_devices = opt_num_devices
        config.num_epochs = num_epochs
        config.log_every_steps = log_every_steps

        # If num_train_steps==-1 then the number of training steps is calculated from
        # num_epochs using the entire dataset. Similarly for steps_per_eval.
        config.num_train_steps = num_train_steps
        config.steps_per_eval = steps_per_eval

        if batch_size == -1:
            config.batch_size = max(config.opt_num_devices * 256, 8 * 1024)
        else:
            config.batch_size = batch_size

        return config
    elif config is not None and type(config) is dict:
        return ml_collections.ConfigDict(initial_dictionary=config)
