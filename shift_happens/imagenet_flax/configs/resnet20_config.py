import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = 'ResNet18'
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = 'cifar10'
    config.image_size = -1

    config.learning_rate = 1e-7
    config.batch_size = 80

    config.num_epochs = 300
    config.log_every_steps = 100

    config.momentum_decay = 0.9
    config.weight_decay = 100.

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
