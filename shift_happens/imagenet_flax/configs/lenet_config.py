import ml_collections


def get_config(seed=2):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = 'LeNet'
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = 'mnist'
    config.seed = seed
    config.image_size = -1

    config.learning_rate = 0.001
    config.batch_size = 80
    config.train_freq = 5

    config.num_epochs = 200
    config.log_every_steps = 100

    config.momentum_decay = 0.9
    config.weight_decay = 0.001

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
