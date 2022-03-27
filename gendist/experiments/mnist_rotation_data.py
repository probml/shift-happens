import os
import jax
import optax
import pickle
import gendist
import torchvision
import numpy as np
from augly import image
from datetime import datetime
from tqdm import tqdm
from loguru import logger

def eval_acc(y, yhat):
    return (y.argmax(axis=1) == yhat.argmax(axis=1)).mean().item()


def processor(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift


def create_experiment_path(base_path, experiment_name):
    base_path = os.path.join(base_path, experiment_name)
    path_output = os.path.join(base_path, "output")
    path_logs = os.path.join(base_path, "logs")

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    if not os.path.exists(path_logs):
        os.makedirs(path_logs)

    return base_path


def training_loop(key, X, y, configs, trainer, n_epochs, batch_size, evalfn, logger):
    """
    Train a collection of models with different configurations

    Parameters
    ----------
    X : ndarray
        Input data
    y : ndarray
        Target data
    configs : list
        List of configurations to train.
        Each configuration is a dictionary
    trainer: gendist.training
        Trainer object
    n_epochs: int
        Number of epochs to train
    batch_size: int
        Batch size
    evalfn : function
        Function to evaluate the model 
    
    Returns
    -------
    results : dict
        Dictionary with the results of the experiments
    """
    configs_params = []
    configs_losses = []
    configs_metric = []

    for config in tqdm(configs):
        train_output = trainer.fit(key, X, y, config, n_epochs, batch_size, evalfn)
        configs_params.append(train_output["params"])
        configs_losses.append(train_output["losses"])
        configs_metric.append(train_output["metric"])

        name, value = config.popitem()
        logger.info(f"{name}={value:0.3f} | {train_output['metric']:.4f}")
    
    output = {
        "params": configs_params,
        "losses": configs_losses,
        "metric": configs_metric
    }

    return output
    

def main(key, base_path, trainer, X, y, configs, n_epochs, batch_size, evalfn,
         experiment_path=None, logname=None, filename=None):
    filename = "data-model-result.pkl" if filename is None else filename
    logname = "log-data.log" if logname is None else logname

    if experiment_path is None:
        date_str = datetime.now().strftime("%y%m%d%H%M")
        experiment_path = create_experiment_path(base_path, date_str)

    logs_path = os.path.join(experiment_path, "logs")
    logs_path = os.path.join(logs_path, logname)
    logger.add(logs_path, rotation="5mb")

    experiment_results = training_loop(key, X, y, configs, trainer, n_epochs, batch_size, evalfn, logger)
    experiment_results["configs"] = configs

    filename = os.path.join(experiment_path, "output", filename)
    with open(filename, "wb") as f:
        pickle.dump(experiment_results, f)

    print(f"Experiment path: {experiment_path}")


if __name__ == "__main__":
    logger.remove() # avoid output to terminal
    n_configs, n_classes = 150, 10
    batch_size = 2000
    n_epochs = 50
    alpha = 0.001
    tx = optax.adam(learning_rate=alpha)
    # model = gendist.models.MLPDataV1(num_outputs=10)
    model = gendist.models.LeNet5(n_classes)
    processing_class = gendist.processing.Factory(processor)
    loss = gendist.training.make_cross_entropy_loss_func
    trainer = gendist.training.TrainingBase(model, processing_class, loss, tx)

    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    X_train = np.array(mnist_train.data) / 255.0
    y_train = np.array(mnist_train.targets)
    y_train_ohe = jax.nn.one_hot(y_train, n_classes)

    degrees = np.linspace(0, 360, n_configs)
    configs = [{"angle": angle.item()} for angle in degrees]

    base_path = "./outputs"
    key = jax.random.PRNGKey(314)
    main(key, base_path, trainer, X_train, y_train_ohe, configs, n_epochs, batch_size, eval_acc)
