import os
import pickle
import gendist
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from augly import image
from jax.flatten_util import ravel_pytree


def processor(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift


def predict_shifted_dataset(ix_seed, X_batch, processor, config, wmodel, wparams, dmodel, proj, fn_reconstruct):
    """
    Parameters
    ----------
    ix_seed: array
    X_batch: array
    ...
    wmodel: model for the latent space
    wparams: trained weights for the latent space
    dmodel: model for the observed space
    dparams: trained model for the observed weights
    """
    x_seed = X_batch[ix]
    x_shift = processor.process_single(x_seed, **config).ravel()
    predicted_weights = wmodel.apply(wparams, x_shift)
    predicted_weights = proj.inverse_transform(predicted_weights)
    predicted_weights = fn_reconstruct(predicted_weights)
    
    X_batch_shift = processor(X_batch, config)
    y_batch_hat = dmodel.apply(predicted_weights, X_batch_shift)
    
    return y_batch_hat


path_experiment = "./outputs/2203221129/"
path_data_model = os.path.join(path_experiment, "output", "data-model-result.pkl")
path_meta_model = os.path.join(path_experiment, "output", "meta-model.pkl")
path_results = os.path.join(path_experiment, "output", "accuracy.pkl")


with open(path_data_model, "rb") as f:
    data_model_results = pickle.load(f)

with open(path_meta_model, "rb") as f:
    meta_model_results = pickle.load(f)

now_str = datetime.now().strftime("%Y%m%d%H%M")
file_log = f"trench_test_{now_str}.log"
path_logger = os.path.join(path_experiment, "logs", file_log)
logger.remove()
logger.add(path_logger, rotation="5mb")

mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=True)
X_test = np.array(mnist_test.data) / 255
y_test = np.array(mnist_test.targets)

proc_class = gendist.processing.Factory(processor)
pca = meta_model_results["projection_model"]

meta_model = gendist.models.MLPWeightsV1(pca.n_components)
data_model = gendist.models.MLPDataV1(10)

_, fn_reconstruct_params = ravel_pytree(data_model_results["params"][0])

accuracy_configs_learned = []
ixs = np.arange(5)

for config in tqdm(data_model_results["configs"]):
    acc_dict = {}
    for ix in ixs:
        y_test_hat = predict_shifted_dataset(ix, X_test, proc_class, config,
                                             meta_model, meta_model_results["params"],
                                             data_model, pca, fn_reconstruct_params)
        y_test_hat = y_test_hat.argmax(axis=1)
        accuracy_learned = (y_test_hat == y_test).mean().item()
        acc_dict[ix] = accuracy_learned
        
    accuracy_configs_learned.append(acc_dict)
    
    angle = config["angle"]
    logger_row = "|".join([format(v, "0.2%") for v in acc_dict.values()])
    logger_row = f"{angle=:0.4f} | " + logger_row 
    
    logger.info(logger_row)

pd.DataFrame(acc_dict).to_pickle(path_results)
