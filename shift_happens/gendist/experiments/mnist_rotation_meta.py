import os
import jax
import optax
import gendist
import pickle
import torchvision
import numpy as np
from augly import image
from sklearn.decomposition import PCA


def rotate(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift


def load_train_combo(filename):
    """
    Load the parameters and the configurations from a file
    of trained models. We return the flatten weights,
    the reconstruct function and the configurations.
    """
    with open(filename, "rb") as f:
        params = pickle.load(f)
        list_params = params["params"]
        list_configs = params["configs"]
    
    target_params, fn_reconstruct = gendist.processing.flat_and_concat_params(list_params)

    output = {
        "params": target_params,
        "configs": list_configs,
        "fn_reconstruct": fn_reconstruct
    }

    return output


def configure_covariates(key, processor, X, configs, n_subset):
    """
    Given a dataset with shape (n_train, ...), a subset size n_subset,
    and a list of configurations to transform the dataset, we transform the
    dataset in an array of shape (n_subset, n_features, ...).
    """
    n_configs = len(configs)
    n_train, *elem_dims = X.shape

    imap = np.ones((n_configs, 1, *elem_dims))
    configs_transform = np.repeat(configs, n_subset)
    subset_ix = jax.random.choice(key, n_train, (n_subset,), replace=False).to_py()
    X = X[subset_ix, ...] * imap
    X = processor(X.reshape(-1, *elem_dims), configs_transform)
    X = X.reshape((n_subset, n_configs, -1), order="F")

    return X


def predict_shifted_dataset(ix_seed, X_batch, processor, config, meta_model,
                            meta_params, dmodel, proj, fn_reconstruct):
    """
    Predict weights and estimate the values
    
    Parameters
    ----------
    ix_seed: array
    X_batch: array
    ...
    meta_model: model for the latent space
    meta_params: trained weights for the latent space
    dmodel: model for the observed space
    dparams: trained model for the observed weights
    """
    x_seed = X_batch[ix_seed]
    x_shift = processor.process_single(x_seed, **config).ravel()
    predicted_weights = meta_model.apply(meta_params, x_shift)
    predicted_weights = proj.inverse_transform(predicted_weights)
    predicted_weights = fn_reconstruct(predicted_weights)
    
    X_batch_shift = processor(X_batch, config)
    y_batch_hat = dmodel.apply(predicted_weights, X_batch_shift)
    
    return y_batch_hat


processing_class = gendist.processing.Factory(rotate)


if __name__ == "__main__":
    import sys

    _, filename_data_model = sys.argv
    experiment_path, _ = os.path.split(filename_data_model)

    output = load_train_combo(filename_data_model)
    target_params = output["params"]
    list_configs = output["configs"]
    fn_reconstruct_params = output["fn_reconstruct"]

    processing_class = gendist.processing.Factory(rotate)
    key = jax.random.PRNGKey(314)
    key, key_subset = jax.random.split(key)

    mnist_train = torchvision.datasets.MNIST(root=".", train=True, download=True)
    X_train = np.array(mnist_train.data) / 255
    X_train = configure_covariates(key_subset, processing_class, X_train, list_configs, n_train_subset)

    n_components = 60
    n_classes = 10
    n_train_subset = 6_000
    n_train, *elem_dims = X_train.shape
    n_configs = len(list_params)

    pca = PCA(n_components=n_components)
    projected_params = pca.fit_transform(target_params)[None, ...]

    alpha = 0.01
    n_epochs = 150
    batch_size = 2000
    tx = optax.adam(learning_rate=alpha)
    lossfn = gendist.training.make_multi_output_loss_func
    weights_model = gendist.models.MLPWeightsV1(n_components)
    trainer = gendist.training.TrainingMeta(weights_model, lossfn, tx)

    meta_output = trainer.fit(key, X_train, projected_params, n_epochs, batch_size)
    meta_output["projection_model"] = pca

    filename_meta_model = "meta-model.pkl"
    filename_meta_model = os.path.join(experiment_path, filename_meta_model)
    with open(filename_meta_model, "wb") as f:
        pickle.dump(meta_output, f)
