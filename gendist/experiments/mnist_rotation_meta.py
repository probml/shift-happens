import os
import jax
import optax
import gendist
import pickle
import torchvision
import numpy as np
from augly import image
from sklearn.decomposition import PCA


def processor(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift


if __name__ == "__main__":
    import sys
    _, filename_data_model = sys.argv
    experiment_path, _ = os.path.split(filename_data_model)

    with open(filename_data_model, "rb") as f:
        data_model_config = pickle.load(f)
        list_params = data_model_config["params"]
        configs = data_model_config["configs"]

    target_params, fn_recontruct_params = gendist.processing.flat_and_concat_params(list_params)
    processing_class = gendist.processing.Factory(processor)
    key = jax.random.PRNGKey(314)
    key, key_subset = jax.random.split(key)

    mnist_train = torchvision.datasets.MNIST(root=".", train=True, download=True)
    X_train = np.array(mnist_train.data) / 255

    n_components = 60
    n_classes = 10
    n_train_subset = 6_000
    n_train, *elem_dims = X_train.shape
    n_configs = len(list_params)

    pca = PCA(n_components=n_components)
    projected_params = pca.fit_transform(target_params)[None, ...]

    imap = np.ones((n_configs, 1, *elem_dims))
    configs_transform = np.repeat(configs, n_train_subset)
    subset_ix = jax.random.choice(key_subset, n_train, (n_train_subset,), replace=False).to_py()
    X_train = X_train[subset_ix, ...] * imap
    X_train = processing_class(X_train.reshape(-1, *elem_dims), configs_transform)
    X_train = X_train.reshape((n_train_subset, n_configs, -1), order="F")

    alpha = 0.01
    n_epochs = 100
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