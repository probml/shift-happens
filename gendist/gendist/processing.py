"""
This library contains functions to process image data used by GenDist
"""
import jax
import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool
from augly import image 

class Factory:
    """
    This is a base library to process / transform the elements of a numpy
    array according to a given function. To be used with gendist.TrainingConfig
    """
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, img, configs, n_processes=90):
        return self.process_multiple_multiprocessing(img, configs, n_processes)

    def process_single(self, X, *args, **kwargs):
        """
        Process a single element.

        Paramters
        ---------
        X: np.array
            A single numpy array
        kwargs: dict/params
            Processor's configuration parameters
        """
        return self.processor(X, *args, **kwargs)

    def process_multiple(self, X_batch, configurations):
        """
        Process all elements of a numpy array according to a list
        of configurations.
        Each image is processed according to a configuration.
        """
        X_out = []
        n_elements = len(X_batch)
                    
        for X, configuration in zip(X_batch, configurations):
            X_processed = self.process_single(X, **configuration)
            X_out.append(X_processed)
            
        X_out = np.stack(X_out, axis=0)
        return X_out
    
    def process_multiple_multiprocessing(self, X_dataset, configurations, n_processes):
        """
        Process elements in a numpy array in parallel.

        Parameters
        ----------
        X_dataset: array(N, ...)
            N elements of arbitrary shape
        configurations: list
            List of configurations to apply to each element. Each
            element is a dict to pass to the processor.
        n_processes: int
            Number of cores to use
        """
        num_elements = len(X_dataset)
        if type(configurations) == dict:
            configurations = [configurations] * num_elements

        dataset_proc = np.array_split(X_dataset, n_processes)
        config_split = np.array_split(configurations, n_processes)
        elements = zip(dataset_proc, config_split)

        with Pool(processes=n_processes) as pool:    
            dataset_proc = pool.starmap(self.process_multiple, elements)
            dataset_proc = np.concatenate(dataset_proc, axis=0)

        return dataset_proc.reshape(num_elements, -1)


def flat_and_concat_params(params_hist):
    """
    Flat and concat a list of parameters trained using
    a Flax model

    Parameters
    ----------
    params_hist: list of flax FrozenDicts
        List of flax FrozenDicts containing trained model
        weights.

    Returns
    -------
    jnp.array: flattened and concatenated weights
    function: function to unflatten (reconstruct) weights
    """
    _, recontruct_pytree_fn = jax.flatten_util.ravel_pytree(params_hist[0])
    flat_params = [jax.flatten_util.ravel_pytree(params)[0] for params in params_hist]
    flat_params  = jnp.r_[flat_params]
    return flat_params, recontruct_pytree_fn