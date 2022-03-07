"""
This library contains functions to process image data used by GenDist
"""

class BlurRad:        
    def __call__(self, img, radius):
        return self.blur_multiple_multiprocessing(img, radius)

    def blur(self, X, radius):
        """
        Blur an image using the augly library

        Paramters
        ---------
        X: np.array
            A single NxM-dimensional array
        radius: float
            The amout of blurriness
        """
        return image.aug_np_wrapper(X, image.blur, radius=radius)

    def blur_multiple(self, X_batch, radii):
        images_out = []
        n_elements = len(X_batch)
                    
        for X, radius in zip(X_batch, radii):
            img_blur = self.blur(X, radius)
            images_out.append(img_blur)
            
        images_out = np.stack(images_out, axis=0)
        return images_out
    
    def blur_multiple_multiprocessing(self, X_dataset, radii, n_processes=90):
        """
        Blur all images of a dataset stored in a numpy array with variable
        radius.

        Parameters
        ----------
        radius: array(N,) or float
            Intensity of bluriness. One per image. If
            float, the same value is used for all images.
        img_dataset: array(N, L, K)
            N images of size LxK
        n_processes: int
            Number of processes to blur over
        """
        num_elements = len(X_dataset)
        if type(radii) in [float, np.float_]:
            radii = radii * np.ones(num_elements)

        dataset_proc = np.array_split(X_dataset, n_processes)
        radii_split = np.array_split(radii, n_processes)
        elements = zip(dataset_proc, radii_split)

        with Pool(processes=n_processes) as pool:    
            dataset_proc = pool.starmap(self.blur_multiple, elements)
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