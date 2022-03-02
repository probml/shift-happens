# Library for domain shift in Jax
import jax
import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool
from augly import image


def rotation_matrix(angle):
    """
    Create a rotation matrix that rotates the
    space 'angle'-radians.
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return R


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
  """
  _, recontruct_pytree_fn = jax.flatten_util.ravel_pytree(params_hist[0])
  flat_params = [jax.flatten_util.ravel_pytree(params)[0] for params in params_hist]
  flat_params  = jnp.r_[flat_params]
  return flat_params, recontruct_pytree_fn


def make_mse_func(model, x_batched, y_batched):
  def mse(params):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
      pred = model.apply(params, x)
      residual = pred - y
      return residual @ residual / 2.0
    # We vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
  return jax.jit(mse)


### Elements for shifting-mnist experiments ###

class BlurRad:
    def __init__(self, rad):
        self.rad = rad
        
    def __call__(self, img):
        return self.blur_multiple(img)

    def blur(self, X):
        """
        Blur an image using the augly library

        Paramters
        ---------
        X: np.array
            A single NxM-dimensional array
        radius: float
            The amout of blurriness
        """
        return image.aug_np_wrapper(X, image.blur, radius=self.rad)

    def blur_multiple(self, X_batch):
        images_out = []
        for X in X_batch:
            img_blur = self.blur(X)
            images_out.append(img_blur)
        images_out = np.stack(images_out, axis=0)
        return images_out


def blur_multiple(radii, img_dataset):
    """
    Blur every element of `img_dataset` given an element
    of `radii`.
    """
    imgs_out = []
    for radius, img in zip(radii, img_dataset):
        img_proc = BlurRad(radius).blur(img)
        imgs_out.append(img_proc)
    imgs_out = np.stack(imgs_out, axis=0)
    
    return imgs_out


# To-do: Modify proc_dataset and proc_dataset_multiple to use
# a function that modifies the image.

def proc_dataset(radius, img_dataset, n_processes=90):
    """
    Blur all images of a dataset stored in a numpy array.
    
    Parameters
    ----------
    radius: float
        Intensity of bluriness
    img_dataset: array(N, L, K)
        N images of size LxK
    n_processes: int
        Number of processes to blur over
    """
    with Pool(processes=n_processes) as pool:
        dataset_proc = np.array_split(img_dataset, n_processes)
        dataset_proc = pool.map(BlurRad(radius), dataset_proc)
        dataset_proc = np.concatenate(dataset_proc, axis=0)
    pool.terminate()
    pool.join()
    
    return dataset_proc


def proc_dataset_multiple(radii, img_dataset, n_processes=90):
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

    if type(radii) in [float, np.float_]:
        radii = radii * np.ones(len(img_dataset))
    
    with Pool(processes=n_processes) as pool:
        dataset_proc = np.array_split(img_dataset, n_processes)
        radii_split = np.array_split(radii, n_processes)
        
        elements = zip(radii_split, dataset_proc)
        dataset_proc = pool.starmap(blur_multiple, elements)
        dataset_proc = np.concatenate(dataset_proc, axis=0)
    pool.terminate()
    pool.join()

    return dataset_proc
