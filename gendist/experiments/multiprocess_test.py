"""
Example of the multiprocessing module in gendist for
transforming multiple images.

In this example, we transform each image in a
dataset by a different angle. The first image
is rotated by 0 degrees and the last image
is rotated by 360 degrees.
"""

import gendist
import torchvision
import numpy as np
from augly import image

def processor(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift


if __name__ == "__main__":
    from time import time

    init_time = time()
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    images = np.array(mnist_train.data) / 255.0

    n_configs = len(images)
    degrees = np.linspace(0, 360, n_configs)
    configs = [{"angle": float(angle)} for angle in degrees]
    process = gendist.processing.Factory(processor)
    images_proc = process(images, configs, n_processes=90)
    end_time = time()
    
    print(f"Time elapsed: {end_time - init_time:.2f}s")
    print(images_proc.shape)
