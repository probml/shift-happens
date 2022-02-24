import dojax
import numpy as np
from augly import image
from multiprocessing import Pool
import torchvision


def main():
    size_subset = 4
    n_processes = size_subset
    radii = np.random.rand(size_subset) * 5

    mnist_train = torchvision.datasets.MNIST(root=".", train=True, download=True)
    X_train = np.array(mnist_train.data)
    X_train_subset = X_train[:size_subset, ...]

    res = dojax.proc_dataset_multiple(radii, X_train_subset, n_processes)
    return res

if __name__ == "__main__":
    np.random.seed(314)
    res = main()
    print("Running")
    print(res)