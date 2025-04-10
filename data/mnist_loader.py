import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist(normalize=True, flatten=True, one_hot=True):
    """
    Load MNIST data, with options for normalization, flattening, and one-hot encoding.

    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize inputs to range [0, 1]
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    # Flatten 28x28 images to 784
    if flatten:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)

    # One-hot encode labels
    if one_hot:
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print("Train set:", x_train.shape, y_train.shape)
    print("Test set:", x_test.shape, y_test.shape)
