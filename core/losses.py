import numpy as np

# === Mean Squared Error (MSE) ===
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# === Binary Cross-Entropy ===
def binary_cross_entropy(y_true, y_pred):
    # Add small epsilon to prevent log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


# === Categorical Cross-Entropy (for Softmax output) ===
def categorical_cross_entropy(y_true, y_pred):
    # One-hot y_true expected
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# No need for derivative if combined with softmax layer during backprop
# It's handled in a numerically stable way during gradient calculation

class MSE:
    def __call__(self, y_true, y_pred):
        return mse(y_true, y_pred)

    def backward(self, y_true, y_pred):
        return mse_derivative(y_true, y_pred)

class CrossEntropy:
    def __call__(self, y_true, y_pred):
        return categorical_cross_entropy(y_true, y_pred)

    def backward(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]  # softmax + cross-entropy shortcut

def get_loss(name):
    """
    Return the loss function and its derivative based on the name.
    """
    name = name.lower()
    if name == "mse":
        return mse, mse_derivative
    elif name in ["binary_crossentropy", "bce"]:
        return binary_cross_entropy, binary_cross_entropy_derivative
    elif name in ["categorical_crossentropy", "categorical_cross_entropy", "cce"]:
        return categorical_cross_entropy, None
    else:
        raise ValueError(f"[get_loss] Unknown loss function: '{name}'")


