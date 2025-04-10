import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self.x > 0, 1, self.alpha)

class Sigmoid(Activation):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)

class Tanh(Activation):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)

class Softmax(Activation):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        # This assumes softmax used with cross-entropy, where the gradient simplifies
        return grad

class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad):
        dx = np.where(self.x >= 0, 1, self.alpha * np.exp(self.x))
        return grad * dx

class Swish(Activation):
    def forward(self, x):
        self.x = x
        self.sigmoid = 1 / (1 + np.exp(-x))
        return x * self.sigmoid

    def backward(self, grad):
        return grad * (self.sigmoid + self.x * self.sigmoid * (1 - self.sigmoid))
    
class Identity(Activation):
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return ReLU()
    elif name == "leaky_relu":
        return LeakyReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "softmax":
        return Softmax()
    elif name == "elu":
        return ELU()
    elif name == "swish":
        return Swish()
    elif name == "identity":
        return Identity()
    else:
        raise ValueError(f"Unknown activation function: {name}")


