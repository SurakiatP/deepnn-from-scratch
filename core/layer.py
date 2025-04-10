import numpy as np

class Linear:
    def __init__(self, in_features, out_features, activation=None):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X  # shape: (batch_size, in_features)
        self.Z = self.X @ self.W + self.b  # (batch_size, out_features)
        return self.activation.forward(self.Z) if self.activation else self.Z

    def backward(self, dY):
        dZ = self.activation.backward(dY) if self.activation else dY
        batch_size = self.X.shape[0]

        self.dW = (self.X.T @ dZ) / batch_size  # (in_features, out_features)
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size  # (1, out_features)
        return dZ @ self.W.T  # (batch_size, in_features)

    def parameters(self):
        return [{'value': self.W, 'grad': self.dW}, {'value': self.b, 'grad': self.db}]

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)

