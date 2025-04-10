import numpy as np
import os
from core.layer import Linear
from core.activations import get_activation
from core.losses import get_loss

class NeuralNetwork:
    def __init__(self, layers_config, loss_name="mse", activation="relu"):
        self.layers = []
        self.loss_fn, self.loss_derivative = get_loss(loss_name)

        for i in range(len(layers_config) - 1):
            act_obj = get_activation(activation if i < len(layers_config) - 2 else "identity")
            self.layers.append(
                Linear(
                    in_features=layers_config[i],
                    out_features=layers_config[i + 1],
                    activation=act_obj
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


    def update(self, optimizer):
        for layer in self.layers:
            optimizer.step(layer)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def fit(self, X, y, epochs=10, lr=0.01, optimizer=None, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                x_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = self.forward(x_batch)
                loss = self.loss_fn(y_batch, output)
                total_loss += loss

                # Fallback for softmax + cross-entropy
                if self.loss_derivative is not None:
                    grad = self.loss_derivative(y_batch, output)
                else:
                    grad = output - y_batch

                self.zero_grad()
                self.backward(grad)
                self.update(optimizer)

            if verbose:
                avg_loss = total_loss / (n_samples // batch_size)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y, metric_fn):
        y_pred = self.predict(X)
        return metric_fn(y, y_pred)

    def save(self, filepath):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b
        np.savez_compressed(filepath, **weights)
        print(f"[INFO] Model weights saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] File '{filepath}' not found!")

        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        print(f"[INFO] Model weights loaded from {filepath}")
