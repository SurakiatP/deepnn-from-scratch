import numpy as np
import yaml
import os
from data.mnist_loader import load_mnist
from models.neural_network import NeuralNetwork
from core.optimizers import SGD, Momentum, Adam
from utils import compute_accuracy, plot_loss_curve, plot_accuracy_curve

# === Load config ===
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

layers_config = config["model"]["layers"]
loss_name = config["loss"]["name"]
activation = config["model"]["activation"]

optimizer_name = config["training"]["optimizer"]
lr = config["training"]["learning_rate"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]

# === Load data ===
(x_train, y_train), (x_test, y_test) = load_mnist()

# === Select optimizer ===
if optimizer_name == "sgd":
    optimizer = SGD(lr=lr)
elif optimizer_name == "momentum":
    optimizer = Momentum(lr=lr)
elif optimizer_name == "adam":
    optimizer = Adam(lr=lr)
else:
    raise ValueError(f"Unknown optimizer: {optimizer_name}")

# === Initialize model ===
model = NeuralNetwork(layers_config=layers_config,
                      loss_name=loss_name,
                      activation=activation)

# === Train ===
history_loss = []
history_acc = []

for epoch in range(epochs):
    model.fit(x_train, y_train, epochs=1, lr=lr, optimizer=optimizer,
              batch_size=batch_size, verbose=False)

    y_pred = model.predict(x_test)
    acc = compute_accuracy(y_test, y_pred)
    loss = model.loss_fn(y_test, y_pred)

    history_loss.append(loss)
    history_acc.append(acc)
    print(f"Epoch {epoch+1}/{epochs} => Val Loss: {loss:.4f} | Val Acc: {acc:.4f}")

# === Save model ===
os.makedirs("experiments", exist_ok=True)
model.save("experiments/model_weights.npz")

# === Plot ===
plot_loss_curve(history_loss)
plot_accuracy_curve(history_acc)
