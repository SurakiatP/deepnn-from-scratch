import numpy as np
from data.mnist_loader import load_mnist
from models.neural_network import NeuralNetwork
from core.activations import get_activation
from core.losses import get_loss
from core.optimizers import SGD
from utils import compute_accuracy
import yaml

# --- Load config ---
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Load Data ---
(_, _), (x_test, y_test) = load_mnist(
    normalize=config["data"]["normalize"],
    one_hot=config["data"]["one_hot_labels"]
)

# --- Build Model ---
input_size = config["model"]["input_size"]
hidden_sizes = config["model"]["hidden_sizes"]
output_size = config["model"]["output_size"]
activation_name = config["model"]["activation"]
output_activation = config["model"].get("output_activation", "identity")
layers_config = [input_size] + hidden_sizes + [output_size]

model = NeuralNetwork(
    layers_config=layers_config,
    loss_name=config["loss"]["name"],
    activation=activation_name
)

# --- Load Weights ---
model.load("experiments/model_weights.npz")

# --- Predict ---
y_pred = model.predict(x_test)

# --- Evaluate ---
loss_fn, _ = get_loss(config["loss"]["name"])
loss = loss_fn(y_test, y_pred)
acc = compute_accuracy(y_test, y_pred)

print(f"[TEST] Loss: {loss:.4f} | Accuracy: {acc:.4f}")
