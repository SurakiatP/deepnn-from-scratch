import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes):
    """Convert label (y) to one-hot vector"""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def compute_accuracy(y_true, y_pred):
    """Calculate the accuracy between y_true (one-hot) and y_pred (probabilities)"""
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)

def plot_loss_curve(losses, title="Training Loss"):
    plt.plot(losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(accs, title="Validation Accuracy"):
    plt.plot(accs, label="accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sample_predictions(x, y_true, y_pred, num_samples=10):
    """
    Visualize predictions: Show num_samples input images with predicted and true labels.
    Assumes x is flattened if shape is (N, 784).
    """
    assert x.shape[0] >= num_samples, "Not enough samples to visualize."

    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)

    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = x[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()