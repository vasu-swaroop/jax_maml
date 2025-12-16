from matplotlib import pyplot as plt
from typing import Dict, List
from jaxtyping import Float, Array
from dataclasses import dataclass
import jax
from model import LinearParams


def plot_training_results(
    train_data: Dict[str, Float[Array, '...']],
    val_data: Dict[str, Float[Array, '...']],
    model: List[LinearParams],
    train_loss_history: List[float],
    val_loss_history: List[float],
    log_every_n_epochs: int,
    mlp_forward: callable
):
    # Calculate predictions for plotting
    pred = mlp_forward(train_data['input'], model)
    orig = train_data['target']
    train_x_plot = train_data['original_x'][:, 0]

    val_pred = mlp_forward(val_data['input'], model)
    val_orig = val_data['target']
    val_x_plot = val_data['original_x'][:, 0]

    # Plotting predictions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_x_plot, pred[:, 0], label='Train Pred')
    plt.plot(train_x_plot, orig, label='Train Orig')
    plt.title('Training Data Prediction vs. Original (vs. x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_x_plot, val_pred[:, 0], label='Val Pred')
    plt.plot(val_x_plot, val_orig, label='Val Orig')
    plt.title('Validation Data Prediction vs. Original (vs. x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plotting loss history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss')
    plt.title('Training Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    val_epochs = [i * log_every_n_epochs for i in range(len(val_loss_history))]
    plt.plot(val_epochs, val_loss_history, label='Validation Loss', color='orange')
    plt.title('Validation Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

