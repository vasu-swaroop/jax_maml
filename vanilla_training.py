from data import get_data
from model import mlp_forward, loss_fn, LinearParams
import jax
from training_ops import train_step
from jaxtyping import Float, Array, PRNGKeyArray
from typing import List
from tqdm import tqdm
from utils import plot_training_results
import optax

def vanilla_training(
    model: List[LinearParams],
    opt_state,
    optimizer,
    train_data_noise_key: PRNGKeyArray,
    val_data_noise_key: PRNGKeyArray,
    data_gen_key: PRNGKeyArray,
    epochs: int,
    start: Float[Array, "D_in"],
    end: Float[Array, "D_in"],
    h: Float[Array, "D_in"],
    lift_dim: int,
    log_every_n_epochs: int,
    min_val_loss: float,
    train_loss_history: List[float],
    val_loss_history: List[float]
):
    for epoch in tqdm(range(epochs)):
        # Generate new data for each epoch (or batch)
        train_data_noise_key, _ = jax.random.split(train_data_noise_key)

        data_gen_key, _ = jax.random.split(data_gen_key)
        data = get_data(start, end, h, train_data_noise_key, data_gen_key, lift_dim=lift_dim)

        # For validation data, we want to keep it consistent or use a separate key stream
        val_data = get_data(start, end, h, val_data_noise_key, data_gen_key, lift_dim=lift_dim)

        # Perform a training step
        model, opt_state, loss, grad_norm_val = train_step(data, model, opt_state, optimizer)
        train_loss_history.append(loss)

        # Log and plot results periodically
        if epoch % log_every_n_epochs == 0:
            val_loss = loss_fn(val_data, model)
            val_loss_history.append(val_loss)
            print(f"Epoch: {epoch}, Train Loss: {loss:.4f}, Grad Norm: {grad_norm_val:.4f}, Val Loss: {val_loss:.4f}")

            plot_training_results(
                train_data=data,
                val_data=val_data,
                model=model,
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                log_every_n_epochs=log_every_n_epochs,
                mlp_forward=mlp_forward
            )

            # Early stopping condition
            if val_loss < min_val_loss:
                print(f"Validation loss {val_loss:.4f} reached below {min_val_loss}. Stopping training.")
                # Additional check as in original code (loop through some keys)
                for key_val_extra in range(46,56):
                  extra_val_key=jax.random.PRNGKey(key_val_extra)
                  extra_val_data = get_data(start, end, h, extra_val_key, data_gen_key, lift_dim=lift_dim)
                  extra_val_loss=loss_fn(extra_val_data, model)
                  print(f"Extra Val (key={key_val_extra}): {extra_val_loss:.4f}")
                break
    return model, train_loss_history, val_loss_history

