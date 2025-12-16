from typing import List
from jaxtyping import Float, Array, PRNGKeyArray
from tqdm import tqdm
from utils import plot_training_results
from data import get_query_support
from model import mlp_forward, loss_fn, LinearParams
import jax
from training_ops import train_step, update_params
from jax.tree_util import tree_leaves
from jax import numpy as jnp
import optax

def fo_maml_training(
    model: List[LinearParams],
    opt_state,
    optimizer,
    master_task_key: PRNGKeyArray, # This key will be split for each task
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
        # Generate a new task_specific_key for the current training task
        master_task_key, current_task_key = jax.random.split(master_task_key)
        data_obj = get_query_support(start, end, h, current_task_key, lift_dim=lift_dim)
        support_set= data_obj.train_data
        query_set=data_obj.val_data

        # Perform a training step (inner loop)
        inner_model, inner_opt_state, inner_loss, inner_grad_norm = train_step(support_set, model, opt_state, optimizer)
        
        # Calculate outer loop loss and gradients
        outer_loss, outer_grads = jax.value_and_grad(loss_fn,argnums=1)(query_set, inner_model)
        train_loss_history.append(outer_loss) # Log outer loss as training loss
        
        # Update meta-parameters
        model, opt_state = update_params(outer_grads, model, opt_state, optimizer)
        grads_flat = jnp.concatenate([jnp.ravel(g) for g in tree_leaves(outer_grads)])
        grad_norm = jnp.linalg.norm(grads_flat, ord=2) # ord=2 is default for L2 norm

        # Log and plot results periodically
        if epoch % log_every_n_epochs == 0:
            # Generate a new task_specific_key for the validation task
            master_task_key, val_task_key = jax.random.split(master_task_key)
            val_task_obj = get_query_support(start, end, h, val_task_key, lift_dim=lift_dim)
            
            # Apply inner loop adaptation to the meta-model for the validation task
            val_inner_model, _, _, _ = train_step(val_task_obj.train_data, model, opt_state, optimizer)
            val_loss = loss_fn(val_task_obj.val_data, val_inner_model)
            val_loss_history.append(val_loss)
            print(f"Epoch: {epoch}, Train Loss (Outer): {outer_loss:.4f}, Grad Norm: {grad_norm:.4f}, Val Loss: {val_loss:.4f}")

            plot_training_results(
                train_data=query_set, # Use query set of current task for plotting training performance
                val_data=val_task_obj.val_data, # Use query set of validation task for plotting validation performance
                model=inner_model, # Plot with the adapted model for the current task
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                log_every_n_epochs=log_every_n_epochs,
                mlp_forward=mlp_forward
            )

            # Early stopping condition
            if val_loss < min_val_loss:
                print(f"Validation loss {val_loss:.4f} reached below {min_val_loss}. Stopping training.")
                # Additional check as in original code (loop through some keys)
                for key_val_extra_seed in range(46,56):
                  extra_val_key=jax.random.PRNGKey(key_val_extra_seed)
                  extra_val_data_obj = get_query_support(start, end, h, extra_val_key, lift_dim=lift_dim)
                  extra_val_inner_model, _, _, _ = train_step(extra_val_data_obj.train_data, model, opt_state, optimizer)
                  extra_val_loss=loss_fn(extra_val_data_obj.val_data, extra_val_inner_model)
                  print(f"Extra Val (key={key_val_extra_seed}): {extra_val_loss:.4f}")
                break

    return model, train_loss_history, val_loss_history

