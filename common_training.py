from typing import List, Dict 
from jaxtyping import Float, Array
from jax.tree_util import tree_leaves
import jax
import optax
from model import loss_fn, LinearParams
from jax import numpy as jnp
from model import LinearParams, init_weights_dataclass
from vanilla_training import vanilla_training
from maml_training import fo_maml_training
from training_ops import train_step, update_params

def run_training_loop(
    input_shape: List[int],
    output_dim: int,
    hidden_dims: List[int],
    epochs: int,
    learning_rate: float,
    log_every_n_epochs: int,
    random_seed: int = 42,
    data_range_start: float = -100.0,
    data_range_end: float = 100.0,
    data_h: float = 1.0,
    lift_dim: int = 1000,
    min_val_loss: float = 0.5,
    training_type: str = 'vanilla'
):
    # Initialize model parameters
    master_key = jax.random.PRNGKey(random_seed)
    model_dims = [lift_dim] + hidden_dims + [output_dim]
    model_key, task_generation_master_key = jax.random.split(master_key) # Split for model init and task generation
    model_keys= jax.random.split(model_key, len(model_dims)-1)
    model = init_weights_dataclass(model_keys, model_dims)

    # Setup optimizer (re-initialize for this run)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)

    # Data parameters
    start = jnp.array([data_range_start])
    end = jnp.array([data_range_end])
    h = jnp.array([data_h])

    # Lists to store loss history
    train_loss_history = []
    val_loss_history = []

    if training_type =='vanilla':
      # Split the task_generation_master_key for vanilla training's specific needs
      task_generation_master_key, data_gen_key_vanilla, train_noise_key_vanilla, val_noise_key_vanilla = jax.random.split(task_generation_master_key, 4)
      model, train_loss_history, val_loss_history = vanilla_training(
          model,
          opt_state,
          optimizer,
          train_noise_key_vanilla,
          val_noise_key_vanilla,
          data_gen_key_vanilla,
          epochs,
          start,
          end,
          h,
          lift_dim,
          log_every_n_epochs,
          min_val_loss,
          train_loss_history,
          val_loss_history
      )
    elif training_type =='fomaml':
      model, train_loss_history, val_loss_history = fo_maml_training(
                model,
                opt_state,
                optimizer,
                task_generation_master_key, # Pass the master key for task generation
                epochs,
                start,
                end,
                h,
                lift_dim,
                log_every_n_epochs,
                min_val_loss,
                train_loss_history,
                val_loss_history
            )
    return model, train_loss_history, val_loss_history

