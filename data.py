import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Array, PRNGKeyArray
from jaxtyping import jaxtyped
from beartype import beartype as typechecked
from typing import Dict, List, Tuple
from dataclasses import dataclass

@jax.jit
@jaxtyped(typechecker=typechecked)
def lift_linear(
    x: Float[Array, "B D_in"],
    W: Float[Array, "D_in D_out"],
) -> Float[Array, "B D_out"]:
    # Removed print for cleaner output during training
    return x @ W

@dataclass
class DataObject():
  train_data: Dict[str, Float[Array, "B ..."]]
  val_data: Dict[str, Float[Array, "B ..."]]

@jaxtyped(typechecker=typechecked)
def get_data(
    start: Float[Array, "D_in"],
    end: Float[Array, "D_in"],
    h: Float[Array, "D_in"],
    data_noise_key: PRNGKeyArray,
    data_gen_key: PRNGKeyArray,
    lift_dim: int,
) -> Dict[str, Float[Array, "B ..."]]:

    grids = [
        jnp.linspace(start[d], end[d], int((end[d] - start[d]) / h[d]))
        for d in range(start.shape[0])
    ]
    x = jnp.stack(grids, axis=1)   # (B, D_in)
    y = jnp.sin(x) + jnp.maximum(x, 0)

    W = jax.random.normal(data_gen_key, (x.shape[1], lift_dim))
    x_lifted = lift_linear(x, W)


    x_lifted= x_lifted+ jax.random.normal(data_noise_key, x_lifted.shape)
    return {"input": x_lifted, "target": y, "original_x": x}


@jaxtyped(typechecker=typechecked)
def get_query_support(
    start: Float[Array, "D_in"],
    end: Float[Array, "D_in"],
    h: Float[Array, "D_in"],
    task_specific_key: PRNGKeyArray, # This key defines the entire task
    lift_dim: int,
)-> DataObject:
    data_gen_key, data_noise_key_for_task = jax.random.split(task_specific_key, 2)
    train_data_noise_key, val_data_noise_key = jax.random.split(data_noise_key_for_task, 2)

    # Use the task_specific data_gen_key for W in get_data
    train_data = get_data(start, end, h, train_data_noise_key, data_gen_key, lift_dim=lift_dim)
    val_data = get_data(start, end, h, val_data_noise_key, data_gen_key, lift_dim=lift_dim)

    data_obj= DataObject(train_data, val_data)
    return data_obj
