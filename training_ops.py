import jax   
import jax.numpy as jnp
import optax
from typing import List, Dict, Tuple
from jaxtyping import Float, Array
from jax.tree_util import tree_leaves
from model import loss_fn, LinearParams

# Remove @jax.jit here if optimizer is passed as an argument because it might not be hashable or Pytree.
# However, optax optimizers are usually just a pair of pure functions (init, update), which are hashable/static.
# But `optimizer` object itself (GradientTransformation) is a NamedTuple of two Paytree structures (init, update) which are functions.
# Passing them as static_argnums or just let JIT handle it if they are globals is common. 
# Since we are passing it, best to mark it as static if it's not a pytree. 
# Actually, optax.GradientTransformation is a NamedTuple, so it is a valid PyTree.
# But we can just NOT JIT the top level function if it causes issues, or JIT the inner update part.
# For safety and guaranteed JAX behavior with function arguments, let's keep logic simple.
# The original code JIT-ed `train_step`.

from functools import partial

# Mark optimizer as static because it contains functions (init, update), which are not valid Pytree leaves.
@partial(jax.jit, static_argnums=(3,))
def train_step(inp: Dict[str, Float[Array, '...']], model: List[LinearParams], opt_state, optimizer: optax.GradientTransformation):
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(inp, model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = optax.apply_updates(model, updates)
    grads_flat = jnp.concatenate([jnp.ravel(g) for g in tree_leaves(grads)])
    grad_norm = jnp.linalg.norm(grads_flat, ord=2)
    return model, opt_state, loss, grad_norm

@partial(jax.jit, static_argnums=(3,))
def update_params(grads, model: List[LinearParams], opt_state, optimizer: optax.GradientTransformation):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = optax.apply_updates(model, updates)
    return model, opt_state
