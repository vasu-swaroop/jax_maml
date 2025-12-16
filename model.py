import jax
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecked

from typing import List, Dict
from dataclasses import dataclass
from jaxtyping import PRNGKeyArray
from jax import numpy as jnp

@jax.tree_util.register_pytree_node_class
@dataclass
class LinearParams:
    W: Float[Array, "in out"]
    B: Float[Array, "out"]

    def tree_flatten(self):
        return (self.W, self.B), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

def init_weights_dataclass(
    keys: PRNGKeyArray,
    sizes: List[int],
) -> List[LinearParams]:
  weights=[]
  for layer_num, (inp_dim, out_dim, rng_key) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
    fan=inp_dim+out_dim
    std=jnp.sqrt(2/fan)
    weight=jax.random.normal(rng_key, (inp_dim,out_dim))*std
    bias=jax.random.normal(rng_key, (out_dim,))
    weights.append(LinearParams(weight, bias))
  return weights

@jax.jit
@jaxtyped(typechecker=typechecked)
def mlp_forward(inp:Float[Array, 'B D_in'], params:List[LinearParams])-> Float[Array, '...']:
  for param in params[:-1]:
    weight,bias=param.W, param.B
    inp=inp@weight+bias
    inp=jax.nn.relu(inp)
  param=params[-1]
  weight,bias=param.W, param.B
  inp=inp@weight+bias
  return inp

@jax.jit
def loss_fn(inp:Dict[str, Float[Array, '...']], model_params:List[LinearParams]):
  return jnp.mean((inp['target']-mlp_forward(inp['input'],model_params))**2)

