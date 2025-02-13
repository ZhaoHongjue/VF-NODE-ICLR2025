'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - Neural Flows: Efficient Alternative to Neural ODEs
'''


from typing import Callable
import jax, jax.numpy as jnp,  equinox as eqx
from jaxtyping import Float, Array, Key


class ResNetFlow(eqx.Module):
    '''
    ResNet Flow model.
    
    Reference: 
        - Neural Flows: Efficient Alternative to Neural ODEs
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `width_size`: `int` - Width size
        - `depth`: `int` - Depth
        - `activation`: `Callable` - Activation function
    '''
    mlp: eqx.nn.MLP
    
    def __init__(
        self, key: Key, 
        obs_size: int, 
        width_size: int = 64, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        **kwargs,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            obs_size + 1, obs_size, width_size = width_size, 
            depth = depth, activation = activation, key = key,
        )
    
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        ys: Float[Array, 'tspan obs'],
        key: Key = None,
        **kwargs,
    ):
        ts, y0 = ts - ts[0], ys[0]
        def _singel_eval(t):
            phi = jax.nn.tanh(t)
            return y0 + phi * self.mlp(jnp.append(y0, t))
        return jax.vmap(_singel_eval)(ts)