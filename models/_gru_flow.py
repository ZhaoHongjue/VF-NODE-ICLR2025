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

class GRUFlow(eqx.Module):
    '''
    Implementation of GRU Flow in Neural Flows: Efficient Alternative to Neural ODEs.
    
    Reference: 
        - Neural Flows: Efficient Alternative to Neural ODEs
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `width_size`: `int` - Width size. Default: 74
        - `depth`: `int` - Depth. Default: 4
        - `activation`: `Callable` - Activation function. Default: `jax.nn.elu`
    '''
    
    mlp_z: eqx.nn.MLP
    mlp_r: eqx.nn.MLP
    mlp_c: eqx.nn.MLP
    
    def __init__(
        self, key: Key, 
        obs_size: int, 
        width_size: int = 74, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        **kwargs,
    ):
        super().__init__()
        self.mlp_z = eqx.nn.MLP(
            obs_size + 1, obs_size, width_size = width_size, 
            depth = depth, activation = activation, key = key,
        )
        self.mlp_r = eqx.nn.MLP(
            obs_size + 1, obs_size, width_size = width_size, 
            depth = depth, activation = activation, key = key,
        )
        self.mlp_c = eqx.nn.MLP(
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
        y0 = ys[0]
        ts = ts - ts[0]
        def _singel_eval(t):
            alpha, beta = 0.4, 0.8
            z = alpha * jax.nn.sigmoid(self.mlp_z(jnp.append(y0, t)))
            r = beta * jax.nn.sigmoid(self.mlp_r(jnp.append(y0, t)))
            c = jax.nn.tanh(self.mlp_c(jnp.append(y0 * r, t)))
            phi = jax.nn.tanh(t)
            return y0 + phi * (1 - z) * (c - y0)
        return jax.vmap(_singel_eval)(ts)