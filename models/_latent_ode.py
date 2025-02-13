'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - Latent Ordinary Differential Equations for Irregularly-Sampled Time Series
    - https://docs.kidger.site/diffrax/examples/latent_ode/
'''

import jax, jax.numpy as jnp, equinox as eqx
from jaxtyping import Float, Array, Key

from ._neural_ode import NeuralODE
from ._rnn import RNN
from ._ode_rnn import ODE_RNN
from ._nde_solver import SolverKwargs


class LatentODE(eqx.Module):
    '''
    Implementation of the Latent ODE model based on JAX.
    
    Reference: 
        - https://docs.kidger.site/diffrax/examples/latent_ode/
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `hidden_size`: `int` - Hidden size
        - `latent_size`: `int` - Latent size
        - `node_width_size`: `int` - Node width size
        - `node_depth`: `int` - Node depth
        - `enc_type`: `str` - Encoder type. Default: `RNN`
    '''
    enc: RNN | ODE_RNN
    neural_ode: NeuralODE
    dec: eqx.nn.Linear
    hidden_size: int
    latent_size: int
    
    def __init__(
        self,
        key: Key,
        obs_size: int, 
        hidden_size: int, 
        latent_size: int, 
        node_width_size: int, 
        node_depth: int,
        enc_type: str = 'RNN',
        solver_kws: SolverKwargs = SolverKwargs()
    ):
        super().__init__()
        enc_key, node_key, dec_key = jax.random.split(key, 3)
        if enc_type == 'RNN':
            self.enc = RNN(
                enc_key, obs_size, 2 * latent_size, hidden_size, 
                use_gru = True, including_dts = True
            )
        elif enc_type == 'ODE_RNN':
            self.enc = ODE_RNN(
                enc_key, obs_size, 2 * latent_size, hidden_size, 
                node_width_size, node_depth, SolverKwargs('Midpoint')
            )
            
        self.neural_ode = NeuralODE(
            node_key, latent_size, 
            node_width_size, node_depth,
            activation = jax.nn.softplus, 
            final_activation = jax.nn.tanh,
            solver_kws = solver_kws
        )
        self.dec = eqx.nn.Linear(latent_size, obs_size, key = dec_key)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
    
    @eqx.filter_jit
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        ys: Float[Array, 'tspan obs'],
        key: Key,
    ):
        return self.decode(key, ts, *self.encode(ts, ys))
    
    def encode(
        self, 
        ts: Float[Array, 'tspan'], 
        ys: Float[Array, 'tspan obs']
    ):
        latent = self.enc(
            ts, ys, reverse = True, evolving_out = False
        )
        mean = latent[:self.latent_size]
        std = jnp.exp(latent[self.latent_size:])
        return mean, std
    
    def decode(self, key, ts, mean, std):
        z0 = mean + jax.random.normal(key, (self.latent_size,)) * std
        zs = self.neural_ode(ts, z0)
        pred_ys = jax.vmap(self.dec)(zs)
        return pred_ys
    
    def _loss(self, ts, ys, key):
        mean, std = self.encode(ts, ys)
        pred_ys = self.decode(key, ts, mean, std)
        
        non_mask = jnp.astype(~jnp.isnan(ys), int)
        ys = jnp.nan_to_num(ys)
        pred_ys = pred_ys * non_mask
        
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        kld_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + kld_loss