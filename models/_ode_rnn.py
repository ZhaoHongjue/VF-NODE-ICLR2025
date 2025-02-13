'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - Latent Ordinary Differential Equations for Irregularly-Sampled Time Series
'''

from typing import Callable
from functools import partial

import jax, jax.numpy as jnp, equinox as eqx
from jaxtyping import Key, Float, Array

from ._neural_ode import NeuralODE
from ._nde_solver import SolverKwargs


class ODE_RNN(eqx.Module):
    '''
    ODE-RNN model.
    
    Reference: 
        - Latent Ordinary Differential Equations for Irregularly-Sampled Time Series
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `out_size`: `int` - Output size
        - `hidden_size`: `int` - Hidden size
        - `node_width_size`: `int` - Node width size
        - `node_depth`: `int` - Node depth
        - `solver_kws`: `SolverKwargs` - Solver arguments
    '''
    node: NeuralODE
    rnn_cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear
    
    def __init__(
        self, 
        key: Key,
        obs_size: int,
        out_size: int = None,
        hidden_size: int = 25,
        node_width_size: int = 64,
        node_depth: int = 4,
        solver_kws: SolverKwargs = SolverKwargs(
            solver_type = 'Midpoint'
        )
    ):
        super().__init__()
        if out_size is None: out_size = obs_size
        rnn_key, node_key, dec_key = jax.random.split(key, 3)
        self.rnn_cell = eqx.nn.GRUCell(
            2 * obs_size, hidden_size, key = rnn_key
        )
        self.node = NeuralODE(
            node_key, hidden_size, node_width_size, 
            node_depth, activation = jax.nn.tanh,
            solver_kws = solver_kws
        )
        self.linear = eqx.nn.Linear(
            hidden_size, out_size, key = dec_key
        )

    @eqx.filter_jit
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        ys: Float[Array, 'tspan obs'],
        key: Key = None,
        reverse: bool = False,
        evolving_out: bool = True,
    ):
        mask = ~jnp.isnan(ys)
        new_ys = jnp.concatenate([jnp.nan_to_num(ys), mask], axis = -1)
        
        ts = jax.lax.cond(reverse, partial(jnp.flip, axis = 0), lambda x: x, ts)
        new_ys = jax.lax.cond(reverse, partial(jnp.flip, axis = 0), lambda x: x, new_ys)
        in_ = jnp.concatenate([
            ts[:-1, None], ts[1:, None], new_ys[1:]
        ], axis = 1)
        _, hidden_ts = jax.lax.scan(
            partial(self.forward_step, 
                    rnn_cell = self.rnn_cell, 
                    node = self.node),
            jnp.zeros(self.rnn_cell.hidden_size), in_
        )
        if evolving_out:
            out = jax.vmap(self.linear)(hidden_ts)
            out = jnp.concatenate([ys[0][None, :], out])
        else: 
            out = self.linear(hidden_ts[-1])
        return out

    @staticmethod
    def forward_step(hidden, in_, node, rnn_cell):
        tspan, ys = in_[:2], in_[2:]
        hidden_ = node(tspan, hidden)[-1]
        hidden = rnn_cell(ys, hidden_)
        return hidden, hidden