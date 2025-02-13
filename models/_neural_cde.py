'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - Neural Controlled Differential Equations for Irregularly-Sampled Time-Series
    - https://docs.kidger.site/diffrax/examples/neural_cde/
'''
from typing import Callable

import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, diffrax as dfx
from jaxtyping import Key, Float, Array

from ._nde_solver import *


class CDE_Func(eqx.Module):
    '''
    Vector field function for Neural CDEs.
    
    Reference: 
        - https://docs.kidger.site/diffrax/examples/neural_cde/
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `hidden_size`: `int` - Hidden size
        - `width_size`: `int` - Width size
    '''
    mlp: eqx.nn.MLP

    def __init__(
        self, 
        key: Key,
        obs_size: int, 
        hidden_size: int = 25, 
        width_size: int = 64, 
        depth: int = 4,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            hidden_size, hidden_size * obs_size, width_size,
            depth, jax.nn.softplus, jax.nn.tanh, key = key,
        )

    def __call__(self, t, y, args):
        hidden_size = self.mlp.in_size
        obs_size = self.mlp.out_size // self.mlp.in_size
        return self.mlp(y).reshape(hidden_size, obs_size)


class NeuralCDE(eqx.Module):
    '''
    Neural CDE model.
    
    Reference: 
        - Neural Controlled Differential Equations for Irregularly-Sampled Time-Series
        - https://docs.kidger.site/diffrax/examples/neural_cde/
        
    Args:
        - `key`: `Key` - Random key
        - `in_size`: `int` - Input size
        - `out_size`: `int` - Output size
        - `hidden_size`: `int` - Hidden size
        - `width_size`: `int` - Width size
        - `depth`: `int` - Depth
        - `solver_kws`: `SolverKwargs` - Solver arguments
    '''
    initial: eqx.nn.MLP
    cde_func: CDE_Func
    linear: eqx.nn.Linear
    solver_fn: Callable

    def __init__(
        self, 
        key: Key,
        in_size: int, 
        out_size: int = None,
        hidden_size: int = 25, 
        width_size: int = 64, 
        depth: int = 4,
        solver_kws: SolverKwargs = SolverKwargs()
    ):
        super().__init__()
        initial_key, func_key, linear_key = jr.split(key, 3)
        if out_size is None: out_size = in_size
        self.initial = eqx.nn.MLP(
            in_size + 1, hidden_size, width_size, depth, key = initial_key
        )
        self.cde_func = CDE_Func(
            func_key, in_size + 1, hidden_size, width_size, depth
        )
        self.linear = eqx.nn.Linear(hidden_size, out_size, key = linear_key)
        self.solver_fn = generate_nde_solver_fn(solver_kws)

    @eqx.filter_jit
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        ys: Float[Array, 'tspan obs'], 
        key: Key = None,
        evolving_out: bool = True,
    ):
        tys = jnp.concatenate([ts[..., None], ys], axis = -1)
        coeffs = dfx.backward_hermite_coefficients(ts, tys)

        control = dfx.CubicInterpolation(ts, coeffs)
        term = dfx.ControlTerm(self.cde_func, control).to_ode()
        y0 = self.initial(control.evaluate(ts[0]))
        saveat = dfx.SaveAt(ts = ts)
        sol = self.solver_fn(
            terms = term, y0 = y0, t0 = ts[0], 
            t1 = ts[-1], saveat = saveat, 
        )
        if evolving_out:
            out = jax.vmap(self.linear)(sol.ys)
        else:
            out = self.linear(sol.ys[-1])
        return out 
