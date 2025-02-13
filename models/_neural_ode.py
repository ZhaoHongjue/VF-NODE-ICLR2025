'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - Neural Ordinary Differential Equations
    - https://docs.kidger.site/dfx/examples/neural_ode/
'''

from typing import Callable

import jax, equinox as eqx, diffrax as dfx
from jaxtyping import Float, Array, Key

from ._nde_solver import SolverKwargs, generate_nde_solver_fn


class ODE_Func(eqx.Module):
    '''
    Vector field function for Neural ODEs.
    
    Reference: 
        - Neural Ordinary Differential Equations
        - https://docs.kidger.site/dfx/examples/neural_ode/
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `width_size`: `int` - Width size
        - `depth`: `int` - Depth
        - `activation`: `Callable` - Activation function
        - `final_activation`: `Callable` - Final activation function
    '''
    mlp: eqx.nn.MLP
    
    def __init__(
        self, 
        key: Key, 
        obs_size: int, 
        width_size: int = 64, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        final_activation: Callable = lambda x: x,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            obs_size, obs_size, width_size, depth, 
            activation, final_activation, key = key
        )
        
    def __call__(self, t, y, args = None):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    '''
    Neural ODE model.
    
    Reference: 
        - Neural Ordinary Differential Equations
        - https://docs.kidger.site/dfx/examples/neural_ode/
    
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `width_size`: `int` - Width size
        - `depth`: `int` - Depth
        - `activation`: `Callable` - Activation function
        - `final_activation`: `Callable` - Final activation function
        - `solver_kws`: `SolverKwargs` - Solver arguments
    '''
    ode_func: ODE_Func
    solver_fn: Callable
    
    def __init__(
        self, 
        key: Key, 
        obs_size: int, 
        width_size: int = 64, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        final_activation: Callable = lambda x: x,
        solver_kws: SolverKwargs = SolverKwargs()
    ):
        super().__init__()
        self.ode_func = ODE_Func(
            key, obs_size, width_size, depth, 
            activation, final_activation
        )
        self.solver_fn = generate_nde_solver_fn(solver_kws)
            
    @eqx.filter_jit
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        y0: Float[Array, 'obs'],
        key: Key = None,
    ):
        sol = self.solver_fn(
            terms = dfx.ODETerm(self.ode_func), y0 = y0,
            t0 = ts[0], t1 = ts[-1], saveat = dfx.SaveAt(ts = ts)
        )
        return sol.ys
    
    def vector_field(self, ts, ys):
        return jax.vmap(self.ode_func)(ts, ys)