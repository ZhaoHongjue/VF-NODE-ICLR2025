'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from typing import Callable
import jax, jax.numpy as jnp
import equinox as eqx, diffrax as dfx
from jaxtyping import Array, Float, Key
from ..interp import natural_cubic_spline_coeffs

@eqx.filter_jit
def single_grad_matching_loss(ts, ys, pred_fs):
    coeffs = jnp.stack(natural_cubic_spline_coeffs(ts, ys), axis = 0)
    cubic_interp = dfx.CubicInterpolation(ts, coeffs)
    spline_fs = jax.vmap(cubic_interp.derivative)(ts)
    return jnp.mean((spline_fs - pred_fs)**2)

@eqx.filter_jit
def grad_matching_loss(
    model: Callable,
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    smoothing: float = 0.99999
):
    '''
    Compute the loss of the gradient matching in the Appendix.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
        - `smoothing`: `float` - Smoothing parameter.
    '''
    batch_ts = batch_ts - batch_ts[:, 0][:, None]
    batch_fs = jax.vmap(model.vector_field)(batch_ts, batch_ys)
    return jax.vmap(single_grad_matching_loss)(batch_ts, batch_ys, batch_fs).mean()