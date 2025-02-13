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
def single_spline_integ_loss(ts, ys, fs):
    coeffs = jnp.stack(natural_cubic_spline_coeffs(ts, fs), axis = 0)
    dts = ts[1:] - ts[:-1]
    concat_dts = jnp.stack([dts**i / i for i in range(4, 0, -1)])
    interval_integ = jnp.einsum('ijk, ij -> jk', coeffs, concat_dts)
    cusum_integ = jnp.cumsum(interval_integ, axis = 0)
    pred_ys = jnp.concat([jnp.zeros((1, ys.shape[-1])), cusum_integ], axis = 0) + ys[0][None]
    
    non_mask = jnp.astype(~jnp.isnan(ys), int)
    ys = jnp.nan_to_num(ys)
    pred_ys = non_mask * pred_ys
    return jnp.mean((ys - pred_ys)**2)

@eqx.filter_jit
def spline_integ_loss(
    model: Callable,
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    smoothing: float = 0.99999
):
    '''
    Compute the spline integration loss in Appendix.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
        - `smoothing`: `float` - Smoothing parameter.
    '''
    batch_ts = batch_ts - batch_ts[:, 0][:, None]
    batch_fs = jax.vmap(model.vector_field)(batch_ts, batch_ys)
    return jax.vmap(single_spline_integ_loss)(batch_ts, batch_ys, batch_fs).mean()