'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
import jax, jax.numpy as jnp, jax.lax as lax, equinox as eqx
from jaxtyping import Array, Float, Key

from ..interp import spline_integ, natural_cubic_spline_coeffs

@eqx.filter_jit
def single_spline_vf_loss(ts, ys, fs, func_num):
    ws = jnp.pi / ts[-1] * (jnp.arange(func_num) + 1)
    spline_sin_integrals = jax.vmap(partial(
        spline_integ, ws = ws, ts = ts, use_sin = True
    ))(k = jnp.array([3, 2, 1, 0]))
    coeffs_fs = jnp.stack(natural_cubic_spline_coeffs(ts, fs), axis = 0)
    term1 = jnp.einsum('abc, abd -> cd', spline_sin_integrals, coeffs_fs)

    spline_cos_integrals = jax.vmap(partial(
        spline_integ, ws = ws, ts = ts, use_sin = False
    ))(k = jnp.array([3, 2, 1, 0]))
    coeffs_ys = jnp.stack(natural_cubic_spline_coeffs(ts, ys), axis = 0)
    term2 = jnp.einsum('abc, abd -> cd', spline_cos_integrals, coeffs_ys)
    
    coeff = jnp.sqrt(2 / ts[-1])
    res = coeff * (term1 + ws[:, None] * term2)
    return jnp.sum(res**2, axis = 0)

@eqx.filter_jit
def spline_vf_loss(
    model: eqx.Module, 
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    func_num: int = 100,
    smoothing: float = 0.99,
    **kwargs
):
    '''
    Implementation of proposed the VF loss in our paper.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    batch_ts = batch_ts - batch_ts[:, 0][:, None]
    batch_fs = jax.vmap(model.vector_field)(batch_ts, batch_ys)
    return jax.vmap(
        partial(single_spline_vf_loss, func_num = func_num)
    )(batch_ts, batch_ys, batch_fs).mean()