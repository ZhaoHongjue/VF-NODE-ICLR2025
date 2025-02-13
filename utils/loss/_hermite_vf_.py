'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
import jax, jax.numpy as jnp, jax.lax as lax
import equinox as eqx, diffrax as dfx
from jaxtyping import Array, Float, Key

from ..interp import spline_integ

@eqx.filter_jit
def _hermite_vf_loss(dynamic, func_num, ts, ys):
    ts = ts - ts[0]
    coeffs_ys = dfx.backward_hermite_coefficients(ts, ys)
    interp = dfx.CubicInterpolation(ts, coeffs_ys)
    
    ys = lax.cond(
        jnp.any(jnp.isnan(ys)), 
        lambda ts, ys: jax.vmap(interp.evaluate)(ts),
        lambda ts, ys: ys, ts, ys
    )
    fs = dynamic.vector_field(ts, ys)
    coeffs_fs = dfx.backward_hermite_coefficients(ts, fs)
    
    ws = jnp.pi / ts[-1] * (jnp.arange(func_num) + 1)
    spline_sin_integrals = jax.vmap(partial(
        spline_integ, ws = ws, ts = ts, use_sin = True
    ))(k = jnp.array([3, 2, 1, 0]))
    spline_cos_integrals = jax.vmap(partial(
        spline_integ, ws = ws, ts = ts, use_sin = False
    ))(k = jnp.array([3, 2, 1, 0]))
    
    coeffs_fs = jnp.stack(coeffs_fs, axis = 0)
    coeffs_ys = jnp.stack(coeffs_ys, axis = 0)
    
    term1 = jnp.einsum('abc, abd -> cd', spline_sin_integrals, coeffs_fs)
    term2 = jnp.einsum('abc, abd -> cd', spline_cos_integrals, coeffs_ys)
    coeff = jnp.sqrt(2 / ts[-1])
    res = coeff * (term1 + ws[:, None] * term2)
    return jnp.sum(res**2, axis = 0)

@eqx.filter_jit
def hermite_vf_loss(
    model: eqx.Module,
    batch_ts: Float[Array, 'traj tspan'],
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    func_num: int = 100,
    smoothing: float = 0.99,
    **kwargs
):
    '''
    Compute the VF loss that uses Hermite interpolation to compute the integrals in the ablation study.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
        - `func_num`: `int` - Number of basis functions.
        - `smoothing`: `float` - Smoothing parameter.
    '''
    batch_ts = batch_ts = batch_ts - batch_ts[:, 0][:, None]
    return jax.vmap(
        partial(_hermite_vf_loss, model, func_num)
    )(batch_ts, batch_ys).mean()