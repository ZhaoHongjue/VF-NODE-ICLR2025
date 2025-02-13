'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
import jax, jax.numpy as jnp, equinox as eqx
from jax.scipy.special import gamma
from jaxtyping import Array, Float, Key
from utils.interp import natural_cubic_spline_coeffs

def single_poly_vf_loss(ts, ys, fs, func_num):
    T = ts[-1]
    func = lambda s: (ts / T) * (ts / T - 1) *(ts / T - s / func_num)
    d_func = lambda s: ((ts/T)*(ts/T - 1) + (ts/T)*(ts/T - s/func_num) + (ts/T - 1)*(ts/T - s/func_num)) / T

    fg = jax.vmap(lambda s: fs * func(s)[:, None])(jnp.arange(func_num) + 1)
    ydg = jax.vmap(lambda s: ys * d_func(s)[:, None])(jnp.arange(func_num) + 1)

    d, c, b, a = jax.vmap(natural_cubic_spline_coeffs)(
        jnp.broadcast_to(ts, (func_num, ts.size)),
        fg + ydg
    )
    dt = jnp.diff(ts)[:, None]
    integ = jnp.sum(d / 4 * (dt ** 4) + c / 3 * (dt ** 3) + b / 2 * (dt ** 2) + a * dt, axis = 1)
    return integ**2

def poly_vf_loss(
    model: eqx.Module, 
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    func_num: int = 50,
    smoothing: float = 1.,
    **kwargs
):
    '''
    Compute the VF loss that uses polynomial basis functions in the ablation study.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
        - `func_num`: `int` - Number of basis functions.
        - `smoothing`: `float` - Smoothing parameter.
    '''
    batch_ts = batch_ts - batch_ts[:, 0][:, None]
    batch_fs = jax.vmap(model.vector_field)(batch_ts, batch_ys)
    return jax.vmap(
        partial(single_poly_vf_loss, func_num = func_num)
    )(batch_ts, batch_ys, batch_fs).mean()