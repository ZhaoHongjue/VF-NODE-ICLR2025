'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
import jax, jax.numpy as jnp, equinox as eqx
from jax.scipy.special import gamma
from jaxtyping import Array, Float, Key
from interpax import CubicHermiteSpline
from utils.interp import natural_cubic_spline_coeffs

def single_herm_vf_loss(ts, ys, fs, func_num):
    T = ts[-1]
    def fg_plus_ydg_func(s, ts, ys, fs, func_num):
        herm = CubicHermiteSpline(
            x = jnp.array([0, T/2, T]),
            y = jnp.array([0, s/func_num, 0]),
            dydx = jnp.array([0, 0, 0]),
            check = False
        )
        dherm = herm.derivative()
        fg = fs * herm(ts)[:, None]
        ydg = ys * dherm(ts)[:, None]
        return fg + ydg

    fg_plus_ydg = jax.vmap(
        lambda s: fg_plus_ydg_func(s, ts, ys, fs, func_num)
    )(jnp.arange(func_num) + 1)

    d, c, b, a = jax.vmap(natural_cubic_spline_coeffs)(
        jnp.broadcast_to(ts, (func_num, ts.size)),
        fg_plus_ydg
    )
    dt = jnp.diff(ts)[:, None]
    integ = jnp.sum(d / 4 * (dt ** 4) + c / 3 * (dt ** 3) + b / 2 * (dt ** 2) + a * dt, axis = 1)
    return integ**2

def herm_vf_loss(
    model: eqx.Module, 
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    func_num: int = 100,
    smoothing: float = 1.,
    **kwargs
):
    '''
    Compute the VF loss that uses Hermite polynomials as basis functions in the ablation study.
    
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
        partial(single_herm_vf_loss, func_num = func_num)
    )(batch_ts, batch_ys, batch_fs).mean()