'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
import jax, jax.numpy as jnp, equinox as eqx
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Float, Key

@eqx.filter_jit
def single_vf_loss(ts, ys, fs, func_num):
    T = ts[-1]
    coeff = jnp.sqrt(2 / T)
    fg = jax.vmap(
        lambda s: fs * coeff * jnp.sin(s * jnp.pi * ts / T)[:, None]
    )(jnp.arange(func_num) + 1)
    ydg = jax.vmap(
        lambda s: ys * coeff * s * jnp.pi / T * jnp.cos(s * jnp.pi * ts / T)[:, None]
    )(jnp.arange(func_num) + 1)
    return jnp.sum(trapezoid(fg + ydg, ts, axis = 1)**2, axis = 0)

@eqx.filter_jit
def vf_loss(
    model: eqx.Module, 
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    func_num: int = 100,
    smoothing: float = 1.,
    **kwargs
):
    '''
    Compute the VF loss using trapezoid rule.
    
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
        partial(single_vf_loss, func_num = func_num)
    )(batch_ts, batch_ys, batch_fs).mean()
    