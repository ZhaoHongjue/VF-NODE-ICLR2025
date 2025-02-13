'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from typing import Callable
from functools import partial

import jax, jax.numpy as jnp, equinox as eqx
from jax.scipy.integrate import trapezoid
from jax.experimental.jet import jet
from jaxtyping import Float, Array, Key

def taylor_k_coeff_jet(
    func: Callable, 
    x0: Float[Array, 'dim'], 
    K: int
) -> jnp.ndarray:
    y0 = func(x0)
    yk = []
    for _ in range(K - 1):
        _, yk = jet(func, (x0,), ([y0,] + yk,))
    return yk[-1]

def taylor_coeffs_jet(func, x0, K):
    y0 = func(x0)
    yk = []
    for _ in range(K - 1):
        _, yk = jet(func, (x0,), ([y0,] + yk,))
    return jnp.asarray([x0, y0] + yk)

def high_order_deriv_regu(
    func: Callable, 
    ts: Float[Array, 'tspan'],
    xs: Float[Array, 'tspan dim'],
    K: int
) -> jnp.ndarray:
    coeff_Ks = jax.vmap(partial(taylor_k_coeff_jet, func = func, K = K))(x0 = xs)
    return trapezoid(jnp.linalg.norm(coeff_Ks, axis = 1)**2, ts)

@eqx.filter_jit
def regu_mse_loss(
    node: Callable, 
    batch_ts: Float[Array, 'traj tspan'], 
    batch_ys: Float[Array, 'traj tspan obs'],
    key: Key,
    regu_k: int = 5,
    lamb: float = 1.,
    **kwargs
):  
    '''
    Compute the MSE loss with high-order derivative regularization.
    
    Reference: 
        - Learning Differential Equations that are Easy to Solve
    
    Args:
        - `node`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
        - `regu_k`: `int` - Order of the Taylor series.
        - `lamb`: `float` - Regularization parameter.
    '''
    assert hasattr(node, 'ode_func')
    pred_batch_ys = jax.vmap(node)(
        batch_ts, batch_ys[:, 0, :], key = jax.random.split(key, len(batch_ts))
    )
    
    # compute regularizer
    func = lambda y: node.ode_func(0, y)
    regu = jnp.mean(jax.vmap(
        partial(high_order_deriv_regu, func = func, K = regu_k)
    )(ts = batch_ts, xs = pred_batch_ys))
    
    # compute mse
    non_mask = jnp.astype(~jnp.isnan(batch_ys), int)
    batch_ys = jnp.nan_to_num(batch_ys)
    pred_batch_ys = pred_batch_ys * non_mask
    mse = jnp.mean((batch_ys - pred_batch_ys)**2)
    return mse + lamb * regu