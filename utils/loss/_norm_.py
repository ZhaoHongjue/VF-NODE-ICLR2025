'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import jax, jax.numpy as jnp, equinox as eqx

@eqx.filter_jit
def l1_loss(model, batch_ts, batch_ys, key, **kwargs):
    '''
    Compute the L1 loss.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    if model.__class__.__name__ == 'NeuralODE':
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys[:, 0, :], key = jax.random.split(key, len(batch_ts))
        )
    else:
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys, key = jax.random.split(key, len(batch_ts))
        )
        
    non_mask = jnp.astype(~jnp.isnan(batch_ys), int)
    batch_ys = jnp.nan_to_num(batch_ys)
    pred_batch_ys = pred_batch_ys * non_mask
    return jnp.mean(jnp.abs(batch_ys - pred_batch_ys))  

@eqx.filter_jit
def mse_loss(model, batch_ts, batch_ys, key, **kwargs):
    '''
    Compute the MSE.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    if model.__class__.__name__ == 'NeuralODE':
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys[:, 0, :], key = jax.random.split(key, len(batch_ts))
        )
    else:
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys, key = jax.random.split(key, len(batch_ts))
        )
    
    non_mask = jnp.astype(~jnp.isnan(batch_ys), int)
    batch_ys = jnp.nan_to_num(batch_ys)
    pred_batch_ys = pred_batch_ys * non_mask
    return jnp.mean((batch_ys - pred_batch_ys)**2)

@eqx.filter_jit
def tumor_rmse_loss(model, batch_ts, batch_ys, key, **kwargs):
    '''
    Compute the RMSE loss for tumor data in the Appendix.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    if model.__class__.__name__ == 'NeuralODE':
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys[:, 0, :], key = jax.random.split(key, len(batch_ts))
        )
    else:
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys, key = jax.random.split(key, len(batch_ts))
        )
    non_mask = jnp.astype(~jnp.isnan(batch_ys), int)
    batch_ys = jnp.nan_to_num(batch_ys)
    pred_batch_ys = pred_batch_ys * non_mask
    return jnp.mean((batch_ys[..., 0] - pred_batch_ys[..., 0])**2)

@eqx.filter_jit
def mape_loss(model, batch_ts, batch_ys, key, **kwargs):
    '''
    Compute the MAPE.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    if model.__class__.__name__ == 'NeuralODE' or \
        model.__class__.__name__ == 'LinearNODE':
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys[:, 0, :], key = jax.random.split(key, len(batch_ts))
        )
    else:
        pred_batch_ys = jax.vmap(model)(
            batch_ts, batch_ys, key = jax.random.split(key, len(batch_ts))
        )
        
    non_mask = jnp.astype(~jnp.isnan(batch_ys), int)
    batch_ys = jnp.nan_to_num(batch_ys)
    pred_batch_ys = pred_batch_ys * non_mask
    return jnp.mean(jnp.abs(batch_ys - pred_batch_ys) / jnp.abs(batch_ys))