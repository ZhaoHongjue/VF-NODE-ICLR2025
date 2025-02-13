'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import jax, jax.numpy as jnp, equinox as eqx

@eqx.filter_jit
def vae_loss(model, batch_ts, batch_ys, key, **kwargs):
    '''
    Compute the ELBO for VAEs.
    
    Args:
        - `model`: `Callable` - Model.
        - `batch_ts`: `Float[Array, 'traj tspan']` - Time points.
        - `batch_ys`: `Float[Array, 'traj tspan obs']` - Values.
        - `key`: `Key` - PRNGKey.
    '''
    return jnp.mean(jax.vmap(model._loss)(
        batch_ts, batch_ys, jax.random.split(key, num = len(batch_ts))
    ))
