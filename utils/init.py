'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: https://docs.kidger.site/equinox/tricks/#custom-parameter-initialisation
'''

from typing import Callable

import jax, jax.numpy as jnp, equinox as eqx
from jax.nn import initializers
from jaxtyping import Float, Array, Key, PyTree


def xavier_uniform_init(
    key: Key, 
    model: eqx.Module
):
    '''
    Initialize the model with Xavier uniform initialization.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `model`: `eqx.Module` - Model.
    '''
    weight_key, bias_key = jax.random.split(key, 2)
    model = init_linear(weight_key, model, xavier_uniform_init_params, is_weight = True)
    model = init_linear(bias_key, model, zero_init_params, is_weight = False)
    return model


def init_linear(
    key: Key, 
    model: eqx.Module, 
    init_fn: Callable, 
    is_weight: bool
):
    '''
    Initialize the linear parameters of the model.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `model`: `eqx.Module` - Model.
    '''
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_params = lambda m: [
        x.weight if is_weight else x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf = is_linear)
        if is_linear(x)
    ]
    params = get_params(model)
    new_params = [
        init_fn(subkey, weight)
        for weight, subkey in zip(
            params, jax.random.split(key, len(params))
    )]
    new_model = eqx.tree_at(get_params, model, new_params)
    return new_model


def xavier_uniform_init_params(
    key: Key, 
    weight: Float[Array, 'out in']
):
    '''
    Initialize the weight of the linear parameters with Xavier uniform initialization.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `weight`: `Float[Array, 'out in']` - Weight.
    '''
    initializer = initializers.glorot_uniform()
    return initializer(key, weight.shape, jnp.float64) 

def zero_init_params(
    key: Key, 
    bias: Float[Array, 'out']
):
    '''
    Initialize the bias of the linear parameters with zero initialization.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `bias`: `Float[Array, 'out']` - Bias.
    '''
    if bias is not None: return jnp.zeros_like(bias)
    else: return