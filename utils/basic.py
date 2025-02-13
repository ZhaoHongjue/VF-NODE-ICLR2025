'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import os, numpy as np
from typing import Callable
import jax, jax.numpy as jnp, equinox as eqx, optax
from jaxtyping import Float, Array, Key, PyTree
from math import floor


class PRNGKey_Manager:
    '''
    Manager for PRNGKey.
    
    Args:
        - `seed`: `int` - Seed for the PRNGKey.
    '''
    def __init__(self, seed: int = 0) -> None:
        self.raw_key = jax.random.PRNGKey(seed = seed)
        
    def new_key(self):
        key, self.raw_key = jax.random.split(self.raw_key, num = 2)
        return key
    
    
@eqx.filter_jit
def make_opt_step(
    key: Key,
    opt: optax.GradientTransformation,
    model: eqx.Module,
    loss_fn: Callable,
    opt_state: PyTree,
    batch_ts: Float[Array, 'batch tspan'],
    batch_ys: Float[Array, 'batch tspan obs'],
):
    '''
    Make an optimization step using the loss function.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `opt`: `optax.GradientTransformation` - Optimization transformation.
    '''
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch_ts, batch_ys, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def parse_ckpt_name(
    ckpt_pth: str, 
    epoch: int
):
    '''
    Parse the checkpoint name.
    
    Args:
        - `ckpt_pth`: `str` - Path to the checkpoint.
        - `epoch`: `int` - Epoch number.
    '''
    ckpt_lst = os.listdir(ckpt_pth)
    ckpt_lst.remove('best.eqx')
    idx_lst = [
        int(ckpt_name[ckpt_name.rfind('epoch')+5:ckpt_name.rfind('.')])
        for ckpt_name in ckpt_lst
    ]
    idx_lst.sort()
    idx_np = np.array(idx_lst)
    best_idx = idx_np[idx_np < epoch][-1]
    best_ckpt_name = f'best-epoch{best_idx}.eqx'
    return best_ckpt_name


def save_ckpt(
    filename: str, 
    model: eqx.Module, 
    opt_state: PyTree, 
    epoch: int
):  
    '''
    Save the checkpoint.
    
    Args:
        - `filename`: `str` - Path to the checkpoint.
        - `model`: `eqx.Module` - Model.
        - `opt_state`: `PyTree` - Optimization state.
        - `epoch`: `int` - Epoch number.
    '''
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    ckpt = (params, opt_state, epoch)
    with open(filename, 'wb') as f:
        eqx.tree_serialise_leaves(f, ckpt)
        
        
def load_ckpt(
    filename: str, 
    raw_model: eqx.Module, 
    raw_opt_state: PyTree, 
    raw_epoch: int
) -> eqx.Module:
    '''
    Load the checkpoint.
    
    Args:
        - `filename`: `str` - Path to the checkpoint.
        - `raw_model`: `eqx.Module` - Model.
        - `raw_opt_state`: `PyTree` - Optimization state.
    '''
    raw_params, arch = eqx.partition(raw_model, eqx.is_inexact_array)
    raw_ckpt = (raw_params, raw_opt_state, raw_epoch)
    with open(filename, 'rb') as f:
        params, opt_state, epoch = eqx.tree_deserialise_leaves(f, raw_ckpt)
    model = eqx.combine(params, arch)
    return model, opt_state, epoch

def save_eqx_model(
    filename: str, 
    model: eqx.Module
):
    '''
    Save the model.
    '''
    with open(filename, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
        
def load_eqx_model(
    filename: str, 
    model: eqx.Module
) -> eqx.Module:
    '''
    Load the model.
    '''
    with open(filename, 'rb') as f:
        return eqx.tree_deserialise_leaves(f, model)

def count_params(model: eqx.Module) -> int:
    '''
    Count the number of parameters in the model.
    '''
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def generate_mask(
    key: Key, 
    shape: tuple[int, int, int], 
    nan_p: float = 0.0
):
    '''
    Generate a mask.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `shape`: `Tuple[int]` - Shape of the mask.
        - `nan_p`: `float` - Proportion of NaNs in the mask.
    '''
    assert len(shape) == 3
    return jax.random.choice(
        key, jnp.array([1, jnp.nan]), shape, 
        p = jnp.array([1-nan_p, nan_p])
    ).at[:, jnp.array([0, -1]), :].set(1)

def irregularly_sampling(
    key: Key, 
    ts: Float[Array, 'traj tspan'], 
    ys: Float[Array, 'traj obs'], 
    ratio: float = 0.8
):
    '''
    Irregularly sample the data.
    
    Args:
        - `key`: `Key` - PRNGKey.
        - `ts`: `Float[Array, 'traj tspan']` - Time points.
    '''
    if ratio == 1.0: return ts, ys
    t_len = ts.shape[1]
    indices = jnp.sort(
        jax.random.permutation(
            key, jnp.broadcast_to(
                jnp.arange(t_len), ts.shape
            ), axis = 1, independent = True
        )[:, :floor(t_len * ratio)+1], axis = 1
    )
    sampled_ts = jnp.stack([
        t[index] for t, index in zip(ts, indices)
    ], axis = 0)
    sampled_ys = jnp.stack([
        y[index] for y, index in zip(ys, indices)
    ], axis = 0)
    return sampled_ts, sampled_ys