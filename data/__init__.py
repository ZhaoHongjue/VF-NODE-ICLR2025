'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import os, pickle, pandas as pd
from omegaconf import DictConfig

import jax, jax.random as jr, numpy as np
from jaxtyping import Float, Array, Key
jax.config.update('jax_enable_x64', True)

from csaps import csaps

from utils import generate_mask
from utils.interp import nan_cubic_spline_smoothing
from .ode import *


_general_dynamics_ = [
    'Toggle',
    'Glycolytic',
    'Repressilator2',
    'AgeSIR',
    'Gompertz', 
    'PredatorPrey', 
    'Lorenz', 
]

def _preprocessing_data(
    ts: Float[Array, "traj tspan"],
    ys: Float[Array, "traj n"],
    cfg: DictConfig,
):
    '''
    Preprocess the data. If the model is the VF-NODE (or other models related to ablation studies), 
    the data is processed using spline regression (key technique in VF-NODE). Otherwise, the data is not processed.
    
    Args:
        - `ts`: `jnp.ndarray` - Times
        - `ys`: `jnp.ndarray` - States
    '''
    conds = [
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'spline_vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'bern_vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'herm_vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'hermite_vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'poly_vf_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'spline_integ_loss',
        cfg.model.type == 'NeuralODE'     and     cfg.loss.type == 'grad_matching_loss',
    ]
    
    if any(conds): 
        return ts, nan_cubic_spline_smoothing(ts, ys, cfg.loss.kwargs.smoothing)
    return ts, ys


def generate_datasets(
    key: Key, 
    cfg: DictConfig,
):
    '''
    Generate the datasets based on the configuration.
    
    Args:
        - `key`: `jax.random.PRNGKey` - Random key
        - `cfg`: `Dict` - Configuration
    '''
    if cfg.ode.type in _general_dynamics_:
        return load_ode_data(key, cfg)
    elif 'SIRF_' in cfg.ode.type:
        return load_sirf_data(key, cfg)
    else:
        raise ValueError(f'Unknown dataset type: {cfg.ode.type}')


def generate_single_set(
    ts, ys, noise_level, ratio, 
    mask_key, noise_key, 
    split = False, split_num = 10
):
    '''
    Generate a single dataset.
    
    Args:
        - `ts`: `jnp.ndarray` - Times
        - `ys`: `jnp.ndarray` - States
        - `noise_level`: `float` - Noise level
        - `ratio`: `float` - Ratio
    '''
    mask_keys = jr.split(mask_key, split_num)
    ys += noise_level * jnp.std(
        ys, axis = 1, keepdims = True
    ) * jr.normal(noise_key, ys.shape)
    split_ys = jnp.split(ys, split_num, axis = 1)
    split_masks = [
        generate_mask(mask_keys[i], split_ys[0].shape, 1 - ratio)
        for i in range(split_num)
    ]

    masked_ys = jnp.concatenate([
        split_ys[i] * split_masks[i]
        for i in range(split_num)
    ], axis = 0 if split else 1)
    ts = jnp.concatenate(
        jnp.split(ts, split_num, axis = 1), axis = 0
    ) if split else ts
    return ts, masked_ys

def load_ode_data(key, cfg):
    sim_key, mask_key, noise_key = jr.split(key, 3)
    tr_mask_key, vl_mask_key = jr.split(mask_key, 2)
    tr_noise_key, vl_noise_key = jr.split(noise_key, 2)

    val_size = cfg.data.traj_num // 6
    train_size = cfg.data.traj_num - 2 * val_size

    dynamic: DataODE = eval(cfg.ode.type)()
    raw_ts, raw_ys = dynamic.simulate(
        sim_key, cfg.data.T, cfg.data.point_num, cfg.data.traj_num
    )

    tr_ys, vl_ys, te_ys = raw_ys[:train_size], raw_ys[train_size: -val_size], raw_ys[-val_size:]
    tr_ts, vl_ts, te_ts = raw_ts[:train_size], raw_ts[train_size: -val_size], raw_ts[-val_size:]

    tr_set = generate_single_set(
        tr_ts, tr_ys, cfg.data.noise_level, cfg.data.ratio,
        tr_mask_key, tr_noise_key, cfg.data.split, cfg.data.split_num
    )
    tr_set = _preprocessing_data(tr_set[0], tr_set[1], cfg)
    vl_set = generate_single_set(
        vl_ts, vl_ys, cfg.data.noise_level, cfg.data.ratio,
        vl_mask_key, vl_noise_key, cfg.data.split, cfg.data.split_num
    )
    vl_set = _preprocessing_data(vl_set[0], vl_set[1], cfg)
    te_set = (te_ts, te_ys)
    return tr_set, vl_set, te_set


def generate_eval_datasets(key, cfg):
    traj_key, noise_key, mask_key = jr.split(key, 3)
    eval_size = cfg.data.traj_num // 6
    
    if cfg.ode.type in _general_dynamics_:
        traj_keys = jr.split(traj_key, 3)
        dynamic: DataODE = eval(cfg.ode.type)()
        batch_ts1 = jnp.sort(jr.uniform(
            traj_keys[0], (cfg.data.traj_num, cfg.data.point_num), 
            jnp.float64, 0, cfg.data.T
        ), axis = 1)
        batch_ts2 = jnp.sort(jr.uniform(
            traj_keys[1], (cfg.data.traj_num, cfg.data.point_num), 
            jnp.float64, cfg.data.T, 2*cfg.data.T
        ), axis = 1)
        batch_ts = jnp.concatenate([batch_ts1, batch_ts2], axis = 1)
        batch_ys = dynamic.simulate_ts(traj_keys[2], batch_ts)
        
    sampled_bys = batch_ys + cfg.data.noise_level * jnp.std(
        batch_ys, axis = 1, keepdims = True
    ) * jr.normal(noise_key, batch_ys.shape)
    mask = generate_mask(
        mask_key, sampled_bys.shape, 1 - cfg.data.ratio
    )
    sampled_bys = sampled_bys * mask
    _, sampled_bys = _preprocessing_data(batch_ts, sampled_bys, cfg)
    
    if not cfg.ode.auto:
        batch_ys = jnp.concatenate((batch_ts[..., None], batch_ys), axis = -1)
        sampled_bys = jnp.concatenate((batch_ts[..., None], sampled_bys), axis = -1)
    
    return batch_ts, batch_ys, sampled_bys


def load_sirf_data(key, cfg):
    country = cfg.ode.type.split('_')[-1]
    
    df = pd.read_csv(f'./data/sirf/{country}.csv', index_col = 0)
    ts = jnp.linspace(0, 1, 100)
    raw_ys = df.values[:100] / df.values[0].sum() # * 1e5
    ys = csaps(ts, raw_ys.T, ts, smooth = 0.99999).T
    
    mu, std = ys[:90].mean(axis = 0), ys[:90].std(axis = 0)
    ys = (ys - mu[None]) / std[None]
    
    tr_bts, tr_bys = ts[:80][None], ys[:80][None]
    if cfg.data.split:
        tr_bts = tr_bts.reshape(cfg.data.split_num, -1)
        tr_bys = tr_bys.reshape(cfg.data.split_num, -1, ys.shape[-1])
    vl_bts, vl_bys = ts[80:90][None], ys[80:90][None]
    te_bts, te_bys = ts[90:][None], ys[90:][None]
    tr_set, vl_set, te_set = (tr_bts, tr_bys), (vl_bts, vl_bys), (te_bts, te_bys)
    return tr_set, vl_set, te_set
    

def load_tumor_data():
    with open('./data/tumor/real_data_c1.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    with open('./data/tumor/real_data_mask_c1.pkl', 'rb') as f:
        mask = pickle.load(f)

    raw_bys = raw_data.transpose(1, 0, 2)
    raw_bts = raw_data[..., 1].T
    
    mask = mask.T[250:].astype(float)
    mask[:, 0] = 1.
    mask[mask == 0] = float('NaN')
    raw_bys[250:, :, 0] *= mask

    train_set = raw_bts[:200], raw_bys[:200]
    val_set = raw_bts[200:250], raw_bys[200:250]
    test_set = raw_bts[250:], raw_bys[250:]
    return train_set, val_set, test_set
