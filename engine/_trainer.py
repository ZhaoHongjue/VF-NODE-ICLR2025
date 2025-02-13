'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import logging, time, os, wandb
from typing import Callable
from functools import partial
log = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax
jax.config.update('jax_enable_x64', True)

from models import *
import utils, data, utils.loss
from utils.init import xavier_uniform_init
from utils.visualize import *

class Trainer:
    def __init__(
        self, 
        cfg: DictConfig, 
        output_dir: str
    ) -> None:
        self.cfg = cfg
        self.km = utils.PRNGKey_Manager(seed = cfg.seed)
        
        self.train_set, self.val_set, self.test_set = data.generate_datasets(self.km.new_key(), cfg)
        self.model = self.create_model()
        self.opt, self.scheduler = self.create_optimizer()
        self.loss_fn = self.create_loss_fn()
        self.make_output_dirs(output_dir)
        
        if cfg.use_wandb:
            wandb.init(
                project = f'VF-NODE-ICLR-{self.cfg.project}',
                name = f'{cfg.model}-{cfg.ode}-{cfg.loss}-{cfg.scheduler}-{cfg.data}-{cfg.opt_type}-seed{cfg.seed}',
                config = OmegaConf.to_container(cfg)
            )
    
    
    def fit(self):
        min_loss = float('inf')
        overall_time, best_epoch, avg_time, init_epoch = 0, 0, float('NaN'), 0
        opt_state = self.opt.init(eqx.filter(self.model, eqx.is_inexact_array))
        
        if self.cfg.load:
            ckpt_pth = os.path.join(self.ckpt_dir, 'current_ckpt')
            if os.path.exists(ckpt_pth):
                self.model, opt_state, init_epoch = utils.load_ckpt(
                    ckpt_pth, self.model, opt_state, init_epoch
                )
                
        for epoch in range(init_epoch, self.cfg.epochs):
            utils.save_ckpt(
                os.path.join(self.ckpt_dir, 'current_ckpt'),
                self.model, opt_state, epoch
            )
            start = time.perf_counter()
            self.model, opt_state, train_loss = utils.make_opt_step(
                key = self.km.new_key(), opt = self.opt,
                model = self.model, loss_fn = self.loss_fn, 
                opt_state = opt_state, batch_ts = self.train_set[0], 
                batch_ys = self.train_set[1],
            )
            val_loss = self.loss_fn(
                self.model, self.val_set[0], self.val_set[1], 
                key = self.km.new_key()
            )
            finish = time.perf_counter()
            if epoch != 0: 
                overall_time += finish - start
                avg_time = overall_time / epoch
            
            if val_loss < min_loss:
                utils.save_eqx_model(os.path.join(self.ckpt_dir, 'best.eqx'), self.model)
                # utils.save_eqx_model(os.path.join(self.ckpt_dir, f'best-epoch{epoch}.eqx'), self.model)
    
            log_info = f'epoch: [{epoch+1:5}/{self.cfg.epochs:5}], ' + \
                f'train loss: {train_loss:.4e}, val loss: {val_loss:.4e}, ' + \
                f'lr: {self.scheduler(epoch):3e}, avg_time: {avg_time:.4f}, '
            logging.info(log_info)
            if self.cfg.use_wandb:
                wandb.log({
                    'epoch': epoch+1, 'lr': self.scheduler(epoch), 'train loss': train_loss, 
                    'val_loss': val_loss, 'avg_time': avg_time 
                })
        logging.info(f'Training time: {overall_time:.4e}')
        if self.cfg.use_wandb:
            wandb.log({'Training time': overall_time})
    
    
    def create_model(self) -> eqx.Module:
        solver_kws = SolverKwargs(**self.cfg.model.solver)
        tot_kwargs = {}
        tot_kwargs.update(self.cfg.model.kwargs)
        if self.cfg.model.type == 'NeuralODE' or \
            self.cfg.model.type == 'ODE_RNN' or \
            'LatentODE' in self.cfg.model.type or \
            'Flow' in self.cfg.model.type:
            tot_kwargs.update({
                'obs_size': self.cfg.ode.obs_size
            })
            
        elif self.cfg.model.type == 'NeuralCDE':
            tot_kwargs.update({
                'in_size': self.cfg.ode.obs_size,
                'out_size': self.cfg.ode.obs_size,
            })
            
        else: raise NotImplementedError
        
        return xavier_uniform_init(
            self.km.new_key(),
            eval(self.cfg.model.type)(
                key = self.km.new_key(), 
                solver_kws = solver_kws,
                **tot_kwargs
            )
        )
    
    
    def create_optimizer(self) -> optax.GradientTransformation:
        scheduler = eval(
            f'optax.{self.cfg.scheduler.type}'
        )(**self.cfg.scheduler.kwargs)
        return eval(f'optax.{self.cfg.opt_type}')(scheduler), scheduler
    
    
    def create_loss_fn(self) -> Callable:
        return partial(
            eval(f'utils.loss.{self.cfg.loss.type}'), 
            **self.cfg.loss.kwargs
        )
        
        
    def make_output_dirs(self, output_dir):
        self.output_dir = output_dir            
        self.ckpt_dir = os.path.join(output_dir, 'ckpt')
        self.fig_dir = os.path.join(output_dir, 'figures')
        self.eval_dir = os.path.join(output_dir, 'eval')
        dirs = [self.ckpt_dir, self.fig_dir, self.eval_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
                
                
    def load_model(self, epoch = None):
        if epoch is not None:
            ckpt_name = utils.parse_ckpt_name(self.ckpt_dir, epoch)
        else:
            ckpt_name = 'best.eqx'
        self.model = utils.load_eqx_model(
            os.path.join(self.ckpt_dir, ckpt_name), self.model
        )
        
        
    def evaluate(self, key, epoch = None):
        if 'SIR' in self.cfg.ode.type and self.cfg.ode.type != 'AgeSIR': 
            self.eval_sir(epoch)
            return
        
        def eval_loss(ys, pred_ys, use_mse = True):
            mask = jnp.astype(~jnp.isnan(ys), int)
            ys = jnp.nan_to_num(ys)
            pred_ys = pred_ys * mask
            if use_mse: return jnp.mean((pred_ys - ys)**2)
            else: return jnp.mean(
                jnp.abs(pred_ys - ys) / (jnp.abs(ys) + 1e-8)
            )
        
        def evaluate_model(key, model, bts, bys):
            if model.__class__.__name__ == 'NeuralODE':
                pred_bys = jax.vmap(model)(
                    bts, bys[:, 0, :], key = jax.random.split(key, len(bts))
                )
            else:
                pred_bys = jax.vmap(model)(
                    bts, bys, key = jax.random.split(key, len(bts))
                )
                
            in_bys, ex_bys = jnp.split(bys, 2, axis = 1)
            in_pred_bys, ex_pred_bys = jnp.split(pred_bys, 2, axis = 1)
                        
            return {
                'in_mape_loss': float(eval_loss(in_bys, in_pred_bys, False)),
                'in_mse_loss': float(eval_loss(in_bys, in_pred_bys, True)),
                'ex_mape_loss': float(eval_loss(ex_bys, ex_pred_bys, False)),
                'ex_mse_loss': float(eval_loss(ex_bys, ex_pred_bys, True))
            }
        
        self.load_model(epoch)
        bts, bys, sampled_bys = data.generate_eval_datasets(key, self.cfg)
        
        eval_clean = evaluate_model(self.km.new_key(), self.model, bts, bys)
        eval_noisy = evaluate_model(self.km.new_key(), self.model, bts, sampled_bys)
        
        logging.info(f'clean: {eval_clean}, noisy: {eval_noisy}')
        if self.cfg.use_wandb:
            wandb.log({'clean': eval_clean, 'noisy': eval_noisy})    

        self.eval_visualize(bts[0], bys[0], sampled_bys[0], epoch)
    
    
    def eval_visualize(self, ts, ys, sampled_ys, epoch = None):
        self.load_model(epoch)
        in_ = (ys[0], sampled_ys[0]) if self.cfg.model.type == 'NeuralODE' else (ys, sampled_ys)
        pred_clean_ys = self.model(ts, in_[0], key = self.km.new_key())
        pred_noisy_ys = self.model(ts, in_[1], key = self.km.new_key())

        if epoch is None:
            fig_name = f'/{self.cfg.ode.type}-{self.cfg.model.type}'
        else:
            fig_name = f'/{self.cfg.ode.type}-{self.cfg.model.type}-epoch{epoch}'
        
        if self.cfg.model.type == 'NeuralODE':
            fig_name += f'-{self.cfg.loss.type}'
        elif self.cfg.model.type == 'LatentODE':
            fig_name += f'-{self.cfg.loss.type}-{self.cfg.model.kwargs.enc_type}'
        
        plot_eval_phase(ys, pred_clean_ys, save_pth = self.eval_dir + fig_name)
        plot_eval_tspan(ts, ys, pred_clean_ys, save_pth = self.eval_dir + fig_name)
        plot_eval_phase(ys, pred_noisy_ys, sampled_ys, save_pth = self.eval_dir + fig_name)
        plot_eval_tspan(ts, ys, pred_noisy_ys, sampled_ys, save_pth = self.eval_dir + fig_name)
        
    def eval_sir(self, epoch):
        def eval_loss(ys, pred_ys, use_mse = True):
            mask = jnp.astype(~jnp.isnan(ys), int)
            ys = jnp.nan_to_num(ys)
            pred_ys = pred_ys * mask
            if use_mse: return jnp.mean((pred_ys - ys)**2)
            else: return jnp.mean(
                jnp.abs(pred_ys - ys) / (jnp.abs(ys) + 1e-8)
            )
            
        self.load_model(epoch)
        
        ## Separate Trajectories
        for mode in ['train', 'val', 'test']:
            bts, bys = getattr(self, f'{mode}_set')
            
            if self.model.__class__.__name__ == 'NeuralODE':
                pred_bys = jax.vmap(self.model)(bts, bys[:, 0, :])
            else:
                pred_bys = jax.vmap(self.model)(
                    bts, bys, key = jax.random.split(self.km.new_key(), len(bts))
                )
            
            eval_mse = eval_loss(bys, pred_bys, use_mse = True)
            eval_mape = eval_loss(bys, pred_bys, use_mse = False)
            logging.info(f'eval_{mode}_mse: {eval_mse:.4e},  eval_{mode}_mape: {eval_mape:.4e}')
            if self.cfg.use_wandb:
                wandb.log({
                    f'eval_{mode}_mse': eval_mse,
                    f'eval_{mode}_mape': eval_mape,
                })    
            
            figure = plt.figure(figsize = (15, 5))
            ts, ys, pred_ys = bts[0], bys[0], pred_bys[0]
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.plot(ts, ys[:, i], label = 'Ground Truth')
                plt.plot(ts, pred_ys[:, i], label = 'Prediction')
                plt.legend()
            plt.tight_layout()
            plt.savefig(self.fig_dir + f'/covid-{mode}-separate.png', bbox_inches = 'tight') 