'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025

Reference: 
    - https://docs.kidger.site/equinox/examples/train_rnn/
'''

from functools import partial
import jax, jax.numpy as jnp, equinox as eqx
from jaxtyping import Float, Array, Key


class RNN(eqx.Module):
    '''
    RNN model.
    
    Reference: 
        - https://docs.kidger.site/equinox/examples/train_rnn/
        
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `out_size`: `int` - Output size
        - `hidden_size`: `int` - Hidden size
        - `use_gru`: `bool` - Use GRU cell. Default: `True`
        - `including_dts`: `bool` - Whether to include time differences. Default: `False`
    '''
    rnn_cell: eqx.nn.GRUCell | eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    hidden_size: int
    including_dts: bool
    
    def __init__(
        self,
        key: Key, 
        obs_size: int, 
        out_size: int,
        hidden_size: int,
        use_gru: bool = True,
        including_dts: bool = False,
    ): 
        super().__init__()
        self.hidden_size = hidden_size
        self.including_dts = including_dts
        
        in_size = 2 * obs_size
        if including_dts: in_size += 1
        
        rnn_key, linear_key = jax.random.split(key, 2)
        rnn_key_kwargs = {
            'input_size': in_size, 'hidden_size': hidden_size,
            'key': rnn_key
        }
        self.rnn_cell = eqx.nn.GRUCell(**rnn_key_kwargs) if use_gru \
            else eqx.nn.LSTMCell(**rnn_key_kwargs)
        self.linear = eqx.nn.Linear(
            hidden_size, out_size, key = linear_key
        )
        
    def __call__(
        self,
        ts: Float[Array, 'tspan'],
        ys: Float[Array, 'tspan obs'],
        reverse: bool = False,
        evolving_out: bool = True
    ):
        mask = ~jnp.isnan(ys)
        ys = jnp.concatenate([jnp.nan_to_num(ys), mask], axis = -1)
        
        if self.including_dts:
            dts = jnp.append(ts[1:] - ts[:-1], 0.0)
            ys = jnp.concatenate([dts[..., None], ys], axis = -1)
        _, hidden_ts = jax.lax.scan(
            partial(self.forward_step, rnn_cell = self.rnn_cell),
            jnp.zeros((self.hidden_size,)), ys, reverse = reverse
        )
        if evolving_out:
            out = jax.vmap(self.linear)(hidden_ts)
        else: 
            out = self.linear(hidden_ts[-1])
        return out
        
    @staticmethod
    def forward_step(hidden, in_, rnn_cell):
        hidden = rnn_cell(in_, hidden)
        return hidden, hidden