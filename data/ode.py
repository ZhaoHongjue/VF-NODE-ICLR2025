'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from typing import Tuple
from abc import abstractmethod
from functools import partial

import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, diffrax as dfx
from jaxtyping import Key, Float, Array

jax.config.update('jax_enable_x64', True)


class DataODE(eqx.Module):
    '''
    Base class for ODE models.
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float]
    y0_range: Tuple[Tuple[float]]
    
    @abstractmethod
    def __call__(self, t, y, args):
        '''
        Compute the vector field of the ODE.
        
        Args:
            - `t`: `float` - Time
            - `y`: `Array[float, "n"]` - State
            - `args`: `Tuple[float]` - Parameters
        '''
        raise NotImplementedError
    
    def simulate(
        self,
        key: Key, 
        T: float, 
        point_num: int = 100, 
        traj_num: int = 25,
        **kwargs
    ):
        '''
        Simulate the ODE.
        
        Args:
            - `key`: `Key` - Random key
            - `T`: `float` - Time horizon
            - `point_num`: `int` - Number of points. Default: 100
            - `traj_num`: `int` - Number of trajectories. Default: 25
        '''
        y0_key, sample_key = jr.split(key, num = 2)
        y0_range_jnp = jnp.asarray(self.y0_range)
        y0_ratio = jr.uniform(y0_key, (traj_num, y0_range_jnp.shape[0]))
        interval = y0_range_jnp[:, 1] - y0_range_jnp[:, 0]
        y0 = y0_ratio * interval + y0_range_jnp[:, 0]
        ts = jnp.sort(jr.uniform(
            sample_key, (traj_num, point_num),
            minval = 0., maxval = T
        ), axis = 1)
        return jax.vmap(self._solve_)(ts, y0)
    
    def simulate_ts(
        self, key: Key,
        ts: Float[Array, "traj tspan"],
    ):
        '''
        Simulate the ODE at specific times.
        
        Args:
            - `key`: `Key` - Random key
            - `ts`: `Float[Array, "traj tspan"]` - Times
        '''
        traj_num = ts.shape[0]
        y0_range_jnp = jnp.asarray(self.y0_range)
        y0_ratio = jr.uniform(key, (traj_num, y0_range_jnp.shape[0]))
        interval = y0_range_jnp[:, 1] - y0_range_jnp[:, 0]
        y0 = y0_ratio * interval + y0_range_jnp[:, 0]
        _, ys = jax.vmap(self._solve_)(ts, y0)
        return ys
    
    def vector_field(self, ts, ys):
        '''
        Compute the vector field of the ODE.
        
        Args:
            - `ts`: `Float[Array, "traj tspan"]` - Times
            - `ys`: `Float[Array, "traj n"]` - States
        '''
        return jax.vmap(partial(self, args = self.args))(ts, ys)
        
    def _solve_(self, ts, y0):
        '''
        Solve the ODE.
        
        Args:
            - `ts`: `Float[Array, "traj tspan"]` - Times
            - `y0`: `Float[Array, "traj n"]` - Initial states
        '''
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self),
            dfx.Dopri5(),
            t0 = ts[0], t1 = ts[-1], dt0 = None,
            y0 = y0, saveat = dfx.SaveAt(ts = ts),
            args = self.args, max_steps = 8192,
            stepsize_controller = dfx.PIDController(1e-8, 1e-10)
        )
        return sol.ts, sol.ys
 

class Toggle(DataODE):
    '''
    Toggle model for gene expression.
    
    Reference: https://www.nature.com/articles/35002131
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (4., 4., 3., 3.)
    y0_range: Tuple[Tuple[float]] = ((0.1, 4.),)*2
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        alpha_1, alpha_2, beta, gamma = args
        dy0 = alpha_1 / (1 + y[1] ** beta) - y[0]
        dy1 = alpha_2 / (1 + y[0] ** gamma) - y[1]
        return jnp.array([dy0, dy1])


class Glycolytic(DataODE):
    '''
    Glycolytic model.
    
    Reference: https://febs.onlinelibrary.wiley.com/doi/full/10.1111/j.1432-1033.1968.tb00175.x
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (0.75, 0.1, 0.1)
    y0_range: Tuple[Tuple[float]] = ((0.1, 1.1),)*2
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        th1, th2, th3 = args
        dy0 = th1 - th2 * y[0] - y[0] * y[1] ** 2
        dy1 = -y[1] + th3 * y[0] + y[0] * y[1] ** 2
        return jnp.array([dy0, dy1])
    

class Repressilator2(DataODE):
    '''
    Repressilator model.
    
    Reference: https://www.nature.com/articles/35002125
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    
    args: Tuple[float] = (10., 1e-5, 1., 3.)
    y0_range: Tuple[Tuple[float]] = ((0., 5.),)*6
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        alpha, alpha_0, beta, n = args
        dy0 = - y[0] + alpha / (1 + y[5] ** n) + alpha_0
        dy1 = - y[1] + alpha / (1 + y[3] ** n) + alpha_0
        dy2 = - y[2] + alpha / (1 + y[4] ** n) + alpha_0
        dy3 = beta * ( y[0] - y[3] )
        dy4 = beta * ( y[1] - y[4] )
        dy5 = beta * ( y[2] - y[5] )
        return jnp.array([dy0, dy1, dy2, dy3, dy4, dy5])
    

class AgeSIR(DataODE):
    '''
    Age-structured SIR model.
    
    Reference: https://www.nature.com/articles/s41598-021-94609-3
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (0.8, 0.5)
    y0_range: Tuple[Tuple[float]] = ((0.1, 10.1),)*27
    
    def __call__(self, t, y, args):
        (S, I, R), N = jnp.split(y, 3), y.sum()
        beta, gamma = args
        M = jnp.array([
            [19.2,  4.8,  3. ,  7.1,  3.7,  3.1,  2.3,  1.4,  1.4],
            [ 4.8, 42.4,  6.4,  5.4,  7.5,  5. ,  1.8,  1.7,  1.7],
            [ 3. ,  6.4, 20.7,  9.2,  7.1,  6.3,  2. ,  0.9,  0.9],
            [ 7.1,  5.4,  9.2, 16.9, 10.1,  6.8,  3.4,  1.5,  1.5],
            [ 3.7,  7.5,  7.1, 10.1, 13.1,  7.4,  2.6,  2.1,  2.1],
            [ 3.1,  5. ,  6.3,  6.8,  7.4, 10.4,  3.5,  1.8,  1.8],
            [ 2.3,  1.8,  2. ,  3.4,  2.6,  3.5,  7.5,  3.2,  3.2],
            [ 1.4,  1.7,  0.9,  1.5,  2.1,  1.8,  3.2,  7.2,  7.2],
            [ 1.4,  1.7,  0.9,  1.5,  2.1,  1.8,  3.2,  7.2,  7.2],
        ])
        dS = -beta * S/N * (M @ I)
        dI = beta * S/N * (M @ I) - gamma * I
        dR = gamma * I
        return jnp.concatenate([dS, dI, dR])

    
class Gompertz(DataODE):
    '''
    Gompertz model.
    
    Reference: https://en.wikipedia.org/wiki/Gompertz_function
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (1.5, 1.5)
    y0_range: Tuple[Tuple[float]] = ((0.1, 1.1),)
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        th1, th2 = args
        return -th1 * y * jnp.log(th2 * y)

    
class PredatorPrey(DataODE):
    '''
    Predator-prey model/Lotka-Volterra equations.
    
    Reference: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (1.0, 0.3, 0.1, 3.0)
    y0_range: Tuple[Tuple[float]] = ((10.0, 20.0), (5.0, 10.0))
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        alpha, beta, delta, gamma = args # [1.0, 0.3, 0.1, 3.0]
        dy0 = alpha*y[0] - beta*y[0]*y[1]
        dy1 = delta*y[0]*y[1] - gamma*y[1]
        return jnp.array([dy0, dy1])


class Lorenz(DataODE):
    '''
    Lorenz system.
    
    Reference: https://en.wikipedia.org/wiki/Lorenz_system
    
    Args:
        - `args`: `Tuple[float]` - Parameters
        - `y0_range`: `Tuple[Tuple[float]]` - Initial state range
    '''
    args: Tuple[float] = (10.0, 28.0, 2.66667)
    y0_range: Tuple[Tuple[float]] = ((-10., 10.),)*3
    
    @partial(jax.jit, static_argnums = (0,))
    def __call__(self, t, y, args):
        P, Ra, b = args
        dy0 = P*(y[1] - y[0])
        dy1 = Ra*y[0] - y[1] - y[0]*y[2]
        dy2 = y[0]*y[1] - b*y[2]
        return jnp.array([dy0, dy1, dy2])