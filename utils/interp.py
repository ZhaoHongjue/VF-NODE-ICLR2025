'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import jax, jax.numpy as jnp, jax.lax as lax
import equinox as eqx, diffrax as dfx
from csaps import csaps


@eqx.filter_jit
def tridiagonal_solve(dl, d, du, b): 
   """
   Pure JAX implementation of `tridiagonal_solve`.
   
   Reference: https://github.com/google/jax/discussions/10339
    
   Args:
       - `dl`: `Float[Array, 'n']` - Lower diagonal.
       - `d`: `Float[Array, 'n']` - Diagonal.
       - `du`: `Float[Array, 'n']` - Upper diagonal.
       - `b`: `Float[Array, 'n']` - Right-hand side.
   """ 
   prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1]) 
   fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_) 
   fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2]) 
   bwd1 = lambda x_, x: x[0] - x[1] * x_ 
   double = lambda f, args: (f(*args), f(*args)) 
  
   # Forward pass. 
   _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)), 
                     du[0] / d[0], 
                     (d, du, dl), 
                     unroll=32) 
  
   _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)), 
                    b[0] / d[0], 
                    (b, d, prepend_zero(tu_), dl), 
                    unroll=32) 
  
   # Backsubstitution. 
   _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)), 
                    b_[-1], 
                    (b_[::-1], tu_[::-1]), 
                    unroll=32) 
  
   return x_[::-1] 


@eqx.filter_jit
def natural_cubic_spline_coeffs(ts, ys):
    '''
    Compute the coefficients of the natural cubic spline. This is compatible with `CubicInterpolation` in Diffrax.
    
    Reference: https://en.wikipedia.org/wiki/Cubic_spline
    
    Args:
        - `ts`: `Float[Array, 'n']` - Time points.
        - `ys`: `Float[Array, 'n']` - Values.
    '''
    h = ts[1:] - ts[:-1]
 
    c_mat_d = jnp.ones(len(ts)).at[1:-1].set(2 * (h[1:] + h[:-1]))
    c_mat_du = jnp.zeros(len(ts)).at[1:-1].set(h[1:])
    c_mat_dl = jnp.zeros(len(ts)).at[1:-1].set(h[:-1])
    
    diff1 = (ys[1:-1] - ys[0:-2]) / (ts[1:-1] - ts[0:-2])[:, None]
    diff2 = (ys[2:] - ys[1:-1]) / (ts[2:] - ts[1:-1])[:, None]
    c_vecs = jnp.zeros((len(ts), ys.shape[1])).at[1:-1].set(3 * (diff2 - diff1))
    
    c = tridiagonal_solve(c_mat_dl, c_mat_d, c_mat_du, c_vecs)
    
    h = h[:, None]
    d = (c[1:] - c[:-1]) / (3 * h)
    b = (ys[1:] - ys[:-1]) / h - h / 3 * (2*c[:-1] + c[1:])
    a, c = ys[:-1], c[:-1]
    return (d, c, b, a)


def nan_cubic_interp(ts, ys):
    '''
    Interpolate the values of the natural cubic spline. 
    
    Args:
        - `ts`: `Float[Array, 'n']` - Time points.
        - `ys`: `Float[Array, 'n']` - Values.
    '''
    if ts.ndim == 1 and ys.ndim == 1:
        indices = jnp.arange(len(ts))[~jnp.isnan(ys)]
        sampled_ts = ts[indices]
        sampled_ys_i = ys[indices]
        
        coeffs = natural_cubic_spline_coeffs(
            sampled_ts, sampled_ys_i[:, None]
        )
        interp = dfx.CubicInterpolation(sampled_ts, coeffs) 
        return jax.vmap(interp.evaluate)(ts)[:, 0]
    
    elif ts.ndim == 1 and ys.ndim == 2:
        return jnp.stack([
            nan_cubic_interp(ts, ys_i)
            for ys_i in ys.T
        ], axis = -1)
        
    elif ts.ndim == 2 and ys.ndim == 3:
        return jnp.stack([
            nan_cubic_interp(ts[i], ys[i])
            for i in range(len(ts))
        ], axis = 0)
    
    else:
        raise ValueError
    
    
def nan_cubic_spline_smoothing(ts, ys, smooth = 0.99):
    '''
    Smooth the values of the natural cubic spline using `csaps`.
    
    Args:
        - `ts`: `Float[Array, 'n']` - Time points.
        - `ys`: `Float[Array, 'n']` - Values.
        - `smooth`: `float` - Smoothing parameter.
    '''
    if ts.ndim == 1 and ys.ndim == 1:
        indices = jnp.arange(len(ts))[~jnp.isnan(ys)]
        return jnp.asarray(csaps(ts[indices], ys[indices], ts, smooth = smooth))
    
    elif ts.ndim == 1 and ys.ndim == 2:
        return jnp.stack([
            nan_cubic_spline_smoothing(ts, ys_i, smooth)
            for ys_i in ys.T
        ], axis = -1)
        
    elif ts.ndim == 2 and ys.ndim == 3:
        return jnp.stack([
            nan_cubic_spline_smoothing(ts[i], ys[i], smooth)
            for i in range(len(ts))
        ], axis = 0)


@eqx.filter_jit
def spline_integ(ws, ts, k, use_sin = True):
    '''
    Compute the integrals used in VF-NODE.
    
    Args:
        - `ws`: `Float[Array, 'n']` - Time points.
        - `ts`: `Float[Array, 'n']` - Values.
        - `k`: `int` - Order of the integration.
        - `use_sin`: `bool` - Whether to use sine or cosine.
    '''
    func = jnp.sin if use_sin else jnp.cos
    half_pi, ti, tj = jnp.pi / 2, ts[:-1], ts[1:]
    integ_0 = jax.vmap(lambda w: (
        func(w*ti + half_pi) - func(w*tj + half_pi)
    ) / w, out_axes = -1)
    integ_1 = jax.vmap(lambda w: (
        w * (ti - tj) * func(w*tj + half_pi) + func(w*tj) - func(w*ti)
    ) / w**2, out_axes = -1)
    integ_2 = jax.vmap(lambda w: (
        - w**2 * (ti - tj)**2 * func(w*tj + half_pi) \
        - 2*w * (ti - tj) * func(w*tj) \
        + 2 * (func(w*tj + half_pi) - func(w*ti + half_pi))
    ) / w**3, out_axes = -1)
    integ_3 = jax.vmap(lambda w: (
        w**3 * (ti - tj)**3 * func(w*tj + half_pi) \
        + 3*w**2 * (ti - tj)**2 * func(w*tj) \
        + 6*w * (tj - ti) * func(w*tj + half_pi) \
        + 6 * (func(w*ti) - func(w*tj))
    ) / w**4, out_axes = -1)
    return jax.lax.switch(
        k, [integ_0, integ_1, integ_2, integ_3], ws
    )