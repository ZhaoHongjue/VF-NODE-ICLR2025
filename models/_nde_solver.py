'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from functools import partial
from dataclasses import dataclass

import optimistix as optx, diffrax as dfx


@dataclass
class SolverKwargs:
    '''
    Solver arguments for ODE solvers based on Diffrax.
    
    Args:
        - `solver_type`: `str` - Solver type. Default: `Dopri5`
        - `adjoint_mode`: `int` - Adjoint mode. Default: 0 (RecursiveCheckpointAdjoint)
            - 0: RecursiveCheckpointAdjoint
            - 1: BacksolveAdjoint (Seminorm) "Hey, that’s not an ODE": Faster ODE Adjoints via Seminorms
            - 2: BacksolveAdjoint (RMSNorm)
        - `rtol`: `float` - Relative tolerance. Default: 1e-3
        - `atol`: `float` - Absolute tolerance. Default: 1e-6
    '''
    solver_type: str = 'Dopri5'
    adjoint_mode: int = 0
    rtol: float = 1e-3
    atol: float = 1e-6


def generate_nde_solver_fn(solver_kws: SolverKwargs):
    '''
    Generate a solver function for Neural ODEs based on Diffrax.
    
    Args:
        - `solver_kws`: `SolverKwargs` - Solver arguments
    '''
    # Initialize the solver
    solver = eval(f'dfx.{solver_kws.solver_type}')()
    if solver_kws.adjoint_mode == 0:
        adjoint = dfx.RecursiveCheckpointAdjoint()
    else:
        if solver_kws.adjoint_mode == 1:
            norm = dfx.adjoint_rms_seminorm 
        else:
            norm = optx.rms_norm
            
        if solver_kws.solver_type == 'Euler':
            adjoint_step_controller = dfx.ConstantStepSize()
        else:
            adjoint_step_controller = dfx.PIDController(
                solver_kws.rtol, solver_kws.atol, 
                norm = norm 
            )
        adjoint = dfx.BacksolveAdjoint(
            stepsize_controller = adjoint_step_controller,
            solver = solver
        )
        
    if solver_kws.solver_type == 'Euler':
        stepsize_controller = dfx.ConstantStepSize()
        return partial(
            dfx.diffeqsolve, solver = solver, adjoint = adjoint, 
            stepsize_controller = stepsize_controller,
            max_steps = 65536, dt0 = 0.01
        )
        
    else:
        stepsize_controller = dfx.PIDController(
            solver_kws.rtol, solver_kws.atol
        )
        return partial(
            dfx.diffeqsolve, solver = solver, adjoint = adjoint, 
            stepsize_controller = stepsize_controller,
            max_steps = 65536, dt0 = None
        )