""" 1D diffusion solvers: FTCS, BTCS, Crank-Nikolson """

import numpy as np
from src.linalg import thomas

def ftcs_step(u, d):
    """Advance one step with explicit FTCS (forward time, central space).

    u : solution at the current step; u[0] and u[-1] are Dirichlet
        boundary values and are held fixed
    d : diffusion number alpha*dt/dx**2
    Returns the solution at the next step.
    Stable only for d <= 0.5; first order in time, second order in space.
    """
    u_new = u.copy()
    # update the interior points u_new[1:-1] in one line using slicing (no loop)
    u_new[1:-1] = u[1:-1] + d*(u[2:] - 2*u[1:-1] + u[:-2])
    return u_new

def backward_euler_step(u, d):
    """Advance one step with implicit backward Euler.

    Solves the tridiagonal system for the new interior values with thomas().
    u : solution at the current step; u[0] and u[-1] are Dirichlet
        boundary values and are held fixed
    d : diffusion number alpha*dt/dx**2
    Returns the solution at the next step.
    Unconditionally stable and non-oscillatory for any d, but only first
    order in time.
    """
    n = len(u) - 2                      # number of interior unknowns
    a = np.full(n, -d)
    b = np.full(n, 1+2*d)
    c = np.full(n, -d)
    r = u[1:-1].copy()
    r[0] += d*u[0]
    r[-1] += d*u[-1]

    u_new = u.copy()
    u_new[1:-1] = thomas(a, b, c, r)
    return u_new

def crank_nicolson_step(u, d):
    """Advance one step with implicit Crank-Nicolson (trapezoidal in time).

    Averages the old and new time levels; solves a tridiagonal system with
    thomas().
    u : solution at the current step; u[0] and u[-1] are Dirichlet
        boundary values and are held fixed
    d : diffusion number alpha*dt/dx**2
    Returns the solution at the next step.
    Unconditionally stable and second order in time, but for large d it
    barely damps grid-scale modes: sharp data oscillates and can go
    negative (positivity only guaranteed for d <= 1).
    """
    n = len(u) - 2                      # number of interior unknowns
    a = np.full(n, -d/2)
    b = np.full(n, 1+d)
    c = np.full(n, -d/2)
    
    # r uses the old-level combination (slicing, like ftcs_step)
    r = (d/2)*u[:-2] + (1-d)*u[1:-1] + (d/2)*u[2:]
    r[0] += (d/2)*u[0]
    r[-1] += (d/2)*u[-1]

    u_new = u.copy()
    u_new[1:-1] = thomas(a, b, c, r)
    return u_new
    

def solve(step, u0, d, n_steps):
    """Advance an initial condition n_steps times with a given step function.

    step    : function step(u, d) -> u_next, e.g. ftcs_step
    u0      : initial condition (not modified)
    d       : diffusion number alpha*dt/dx**2
    n_steps : number of time steps; final time is n_steps*dt
    Returns the solution after n_steps.
    """
    u = u0.copy()
    for _ in range(n_steps):
        u = step(u, d)
    return u