import numpy as np
from src.linalg import thomas

def ftcs_step(u, d):
    """Advance one timestep. Boundary values stay fixed."""
    u_new = u.copy()
    # update the interior points u_new[1:-1] in one line using slicing (no loop)
    u_new[1:-1] = u[1:-1] + d*(u[2:] - 2*u[1:-1] + u[:-2])
    return u_new

def backward_euler_step(u, d):
    """One BE step. Dirichlet boundaries held at u[0], u[-1]."""
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