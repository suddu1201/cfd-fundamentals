"""1D linear advection u_t + c*u_x = 0 on a periodic domain: upwind,
central (FTCS) and Lax-Wendroff schemes, plus test profiles."""

import numpy as np

def upwind_advection(u0, c, dx, dt, nsteps, periodic=True, check_cfl=True):
    """Solve linear advection with first-order upwind (assumes c > 0).

    u0        : initial condition on a periodic grid (not modified)
    c         : advection speed
    dx, dt    : grid spacing and time step
    nsteps    : number of time steps
    check_cfl : if True, assert C = c*dt/dx <= 1; set False for
                deliberate stability experiments
    Returns the solution after nsteps.
    Stable for C <= 1. Numerical diffusion D = (c*dx/2)*(1 - C);
    exact at C = 1.
    """
    u = u0.copy()
    C = c * dt / dx
    if check_cfl:
        assert C <= 1.0, f"CFL violated: C={C:.3f}"
    for _ in range(nsteps):
        if periodic:
            u_upwind = np.roll(u, 1)
        else:
            u_upwind = np.empty_like(u)
            u_upwind[1:] = u[:-1]
            u_upwind[0] = u[0]
        u = u - C * (u - u_upwind)
    return u

def square_wave(x, x0=0.2, x1=0.4, amplitude=1.0):
    """Square-wave test profile: 1 inside the pulse, 0 elsewhere."""
    return np.where((x >= x0) & (x <= x1), amplitude, 0.0)

def gaussian(x, x0=0.3, sigma=0.05, amplitude=1.0):
    """Gaussian test profile """
    return amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def exact_shift(u0_func, x, c, t, L):
    """Exact solution of linear advection: the initial profile shifted by c*t.

    func : initial-condition function, e.g. square_wave
    x    : grid points
    c, t : advection speed and time
    L    : domain length (periodic wrap-around)
    Returns func evaluated at (x - c*t) mod L.
    """
    x_shifted = (x - c * t) % L
    return u0_func(x_shifted)

def central_advection(u0, c, dx, dt, nsteps):
    """Solve linear advection with FTCS (forward time, central space).

    Arguments as in upwind_advection. Returns the solution after nsteps.
    Unconditionally unstable for C > 0 (|G|^2 = 1 + C^2 sin^2(theta));
    kept for demonstration only.
    """
    u = u0.copy()
    C = c * dt / dx
    for _ in range(nsteps):
        u = u - (C / 2) * (np.roll(u, -1) - np.roll(u, 1))
    return u

def lax_wendroff_advection(u0, c, dx, dt, nsteps, check_cfl=True):
    """Solve linear advection u_t + c*u_x = 0 with the Lax-Wendroff scheme.

    Periodic boundaries (via np.roll).
    u0     : initial condition on a periodic grid (not modified)
    c      : advection speed
    dx, dt : grid spacing and time step
    nsteps : number of time steps
    Returns the solution after nsteps.
    check_cfl : if True, assert C <= 1; set False for deliberate stability experiments

    Central difference plus a (C^2/2) second-difference correction, which
    cancels FTCS's anti-diffusion. Second order in space and time, stable
    for C <= 1. Leading error is dispersive: over/undershoots (min < 0,
    max > 1) appear behind discontinuities.
    """
    u = u0.copy()
    C = c * dt / dx
    if check_cfl:
        assert C <= 1.0, f"CFL violated: C={C:.3f}"
    for _ in range(nsteps):
        up, um = np.roll(u, -1), np.roll(u, 1)
        u = u - (C / 2) * (up - um) + (C**2 / 2) * (up - 2 * u + um)
    return u
