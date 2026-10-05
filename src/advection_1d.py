import numpy as np

def upwind_advection(u0, c, dx, dt, nsteps, periodic=True):
    """First-order upwind for u_t + c u_x = 0."""
    u = u0.copy()
    C = c * dt / dx
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
    return np.where((x >= x0) & (x <= x1), amplitude, 0.0)

def gaussian(x, x0=0.3, sigma=0.05, amplitude=1.0):
    return amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def exact_shift(u0_func, x, c, t, L):
    x_shifted = (x - c * t) % L
    return u0_func(x_shifted)
