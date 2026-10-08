import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import upwind_advection, square_wave, exact_shift

L, N = 1.0, 200
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
c = 1.0
t_final = 0.3

Cs = [0.95, 1.05]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
u0 = square_wave(x)

for ax, C in zip(axes, Cs):
    dt = C * dx / c
    nsteps = round(t_final / dt)
    u_num = upwind_advection(u0, c, dx, dt, nsteps, check_cfl=False)
    u_exact = exact_shift(square_wave, x, c, nsteps * dt, L)

    ax.plot(x, u0, "k--", alpha=0.3, label="initial")
    ax.plot(x, u_exact, "g-", lw=2, label="exact")
    ax.plot(x, u_num, "o-", markersize=2, label=f"upwind, C={C}")
    ax.set_title(f"Upwind, C = {C}, {nsteps} steps")
    ax.legend()
    print(f"C = {C}   steps = {nsteps}   max|u| = {np.max(np.abs(u_num)):.3e}")

plt.tight_layout()
plt.savefig("results/cfl_sweep.png", dpi=120)
