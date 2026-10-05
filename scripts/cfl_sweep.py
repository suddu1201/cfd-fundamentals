import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import upwind_advection, square_wave, exact_shift

L, N = 1.0, 200
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
c = 1.0
t_final = 0.3

Cs = [0.1, 0.25, 0.5, 0.75, 1.0]

fig, ax = plt.subplots()
u0 = square_wave(x)
ax.plot(x, u0, "k--", alpha=0.3, label="initial")

for C in Cs:
    dt = C * dx / c
    nsteps = round(t_final / dt)
    u_num = upwind_advection(u0, c, dx, dt, nsteps)
    ax.plot(x, u_num, label=f"C={C}")

u_exact = exact_shift(square_wave, x, c, nsteps * dt, L)
ax.plot(x, u_exact, "g-", lw=2, label="exact (last C's time)")
ax.legend()
ax.set_title("Upwind: effect of Courant number on numerical diffusion")
plt.savefig("cfl_sweep.png", dpi=120)
