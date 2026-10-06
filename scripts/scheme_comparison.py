import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import (upwind_advection, central_advection,
                              lax_wendroff_advection, square_wave, gaussian,
                              exact_shift)

L, N, c, C = 1.0, 200, 1.0, 0.5
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
dt = C * dx / c
nsteps = round(0.3 / dt)

schemes = {
    "upwind": upwind_advection,
    "central (FTCS)": central_advection,
    "Lax-Wendroff": lax_wendroff_advection,
}
initial_conditions = [("square wave", square_wave), ("gaussian", gaussian)]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for row, (ic_name, ic) in enumerate(initial_conditions):
    u0 = ic(x)
    u_exact = exact_shift(ic, x, c, nsteps * dt, L)
    for col, (name, scheme) in enumerate(schemes.items()):
        u = scheme(u0, c, dx, dt, nsteps)
        ax = axes[row, col]
        ax.plot(x, u_exact, "g-", label="exact")
        ax.plot(x, u, "r.-", label=name)
        ax.set_title(f"{name}  (min={u.min():.2f}, max={u.max():.2f})")
        ax.set_ylim(-0.5, 1.5)
        ax.legend()
    axes[row, 0].set_ylabel(ic_name)

fig.suptitle(f"Scheme comparison at C={C}, t={nsteps * dt:.2f}")
fig.tight_layout()
plt.savefig("scheme_comparison.png", dpi=120)