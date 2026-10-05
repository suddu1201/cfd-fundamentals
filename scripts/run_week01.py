import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import upwind_advection, square_wave, gaussian, exact_shift

L, N = 1.0, 200
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
c = 1.0
C_courant = 0.5
dt = C_courant * dx / c
t_final = 0.3
nsteps = int(t_final / dt)

for name, ic_func in [("square", square_wave), ("gaussian", gaussian)]:
    u0 = ic_func(x)
    u_num = upwind_advection(u0, c, dx, dt, nsteps)
    u_exact = exact_shift(ic_func, x, c, nsteps * dt, L)

    plt.figure()
    plt.plot(x, u0, "k--", alpha=0.4, label="initial")
    plt.plot(x, u_exact, "g-", label="exact")
    plt.plot(x, u_num, "r.-", label=f"upwind, C={C_courant}")
    plt.title(f"{name} advection, t={nsteps*dt:.2f}")
    plt.legend()
    plt.savefig(f"advection_{name}.png", dpi=120)
    print(f"saved advection_{name}.png")
