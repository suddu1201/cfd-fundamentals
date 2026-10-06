# scripts/central_blowup.py
import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import central_advection, gaussian

L, N, c = 1.0, 200, 1.0
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
u0 = gaussian(x)

fig, ax = plt.subplots()
for C in [0.1, 0.5, 0.9]:
    dt = C * dx / c
    peak = []
    u = u0.copy()
    for step in range(200):
        u = central_advection(u, c, dx, dt, 1)
        peak.append(np.max(np.abs(u)))
    ax.semilogy(peak, label=f"C={C}")

ax.set_xlabel("step")
ax.set_ylabel("max|u|")
ax.set_title("FTCS: growth at every Courant number")
ax.legend()
plt.savefig("central_blowup.png", dpi=120)