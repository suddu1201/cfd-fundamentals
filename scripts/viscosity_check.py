import numpy as np
import matplotlib.pyplot as plt
from src.advection_1d import upwind_advection, gaussian

L, N, c = 1.0, 200, 1.0
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]

def variance(x, u):
    m = np.sum(x * u) / np.sum(u)
    return np.sum((x - m) ** 2 * u) / np.sum(u)

Cs = [0.25, 0.5, 0.75, 1.0]
D_meas, D_theory = [], []

for C in Cs:
    dt = C * dx / c
    total_steps = round(0.3 / dt)          # same physical time for every C
    chunk = max(1, total_steps // 15)      # ~15 measurement points
    u = gaussian(x)
    ts, vs = [0.0], [variance(x, u)]
    for _ in range(15):
        u = upwind_advection(u, c, dx, dt, chunk)
        ts.append(ts[-1] + chunk * dt)
        vs.append(variance(x, u))
    D_meas.append(np.polyfit(ts, vs, 1)[0] / 2)
    D_theory.append((c * dx / 2) * (1 - C))
    print(f"C={C}: measured D={D_meas[-1]:.6f}, theory D={D_theory[-1]:.6f}")

plt.plot(Cs, D_theory, "g-", label="theory: (c·dx/2)(1−C)")
plt.plot(Cs, D_meas, "ro", label="measured")
plt.xlabel("C"); plt.ylabel("D"); plt.legend()
plt.savefig("viscosity_check.png", dpi=120)
