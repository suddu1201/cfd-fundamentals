import numpy as np
import matplotlib.pyplot as plt
from src.diffusion_1d import ftcs_step, backward_euler_step, crank_nicolson_step

def solve(step, u0, d, n_steps):
    u = u0.copy()
    for _ in range(n_steps):
        u = step(u, d)
    return u

N = 51
x = np.linspace(0, 1, N)
alpha = 1
dx = x[1] - x[0]

#Run A
u0 = np.sin(np.pi * x)
d = 0.4
dt = d * dx**2 / alpha
t_final = 0.1
n_steps = round(t_final/dt)
schemes = [ftcs_step, backward_euler_step, crank_nicolson_step]
t_actual = n_steps*dt
u_exact = np.exp(-alpha*np.pi**2*t_actual) * np.sin(np.pi*x)

for scheme in schemes:
    u_num = solve(scheme, u0, d, n_steps)
    err = np.max(np.abs(u_num - u_exact)) 
    print(f"Max Error = {err:.3e} for scheme {scheme.__name__:22s}")

# Run B
u0 = np.where((x > 0.4) & (x < 0.6), 1.0, 0.0)
d_values = [0.4, 5, 50]
t_final = 0.04

d_ref = 0.1
dt_ref = d_ref * dx**2 / alpha
u_ref = solve(ftcs_step, u0, d_ref, round(t_final / dt_ref))

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, d in zip(axes, d_values):
    dt = d * dx**2 / alpha
    n_steps = round(t_final / dt)
    ax.plot(x, u_ref, "k-", lw=1, label="reference")

    for scheme in schemes:
        if scheme is ftcs_step and d > 0.5:
            continue                      # FTCS is unstable here, skip it
        u_num = solve(scheme, u0, d, n_steps)
        err = np.max(np.abs(u_num - u_ref))
        print(f"d = {d:5}  steps = {n_steps:4d}  {scheme.__name__:22s}"
              f"  max = {u_num.max():.3f}  min = {u_num.min():+.3f}  err = {err:.3e}")
        ax.plot(x, u_num, "o-", markersize=3, label=scheme.__name__)

    ax.set_title(f"d = {d}, {n_steps} steps")
    ax.set_xlabel("x")
    ax.legend()

plt.tight_layout()
plt.savefig("results/implicit_runB.png", dpi=150)


