import numpy as np
import matplotlib.pyplot as plt

def ftcs_step(u, d):
    """Advance one timestep. Boundary values stay fixed."""
    u_new = u.copy()
    # update the interior points u_new[1:-1] in one line using slicing (no loop)
    u_new[1:-1] = u[1:-1] + d*(u[2:] - 2*u[1:-1] + u[:-2])
    return u_new

def solve(u0, d, n_steps):
    u = u0.copy()
    for _ in range(n_steps):
        u = ftcs_step(u, d)
    return u

N = 51
x = np.linspace(0, 1, N)
alpha = 1.0
dx = x[1] - x[0]

# Run A: compare numerical and exact solutions at several times
u0 = np.sin(np.pi * x)
d = 0.4
dt = d * dx**2 / alpha

times = [0.0, 0.02, 0.05, 0.1]
colors = ["C0", "C1", "C2", "C3"]   # matplotlib's default color cycle

plt.figure()
for t_target, c in zip(times, colors):
    n_steps = round(t_target / dt)
    t_actual = n_steps * dt

    u_num = solve(u0, d, n_steps)
    u_exact = np.exp(-alpha * np.pi**2 * t_actual) * np.sin(np.pi * x)
    err = np.max(np.abs(u_num - u_exact))
    print(f"t = {t_actual:.4f}   steps = {n_steps:4d}   max error = {err:.3e}")

    plt.plot(x, u_exact, "-", color=c, label=f"exact, t = {t_actual:.3f}")
    plt.plot(x, u_num, "o", color=c, markersize=3)

plt.xlabel("x")
plt.ylabel("u")
plt.title(f"FTCS vs exact, d = {d}, N = {N}  (markers = numerical)")
plt.legend()
plt.savefig("results/heat_runA.png", dpi=150)


# TODO: Run B, the stability test
# Initial condition: a hat or step (e.g. u0 = 1 for 0.4 < x < 0.6, else 0)
# Run d = 0.49 and d = 0.51 for the same physical time. Plot side by side.
u0 = np.where((x > 0.4) & (x < 0.6), 1.0, 0.0)
d_values = [0.49, 0.51]
steps = [0, 50, 200, 500]
colors = ["C0", "C1", "C2", "C3"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, d in zip(axes, d_values):
    for step, c in zip(steps, colors):
        u_num = solve(u0, d, step)
        print(f"max = {np.max(np.abs(u_num))}")
        ax.plot(x, u_num, "o-", color=c, markersize=3, label=f"step {step}")
    
    ax.set_title(f"FTCS, d = {d}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()

plt.tight_layout()
plt.savefig("results/heat_runB.png", dpi=150)