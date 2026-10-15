import numpy as np
from src.linalg import thomas

n = 8
rng = np.random.default_rng(0)
a, c, r = rng.random(n), rng.random(n), rng.random(n)
b = 2 + rng.random(n)                       # diagonally dominant
A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
print(np.max(np.abs(thomas(a, b, c, r) - np.linalg.solve(A, r))))   # expect ~1e-16