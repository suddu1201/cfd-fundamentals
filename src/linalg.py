"""Linear-algebra helpers: tridiagonal solver."""

import numpy as np

def thomas(a, b, c, r):
    """Solve a tridiagonal system A x = r with the Thomas algorithm.

    Gaussian elimination specialised to tridiagonal matrices: O(n) work and
    memory instead of O(n^3) for a dense solve. No pivoting, so A should be
    diagonally dominant (|b_i| >= |a_i| + |c_i|), which the implicit
    diffusion matrices always are.

    a : sub-diagonal, length n (a[0] unused)
    b : main diagonal, length n
    c : super-diagonal, length n (c[-1] unused)
    r : right-hand side, length n
    Returns x, length n.
    """
    
    n = len(b)
    cp = np.zeros(n)
    rp = np.zeros(n)
    x = np.zeros(n)
    
    # Forward sweep to eliminate the sub-diagonal
    cp[0] = c[0]/b[0]
    rp[0] = r[0]/b[0]
    for i in range(1, n, 1):
        m = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/m
        rp[i] = (r[i] - a[i]*rp[i-1])/m
    
    # Back substitution
    x[n-1] = rp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = rp[i] - cp[i]*x[i+1]

    return x