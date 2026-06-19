"""run20 min-curvature racing line -- rolled our own (scipy/numpy, MIT-clean; NOT the LGPL
TUM package). The repo is MIT, so we implement the classic first-pass method directly.

Spatial half of the racing line: given a closed centerline and a lateral corridor (half-width
per point, minus a car/safety margin), find the path p_i = c_i + alpha_i * n_i that minimizes
total discrete curvature -- the sum of squared path second-differences ||p_{i-1} - 2p_i +
p_{i+1}||^2 -- subject to the corridor box bounds lo_i <= alpha_i <= hi_i. Because p is affine
in alpha, the curvature objective is quadratic in alpha and the box-constrained minimum is a
bounded linear least-squares, solved exactly by scipy.optimize.lsq_linear. The speed profile
(envs/speed_profile.py) is the other (longitudinal) half.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear


def left_normals(xy):
    """Unit LEFT-normals from central-difference tangents (closed loop). (n,2)->(n,2)."""
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)        # c_{i+1} - c_{i-1}
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)               # (-ty, tx) = left


def resample_closed(centerline, spacing):
    """Resample a closed (x,y,z) path to uniform arc-length `spacing` (m). The raw 2nd-difference
    is only a clean curvature proxy at uniform spacing; the saved centerline varies 4-7 m/pt."""
    cl = np.asarray(centerline, float)
    seg = np.linalg.norm(np.roll(cl, -1, axis=0) - cl, axis=1)  # closes the loop
    s = np.concatenate([[0.0], np.cumsum(seg)])                # len n+1, s[-1]=perimeter
    total = s[-1]
    m = max(3, int(round(total / spacing)))
    targ = np.linspace(0.0, total, m, endpoint=False)
    loop = np.vstack([cl, cl[0]])                              # wrap for interp
    return np.column_stack([np.interp(targ, s, loop[:, k]) for k in range(3)])


def min_curvature_line(centerline, half_width, margin, ridge=0.01):
    """centerline: (n,3) UNIFORMLY spaced. half_width: scalar or (n,) (m). margin: corridor inset (m).
    ridge: weak Tikhonov pull of alpha->0 where curvature is insensitive to offset (kills the
    closed-loop translational nullspace drift; in corners the curvature term dominates so the apex
    still clips). Returns (raceline (n,3), alpha (n,) +left, normals (n,2), bound (n,))."""
    cl = np.asarray(centerline, float)
    xy = cl[:, :2]
    n = len(xy)
    nrm = left_normals(xy)
    half = np.full(n, float(half_width)) if np.isscalar(half_width) else np.asarray(half_width, float)
    bound = np.maximum(half - margin, 0.0)                     # |alpha| <= bound

    im1 = np.roll(np.arange(n), 1)
    ip1 = np.roll(np.arange(n), -1)
    # minimize || (n_{i-1} a_{i-1} - 2 n_i a_i + n_{i+1} a_{i+1}) - b_i ||^2 + ridge*||a||^2,
    # b_i = -(c_{i-1} - 2 c_i + c_{i+1})  (so the full path 2nd-diff -> 0).
    b = -(xy[im1] - 2.0 * xy + xy[ip1])                        # (n,2)
    rows, cols, vals = [], [], []
    for comp in (0, 1):                                        # x then y component
        for i in range(n):
            r = 2 * i + comp
            rows += [r, r, r]
            cols += [im1[i], i, ip1[i]]
            vals += [nrm[im1[i], comp], -2.0 * nrm[i, comp], nrm[ip1[i], comp]]
    # ridge rows: sqrt(ridge) * I  -> appended residual sqrt(ridge)*alpha_i
    sr = float(np.sqrt(ridge))
    for i in range(n):
        rows.append(2 * n + i); cols.append(i); vals.append(sr)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(3 * n, n))
    d = np.concatenate([b.reshape(-1), np.zeros(n)])           # curvature target + ridge target 0

    res = lsq_linear(A, d, bounds=(-bound, bound), max_iter=1000, lsq_solver="lsmr")
    alpha = res.x
    rl_xy = xy + alpha[:, None] * nrm
    raceline = np.column_stack([rl_xy, cl[:, 2]])              # carry centerline elevation
    return raceline, alpha, nrm, bound
