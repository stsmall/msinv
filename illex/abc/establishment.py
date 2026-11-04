"""Exact establishment probability u(theta): one copy reaching ESTABLISHED copies under binomial Wright-Fisher.

This is the quantity the importance weight ``w_est`` needs, and it is exactly what ``trajectory._establish`` simulates:
each generation p = k / 2N, p' = clip(p + delta(p), 0, 1), k' ~ Binomial(2N, p'), with the model's own delta
(genic s p (1 - p) for N / D1 / D2, s = 0 for N; for B, s p (1 - p) (p* - p) / p*), and the attempt succeeds the first
time k >= ESTABLISHED (overshoot included, which Kimura's diffusion ignores: it is 0.3-2.5% high). u is the first-passage
probability from k = 1 of the transient chain k = 1 .. ESTABLISHED - 1 (absorbing at 0 and at >= ESTABLISHED):
(I - Q) u = r, solved directly (``direct_u``).

Over the prior's range (t_inv 1e5 to 1.5e6) 2N runs 1.1e6 to 9.8e6; u depends on N only at the 1e-4 level, but it is kept exactly: a cached
grid (``establishment_grid.npz``: log-spaced s 1e-6 .. 1e-2, plus s = 0, x 5 values of 2N [x 4 values of 1 / p* for B])
is interpolated (cubic spline in log s, linear in 1 / 2N and 1 / p*, both exactly smooth). The grid is built by
``build_grid`` and regenerated if absent. Outside the grid's 2N range (tests with a small constant N) ``direct_u``
is used per unique (2N, s, p*).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import binom

ESTABLISHED = 500
S_LO, S_HI, N_S = 1e-6, 1e-2, 200
TWO_N_NODES = (1.0e6, 2.2e6, 5.2e6, 1.0e7, 1.4e7)   # to 1.4e7 = above 2 N0, so every prior t_inv is on the grid
INV_PSTAR_NODES = (1.0, 1.0 / 0.95 * 1.0, 1 / 0.6, 1 / 0.3)   # 1 / p* over the prior (0.3, 0.95); sorted below
GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "establishment_grid.npz")


def direct_u(two_n, s, p_star=None, established=ESTABLISHED):
    """Exact P(one copy reaches >= ``established`` copies); p_star None = genic selection, else the B form."""
    two_n = int(two_n)
    k = np.arange(1, established, dtype=float)
    p = k / two_n
    d = s * p * (1.0 - p)
    if p_star is not None:
        d = d * (p_star - p) / p_star
    pn = np.clip(p + d, 0.0, 1.0)
    j = np.arange(0, established)
    q = binom.pmf(j[None, :], two_n, pn[:, None])[:, 1:]             # transient targets 1 .. established - 1
    r = binom.sf(established - 1, two_n, pn)
    return float(np.linalg.solve(np.eye(established - 1) - q, r)[0])


def _s_nodes():
    return np.logspace(np.log10(S_LO), np.log10(S_HI), N_S)


def build_grid(path=GRID_PATH):
    ip = np.array(sorted(INV_PSTAR_NODES))
    s = _s_nodes()
    sg = np.concatenate([[0.0], s])
    genic = np.array([[direct_u(n, x) for n in TWO_N_NODES] for x in sg])                   # (N_S+1, nN)
    bal = np.array([[[direct_u(n, x, 1.0 / q) for n in TWO_N_NODES] for q in ip] for x in sg])  # (N_S+1, nP, nN)
    np.savez(path, s=sg, two_n=np.array(TWO_N_NODES), inv_pstar=ip, genic=genic, balancing=bal)
    return path


_GRID = {}


def _grid():
    if "g" not in _GRID:
        if not os.path.exists(GRID_PATH):
            build_grid()
        z = np.load(GRID_PATH)
        s = z["s"]
        _GRID["g"] = dict(
            s=s, inv_n=1.0 / z["two_n"], inv_p=z["inv_pstar"], u0=z["genic"][0], b0=z["balancing"][0],
            gs=[CubicSpline(np.log(s[1:]), z["genic"][1:, i]) for i in range(z["genic"].shape[1])],
            bs=[[CubicSpline(np.log(s[1:]), z["balancing"][1:, j, i]) for i in range(z["genic"].shape[1])]
                for j in range(len(z["inv_pstar"]))],
            genic=z["genic"], balancing=z["balancing"])
    return _GRID["g"]


def _interp_s(spl, u_zero, u_first, s):
    """Cubic spline in log s on [S_LO, S_HI]; linear in s between s = 0 and S_LO below it."""
    s = np.asarray(s, float)
    lo = s < S_LO
    out = spl(np.log(np.clip(s, S_LO, S_HI)))
    return np.where(lo, u_zero + (u_first - u_zero) * s / S_LO, out)


def u_grid(s, two_n, p_star=None):
    """Grid-interpolated u for arrays s, two_n (and p_star for B); requires two_n inside the grid's range."""
    g = _grid()
    s = np.atleast_1d(np.asarray(s, float))
    inv_n = 1.0 / np.broadcast_to(np.asarray(two_n, float), s.shape)
    nodes = g["inv_n"][::-1]                                         # ascending 1 / 2N
    if p_star is None:
        cols = np.stack([_interp_s(g["gs"][i], g["u0"][i], g["genic"][1, i], s) for i in range(len(nodes))[::-1]], -1)
        # cols[..., m] corresponds to node index (n - 1 - m) of the original order, i.e. ascending 1 / 2N
        return _lin1(inv_n, nodes, cols)
    ip = 1.0 / np.broadcast_to(np.asarray(p_star, float), s.shape)
    out = np.empty(s.shape)
    colsp = []
    for j in range(len(g["inv_p"])):
        cj = np.stack([_interp_s(g["bs"][j][i], g["b0"][j, i], g["balancing"][1, j, i], s)
                       for i in range(len(nodes))[::-1]], -1)
        colsp.append(_lin1(inv_n, nodes, cj))
    colsp = np.stack(colsp, -1)                                      # (..., nP)
    ipn = g["inv_p"]
    i = np.clip(np.searchsorted(ipn, ip) - 1, 0, len(ipn) - 2)
    w = (ip - ipn[i]) / (ipn[i + 1] - ipn[i])
    idx = np.arange(s.size)
    cs = colsp.reshape(-1, len(ipn))
    out = cs[idx, i.ravel()] * (1 - w.ravel()) + cs[idx, i.ravel() + 1] * w.ravel()
    return out.reshape(s.shape)


def _lin1(x, nodes, cols):
    """Linear interpolation of cols[..., m] (m over ascending nodes) at x."""
    i = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
    w = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    a = np.take_along_axis(cols, i[..., None], -1)[..., 0]
    b = np.take_along_axis(cols, (i + 1)[..., None], -1)[..., 0]
    return a * (1 - w) + b * w


def establishment_prob(model, s, t_inv, n_e_fn=None, p_star=None):
    """u(theta) per draw (vectorised). ``s`` is ignored for model N; ``p_star`` is used only for model B."""
    from illex.theory import N_growth
    n_e_fn = N_growth if n_e_fn is None else n_e_fn
    t_inv = np.atleast_1d(np.asarray(t_inv, float))
    two_n = np.maximum(np.floor(2.0 * np.asarray(n_e_fn(t_inv), float)), 2.0)
    sv = np.zeros(t_inv.shape) if model == "N" else np.nan_to_num(np.broadcast_to(np.asarray(s, float), t_inv.shape))
    ps = np.asarray(p_star, float) if model == "B" else None
    if ps is not None:
        ps = np.broadcast_to(ps, t_inv.shape)
    inside = (two_n >= TWO_N_NODES[0]) & (two_n <= TWO_N_NODES[-1]) & (sv <= S_HI)
    out = np.empty(t_inv.shape)
    if inside.any():
        out[inside] = u_grid(sv[inside], two_n[inside], None if ps is None else ps[inside])
    off = np.where(~inside)[0]
    keys = {(int(two_n[i]), float(sv[i]), None if ps is None else float(ps[i])) for i in off}
    if len(keys) > 60 and ps is None:      # many distinct s at an off-grid N (tests): local spline in log s, exact nodes
        for n in {k[0] for k in keys}:
            sel = off[two_n[off] == n]
            spl, u0, u1 = _local_spline(n)
            out[sel] = _interp_s(spl, u0, u1, sv[sel])
    else:
        for i in off:
            out[i] = _direct_cached(int(two_n[i]), float(sv[i]), None if ps is None else float(ps[i]))
    return np.clip(out, 0.0, 1.0)


_DC = {}


def _direct_cached(two_n, s, p_star):
    key = (two_n, s, p_star)
    if key not in _DC:
        _DC[key] = direct_u(two_n, s, p_star)
    return _DC[key]


_LS = {}


def _local_spline(two_n, n_nodes=120):
    """(spline in log s, u(0), u(S_LO)) of the genic exact u at one off-grid 2N (cached)."""
    if two_n not in _LS:
        sn = np.logspace(np.log10(S_LO), np.log10(S_HI), n_nodes)
        _LS[two_n] = (CubicSpline(np.log(sn), [direct_u(two_n, x) for x in sn]), direct_u(two_n, 0.0),
                      direct_u(two_n, S_LO))
    return _LS[two_n]
