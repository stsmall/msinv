"""Forward trajectory sampler for the inversion frequency, four selection models.

Models (``params["model"]``), all on a diploid Ne given by ``n_e_fn(t)`` (backward time, generations):

  N   neutral
  D1  genic selection s p (1-p) from t_sel to the present
  D2  genic selection s from t_sel to t_stop, neutral after
  B   overdominance s_het p (1-p) (p* - p) / p*  (the form of illex/balancing.py:_time_to_reach),
      from t_sel to the present (the parameter ``s`` is s_het)

Soft starts: frequency p0 at t_inv, neutral drift until t_sel, then selection. Hard starts (``hard`` True,
t_sel = t_inv): one copy, exact binomial Wright-Fisher until ESTABLISHED copies (attempts redrawn on loss, up to
``max_attempts``, otherwise counted in ``Counts.est_failed``), then the Gaussian diffusion in steps of ``dt``.
Fixation (p >= 1) and loss (p <= 0) are rejections, so no returned frequency is clipped.

The diffusion is vectorised over draws. A hard-start path records only its first copy, the end of
establishment and the grid after that (the establishment phase is not stored generation by generation).
Establishment uses Ne at t_inv, held fixed over the (short) phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from illex.abc.constants import GAMMA_MAX, P_NOW, P_TOL, PRIOR
from illex.theory import N_growth

ESTABLISHED = 500
MODELS = ("N", "D1", "D2", "B")
KEYS = ("t_inv", "hard", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff")


@dataclass
class Path:
    params: dict
    times: np.ndarray        # backward, increasing from 0
    freqs: np.ndarray
    p_now: float


@dataclass
class Counts:
    drawn: int = 0
    lost: int = 0
    fixed: int = 0
    est_failed: int = 0
    accepted: int = 0


def _logu(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def draw_prior(model: str, flux: bool, rng, n: int) -> dict:
    """Draw n parameter sets. Unused keys are NaN. Also returns ``model`` and ``flux`` as scalars."""
    if model not in MODELS:
        raise ValueError(model)
    nan = np.full(n, np.nan)
    t_inv = rng.uniform(*PRIOR["t_inv"], n)
    hard = rng.random(n) < PRIOR["p_hard"]
    p0 = np.where(hard, np.nan, _logu(rng, *PRIOR["p0"], n))
    out = dict(t_inv=t_inv, hard=hard, p0=p0, t_sel=nan.copy(), s=nan.copy(), t_stop=nan.copy(),
               p_star=nan.copy(), gamma_eff=nan.copy())
    if model != "N":
        out["t_sel"] = np.where(hard, t_inv, rng.uniform(0.0, 1.0, n) * t_inv)
        out["s"] = _logu(rng, *PRIOR["s"], n)
    if model == "D2":
        out["t_stop"] = rng.uniform(0.0, 1.0, n) * out["t_sel"]
    if model == "B":
        out["p_star"] = rng.uniform(*PRIOR["p_star"], n)
    if flux:
        out["gamma_eff"] = _logu(rng, 1e-9, GAMMA_MAX, n)
    out["model"] = model
    out["flux"] = flux
    return out


def _delta(model, p, s, p_star):
    """Per-generation change from selection (before multiplying by the on/off mask)."""
    d = s * p * (1.0 - p)
    if model == "B":
        d = d * (p_star - p) / p_star
    return d


def _establish(model, prm, idx, rng, n_e_fn, max_attempts, block=64, flat_max=2_000_000):
    """Exact WF from one copy for the hard draws ``idx``, all draws and attempt blocks at once.

    Returns (g, k, two_n): generations elapsed and copies at the end of establishment, g = -1 where every
    attempt failed (or establishment outlasted t_inv).
    """
    m = idx.size
    t_inv = prm["t_inv"][idx]
    two_n = np.maximum(2.0 * np.asarray(n_e_fn(t_inv), float), 2.0).astype(np.int64)
    s = np.nan_to_num(prm["s"][idx]) if model != "N" else np.zeros(m)
    t_sel = prm["t_sel"][idx] if model != "N" else np.full(m, -1.0)
    t_stop = prm["t_stop"][idx] if model == "D2" else np.full(m, -1.0)
    p_star = prm["p_star"][idx] if model == "B" else np.ones(m)
    g_out = np.full(m, -1, np.int64)
    k_out = np.zeros(m, np.int64)
    used = np.zeros(m, np.int64)
    pending = np.arange(m)
    while pending.size:
        nb = int(min(block, (max_attempts - used[pending]).max())) if max_attempts > 0 else 0
        if nb <= 0:
            break
        for lo in range(0, pending.size, max(1, flat_max // nb)):
            ch = pending[lo:lo + max(1, flat_max // nb)]
            take = np.minimum(nb, max_attempts - used[ch])      # attempts this block, per draw
            ids = np.repeat(ch, take)
            used[ch] += take
            k = np.ones(ids.size, np.int64)
            g = np.zeros(ids.size, np.int64)
            alive = np.arange(ids.size)
            ok = np.zeros(ids.size, bool)
            while alive.size:
                d = ids[alive]
                kk = k[alive]
                p = kk / two_n[d]
                t = t_inv[d] - g[alive]
                on = (t <= t_sel[d]) & (t > t_stop[d]) & (s[d] != 0)
                p = np.clip(p + np.where(on, _delta(model, p, s[d], p_star[d]), 0.0), 0.0, 1.0)
                kk = rng.binomial(two_n[d], p)
                k[alive] = kk
                g[alive] += 1
                done = kk >= ESTABLISHED
                ok[alive[done]] = True
                alive = alive[(kk > 0) & ~done & (g[alive] < t_inv[d])]
            sidx = np.where(ok)[0]
            if sidx.size:
                dd, first = np.unique(ids[sidx], return_index=True)
                sel = sidx[first]
                g_out[dd] = g[sel]
                k_out[dd] = k[sel]
        pending = pending[(g_out[pending] < 0) & (used[pending] < max_attempts)]
    return g_out, k_out, two_n


def simulate_paths(params: dict, rng, dt=10.0, rec_every=1000.0, max_attempts=2_000_000,
                   n_e_fn=N_growth, apply_filter=True):
    """Simulate one path per parameter set. Returns (accepted paths, Counts)."""
    model = params["model"]
    n = len(params["t_inv"])
    counts = Counts(drawn=n)
    rec_k = max(1, int(round(rec_every / dt)))
    rec_step = rec_k * dt
    t_inv = np.asarray(params["t_inv"], float)
    hard = np.asarray(params["hard"], bool)
    p_start = np.where(hard, np.nan, np.asarray(params["p0"], float))
    t_start = t_inv.copy()
    est_pt = {}                                   # hard draws: (t_inv, 1/2N) first point
    ok = np.ones(n, bool)
    h = np.where(hard)[0]
    if h.size:
        g, k, two_n = _establish(model, params, h, rng, n_e_fn, max_attempts)
        bad = g < 0
        counts.est_failed = int(bad.sum())
        ok[h[bad]] = False
        for j in np.where(~bad)[0]:
            i = h[j]
            t_start[i] = t_inv[i] - g[j]
            p_start[i] = k[j] / two_n[j]
            est_pt[i] = 1.0 / two_n[j]
    live = np.where(ok)[0]
    if live.size == 0:
        return [], counts
    s_all = np.nan_to_num(np.asarray(params["s"], float)) if model != "N" else np.zeros(n)
    t_sel = np.asarray(params["t_sel"], float) if model != "N" else np.full(n, -1.0)
    t_stop = np.asarray(params["t_stop"], float) if model == "D2" else np.full(n, -1.0)
    p_star = np.asarray(params["p_star"], float) if model == "B" else np.ones(n)
    start_idx = np.floor(t_start / dt).astype(np.int64)
    i_max = int(start_idx[live].max())
    ne_grid = np.asarray(n_e_fn(np.arange(i_max + 1) * dt), float)
    var_f = dt / (2.0 * ne_grid)
    ncol = i_max // rec_k + 1
    rec = np.full((live.size, ncol), np.nan)
    order = np.argsort(-start_idx[live], kind="stable")
    ptr = 0
    pos = np.full(n, -1, np.int64)
    pos[live] = np.arange(live.size)
    a = np.empty(0, np.int64)               # active draw indices (global)
    pa = np.empty(0)
    stat = np.zeros(n, np.int8)             # 1 lost, 2 fixed
    sa = ta = tb = ps = pa
    for i in range(i_max, 0, -1):
        grown = False
        while ptr < live.size and start_idx[live[order[ptr]]] >= i:
            d = live[order[ptr]]
            a = np.append(a, d)
            pa = np.append(pa, p_start[d])
            ptr += 1
            grown = True
        if grown or sa.size != a.size:
            sa, ta, tb, ps = s_all[a], t_sel[a], t_stop[a], p_star[a]
        if a.size == 0:
            continue
        t = i * dt
        q = pa * (1.0 - pa)
        step = np.sqrt(q * var_f[i]) * rng.standard_normal(pa.size)
        if model != "N":
            on = (t <= ta) & (t > tb)
            step += np.where(on, _delta(model, pa, sa, ps), 0.0) * dt
        pa = pa + step
        if (i - 1) % rec_k == 0:
            rec[pos[a], (i - 1) // rec_k] = pa
        bad = (pa <= 0) | (pa >= 1)
        if bad.any():
            stat[a[bad]] = np.where(pa[bad] <= 0, 1, 2)
            keep = ~bad
            a, pa = a[keep], pa[keep]
            sa, ta, tb, ps = sa[keep], ta[keep], tb[keep], ps[keep]
    counts.lost = int((stat == 1).sum())
    counts.fixed = int((stat == 2).sum())
    surv = live[stat[live] == 0]
    p_now = rec[pos[surv], 0]
    keep = np.abs(p_now - P_NOW) <= P_TOL if apply_filter else np.ones(surv.size, bool)
    paths = []
    scal = {k: v for k, v in params.items() if k not in KEYS}
    for d, pn in zip(surv[keep], p_now[keep]):
        row = rec[pos[d]]
        j = np.where(~np.isnan(row))[0]
        tt = j * rec_step
        ff = row[j]
        m = tt < t_start[d]
        tt, ff = tt[m], ff[m]
        times, freqs = [tt, [t_start[d]]], [ff, [p_start[d]]]
        if d in est_pt and t_inv[d] > t_start[d]:
            times.append([t_inv[d]])
            freqs.append([est_pt[d]])
        times, freqs = np.concatenate(times), np.concatenate(freqs)
        pr = dict(scal, **{k: params[k][d] for k in KEYS})
        pr = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in pr.items()}
        paths.append(Path(params=pr, times=times, freqs=freqs, p_now=float(pn)))
    counts.accepted = len(paths)
    return paths, counts
