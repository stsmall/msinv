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
from illex.abc.establishment import ESTABLISHED, establishment_prob  # noqa: F401 (re-exported)
from illex.theory import N_growth

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
    lost_establishing: int = 0   # hard start: the single copy was lost before ESTABLISHED copies (max_attempts == 1)
    est_failed: int = 0          # hard start: every one of max_attempts > 1 attempts failed
    accepted: int = 0
    attempts: int = 0            # total establishment attempts (condition_on_establishment: up to and incl. each success)
    w_acc: float = 0.0           # sum of w_est over accepted paths (importance-weighted accepted count)
    w_acc_sq: float = 0.0        # sum of w_est^2 over accepted paths (for the SE of the weighted acceptance)


def _logu(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def draw_prior(model: str, flux: bool, rng, n: int, cls: str = "mixed") -> dict:
    """Draw n parameter sets. Unused keys are NaN. Also returns ``model`` and ``flux`` as scalars.

    cls: "hard" forces every draw hard, "soft" every draw soft, "mixed" keeps P(hard) = PRIOR["p_hard"].
    """
    if cls not in ("hard", "soft", "mixed"):
        raise ValueError(cls)
    if model not in MODELS:
        raise ValueError(model)
    nan = np.full(n, np.nan)
    t_inv = rng.uniform(*PRIOR["t_inv"], n)
    hard = rng.random(n) < PRIOR["p_hard"]
    if cls != "mixed":
        hard = np.full(n, cls == "hard")
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

    Returns (g, k, two_n, attempts): generations elapsed and copies at the end of establishment, g = -1 where every
    attempt failed (or establishment outlasted t_inv), and the attempts used per draw (counted up to and including
    the first success, so ``attempts.sum()`` is an unbiased attempt count).
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
    n_att = np.zeros(m, np.int64)
    pending = np.arange(m)
    while pending.size:
        nb = int(min(block, (max_attempts - used[pending]).max())) if max_attempts > 0 else 0
        if nb <= 0:
            break
        for lo in range(0, pending.size, max(1, flat_max // nb)):
            ch = pending[lo:lo + max(1, flat_max // nb)]
            take = np.minimum(nb, max_attempts - used[ch])      # attempts this block, per draw
            ids = np.repeat(ch, take)
            gstart = np.cumsum(take) - take                      # first attempt of each draw within ids
            used[ch] += take
            n_att[ch] += take
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
                ci = np.searchsorted(ch, dd)                    # ch is ascending (pending is a filtered arange)
                n_att[dd] -= take[ci] - (sel - gstart[ci] + 1)  # drop the attempts after the first success
                g_out[dd] = g[sel]
                k_out[dd] = k[sel]
        pending = pending[(g_out[pending] < 0) & (used[pending] < max_attempts)]
    return g_out, k_out, two_n, n_att


def simulate_paths(params: dict, rng, dt=10.0, rec_every=1000.0, max_attempts=1,
                   n_e_fn=N_growth, apply_filter=True,
                   condition_on_establishment=False, est_cap=1_000_000):
    """Simulate one path per parameter set. Returns (accepted paths, {"hard": Counts, "soft": Counts}).

    Default max_attempts=1: each hard draw gets ONE establishment attempt from a single copy, and a loss before
    ESTABLISHED copies is a rejection (Counts.lost_establishing). Accepted rows are then exact prior draws, so
    accepted/drawn includes the establishment probability. max_attempts > 1 redraws lost copies (conditioning
    on establishment; failures of the cap are Counts.est_failed) and is for diagnostics only.

    ``condition_on_establishment=True`` (exact importance sampling, used by run_table): a hard draw whose copy is
    lost is redrawn with the SAME theta until it reaches ESTABLISHED copies (safety cap ``est_cap`` attempts, then
    Counts.est_failed). Every returned Path carries ``params["w_est"]`` = u(theta), ``establishment_prob``
    (hard) or 1 (soft). The prior acceptance is (1 / drawn) * sum(w_est over accepted); Counts.w_acc / w_acc_sq
    hold the sums, Counts.drawn is the number of theta draws, Counts.attempts the establishment attempts.
    """
    model = params["model"]
    n = len(params["t_inv"])
    rec_k = max(1, int(round(rec_every / dt)))
    rec_step = rec_k * dt
    t_inv = np.asarray(params["t_inv"], float)
    hard = np.asarray(params["hard"], bool)
    p_start = np.where(hard, np.nan, np.asarray(params["p0"], float))
    t_start = t_inv.copy()
    est_pt = {}                                   # hard draws: (t_inv, 1/2N) first point
    ok = np.ones(n, bool)
    est_bad = np.zeros(n, bool)
    h = np.where(hard)[0]
    if h.size:
        g, k, two_n, n_att = _establish(model, params, h, rng, n_e_fn,
                                        est_cap if condition_on_establishment else max_attempts)
        bad = g < 0
        ok[h[bad]] = False
        est_bad[h[bad]] = True
        for j in np.where(~bad)[0]:
            i = h[j]
            t_start[i] = t_inv[i] - g[j]
            p_start[i] = k[j] / two_n[j]
            est_pt[i] = 1.0 / two_n[j]
    live = np.where(ok)[0]
    w_est = np.ones(n)
    if condition_on_establishment:
        if h.size:
            assert np.allclose(np.asarray(params["t_sel"], float)[h], t_inv[h]) or model == "N", \
                "hard draws must have t_sel == t_inv"
            w_est[h] = establishment_prob(model, np.asarray(params["s"], float)[h] if model != "N" else 0.0,
                                          t_inv[h], n_e_fn,
                                          np.asarray(params["p_star"], float)[h] if model == "B" else None)
            counts_att = int(n_att.sum())
        else:
            counts_att = 0
    stat = np.zeros(n, np.int8)             # 1 lost, 2 fixed
    counts = {"hard": Counts(drawn=int(hard.sum())), "soft": Counts(drawn=int((~hard).sum()))}
    for c, m in (("hard", hard), ("soft", ~hard)):
        nb = int((est_bad & m).sum())
        if condition_on_establishment:
            counts[c].est_failed = nb
        elif max_attempts == 1:
            counts[c].lost_establishing = nb
        else:
            counts[c].est_failed = nb
    if condition_on_establishment:
        counts["hard"].attempts = counts_att
        counts["soft"].attempts = counts["soft"].drawn
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
    for c, m in (("hard", hard), ("soft", ~hard)):
        counts[c].lost = int(((stat == 1) & m).sum())
        counts[c].fixed = int(((stat == 2) & m).sum())
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
        if condition_on_establishment:
            pr["w_est"] = float(w_est[d])
        paths.append(Path(params=pr, times=times, freqs=freqs, p_now=float(pn)))
    for c in counts:
        counts[c].accepted = sum(bool(q.params["hard"]) == (c == "hard") for q in paths)
    if condition_on_establishment:
        for q in paths:
            c = counts["hard" if q.params["hard"] else "soft"]
            c.w_acc += q.params["w_est"]
            c.w_acc_sq += q.params["w_est"] ** 2
    return paths, counts
