"""Time integration of the tier 0 chain (spec §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply, splu

from ._chain import I, P, S

_NAME = {S: "S", I: "I"}


class ImpossibleHistoryError(ValueError):
    """Ancestry is left in a (population, state) cell with no support."""


@dataclass
class Result:
    branch_lengths: np.ndarray  # (R, n-1): E[L_b], generations
    mean: np.ndarray  # (R,): E[T_MRCA]
    second_moment: np.ndarray  # (R,): E[T_MRCA^2]
    survival: np.ndarray  # (R, len(times)): P(T_MRCA > t)


class Solver:
    """Integrate R replicas of one chain that differ only in the switching
    coefficient (one per site), as one block-diagonal system."""

    def __init__(self, chain, epochs, freq, coeffs, rtol=1e-8, atol=1e-12,
                 force_knots_max=1000):
        self.ch, self.ep, self.fq = chain, epochs, freq
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.R = len(self.coeffs)
        self.rtol, self.atol = rtol, atol
        eye = sp.identity(self.R, format="csr")
        self._AT = {}
        for key, at in chain.AT.items():
            left = sp.diags(self.coeffs) if key[0] == "sw" else eye
            self._AT[key] = sp.kron(left, at, format="csr")
        self._BT = {k: sp.kron(eye, bt, format="csr") for k, bt in chain.BT.items()}
        self.N = chain.size * self.R
        self._rew = sp.kron(eye, sp.csr_matrix(chain.rewards), format="csr")
        self._smat = sp.kron(
            eye, sp.csr_matrix(chain.transient.astype(float)[None, :]), format="csr"
        )
        self._trans = np.tile(chain.transient, self.R)
        self._post = np.tile(chain.post, self.R)
        self._block = np.repeat(np.arange(self.R), chain.size)
        self._n_aux = self.R * (chain.n - 1) + 2 * self.R
        knots = freq.knots if len(freq.knots) <= force_knots_max else []
        self._supp = freq.support_boundaries()
        pts = set(epochs.boundaries()) | set(knots) | {freq.t_inv}
        pts |= {t for t, _, _ in self._supp}
        self.breaks = sorted(t for t in pts if 0.0 < t < np.inf)
        self._memo_t, self._memo_q = None, None

    # -- rates -------------------------------------------------------------

    def _g(self, key, t: float) -> float:
        kind, d = key[0], key[1]
        pre = t < self.fq.t_inv
        if kind == "coal":
            k = key[2]
            n_d = self.ep.size(d, t)
            if k == P:
                return 1.0 / (2.0 * n_d)
            if not pre:
                return 0.0
            f = self.fq.f(d, _NAME[k], t)
            return 1.0 / (2.0 * n_d * f) if f > 0.0 else 0.0
        if kind == "mig":
            d2, k = key[2], key[3]
            m = self.ep.mig_rate(d, d2, t)
            if m == 0.0:
                return 0.0
            if k == P:
                return m
            if not pre:
                return 0.0
            fd = self.fq.f(d, _NAME[k], t)
            return m * self.fq.f(d2, _NAME[k], t) / fd if fd > 0.0 else 0.0
        if not pre:  # "sw": the per-site coefficient is in the matrix
            return 0.0
        return self.fq.f(d, _NAME[1 - key[2]], t)

    def qt(self, t: float):
        """Transpose generator at t: d(pi)/dt = qt(t) @ pi."""
        out = sp.csr_matrix((self.N, self.N))
        for key, at in self._AT.items():
            g = self._g(key, t)
            if g != 0.0:
                out = out + g * at
        return out

    def _qt_memo(self, t: float):
        if t != self._memo_t:
            self._memo_t, self._memo_q = t, self.qt(t)
        return self._memo_q

    def _rhs(self, t, z):
        q = self._qt_memo(t)
        pi = z[: self.N]
        s = self._smat @ pi
        return np.concatenate([q @ pi, self._rew @ pi, s, t * s])

    def _jac(self, t, z):
        q = self._qt_memo(t)
        m = self._n_aux
        top = sp.hstack([q, sp.csr_matrix((self.N, m))])
        low = sp.vstack([self._rew, self._smat, t * self._smat])
        bottom = sp.hstack([low, sp.csr_matrix((m, m))])
        return sp.vstack([top, bottom], format="csc")

    # -- boundaries --------------------------------------------------------

    def _apply_boundaries(self, tb: float, pi):
        for src, dst in self.ep.joins_at(tb):
            bt = self._BT.get(("join", src, dst))
            if bt is not None:
                pi = bt @ pi
        if tb == self.fq.t_inv and "t_inv" in self._BT:
            pi = self._BT["t_inv"] @ pi
        return pi

    def _flush(self, tb: float, cells: list, pi):
        raise NotImplementedError(
            "trajectories whose frequency reaches zero before t_inv are not "
            "supported yet"
        )

    # -- main loop ---------------------------------------------------------

    def run(self, times=None) -> Result:
        times = np.asarray([] if times is None else times, dtype=float)
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be in ascending order")
        surv = np.full((self.R, len(times)), np.nan)
        z = np.zeros(self.N + self._n_aux)
        pi0 = np.zeros(self.ch.size)
        pi0[self.ch.init] = 1.0
        z[: self.N] = self._apply_boundaries(0.0, np.tile(pi0, self.R))
        t = 0.0
        for tb in self.breaks:
            cells = [(d, st) for tt, d, st in self._supp if tt == tb]
            end = tb - 1e-9 * max(1.0, tb) if cells else tb
            sel = (times >= t) & (times < tb)
            if end > t:
                t_eval = np.append(np.minimum(times[sel], end), end)
                sol = solve_ivp(self._rhs, (t, end), z, method="BDF", jac=self._jac,
                                rtol=self.rtol, atol=self.atol, t_eval=t_eval)
                if not sol.success:
                    raise RuntimeError(
                        f"ODE solver failed on [{t}, {end}]: {sol.message}"
                    )
                surv[:, sel] = self._smat @ sol.y[: self.N, :-1]
                z = sol.y[:, -1].copy()
            if cells:
                z[: self.N] = self._flush(tb, cells, z[: self.N])
            z[: self.N] = self._apply_boundaries(tb, z[: self.N])
            t = tb
        return self._finish(t, z, times, surv)

    def _finish(self, t: float, z, times, surv) -> Result:
        n_rw = self.R * (self.ch.n - 1)
        pi = z[: self.N]
        rw = z[self.N : self.N + n_rw].copy()
        m1 = z[self.N + n_rw : self.N + n_rw + self.R].copy()
        m2 = z[self.N + n_rw + self.R :].copy()
        # After the last boundary (always >= t_inv) only post-t_inv states hold
        # mass; pre-t_inv states have zero rates there and would make the
        # system singular.
        idx = np.flatnonzero(self._trans & self._post)
        pi_t = pi[idx]
        late = np.flatnonzero(times >= t)
        if pi_t.sum() > self.atol:
            qtt = self.qt(t)[idx][:, idx].tocsc()
            try:
                lu = splu((-qtt.T).tocsc())
            except RuntimeError as exc:
                raise ValueError(
                    "lineages can never coalesce after the last demographic "
                    "event: no migration or population join connects their "
                    "populations"
                ) from exc
            blk = self._block[idx]
            v1 = lu.solve(np.ones(len(idx)))
            v2 = lu.solve(v1)
            rw += pi_t @ lu.solve(self._rew[:, idx].T.toarray())
            add1 = np.bincount(blk, weights=pi_t * v1, minlength=self.R)
            m1 += add1
            m2 += t * add1 + np.bincount(blk, weights=pi_t * v2, minlength=self.R)
            for j in late:
                v = expm_multiply(qtt * (times[j] - t), pi_t)
                surv[:, j] = np.bincount(blk, weights=v, minlength=self.R)
        else:
            surv[:, late] = 0.0
        return Result(rw.reshape(self.R, self.ch.n - 1), m1, 2.0 * m2, surv)
