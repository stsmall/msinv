"""Time integration of the tier 0 chain (spec §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import BDF, solve_ivp
from scipy.sparse.linalg import expm_multiply, splu

from ._chain import I, P, S

_NAME = {S: "S", I: "I"}


class _NoPivotBDF(BDF):
    """BDF whose sparse LU takes diagonal pivots (no partial pivoting).

    In the Newton matrix I - h*J the generator block is strictly column
    diagonally dominant (J's generator columns sum to zero) and the auxiliary
    rows (rewards, survival) sit below it with identity diagonal, so diagonal
    pivots are stable. With SuperLU's default threshold pivoting the larger
    auxiliary entries get picked as pivots and the factors fill in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sp.issparse(self.J):

            def lu(a):
                self.nlu += 1
                return splu(a, diag_pivot_thresh=0.0)

            self.lu = lu


# ODE method used by Solver; a module attribute so tests can compare it
# against scipy's default "BDF".
_ODE_METHOD = _NoPivotBDF


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

    def __init__(
        self, chain, epochs, freq, coeffs, rtol=1e-8, atol=1e-12, force_knots_max=1000
    ):
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
        # Cells whose frequency declines continuously to zero at t_inv: their
        # ancestry leaves by the limit chain just before t_inv (spec §3.5).
        self._supp += [(freq.t_inv, d, st) for d, st in freq.vanishing_at_t_inv()]
        pts = set(epochs.boundaries()) | set(knots) | {freq.t_inv}
        pts |= {t for t, _, _ in self._supp}
        self.breaks = sorted(t for t in pts if 0.0 < t < np.inf)
        self._build_patterns()
        self._memo_t, self._memo_q = None, None

    def _build_patterns(self) -> None:
        """Fixed sparsity patterns, so qt(t) and the Jacobian only refill data.

        qt(t) = sum_k g_k(t) AT_k: every AT_k is scattered once onto the union
        pattern, giving coefficients D (nnz x K) with qt(t).data = D @ g(t).
        The augmented Jacobian [[qt, 0], [rew; S; t*S, 0]] gets a fixed CSC
        pattern; position arrays map qt data and the auxiliary rows into it.
        """
        n, m = self.N, self._n_aux
        self._keys = list(self._AT)
        coos = [self._AT[k].tocoo() for k in self._keys]
        lin = np.concatenate(
            [c.row.astype(np.int64) * n + c.col for c in coos] or [np.zeros(0, int)]
        )
        kidx = np.concatenate(
            [np.full(c.nnz, i) for i, c in enumerate(coos)] or [np.zeros(0, int)]
        )
        vals = np.concatenate([c.data for c in coos] or [np.zeros(0)])
        uniq, inv = np.unique(lin, return_inverse=True)  # sorted = CSR order
        self._qD = sp.csr_matrix(
            (vals, (inv, kidx)), shape=(len(uniq), max(len(self._keys), 1))
        )
        rows_u = uniq // n
        self._q_indices = (uniq % n).astype(np.int32)
        self._q_indptr = np.concatenate(
            [[0], np.cumsum(np.bincount(rows_u, minlength=n))]
        ).astype(np.int32)
        nq = len(uniq)
        # Jacobian pattern via markers: qt entries 1..nq, auxiliary nq+1...
        qmark = sp.csr_matrix(
            (np.arange(1, nq + 1, dtype=float), self._q_indices, self._q_indptr),
            shape=(n, n),
        )
        low = sp.vstack([self._rew, self._smat, self._smat], format="csr")
        nl = low.nnz
        lmark = sp.csr_matrix(
            (np.arange(nq + 1, nq + nl + 1, dtype=float), low.indices, low.indptr),
            shape=low.shape,
        )
        jmark = sp.hstack(
            [sp.vstack([qmark, lmark], format="csc"), sp.csc_matrix((n + m, m))],
            format="csc",
        )
        mk = jmark.data.astype(np.int64)
        self._j_indices, self._j_indptr = jmark.indices, jmark.indptr
        self._j_pos_q = np.flatnonzero(mk <= nq)
        self._j_src_q = mk[self._j_pos_q] - 1
        self._j_pos_l = np.flatnonzero(mk > nq)
        src_l = mk[self._j_pos_l] - nq - 1
        self._j_val_l = low.data[src_l]
        # Rows of the last auxiliary block (t * S) scale with t.
        low_rows = np.repeat(np.arange(low.shape[0]), np.diff(low.indptr))
        self._j_t_l = low_rows[src_l] >= low.shape[0] - self.R
        self._j_nnz = len(mk)

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
        g = np.array([self._g(k, t) for k in self._keys] or [0.0])
        return sp.csr_matrix(
            (self._qD @ g, self._q_indices, self._q_indptr), shape=(self.N, self.N)
        )

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
        data = np.empty(self._j_nnz)
        data[self._j_pos_q] = q.data[self._j_src_q]
        data[self._j_pos_l] = np.where(self._j_t_l, t, 1.0) * self._j_val_l
        size = self.N + self._n_aux
        return sp.csc_matrix(
            (data, self._j_indices, self._j_indptr), shape=(size, size)
        )

    # -- boundaries --------------------------------------------------------

    def _apply_boundaries(self, tb: float, pi):
        for src, dst in self.ep.joins_at(tb):
            bt = self._BT.get(("join", src, dst))
            if bt is not None:
                pi = bt @ pi
                if tb < self.fq.t_inv:
                    self._check_join(tb, src, dst, pi)
        if tb == self.fq.t_inv and "t_inv" in self._BT:
            pi = self._BT["t_inv"] @ pi
        return pi

    def _check_join(self, tb: float, src: int, dst: int, pi) -> None:
        """A join may not move ancestry into a cell with zero frequency."""
        for k, st in ((S, "S"), (I, "I")):
            mask = self.ch.cell_mask.get(3 * dst + k)
            if mask is None or self.fq.f(dst, st, tb) > 0.0:
                continue
            mass = self._per_replica(pi, np.tile(mask, self.R))
            if mass.max() > self.atol:
                raise ImpossibleHistoryError(
                    f"history impossible: join {src} -> {dst} at t={tb} moves "
                    f"{mass.max():.3g} of ancestry into population {dst}, state "
                    f"{st}, where its frequency is zero"
                )

    def _flush(self, tb: float, cells: list, pi, allow_stuck: bool = False):
        """Resolve ancestry left in cells whose frequency reaches zero at tb.

        All exit hazards from such a cell scale as 1/f, so as f -> 0 the
        remaining mass leaves by the rescaled chain: coalescence within the
        cell at 1/(2N) and migration at m(d -> d') f_d'(K, tb) (spec §3.5);
        migration into another cell that vanishes at tb has rate 0. This
        assumes the integral of dt/f diverges at tb, which holds for
        piecewise-linear and closed-form trajectories. Mass this chain cannot
        move, or mass in a cell whose frequency jumps to zero, means the
        history is impossible. With ``allow_stuck`` (cells vanishing at t_inv)
        mass the limit chain cannot move stays put and passes through the
        t_inv map.
        """
        tl = tb - 1e-9 * max(1.0, tb)
        dying = set(cells)
        for d, st in cells:
            k = 0 if st == "S" else 1
            c = 3 * d + k
            if c not in self.ch.cell_mask:
                continue
            mask = np.tile(self.ch.cell_mask[c], self.R)
            mass = self._per_replica(pi, mask)
            if mass.max() <= self.atol:
                continue
            f2 = self.fq.f(d, st, tb - 2e-7 * max(1.0, tb))
            if self.fq.left_limit(d, st, tb) > 1e-6 * f2:
                raise ImpossibleHistoryError(
                    f"history impossible: frequency jumps to zero in population "
                    f"{d}, state {st} at t={tb} while {mass.max():.3g} of "
                    "ancestry remains there"
                )
            qts = sp.csr_matrix((self.N, self.N))
            for key, at in self._AT.items():
                if key[0] == "coal" and key[1] == d and key[2] == k:
                    qts = qts + at / (2.0 * self.ep.size(d, tl))
                elif key[0] == "mig" and key[1] == d and key[3] == k:
                    if (key[2], st) in dying:
                        continue
                    rate = self.ep.mig_rate(d, key[2], tl) * self.fq.f(key[2], st, tl)
                    if rate > 0.0:
                        qts = qts + rate * at
            stuck = mask & (qts.diagonal() >= 0.0)
            tr = np.flatnonzero(mask & ~stuck)
            ab = np.flatnonzero(~mask | stuck)
            new = pi.copy()
            if len(tr):
                y = splu((-qts[tr][:, tr]).tocsc()).solve(pi[tr])
                new[tr] = 0.0
                new[ab] += qts[ab][:, tr] @ y
            if allow_stuck:
                pi = new
                continue
            left = self._per_replica(new, stuck)
            if left.max() > self.atol:
                raise ImpossibleHistoryError(
                    f"history impossible: {left.max():.3g} of ancestry remains "
                    f"in population {d}, state {st} at t={tb}, where its "
                    "frequency is zero and no lineage can leave"
                )
            new[stuck] = 0.0
            pi = new
        return pi

    def _per_replica(self, v, mask):
        """Sum of v over the masked entries, per replica."""
        return np.bincount(self._block[mask], weights=v[mask], minlength=self.R)

    # -- main loop ---------------------------------------------------------

    def run(self, times=None) -> Result:
        times = np.asarray([] if times is None else times, dtype=float)
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be in ascending order")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")
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
            if end <= t:
                surv[:, sel] = (self._smat @ z[: self.N])[:, None]
            else:
                t_eval = np.append(np.minimum(times[sel], end), end)
                sol = solve_ivp(
                    self._rhs,
                    (t, end),
                    z,
                    method=_ODE_METHOD,
                    jac=self._jac,
                    rtol=self.rtol,
                    atol=self.atol,
                    t_eval=t_eval,
                )
                if not sol.success:
                    raise RuntimeError(
                        f"ODE solver failed on [{t}, {end}]: {sol.message}"
                    )
                surv[:, sel] = self._smat @ sol.y[: self.N, :-1]
                z = sol.y[:, -1].copy()
            if cells:
                z[: self.N] = self._flush(
                    tb, cells, z[: self.N], allow_stuck=tb == self.fq.t_inv
                )
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
