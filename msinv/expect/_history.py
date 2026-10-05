"""History inputs for tier 0: demography epochs, frequencies, flux coverage."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from msinv._msinv_core import InvTrajectory
from msinv.hull._rust_bridge import inversion_to_dict

_SUPPORTED_EVENTS = ("eN", "en", "eG", "eg", "eM", "em", "ej")


def _size_at(n0: float, growth: float, growth_start: float, t: float) -> float:
    if growth == 0.0:
        return n0
    return n0 * math.exp(-growth * (t - growth_start))


@dataclass
class Epoch:
    """Demographic state from ``start`` until the next epoch."""

    start: float
    sizes: list
    growth: list
    growth_start: list
    mig: list  # msinv convention: mig[dst][src], lineage in src moves to dst
    joins: list = field(default_factory=list)  # (src, dst) applied at start


class Epochs:
    """Piecewise demography read from :class:`msinv.hull.Demography`.

    Events are replayed exactly as ``Demography.apply_event_at`` applies them.
    Backward in time a lineage in population ``d`` moves to ``d2`` at hazard
    ``migration_matrix[d2][d]`` (simulator.rs:887).
    """

    def __init__(self, demography):
        self.n_pops = demography.n_pops
        self.epochs = [
            Epoch(
                0.0,
                [float(x) for x in demography.pop_sizes],
                [float(x) for x in demography.growth_rates],
                [float(x) for x in demography.growth_start],
                [[float(x) for x in row] for row in demography.migration_matrix],
            )
        ]
        events = sorted(demography.events, key=lambda e: e[1])
        for ev in events:
            if ev[0] not in _SUPPORTED_EVENTS:
                raise NotImplementedError(
                    f"tier 0 does not support demographic event {ev[0]!r} "
                    f"(supported: {', '.join(_SUPPORTED_EVENTS)})"
                )
        i = 0
        while i < len(events):
            t = float(events[i][1])
            group = []
            while i < len(events) and abs(events[i][1] - t) <= 1e-9:
                group.append(events[i])
                i += 1
            self.epochs.append(self._apply(self.epochs[-1], t, group))
        self._starts = [e.start for e in self.epochs]

    def _apply(self, prev: Epoch, t: float, group: list) -> Epoch:
        n = self.n_pops
        sizes, growth, g0 = list(prev.sizes), list(prev.growth), list(prev.growth_start)
        mig = [list(row) for row in prev.mig]
        joins = []
        for ev in group:
            kind = ev[0]
            if kind == "eN":
                for p in range(n):
                    sizes[p], growth[p], g0[p] = float(ev[2]), 0.0, t
            elif kind == "en":
                p = ev[2]
                if p < n:
                    sizes[p], growth[p], g0[p] = float(ev[3]), 0.0, t
            elif kind == "eG":
                for p in range(n):
                    sizes[p] = _size_at(sizes[p], growth[p], g0[p], t)
                    growth[p], g0[p] = float(ev[2]), t
            elif kind == "eg":
                p = ev[2]
                if p < n:
                    sizes[p] = _size_at(sizes[p], growth[p], g0[p], t)
                    growth[p], g0[p] = float(ev[3]), t
            elif kind == "eM":
                per = float(ev[2]) / (n - 1) if n > 1 else 0.0
                for a in range(n):
                    for b in range(n):
                        if a != b:
                            mig[a][b] = per
            elif kind == "em":
                _, _, dst, src, rate = ev
                if dst < n and src < n and dst != src:
                    mig[dst][src] = float(rate)
            elif kind == "ej":
                _, _, src, dst = ev
                joins.append((int(src), int(dst)))
                for k in range(n):
                    mig[src][k] = 0.0
                    mig[k][src] = 0.0
        return Epoch(t, sizes, growth, g0, mig, joins)

    def boundaries(self) -> list:
        return self._starts[1:]

    def epoch_at(self, t: float) -> Epoch:
        return self.epochs[bisect.bisect_right(self._starts, t) - 1]

    def size(self, d: int, t: float) -> float:
        e = self.epoch_at(t)
        return _size_at(e.sizes[d], e.growth[d], e.growth_start[d], t)

    def mig_rate(self, d_from: int, d_to: int, t: float) -> float:
        """Backward hazard of a lineage in ``d_from`` moving to ``d_to``."""
        return self.epoch_at(t).mig[d_to][d_from]

    def joins_at(self, t: float) -> list:
        return [j for e in self.epochs if abs(e.start - t) <= 1e-9 for j in e.joins]

    def final_growth_is_zero(self) -> bool:
        return all(g == 0.0 for g in self.epochs[-1].growth)

    def mig_pairs(self) -> set:
        n = self.n_pops
        return {
            (s, d)
            for e in self.epochs
            for d in range(n)
            for s in range(n)
            if s != d and e.mig[d][s] > 0.0
        }


class FrequencyHistory:
    """Inversion frequencies per population, read from the Rust trajectory.

    ``t_inv`` is global (``t_inv_max``). Per-population values of coupled or
    precomputed trajectories are support boundaries of f_d(I, t), not
    separate inversion ages (spec §3.2).
    """

    def __init__(self, inversion, n_pops: int):
        self.n_pops = n_pops
        self._traj = InvTrajectory(inversion_to_dict(inversion, n_pops))
        self.t_inv = float(self._traj.t_inv_max())
        self.t_inv_pop = [float(self._traj.t_inv(d)) for d in range(n_pops)]
        self.knots = sorted(
            {float(k) for k in self._traj.knots() if 0.0 < k < self.t_inv}
        )
        self._cache_t = None
        self._cache_p = None
        self._check_simplex()

    def p_all(self, t: float) -> list:
        """I frequency in every population at backward time ``t``."""
        if t != self._cache_t:
            self._cache_p = [self._traj.p_inv([t], d)[0] for d in range(self.n_pops)]
            self._cache_t = t
        return self._cache_p

    def f(self, d: int, state: str, t: float) -> float:
        if t >= self.t_inv:
            raise ValueError(f"no karyotype states at or beyond t_inv={self.t_inv}")
        p = self.p_all(t)[d]
        if state == "I":
            return p
        if state == "S":
            return 1.0 - p
        raise ValueError(f"state must be 'S' or 'I', got {state!r}")

    def left_limit(self, d: int, state: str, t: float) -> float:
        """f_d(state, t-), by two-point linear extrapolation from the left."""
        delta = 1e-7 * max(1.0, t)
        return 2.0 * self.f(d, state, t - delta) - self.f(d, state, t - 2.0 * delta)

    def turning_knots(self) -> list:
        """Knots where p changes direction in some population.

        The slope sign (-, 0, +) of the piecewise-linear p changes there,
        which includes both edges of a flat dip or plateau. Between two
        turning knots p is monotone, so a change persists and the solver's
        error control sees it; a short dip between unforced stops does not.
        """
        if not self.knots:
            return []
        pts = [0.0, *self.knots, self.t_inv - 1e-9 * max(1.0, self.t_inv)]
        out = set()
        for d in range(self.n_pops):
            p = [self.p_all(t)[d] for t in pts]
            sign = [
                0 if abs(b - a) <= 1e-14 else (1 if b > a else -1)
                for a, b in zip(p, p[1:])
            ]
            # sign[i] is the slope left of knots[i], sign[i + 1] right of it.
            out.update(k for i, k in enumerate(self.knots) if sign[i] != sign[i + 1])
        return sorted(out)

    def _check_simplex(self) -> None:
        # One inversion: f_S = 1 - p is derived, so only p needs checking.
        for t in [0.0, *self.knots]:
            for d, p in enumerate(self.p_all(t)):
                if not -1e-12 <= p <= 1.0 + 1e-12:
                    raise ValueError(
                        f"inversion frequency {p} outside [0, 1] in population "
                        f"{d} at t={t}"
                    )

    def support_boundaries(self) -> list:
        """Sorted (t_b, pop, state): f_pop(state) becomes zero at t_b < t_inv."""
        out = set()
        for d in range(self.n_pops):
            if self.t_inv_pop[d] < self.t_inv:
                out.add((self.t_inv_pop[d], d, "I"))
            prev = self.p_all(0.0)[d]
            for k in self.knots:
                p = self.p_all(k)[d]
                if p <= 0.0 < prev:
                    out.add((k, d, "I"))
                if p >= 1.0 > prev:
                    out.add((k, d, "S"))
                prev = p
        return sorted(out)

    def vanishing_at_t_inv(self) -> list:
        """(pop, state) whose frequency declines continuously to zero at t_inv.

        Same jump test as the solver's flush: the left limit at t_inv is zero
        relative to the frequency just before it.
        """
        out = []
        t2 = self.t_inv - 2e-7 * max(1.0, self.t_inv)
        for d in range(self.n_pops):
            for st in ("S", "I"):
                f2 = self.f(d, st, t2)
                if f2 > 0.0 and self.left_limit(d, st, self.t_inv) <= 1e-6 * f2:
                    out.append((d, st))
        return out

    def draw_conditional(self, d: int, t: float, known: dict, rng) -> dict:
        """Draw the karyotypes not in ``known`` (inv_id -> 'S'/'I')."""
        if 0 in known:
            if self.f(d, known[0], t) <= 0.0:
                raise ValueError(
                    f"conditioning state {known[0]!r} has zero frequency in "
                    f"population {d} at t={t}"
                )
            return dict(known)
        return {0: "I" if rng.random() < self.f(d, "I", t) else "S"}

    def origin_population(self) -> int:
        eps = 1e-9 * max(1.0, self.t_inv)
        pops = [d for d, p in enumerate(self.p_all(self.t_inv - eps)) if p > 0.0]
        if len(pops) != 1:
            raise ValueError(
                "origin='single' needs exactly one population with f(I) > 0 "
                f"just below t_inv; found {pops}"
            )
        return pops[0]


def _phi(u: float, w: float) -> float:
    """Peischl phi, identical to rust/msinv-core/src/phi.rs::phi."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if w >= 1.0:
        return 1.0
    return min(max(min(u, 1.0 - u, w) / (1.0 - w), 0.0), 1.0)


def _cover_prob(x: float, yr: float, length: float, inv_len: float) -> float:
    """P(tract started from x with this length covers yr); mirrors draw_tract."""
    lo, hi = max(x - length, 0.0), min(x, inv_len - length)
    if hi <= lo:
        b1 = min(max(x - length / 2.0, 0.0), inv_len - length)
        return 1.0 if b1 <= yr <= b1 + length else 0.0
    a, b = max(lo, yr - length), min(hi, yr)
    return max(b - a, 0.0) / (hi - lo)


def flux_coverage(
    y: float,
    bp_left: float,
    bp_right: float,
    mean_tract_length: float,
    tract_distribution: str,
) -> float:
    """C(y) of spec §3.4 rule 4: per-site conversion hazard / (gamma * f_other).

    For a lineage carrying the whole inversion, under the tier 2 event rules
    (``sample_flux_position`` + ``draw_tract``). About phi(y) * mean tract
    length away from the breakpoints.
    """
    from scipy.integrate import quad

    inv_len = bp_right - bp_left
    if not bp_left <= y < bp_right or mean_tract_length <= 0.0:
        return 0.0
    w = mean_tract_length / inv_len
    yr = y - bp_left

    def given_length(length: float) -> float:
        length = min(length, 0.99 * inv_len)
        lo, hi = max(0.0, yr - length), min(inv_len, yr + length)
        if hi <= lo:
            return 0.0
        kinks = (
            yr,
            yr - length / 2,
            yr + length / 2,
            length,
            inv_len - length,
            w * inv_len,
            (1 - w) * inv_len,
        )
        pts = [p for p in kinks if lo < p < hi]
        val, _ = quad(
            lambda x: _phi(x / inv_len, w) * _cover_prob(x, yr, length, inv_len),
            lo,
            hi,
            points=pts or None,
            limit=200,
            epsabs=0.0,
            epsrel=1e-10,
        )
        return val

    if tract_distribution == "fixed":
        return given_length(mean_tract_length)
    cap, m = 0.99 * inv_len, mean_tract_length
    body, _ = quad(
        lambda length: given_length(length) * math.exp(-length / m) / m,
        0.0,
        cap,
        limit=200,
        epsabs=0.0,
        epsrel=1e-8,
    )
    return body + math.exp(-cap / m) * given_length(cap)
