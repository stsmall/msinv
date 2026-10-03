"""History inputs for tier 0: demography epochs, frequencies, flux coverage."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

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
