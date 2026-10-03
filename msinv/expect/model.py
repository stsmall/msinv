"""Public tier 0 API (spec §4.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._chain import Chain
from ._history import Epochs, FrequencyHistory, flux_coverage
from ._solve import Solver


@dataclass
class PairwiseTimes:
    mean: float
    var: float
    survival: np.ndarray | None  # P(T > t) at the requested times


class ExpectModel:
    """Exact single-site expectations under one inversion history."""

    def __init__(self, demography, inversion, recombination_rate,
                 origin="barrier", rtol=1e-8, atol=1e-12, force_knots_max=1000):
        if origin not in ("barrier", "single"):
            raise ValueError(f"origin must be 'barrier' or 'single', got {origin!r}")
        if recombination_rate <= 0.0:
            raise ValueError("recombination_rate must be > 0")
        self.epochs = Epochs(demography)
        if not self.epochs.final_growth_is_zero():
            raise NotImplementedError(
                "tier 0 needs zero growth after the last demographic event "
                "(the final constant-rate phase is solved exactly)"
            )
        self.freq = FrequencyHistory(inversion, demography.n_pops)
        self.inversion = inversion
        self.r = float(recombination_rate)
        self.origin = origin
        self.origin_pop = self.freq.origin_population() if origin == "single" else None
        self._opts = {"rtol": rtol, "atol": atol, "force_knots_max": force_knots_max}

    def switching_coefficient(self, y: float) -> float:
        """Per-lineage switching hazard at site y, before the frequency factor."""
        inv = self.inversion
        if inv.bp_left <= y < inv.bp_right:
            return inv.gene_conversion_rate * flux_coverage(
                y, inv.bp_left, inv.bp_right, inv.mean_tract_length,
                inv.tract_distribution)
        delta = inv.bp_left - y if y < inv.bp_left else y - inv.bp_right
        return self.r * delta

    def _check_sample(self, sample) -> None:
        for pop, st in sample:
            if not 0 <= pop < self.epochs.n_pops:
                raise ValueError(f"population {pop} out of range")
            if self.freq.f(pop, st, 0.0) <= 0.0:
                raise ValueError(
                    f"sampled state {st!r} has zero frequency in population "
                    f"{pop} at t=0"
                )

    def _solve(self, sample, ys, times=None, max_states=None):
        sample = list(sample)
        self._check_sample(sample)
        coeffs = [self.switching_coefficient(y) for y in ys]
        joins = [j for e in self.epochs.epochs for j in e.joins]
        chain = Chain(sample, self.epochs.n_pops, self.epochs.mig_pairs(), joins,
                      any(c > 0.0 for c in coeffs), self.origin, self.origin_pop,
                      max_states)
        return Solver(chain, self.epochs, self.freq, coeffs, **self._opts).run(times)

    def pairwise_times(self, y, a, b, times=None) -> PairwiseTimes:
        """Coalescence time of chromosomes a and b, each (pop, 'S'|'I')."""
        res = self._solve([a, b], [y], times)
        mean = float(res.mean[0])
        surv = None if times is None else res.survival[0].copy()
        return PairwiseTimes(mean, float(res.second_moment[0] - mean**2), surv)

    def profile(self, ys, a, b) -> np.ndarray:
        """Mean pairwise time at each site in ys, in one integration."""
        return self._solve([a, b], list(ys)).mean.copy()
