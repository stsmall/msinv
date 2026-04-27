"""InversionSpec: parameters for one chromosomal inversion.

A chromosome may carry multiple non-overlapping inversions, each
described by an InversionSpec. The hull simulator handles each
inversion's class barrier independently — class labels are tagged
with the inversion's id (e.g. 'S0' = S-arrangement inside inversion 0,
'S1' = inside inversion 1).

Frequency model
---------------
Two modes for specifying the inversion frequency:

1. Constant (legacy back-compat): pass ``p_inv`` (float or dict
   pop->float) and ``t_inv`` (float).  Frequency is fixed in time
   per population, with the karyotype barrier dissolving at t_inv.

2. Trajectory (new): pass a ``trajectory=`` dict with one of the
   supported types matching the Rust trajectory module:
     {'type': 'constant',      'p_inv': ..., 't_inv': ...}
     {'type': 'deterministic', 'p_final': ..., 'n_e': ..., 's': ...,
                               ['p_start': ...]}
     {'type': 'stochastic',    'p_final': ..., 'n_e': ..., 's': ..., 'seed': ...}
     {'type': 'integer_wf',    'p_final': ..., 'n_e': ..., 's': ...,
                               ['p_start': ...], ['seed': ...],
                               ['max_attempts': ...]}
     {'type': 'stoch_det',     'p_final': ..., 'n_e': ..., 's': ...,
                               ['p_start': ...], ['det_threshold': ...],
                               ['seed': ...], ['max_attempts': ...]}
     {'type': 'coupled',       'p_final': [..], 'n_e': [..], 's': [..],
                               'm': ..., 'seed': ...}
     {'type': 'precomputed',   'times': [...], 'freqs': [[..], ...],
                               'n_e': [...], ['t_inv': [...]]}
   The Rust simulator builds the matching trajectory and queries it
   at each event time.

   - 'deterministic': closed-form logistic.  Optional ``p_start``
     enables partial-SHIC-style soft-sweep from standing variation
     (default 1/(2N) = hard sweep).
   - 'integer_wf': discrete Wright-Fisher forward simulation with
     selection, rejection-sampling lost paths.  Robust at large N
     (replaces 'stochastic' which uses continuous-diffusion approx
     and breaks at large N).
   - 'stoch_det': discoal-style hybrid — integer-WF in the drift-
     dominated regime, then closed-form logistic once selection is
     deterministic-strong.
"""

from dataclasses import dataclass, field
from typing import Union, Dict, Optional


@dataclass
class InversionSpec:
    """Parameters for one inversion.

    Attributes
    ----------
    bp_left, bp_right : float
        Inversion breakpoints in genomic coordinates.
    p_inv : float or dict[int, float]
        Frequency of the inverted (I) arrangement.
        - float: same frequency in all populations, must be in (0, 1).
        - dict: per-population frequency, e.g. {0: 0.0, 1: 0.73}.
          Each value must be in [0, 1] and at least one must be in (0, 1).
    t_inv : float
        Inversion age in generations. After t >= t_inv, the class
        barrier lifts and inv-internal positions become panmictic.
    gene_conversion_rate : float
        Per-bp per-generation gene-conversion rate γ. Must be > 0.
        Combined with phi(x) for the per-position flux rate.
    flux_window : float
        Tract length as a fraction of the inversion's genomic length.
    inv_id : int
        Identifier; auto-assigned by HullSimulator from the inversions
        list index.
    """

    bp_left: float
    bp_right: float
    p_inv: Union[float, Dict[int, float], None] = None
    t_inv: Optional[float] = None
    gene_conversion_rate: float = 1e-9
    flux_window: float = 0.05
    # Peischl b2 flux model — see docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md.
    # When migration completes, flux_window is removed.
    mean_tract_length: float = 100.0   # bp, replaces flux_window's tract role
    tract_distribution: str = 'geometric'  # 'geometric' or 'fixed'
    inv_id: int = -1   # set by simulator
    # Trajectory dict overrides p_inv/t_inv when provided.  See module
    # docstring for the supported shapes.
    trajectory: Optional[Dict] = None

    @property
    def length(self) -> float:
        return self.bp_right - self.bp_left

    def p_inv_for(self, pop: int) -> float:
        """Return inverted-arrangement frequency for population *pop*."""
        if isinstance(self.p_inv, dict):
            if pop in self.p_inv:
                return self.p_inv[pop]
            # Fallback: use the first entry
            return next(iter(self.p_inv.values()))
        return self.p_inv

    def p_std_for(self, pop: int) -> float:
        """Return standard-arrangement frequency for population *pop*."""
        return 1.0 - self.p_inv_for(pop)

    def set_p_inv_for(self, pop: int, val: float):
        """Set inverted-arrangement frequency for a specific population."""
        if not isinstance(self.p_inv, dict):
            # Convert scalar to dict
            self.p_inv = {0: self.p_inv}
        self.p_inv[pop] = val

    def _p_inv_as_list(self, n_pops: int) -> list:
        """Return p_inv as a list of length n_pops for the Rust bridge."""
        if isinstance(self.p_inv, dict):
            max_pop = max(self.p_inv.keys()) if self.p_inv else 0
            n = max(n_pops, max_pop + 1)
            default = next(iter(self.p_inv.values()))
            result = [default] * n
            for pop, val in self.p_inv.items():
                result[pop] = val
            return result
        return [self.p_inv] * max(n_pops, 1)

    def class_S(self) -> str:
        # inv_id == -1 is the legacy single-inversion sentinel: use
        # plain 'S' (Phases 2-5a) so segment class tags don't carry
        # an id suffix. Multi-inversion (inversions=[...]) uses 'S0'/'S1'/...
        return 'S' if self.inv_id < 0 else f'S{self.inv_id}'

    def class_I(self) -> str:
        return 'I' if self.inv_id < 0 else f'I{self.inv_id}'

    def __post_init__(self):
        if self.bp_right <= self.bp_left:
            raise ValueError(
                f"bp_right must be > bp_left, got "
                f"({self.bp_left}, {self.bp_right}).")
        # If a trajectory dict is provided, p_inv/t_inv are ignored.
        # The trajectory must specify a 'type' field.
        if self.trajectory is not None:
            if 'type' not in self.trajectory:
                raise ValueError("trajectory dict requires 'type' key")
            if self.gene_conversion_rate <= 0.0:
                raise ValueError(
                    f"gene_conversion_rate (gamma) must be > 0, got "
                    f"{self.gene_conversion_rate}.")
            if not (0.0 < self.flux_window < 1.0):
                raise ValueError(
                    f"flux_window must be in (0, 1), got {self.flux_window}.")
        else:
            # Legacy path: p_inv + t_inv required
            if self.p_inv is None or self.t_inv is None:
                raise ValueError(
                    "InversionSpec requires either (p_inv, t_inv) or trajectory.")
            # Validate p_inv
            if isinstance(self.p_inv, dict):
                if not self.p_inv:
                    raise ValueError("p_inv dict must not be empty.")
                for pop, val in self.p_inv.items():
                    if not (0.0 <= val <= 1.0):
                        raise ValueError(
                            f"p_inv[{pop}] must be in [0, 1], got {val}.")
                # At least one pop must have 0 < p_inv < 1 for the inversion
                # to matter (otherwise it's monomorphic everywhere).
                if not any(0.0 < v < 1.0 for v in self.p_inv.values()):
                    raise ValueError(
                        "At least one population must have 0 < p_inv < 1.")
            else:
                if not (0.0 < self.p_inv < 1.0):
                    raise ValueError(
                        f"p_inv must be in (0, 1), got {self.p_inv}.")
            if self.t_inv <= 0.0:
                raise ValueError(f"t_inv > 0 required, got {self.t_inv}.")
            if not (0.0 < self.flux_window < 1.0):
                raise ValueError(
                    f"flux_window must be in (0, 1), got {self.flux_window}.")
            if self.gene_conversion_rate <= 0.0:
                raise ValueError(
                    f"gene_conversion_rate (gamma) must be > 0, got "
                    f"{self.gene_conversion_rate}. Inversions decouple from "
                    f"flanks unless gene flux is allowed; gamma=0 makes the "
                    f"inversion an absolute barrier (often unrealistic).")
        # ---- b2 flux: validate mean_tract_length, tract_distribution ----
        if self.mean_tract_length < 0.0:
            raise ValueError(
                f"mean_tract_length must be >= 0, got {self.mean_tract_length}. "
                f"Use mean_tract_length=0 (or gene_conversion_rate=0) to "
                f"disable flux entirely.")
        if self.tract_distribution not in ('geometric', 'fixed'):
            raise ValueError(
                f"tract_distribution must be 'geometric' or 'fixed', "
                f"got {self.tract_distribution!r}.")
        inv_len_local = self.bp_right - self.bp_left
        if self.mean_tract_length > inv_len_local / 2.0:
            import warnings as _warnings
            _warnings.warn(
                f"mean_tract_length ({self.mean_tract_length:.1f}) exceeds "
                f"inv_length/2 ({inv_len_local/2:.1f}); tracts will frequently "
                f"span much of the inversion. Verify this is intentional.",
                UserWarning, stacklevel=2)
