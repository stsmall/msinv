"""InversionSpec: parameters for one chromosomal inversion.

A chromosome may carry multiple non-overlapping inversions, each
described by an InversionSpec. The hull simulator handles each
inversion's class barrier independently — class labels are tagged
with the inversion's id (e.g. 'S0' = S-arrangement inside inversion 0,
'S1' = inside inversion 1).
"""

from dataclasses import dataclass


@dataclass
class InversionSpec:
    """Parameters for one inversion.

    Attributes
    ----------
    bp_left, bp_right : float
        Inversion breakpoints in genomic coordinates.
    p_inv : float
        Frequency of the inverted (I) arrangement, in (0, 1).
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
    p_inv: float
    t_inv: float
    gene_conversion_rate: float = 1e-9
    flux_window: float = 0.05
    inv_id: int = -1   # set by simulator

    @property
    def length(self) -> float:
        return self.bp_right - self.bp_left

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
        if not (0.0 < self.p_inv < 1.0):
            raise ValueError(f"p_inv must be in (0, 1), got {self.p_inv}.")
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
