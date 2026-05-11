"""v12 demography for the msinv validation suite.

v12 = v11 from examples/kir_fol_demography.py minus Ghost and Moz pops,
minus all K↔F migrations. Two populations (K=0, F=1) at present,
merged into ancestral pop at the K-F split, with a deep ancestral size
change. All Ne(t) stair-steps from v11 are preserved.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md
"""
from __future__ import annotations

from msinv import Demography

# --- v12 parameters (Small 2023 + v11 lock) -------------------------

# Present-day Ne (t=0 ancestrally backward)
NE_K_PRESENT = 126_772
NE_F_PRESENT = 2_496_632

# F at the K-F split
NE_F_AT_SPLIT = 158_711

# Merged KF pop at the moment of split
NE_KF_AT_MERGE = 86_000

# Deep ancestral Ne (after the KF -> Anc rename)
NE_ANC_DEEP = 450_000

# Times (generations backward, present = 0)
T_KF_SPLIT = 9_194
T_ANC_RENAME = 87_163

# Inversion (3Ra)
T_INV_3RA = 330_000
P_INV_F_3RA = 0.73
P_INV_K_3RA = 0.0
P_INV_ANC_3RA = 0.30
GAMMA_3RA = 1.0e-7


def v12_msinv() -> Demography:
    """Build the v12 msinv.Demography object.

    Backward-time events ordered from present to deep past. K = pop 0,
    F = pop 1. After K-F join at T_KF_SPLIT, the merged pop continues
    as pop 0 (the ms convention used by msinv).
    """
    d = Demography(pop_sizes=[NE_K_PRESENT, NE_F_PRESENT])

    # ---- K Ne(t): ABC stair-step (t=200 dip dropped per v11 lock) --
    d.add_event(("en", 400.0,   0, 161_546))
    d.add_event(("en", 600.0,   0, 152_453))
    d.add_event(("en", 1_400.0, 0, 174_800))
    d.add_event(("en", 3_000.0, 0, 182_180))
    d.add_event(("en", 6_200.0, 0, 159_861))

    # ---- F Ne(t): ABC stair-step ----------------------------------
    d.add_event(("en", 400.0,   1, 1_157_768))
    d.add_event(("en", 600.0,   1, 205_260))
    d.add_event(("en", 1_000.0, 1, 1_374_810))
    d.add_event(("en", 1_400.0, 1, 674_766))
    d.add_event(("en", 3_000.0, 1, 340_074))
    d.add_event(("en", 6_200.0, 1, NE_F_AT_SPLIT))

    # ---- K-F split: F (pop 1) joins K (pop 0) at T_KF_SPLIT -------
    d.add_event(("ej", float(T_KF_SPLIT), 1, 0))
    d.add_event(("en", float(T_KF_SPLIT), 0, NE_KF_AT_MERGE))

    # ---- KF Ne(t) trajectory (50k bottleneck floor per v11 lock) --
    d.add_event(("en", 13_000.0, 0, 81_072))
    d.add_event(("en", 20_000.0, 0, 95_546))
    d.add_event(("en", 30_000.0, 0, 73_250))
    d.add_event(("en", 40_000.0, 0, 50_000))
    d.add_event(("en", 50_000.0, 0, 50_000))
    d.add_event(("en", 60_000.0, 0, 50_000))
    d.add_event(("en", 70_000.0, 0, 50_000))

    # ---- KF -> Anc rename: deep ancestral Ne change ---------------
    d.add_event(("en", float(T_ANC_RENAME), 0, NE_ANC_DEEP))

    return d
