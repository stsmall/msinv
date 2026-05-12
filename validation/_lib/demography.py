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
T_INV_3RA = 300_000
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
    d.add_event(("en", 400.0, 0, 161_546))
    d.add_event(("en", 600.0, 0, 152_453))
    d.add_event(("en", 1_400.0, 0, 174_800))
    d.add_event(("en", 3_000.0, 0, 182_180))
    d.add_event(("en", 6_200.0, 0, 159_861))

    # ---- F Ne(t): ABC stair-step ----------------------------------
    d.add_event(("en", 400.0, 1, 1_157_768))
    d.add_event(("en", 600.0, 1, 205_260))
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


# --- v12 reference Ne for discoal CLI scaling ---------------------
# discoal (and ms) use 4*N0-scaled times and N0-ratio sizes.
# N0 = present-day Kiribina, matching the convention used in
# kir_fol_demography.py and the existing _discoal_bench_runner.
V12_DISCOAL_N0 = NE_K_PRESENT  # 126,772


def v12_msprime():
    """Build the v12 demography as an msprime.Demography object.

    Same events as v12_msinv() but expressed using msprime's API: named
    populations, population_parameters_change events, and a population
    split (msprime's natural representation of a backward-time join).

    Ne values are 2x the msinv values per the CLAUDE.md convention:
    msinv treats `population_size=N` as diploid Ne (per-pair coal rate
    = 1/(2N)). msprime with ploidy=1 treats N as haploid Ne (per-pair
    coal rate = 1/N). To match msinv's coalescent rates, msprime must
    receive 2*N. Track 3 confirmed without the doubling: msinv tree
    heights were 3x msprime tree heights (msprime's effective Ne was
    half of msinv's). With the doubling, both engines see the same
    per-pair coal rates.
    """
    import msprime
    H = 2.0  # haploid-Ne doubling factor for msprime ploidy=1
    d = msprime.Demography()
    d.add_population(name="K", initial_size=H * NE_K_PRESENT)
    d.add_population(name="F", initial_size=H * NE_F_PRESENT)
    d.add_population(name="KF", initial_size=H * NE_KF_AT_MERGE)
    # ---- K Ne(t) stair-step ----
    d.add_population_parameters_change(
        time=400, population="K", initial_size=H * 161_546)
    d.add_population_parameters_change(
        time=600, population="K", initial_size=H * 152_453)
    d.add_population_parameters_change(
        time=1_400, population="K", initial_size=H * 174_800)
    d.add_population_parameters_change(
        time=3_000, population="K", initial_size=H * 182_180)
    d.add_population_parameters_change(
        time=6_200, population="K", initial_size=H * 159_861)
    # ---- F Ne(t) stair-step ----
    d.add_population_parameters_change(
        time=400, population="F", initial_size=H * 1_157_768)
    d.add_population_parameters_change(
        time=600, population="F", initial_size=H * 205_260)
    d.add_population_parameters_change(
        time=1_000, population="F", initial_size=H * 1_374_810)
    d.add_population_parameters_change(
        time=1_400, population="F", initial_size=H * 674_766)
    d.add_population_parameters_change(
        time=3_000, population="F", initial_size=H * 340_074)
    d.add_population_parameters_change(
        time=6_200, population="F", initial_size=H * NE_F_AT_SPLIT)
    # ---- K-F split: K and F merge backward into KF ----
    d.add_population_split(
        time=T_KF_SPLIT, derived=["K", "F"], ancestral="KF")
    # ---- KF Ne(t) trajectory ----
    d.add_population_parameters_change(
        time=13_000, population="KF", initial_size=H * 81_072)
    d.add_population_parameters_change(
        time=20_000, population="KF", initial_size=H * 95_546)
    d.add_population_parameters_change(
        time=30_000, population="KF", initial_size=H * 73_250)
    d.add_population_parameters_change(
        time=40_000, population="KF", initial_size=H * 50_000)
    d.add_population_parameters_change(
        time=50_000, population="KF", initial_size=H * 50_000)
    d.add_population_parameters_change(
        time=60_000, population="KF", initial_size=H * 50_000)
    d.add_population_parameters_change(
        time=70_000, population="KF", initial_size=H * 50_000)
    # ---- KF -> Anc deep change ----
    d.add_population_parameters_change(
        time=T_ANC_RENAME, population="KF", initial_size=H * NE_ANC_DEEP)
    # add_population_split internally appends activation/deactivation events
    # that can land out of time-sorted order relative to the manually added
    # stair-steps.  sort_events() is the msprime-documented fix.
    d.sort_events()
    return d


def v12_discoal_events():
    """Build the v12 demography as discoal CLI argument tokens.

    Returns the list of ms-style argument tokens for the v12 events.
    Caller is responsible for prepending `sampleSize numReplicates nSites`
    and any sweep / rate arguments.

    Scaling: times divided by 4*N0, sizes divided by N0, where
    N0 = V12_DISCOAL_N0 = NE_K_PRESENT.
    """
    N0 = V12_DISCOAL_N0

    def t(gens):
        return f"{gens / (4.0 * N0):.10g}"

    def sz(ne):
        return f"{ne / N0:.10g}"

    # discoal uses 0-indexed pop IDs (same as msinv); -ed maps to ms -ej.
    args: list[str] = [
        # K Ne stair-step (population 0)
        "-en", t(400.0),   "0", sz(161_546),
        "-en", t(600.0),   "0", sz(152_453),
        "-en", t(1_400.0), "0", sz(174_800),
        "-en", t(3_000.0), "0", sz(182_180),
        "-en", t(6_200.0), "0", sz(159_861),
        # F Ne stair-step (population 1)
        "-en", t(400.0),   "1", sz(1_157_768),
        "-en", t(600.0),   "1", sz(205_260),
        "-en", t(1_000.0), "1", sz(1_374_810),
        "-en", t(1_400.0), "1", sz(674_766),
        "-en", t(3_000.0), "1", sz(340_074),
        "-en", t(6_200.0), "1", sz(NE_F_AT_SPLIT),
        # K-F split: -ed merges pop 1 into pop 0 (backward)
        "-ed", t(T_KF_SPLIT), "1", "0",
        # KF starts at NE_KF_AT_MERGE
        "-en", t(T_KF_SPLIT), "0", sz(NE_KF_AT_MERGE),
        # KF stair-step
        "-en", t(13_000.0), "0", sz(81_072),
        "-en", t(20_000.0), "0", sz(95_546),
        "-en", t(30_000.0), "0", sz(73_250),
        "-en", t(40_000.0), "0", sz(50_000),
        "-en", t(50_000.0), "0", sz(50_000),
        "-en", t(60_000.0), "0", sz(50_000),
        "-en", t(70_000.0), "0", sz(50_000),
        # Anc deep change
        "-en", t(T_ANC_RENAME), "0", sz(NE_ANC_DEEP),
    ]
    return args


# ====================================================================
# v_simple — 2-population split with optional migration. Used by Tracks
# 1, 2_a/b, 3_a/b, 4, 6.
#
# Single biological "fact": a t_split = 15,000 gen split with N_anc =
# 1e6 ancestrally and asymmetric present-day Ne (1e6 in pop 0, 1e5 in
# pop 1). No stair-steps. Migration symmetric, on or off.
# ====================================================================

V_SIMPLE_T_SPLIT = 15_000.0
V_SIMPLE_T_INV = 30_000.0  # older than t_split; inversion is ancestral
V_SIMPLE_NE_ANC = 100_000
V_SIMPLE_NE_ANC_BN = 10_000  # Option B: bottlenecked ancestor (10× smaller)
V_SIMPLE_NE_POP0 = 100_000
V_SIMPLE_NE_POP1 = 10_000
V_SIMPLE_M_DEFAULT = 1.0e-5  # symmetric per-generation rate; 4Nm≈4 in pop0

# Reference N0 for discoal CLI scaling — match pop 0 present-day Ne so
# the "big" pop is the unit baseline.
V_SIMPLE_DISCOAL_N0 = V_SIMPLE_NE_POP0  # 100,000


def v_simple_msinv(
    *,
    two_pop: bool = True,
    migration: float = 0.0,
    bottleneck: bool = False,
) -> Demography:
    """v_simple msinv.Demography.

    1-pop mode (``two_pop=False``): single population, Ne = 1e5
    constant, no split, no migration (Track 1).

    2-pop mode (``two_pop=True``): pop 0 (Ne=1e5) and pop 1 (Ne=1e4)
    join backward at ``V_SIMPLE_T_SPLIT`` into ancestral pop. Ancestral
    Ne is 1e5 by default (Option A); pass ``bottleneck=True`` to use
    a tighter ancestral Ne of 1e4 (Option B — narrower founder, faster
    deep coalescence).

    ``migration`` is the symmetric per-generation rate; 0 ⇒ strict
    isolation (sub-scenarios `_a`). msinv's matrix M[i][j] is "fraction
    of pop i absorbing from pop j" per generation.
    """
    if not two_pop:
        return Demography(pop_sizes=[V_SIMPLE_NE_POP0])

    ne_anc = V_SIMPLE_NE_ANC_BN if bottleneck else V_SIMPLE_NE_ANC
    if migration > 0.0:
        mig = [[0.0, float(migration)], [float(migration), 0.0]]
    else:
        mig = None
    d = Demography(
        pop_sizes=[V_SIMPLE_NE_POP0, V_SIMPLE_NE_POP1],
        migration_matrix=mig,
    )
    d.add_event(("ej", V_SIMPLE_T_SPLIT, 1, 0))
    d.add_event(("en", V_SIMPLE_T_SPLIT, 0, ne_anc))
    if migration > 0.0:
        d.add_event(("em", V_SIMPLE_T_SPLIT, 0, 1, 0.0))
        d.add_event(("em", V_SIMPLE_T_SPLIT, 1, 0, 0.0))
    return d


def v_simple_msprime(
    *,
    two_pop: bool = True,
    migration: float = 0.0,
    bottleneck: bool = False,
):
    """v_simple as an msprime.Demography (ploidy=1 conventions).

    Ne values doubled to match msinv's diploid-Ne convention (see
    CLAUDE.md: msprime ploidy=1 needs 2·N to give same per-pair coal
    rate as msinv's `population_size`).

    ``bottleneck=True`` selects Option B (N_anc = 1e4 instead of 1e5).
    """
    import msprime
    H = 2.0  # haploid-Ne doubling for msprime ploidy=1
    d = msprime.Demography()
    if not two_pop:
        d.add_population(name="pop0", initial_size=H * V_SIMPLE_NE_POP0)
        return d
    ne_anc = V_SIMPLE_NE_ANC_BN if bottleneck else V_SIMPLE_NE_ANC
    d.add_population(name="pop0", initial_size=H * V_SIMPLE_NE_POP0)
    d.add_population(name="pop1", initial_size=H * V_SIMPLE_NE_POP1)
    d.add_population(name="anc", initial_size=H * ne_anc)
    if migration > 0.0:
        d.set_symmetric_migration_rate(
            populations=["pop0", "pop1"], rate=float(migration),
        )
    d.add_population_split(
        time=V_SIMPLE_T_SPLIT, derived=["pop0", "pop1"], ancestral="anc",
    )
    d.sort_events()
    return d


def v_simple_discoal_events(
    *,
    two_pop: bool = True,
    migration: float = 0.0,
    bottleneck: bool = False,
) -> list[str]:
    """v_simple as discoal CLI argument tokens.

    Scaling: times divided by 4*N0, sizes divided by N0, where
    N0 = V_SIMPLE_DISCOAL_N0 = pop 0 present Ne. Caller prepends
    ``-p N sample_0 sample_1`` etc.

    ``bottleneck=True`` selects Option B (N_anc = 1e4 instead of 1e5).
    """
    N0 = V_SIMPLE_DISCOAL_N0
    ne_anc = V_SIMPLE_NE_ANC_BN if bottleneck else V_SIMPLE_NE_ANC

    def t(gens):
        return f"{gens / (4.0 * N0):.10g}"

    def sz(ne):
        return f"{ne / N0:.10g}"

    if not two_pop:
        return []

    args: list[str] = [
        # Pop 0 starts at sz(NE_POP0)=1.0 (no -en at t=0 needed).
        # Pop 1 starts at sz(NE_POP1)=0.1.
        "-en", "0", "1", sz(V_SIMPLE_NE_POP1),
    ]
    if migration > 0.0:
        m_scaled = f"{4.0 * N0 * migration:.10g}"
        args += [
            "-m", "0", "1", m_scaled,
            "-m", "1", "0", m_scaled,
        ]
    args += [
        "-ed", t(V_SIMPLE_T_SPLIT), "1", "0",
        "-en", t(V_SIMPLE_T_SPLIT), "0", sz(ne_anc),
    ]
    return args
