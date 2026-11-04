"""One ABC row: a simulation of a precomputed 2a trajectory, plus its neutral control, to ``STAT_KEYS``.

Engines (``simulate_stats(..., engine="auto")``):

  * No-flux rows (c = ``params["gamma_eff"]`` NaN): msprime, exact Hudson (``inversion_demography``). Two
    populations I (2a) and S (2+) of p(t) N(t) and (1 - p(t)) N(t) diploids from the present back to t_inv, where
    they merge (``add_population_split``) into an ancestral A of N(t), the growth demography. Without flux the
    arrangements never exchange material, so this is the exact inversion-body genealogy; the margins do not
    affect the body, and the sequence is the body alone (BODY_BP, body window (0, BODY_BP)). This replaced msinv
    for these rows because msinv's barrier-era coalescence (partial merges) under-estimates within-arrangement
    short-range LD about 3x (ld-debug-report.md); its pi / dxy / SFS are correct.
  * Flux rows (c finite): msinv, which has the gene-conversion flux. A BODY_BP inversion body between two
    MARGIN_BP collinear margins; body statistics use only the body.
  * The neutral control is ALWAYS msprime (both kinds of rows): a separate no-inversion CONTROL_BP simulation
    with the same growth demography, sample size, REC_RATE and MU (validated against msinv's no-inversion engine,
    slim_validation scenario illex_growth_noinv), its samples split by index into the two groups (first N_STD
    as S, the next N_INV as I). This mirrors the data's control (chr2 collinear sequence far from the
    breakpoints); the linked margins are not used as the control.

``engine="msprime"`` / ``"msinv"`` force the inversion engine (validation only; msprime refuses a flux row).

msprime conventions: ``ploidy=2`` with diploid sizes, so a pair of lineages coalesces at 1 / (2N) per
generation, and 100 haploid sample chromosomes per class. I and S sizes are piecewise constant on the path's
own time grid, refined so that no interval exceeds MAX_EPOCH generations or a SIZE_STEP-fold size change; each
interval takes the GEOMETRIC MIDPOINT sqrt(size(start) * size(end)) of its end-point sizes, floored at
SIZE_FLOOR. p is interpolated linearly in p between path points (msinv's precomputed-trajectory semantics),
except over a hard start's establishment interval (first copy 1 / 2N at t_inv to ESTABLISHED copies, two path
points), where it is interpolated log-linearly (expected exponential growth from one copy); the I size then
shrinks to ~1 diploid at t_inv, which forces the I lineages to coalesce before t_inv (the 2a TMRCA is <= t_inv;
tested). A soft start's unsorted I lineages
simply join A at t_inv, matching the msinv semantics. A follows N(t) exactly (exponential growth then N_ANC).

Flux: the prior variable c (``params["gamma_eff"]``) is the per-site conversion probability per generation in
heterokaryotypes (the Korunes & Noor quantity). msinv's gamma * phi(x) is a per-bp INITIATION rate of tracts of mean
TRACT_BP (event rate gamma * p_other * integral(phi) * L, ``msinv/hull/simulator.py``), with phi ~ w/(1-w) inside
the body (``_phi``) and w = TRACT_BP / BODY_BP. So the initiation rate is c / TRACT_BP and
gamma = c / TRACT_BP * (1 - w) / w. c NaN (no-flux model) uses GAMMA_FLOOR when msinv is forced.

One integer seed drives everything (both ancestry runs, both mutation drops, the diploid pairings, the LD genotype
errors, p_obs): the same 9 SeedSequence words whichever engine runs. LD diploids carry the data's genotype error
model (``illex.abc.error_model``): body rates in the body, control rates in the control.
"""
from __future__ import annotations

import math

import msprime
import numpy as np

from illex.abc import statvec as SV
from illex.abc.error_model import LD_ERROR
from illex.abc.constants import (BND_MARGIN_BP, BODY_BP, CONTROL_BP, GAMMA_FLOOR, MARGIN_BP, MU, N_INV, N_KARYO,
                                 N_STD, REC_RATE, TRACT_BP)

ENGINES = ("auto", "msprime", "msinv")
MAX_EPOCH = 1000.0      # generations; the longest constant-size interval of the msprime I/S demography
SIZE_STEP = 1.1         # largest size ratio across one interval (refines the hard-start establishment section)
SIZE_FLOOR = 0.5        # diploids; the smallest I/S size (p = 1 / 2N gives N p = 0.5)


def gamma_from_c(c: float) -> float:
    """msinv gene_conversion_rate for per-site conversion probability c (per generation, in heterokaryotypes)."""
    if c is None or not np.isfinite(c):
        return GAMMA_FLOOR
    w = TRACT_BP / BODY_BP
    return float(c) / TRACT_BP * (1.0 - w) / w


def _mutate(ts, seed):
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed, model=msprime.BinaryMutationModel(),
                                 discrete_genome=False)


def _inversion_ts(path, seed, margin=MARGIN_BP):
    """(sim, ts) from msinv, a BODY_BP inversion between two ``margin`` bp collinear margins."""
    from msinv import HullSimulator, InversionSpec

    from illex.demography import PRESENT_NE_GROWTH, growth_demography
    from illex.theory import N_growth

    t_inv = float(path.params["t_inv"])
    spec = InversionSpec(bp_left=margin, bp_right=margin + BODY_BP,
                         gene_conversion_rate=gamma_from_c(path.params.get("gamma_eff", float("nan"))),
                         mean_tract_length=TRACT_BP, tract_distribution="geometric",
                         trajectory={"type": "precomputed", "times": np.asarray(path.times, float),
                                     "freqs": [np.asarray(path.freqs, float)],
                                     "n_e": [float(N_growth(t_inv))], "t_inv": [t_inv]})
    sim = HullSimulator(n_std=N_STD, n_inv=N_INV, population_size=PRESENT_NE_GROWTH,
                        demography=growth_demography(), sequence_length=BODY_BP + 2 * margin,
                        recombination_rate=REC_RATE, inversions=[spec], seed=seed)
    return sim, sim.simulate()


def engine_for(path, engine: str = "auto") -> str:
    """The inversion engine for ``path``: msprime when c (``gamma_eff``) is NaN, msinv when it is finite."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    c = path.params.get("gamma_eff", float("nan"))
    flux = c is not None and np.isfinite(c)
    if engine == "auto":
        return "msinv" if flux else "msprime"
    if engine == "msprime" and flux:
        raise ValueError("the msprime engine has no flux; a finite gamma_eff row needs msinv")
    return engine


def _epochs(path):
    """(edges, p_edges): the refined time grid 0 = t_0 < ... < t_K = t_inv and p at each edge."""
    t_inv = float(path.params["t_inv"])
    T = np.asarray(path.times, float)
    F = np.asarray(path.freqs, float)
    keep = T <= t_inv
    T, F = T[keep], F[keep]
    if T[-1] < t_inv:                       # a path that stops short of t_inv: hold its oldest frequency
        T, F = np.append(T, t_inv), np.append(F, F[-1])
    hard = bool(path.params.get("hard", False))
    edges, pe = [T[0]], [F[0]]
    for j in range(T.size - 1):
        a, b, fa, fb = T[j], T[j + 1], F[j], F[j + 1]
        if b <= a:
            continue
        loglin = hard and j == T.size - 2 and fa > 0 and fb > 0       # the establishment interval
        lr = max(abs(np.log(max(fb, 1e-300) / max(fa, 1e-300))),
                 abs(np.log(max(1 - fb, 1e-300) / max(1 - fa, 1e-300))))
        k = int(max(1, np.ceil((b - a) / MAX_EPOCH), np.ceil(lr / np.log(SIZE_STEP))))
        u = np.linspace(0.0, 1.0, k + 1)[1:]
        edges.extend(a + (b - a) * u)
        pe.extend(np.exp(np.log(fa) + (np.log(fb) - np.log(fa)) * u) if loglin else fa + (fb - fa) * u)
    return np.asarray(edges), np.clip(np.asarray(pe), 0.0, 1.0)


def inversion_demography(path):
    """msprime Demography of the no-flux inversion model for ``path`` (populations I, S, A; diploid sizes)."""
    from illex.theory import ALPHA, N0, N_ANC, N_growth, T_GROW

    t_inv = float(path.params["t_inv"])
    edges, pe = _epochs(path)
    n = np.asarray(N_growth(edges), float)
    size_i = np.maximum(np.sqrt((pe[:-1] * n[:-1]) * (pe[1:] * n[1:])), SIZE_FLOOR)
    size_s = np.maximum(np.sqrt(((1 - pe[:-1]) * n[:-1]) * ((1 - pe[1:]) * n[1:])), SIZE_FLOOR)
    d = msprime.Demography()
    d.add_population(name="I", initial_size=float(size_i[0]))
    d.add_population(name="S", initial_size=float(size_s[0]))
    if t_inv < T_GROW:
        d.add_population(name="A", initial_size=float(N0), growth_rate=float(ALPHA))
        d.add_population_parameters_change(time=float(T_GROW), initial_size=float(N_ANC), growth_rate=0.0,
                                           population="A")
    else:
        d.add_population(name="A", initial_size=float(N_ANC))
    for k in range(1, edges.size - 1):
        d.add_population_parameters_change(time=float(edges[k]), initial_size=float(size_i[k]), population="I")
        d.add_population_parameters_change(time=float(edges[k]), initial_size=float(size_s[k]), population="S")
    d.add_population_split(time=t_inv, derived=["I", "S"], ancestral="A")
    d.sort_events()
    return d


def _inversion_ts_msprime(path, seed):
    """(ts, i_nodes, s_nodes): the no-flux body (BODY_BP), N_INV haploid I and N_STD haploid S samples."""
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(N_INV, population="I", ploidy=1),
                 msprime.SampleSet(N_STD, population="S", ploidy=1)],
        demography=inversion_demography(path), sequence_length=BODY_BP, recombination_rate=REC_RATE,
        ploidy=2, model="hudson", random_seed=int(seed))
    return ts, ts.samples(population=0), ts.samples(population=1)


def _control_ts(seed):
    """msprime no-inversion growth control: N_STD + N_INV haploid samples, CONTROL_BP, ploidy=2 (1 / 2N)."""
    from illex.theory import ALPHA, N0, N_ANC, T_GROW

    d = msprime.Demography()
    d.add_population(name="A", initial_size=float(N0), growth_rate=float(ALPHA))
    d.add_population_parameters_change(time=float(T_GROW), initial_size=float(N_ANC), growth_rate=0.0,
                                       population="A")
    return msprime.sim_ancestry(samples=[msprime.SampleSet(N_STD + N_INV, population="A", ploidy=1)],
                                demography=d, sequence_length=CONTROL_BP, recombination_rate=REC_RATE,
                                ploidy=2, model="hudson", random_seed=int(seed))


def _seed_words(seed):
    """The 9 SeedSequence words of one row (inv, ctl, mut inv, mut ctl, pair inv, pair ctl, p_obs, err inv, err ctl)."""
    ss = np.random.SeedSequence(int(seed))
    return tuple(int(x) for x in ss.generate_state(9) % (2**31 - 1) + 1)


def simulate_stats(path, seed: int, ld_error=LD_ERROR, engine: str = "auto") -> dict[str, float]:
    """``STAT_KEYS`` for one accepted path, plus ``p_obs`` and ``n_nan`` (the count of NaN statistics).

    ``ld_error`` = {"body"|"control": {"I"|"S": (call_rate, h)}} is the data's genotype error model, applied to
    the LD diploids of the body and the control respectively (``illex.abc.error_model``); the default is the
    estimated one. Pass ``{"body": None, "control": None}`` for error-free LD.

    ``engine``: "auto" (msprime for a no-flux row, msinv for a flux row), or "msprime" / "msinv" to force one.

    Never raises on degenerate output (no segregating sites, zero variance): those keys are NaN.
    """
    from illex import stats as S

    # The first 7 words are the Task-3 stream (SeedSequence words are a prefix-stable stream); 8-9 are the LD error
    # draws, added so that every pre-existing draw is unchanged.
    s_inv, s_ctl, s_mi, s_mc, s_pi, s_pc, s_obs, s_ei, s_ec = _seed_words(seed)

    if engine_for(path, engine) == "msprime":
        ts, i_nodes, s_nodes = _inversion_ts_msprime(path, s_inv)
        body_win = (0.0, BODY_BP)
    else:
        sim, ts = _inversion_ts(path, s_inv)
        i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
        body_win = (MARGIN_BP, MARGIN_BP + BODY_BP)
    mts = _mutate(ts, s_mi)
    body = SV.region_stats(mts, i_nodes, s_nodes, body_win, np.random.default_rng(s_pi),
                           ld_err=ld_error["body"], err_rng=np.random.default_rng(s_ei))
    edge = SV.edge_centre(ts, i_nodes, s_nodes, body_win)

    cts = _mutate(_control_ts(s_ctl), s_mc)
    smp = cts.samples()
    control = SV.region_stats(cts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, CONTROL_BP),
                              np.random.default_rng(s_pc), ld_err=ld_error["control"],
                              err_rng=np.random.default_rng(s_ec))

    out = SV.assemble(body, control, edge)
    two_n = 2 * N_KARYO
    out["p_obs"] = float(np.random.default_rng(s_obs).binomial(two_n, float(path.p_now)) / two_n)
    out["n_nan"] = int(sum(math.isnan(out[k]) for k in SV.MODEL_KEYS))  # PPC-only NaNs never flag a row
    return out


# Set to True only once BOTH msinv fixes are in this branch: the barrier coalescence fix (msinv PR #19) and the
# follow-up flank-merge fix (I x S lineage pairs must not merge their shared flank material while each flank is
# still linked to its arrangement; before it, the flanks act panmictic up to the breakpoints). Until then
# boundary_ppc refuses to run unless the caller passes allow_unfixed=True (the tests do).
BND_PPC_READY = False


def boundary_ppc(path, seed: int, engine: str = "msinv", ld_error=LD_ERROR, mask="data",
                 allow_unfixed: bool = False) -> dict:
    """The breakpoint boundary statistic (``boundary.BND_KEYS``) of one path, for posterior predictive checks only.

    msinv only, with a BODY_BP inversion between two BND_MARGIN_BP collinear margins at REC_RATE, the path's
    trajectory and flux (c NaN uses GAMMA_FLOOR), and the seed words of ``simulate_stats`` (ancestry s_inv,
    mutations s_mi, diploid pairing s_pi, genotype errors s_ei). The flanks are collinear, so the genotype errors
    use the control-region parameters ``ld_error["control"]``. msprime is refused because its two-population
    model links the whole sequence to the arrangement, which makes the boundary wrong by construction. So the
    default engine is "msinv", and ``engine_for(path, engine) == "msprime"`` (engine "msprime", or "auto" on a
    no-flux path) raises ValueError.

    ``mask="data"`` (the default) keeps only the simulated sites at the data's accessible offsets from each
    breakpoint (``boundary.load_flank_mask``), so every bin and side has the data's accessible bp. ``mask=None``
    keeps every flank site, and a {"left", "right"} dict is used as given.

    Needs both msinv fixes (the barrier coalescence fix, PR #19, and the follow-up I x S flank-merge fix) before
    it is used for real PPCs. Until BND_PPC_READY is
    True it raises RuntimeError unless ``allow_unfixed=True``.
    """
    from illex import stats as S
    from illex.abc.boundary import boundary_stats, load_flank_mask

    if engine_for(path, engine) == "msprime":
        raise ValueError("boundary_ppc needs msinv: msprime's two-population model links the flanks fully to "
                         "the arrangement")
    if not (BND_PPC_READY or allow_unfixed):
        raise RuntimeError("boundary_ppc needs the msinv barrier coalescence fix (PR #19) and the I x S flank-merge fix, so it runs only "
                           "with allow_unfixed=True until BND_PPC_READY is set")
    if isinstance(mask, str):
        if mask != "data":
            raise ValueError(f"mask must be 'data', None or a dict, got {mask!r}")
        mask = load_flank_mask()
    s_inv, _, s_mi, _, s_pi, _, _, s_ei, _ = _seed_words(seed)
    sim, ts = _inversion_ts(path, s_inv, margin=BND_MARGIN_BP)
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    mts = _mutate(ts, s_mi)
    return boundary_stats(mts, i_nodes, s_nodes, BND_MARGIN_BP, BND_MARGIN_BP + BODY_BP,
                          rng=np.random.default_rng(s_pi), err=ld_error["control"],
                          err_rng=np.random.default_rng(s_ei), mask=mask)
