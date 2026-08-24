"""Balancing selection: the inversion rises under selection, then is held.

WHY THIS MODEL EXISTS, AND WHAT IT REPLACES
-------------------------------------------
Every earlier arm drove the inversion's frequency with msinv's
``deterministic`` logistic from ``p_start`` at ``t_inv`` up to ``p_inv``
today. That trajectory is *still rising at the present moment*: the
inversion spends its whole history on its way to 0.626 and arrives only now.
The selection coefficient it implies is a bookkeeping device for hitting a
target frequency by a target time, not a claim about selection
(``illex/model.py`` says so explicitly).

That is the wrong shape for a balanced polymorphism. Under balancing
selection the inversion rises to an **equilibrium** and then stays there,
so its history has two phases with very different coalescent consequences:

    phase 1  rise      p_start -> p*        duration set by the selection strength
    phase 2  plateau   p == p*              duration t_inv - t_rise

The observable consequences run in opposite directions, which is why this is
worth doing rather than just re-parameterising:

* pi_I/pi_S rises with the plateau's length. A class that has sat at 0.626
  for most of its history carries close to its equilibrium diversity; a class
  that only just arrived carries the bottleneck.
* dxy/pi_I is set mainly by t_inv, since between-class lineages cannot
  coalesce until the origin.

There is a third consequence, which turned out to bind harder than either and
was not anticipated: a LONG plateau also suppresses pi_S, because the standard
arrangement is confined to 1 - p* = 0.374 of the population for that whole
period. Since pi_S is the denominator of the first ratio, a long plateau drives
pi_I/pi_S ABOVE 1 -- the model reaches 1.35 at t_inv = 3e6. The observed 0.744
is therefore evidence that the inversion has NOT been at 0.626 for long on a
coalescent timescale.

THE PREDICTION THIS MAKES, STATED BEFORE IT IS RUN
--------------------------------------------------
The best rising-logistic point (t_inv = 8e5, p_start = 0.15) misses both
targets in OPPOSITE directions -- pi_I/pi_S 0.674 vs 0.744 (-9.4%) and
dxy/pi_I 1.966 vs 1.846 (+6.5%) -- which is the signature of a model-shape
error rather than a scaling error (NOTES sec 7.2). A plateau raises pi_I,
which pushes the first ratio UP and, through the same denominator, the second
ratio DOWN. Both misses should close together. If they do not, this model is
wrong too and the residual is something else.

WHAT IS AND IS NOT IDENTIFIED  (measured 2026-08-07, not assumed)
-----------------------------------------------------------------
My first pass here claimed two free parameters against two targets, hence an
exactly identified point. That was wrong twice over, and the scans that
corrected it are worth recording.

**s_het is not identifiable from its own magnitude.** Over
s_het in [1e-4, 1e-2] at fixed t_inv, the statistics are flat above ~1e-3:
1e-3 and 1e-2 give indistinguishable pi_I/pi_S and dxy/pi_I. Once the rise is
fast relative to t_inv, only its *timing* matters, not its speed. So the
useful parameterisation solves s_het from an arrival condition -- the rise
completes some chosen number of generations before the present -- rather than
fitting it. ``illex/scripts/fit_balancing.py`` does exactly that.

**p_start does NOT collapse to one chromosome.** I predicted that a fast rise
would make a genuine single origin viable, so ``p_start = 1/(2 N(t_inv))``
could replace the phenomenological ~0.15. It does not: at
p_start = 9.1e-7 and t_inv = 8e5 the model gives pi_I/pi_S = 0.60 and
dxy/pi_I = 2.71, and reaching pi_I/pi_S = 0.744 requires t_inv ~ 1.05e6, where
dxy/pi_I ~ 2.8 -- over 50% above target. Strict monophyly caps pi_I at
2 mu t_inv, and that cap binds. The fitted founding frequency is ~0.025, so it
falls 6x from 0.15 but stays a soft origin: at N(7.2e5) ~ 6.9e5 that is
~34,000 founding haplotypes, not one. The caveat in NOTES sec 7.1 stands.

**What IS identified is the age.** Moving the plateau from 0 to 100,000
generations -- the dimension that stays degenerate -- shifts the fitted t_inv
by 1% (726,700 -> 719,900) while p_start moves 22% and s_het 18%. dxy/pi_I
carries the age and is steep in it; pi_I/pi_S carries the founding frequency.
The two are close enough to orthogonal that the age survives the degeneracy.

**The prediction below did hold**, and it is the reason the residual closed:
both misses moved in the predicted directions, together, and overshot at
p_start = 0.15 -- which is what located the fit at a much smaller p_start.

ONE TRAP IN THIS PARAMETERISATION
---------------------------------
"Plateau = 0" does not mean "no plateau". The overdominance curve decelerates
into p* (dp/dt -> 0 as p -> p*), so it is within a few percent of 0.626 for
~1e5 generations before it nominally "arrives" at ``ARRIVAL_TOL``. That is why
this curve at t_inv = 8e5, p_start = 0.15 with plateau = 0 gives
(1.014, 1.504) while the rising *logistic* at the same endpoints gives
(0.674, 1.966): the logistic accelerates through p* instead of settling into
it. The two are not small perturbations of each other, and the difference is
the whole effect being modelled.

THE DYNAMICS
------------
Overdominance, the standard mechanism for a stable intermediate frequency:

    w_II = 1 - s_I        w_IS = 1        w_SS = 1 - s_S

with equilibrium p* = s_S / (s_I + s_S). Fixing p* leaves a single strength
parameter. We use ``s_het`` = s_S, the heterozygote's advantage over the SS
homozygote, which is also the inversion's marginal advantage while it is rare,
so it is the parameter that actually sets the rise rate. Then
s_I = s_het (1 - p*) / p*.

To first order in the selection coefficients the deterministic dynamics
collapse to one line:

    dp/dt = p (1-p) [s_S (1-p) - s_I p]
          = (s_het / p*) p (1-p) (p* - p)

which is integrated in closed form by ``_time_to_reach`` below rather than
iterated, so cost does not scale with t_inv.

CAVEATS THAT TRAVEL WITH ANY RESULT FROM THIS MODULE
----------------------------------------------------
1. **The rise phase is deterministic from p_start.** A newly arisen selected
   variant is in fact drift-dominated until p ~ 1/(2 N s), and this module
   skips that phase (as discoal's ``-ws`` deterministic mode does). At Illex
   sizes the omission is small -- N s_het = 6.8e6 * 1e-4 = 680, so the
   stochastic phase ends at p ~ 7e-4 and occupies a short prefix of the rise
   -- but it is an approximation, and it makes the rise slightly too fast,
   hence t_rise slightly too short and the fitted s_het slightly too small.
2. **Overdominance is one of several mechanisms** that hold a frequency at an
   intermediate value; associative overdominance, frequency-dependent
   selection and antagonistic pleiotropy would all give a plateau. What the
   data can constrain is the *shape* of p(t) -- a fast rise followed by a
   plateau -- not the mechanism producing it. ``s_het`` should be read as "the
   selection strength implied if the mechanism is overdominance".
3. p* is fixed at the observed 0.626 rather than fitted. The model asserts the
   population is AT its equilibrium today. That is the balancing-selection
   hypothesis, not a result of it.
"""
from __future__ import annotations

import numpy as np
from msinv import HullSimulator, InversionSpec

from .demography import PRESENT_NE_GROWTH, growth_demography
from .model import MARGIN_FRACTION, TRACT_FRACTION
from .theory import N_growth, P_I_DEFAULT

# How close to p* counts as "arrived". The approach is logarithmic, so the
# plateau is entered asymptotically and some cutoff is required. 1e-4 of p* is
# far below the resolution of any statistic here.
ARRIVAL_TOL = 1e-4


def s_homozygote(s_het: float, p_star: float = P_I_DEFAULT) -> float:
    """s_I, the inverted homozygote's cost, implied by ``s_het`` and ``p*``.

    From p* = s_S / (s_I + s_S) with s_S = ``s_het``.
    """
    return float(s_het * (1.0 - p_star) / p_star)


def _time_to_reach(p, p_start: float, s_het: float,
                   p_star: float = P_I_DEFAULT):
    """Forward time to go from ``p_start`` to ``p`` under dp/dt above.

    Closed-form integral of  dt = dp / [c p (1-p) (p* - p)],  c = s_het/p*,
    by partial fractions:

        1/[p(1-p)(p*-p)] = (1/p*)/p - (1/(1-p*))/(1-p)
                           + (1/(p*(1-p*)))/(p*-p)

    Vectorised over ``p``. ``p`` must lie in (0, p*).
    """
    p = np.asarray(p, dtype=float)
    c = s_het / p_star
    a = 1.0 / p_star
    b = 1.0 / (1.0 - p_star)
    d = 1.0 / (p_star * (1.0 - p_star))

    def F(x):
        return (a * np.log(x) + b * np.log(1.0 - x)
                - d * np.log(p_star - x))

    return (F(p) - F(p_start)) / c


def rise_time(p_start: float, s_het: float, p_star: float = P_I_DEFAULT,
              tol: float = ARRIVAL_TOL) -> float:
    """Generations from ``p_start`` to within ``tol`` (relative) of ``p*``.

    This is the quantity the fit is really about: it is what divides the
    inversion's history into bottleneck and plateau.
    """
    return float(_time_to_reach(p_star * (1.0 - tol), p_start, s_het, p_star))


def balancing_curve(t_inv: float, s_het: float, p_start: float,
                    p_star: float = P_I_DEFAULT, n_samples: int = 400,
                    tol: float = ARRIVAL_TOL):
    """(times, freqs) in BACKWARD time on [0, t_inv] for msinv.

    ``times[0] = 0`` is the present and ``times[-1] = t_inv`` is the origin,
    matching ``msinv.hull.trajectory_helpers.deterministic_logistic_curve``.

    The sample points are placed on the RISE, not uniformly in time. The
    plateau is constant and needs two points; the rise is where p(t) moves and
    is where msinv's interpolation needs resolution. A uniform time grid at
    t_inv = 1e6 with a 40,000-generation rise would put ~1.6 points in the
    entire rise and turn a gradual sweep into a step function.

    If the rise cannot complete in ``t_inv`` generations the curve is
    truncated: the inversion is still rising at the present, which is the
    old rising-logistic model recovered as a limiting case. Callers that need
    to know should check ``rise_time`` against ``t_inv`` themselves.
    """
    if not 0.0 < p_start < p_star:
        raise ValueError(
            f"p_start ({p_start!r}) must lie in (0, p_star={p_star!r}); "
            "p_start >= p_star has the inversion starting at or above its "
            "equilibrium, which this parameterisation does not describe")
    if s_het <= 0.0:
        raise ValueError(
            f"s_het ({s_het!r}) must be > 0. The neutral case is not a limit "
            "of this trajectory -- a neutral inversion has no equilibrium to "
            "be held at, so p* is undefined. Test neutrality with the "
            "rising-logistic arm in illex.model instead.")

    p_arrive = p_star * (1.0 - tol)
    t_rise = float(_time_to_reach(p_arrive, p_start, s_het, p_star))

    if t_rise >= t_inv:
        # Rise incomplete: solve for the frequency reached by the present, and
        # sample the whole window on the rise.
        p_now = _p_at_forward_time(t_inv, p_start, s_het, p_star)
        p_grid = np.geomspace(p_start, p_now, n_samples)
        t_fwd = _time_to_reach(p_grid, p_start, s_het, p_star)
    else:
        # Geometric in p over the rise: dense where p is small and moving
        # fast in relative terms, which is where the bottleneck is set.
        p_grid = np.geomspace(p_start, p_arrive, n_samples - 1)
        t_fwd = _time_to_reach(p_grid, p_start, s_het, p_star)
        # ... then one point pinning the plateau at the present.
        p_grid = np.append(p_grid, p_arrive)
        t_fwd = np.append(t_fwd, t_inv)

    # Forward time since origin -> backward time before present.
    t_back = t_inv - t_fwd
    order = np.argsort(t_back)
    times = np.clip(t_back[order], 0.0, t_inv)
    freqs = p_grid[order]
    return times.tolist(), freqs.tolist()


def _p_at_forward_time(t, p_start: float, s_het: float,
                       p_star: float = P_I_DEFAULT) -> float:
    """Invert ``_time_to_reach`` by bisection on the monotone t(p)."""
    lo, hi = p_start, p_star * (1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _time_to_reach(mid, p_start, s_het, p_star) < t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def founding_frequency(t_inv: float) -> float:
    """p_start for a genuine single origin: one chromosome at ``t_inv``.

    ``1 / (2 N(t_inv))`` on the growth arm's N(t). Determined by the
    demography, not fitted -- which is the point (see module docstring).
    """
    return float(1.0 / (2.0 * float(N_growth(t_inv))))


def build_balancing_sim(*, seq_length, t_inv, s_het, p_star=P_I_DEFAULT,
                        p_start=None, gamma=1e-15, n_i=100, n_s=100,
                        seed=None, recomb_rate=None, n_samples=400):
    """Growth-arm sim with a rise-then-plateau inversion trajectory.

    Growth arm only: the null has to carry the expansion, and NOTES sec 7.2
    records that the constant-Ne arm's ages do not transfer.

    ``p_start=None`` uses ``founding_frequency(t_inv)`` -- a true single origin.
    ``recomb_rate=None`` uses the measured chr2 collinear male rate.
    """
    if recomb_rate is None:
        from .slim.config import REC_RATE
        recomb_rate = REC_RATE
    if p_start is None:
        p_start = founding_frequency(t_inv)

    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    tract_length = max(1.0, (bp_right - bp_left) * TRACT_FRACTION)

    times, freqs = balancing_curve(t_inv, s_het, p_start, p_star,
                                   n_samples=n_samples)
    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        gene_conversion_rate=gamma,
        mean_tract_length=tract_length,
        tract_distribution="geometric",
        trajectory={
            "type": "precomputed",
            "times": times,
            "freqs": [freqs],
            # n_e is used by PrecomputedTrajectory ONLY to auto-detect t_inv
            # from where p falls to 1/(2 n_e); passing t_inv explicitly makes
            # it inert. Kept faithful anyway so it is not misleading.
            "n_e": [float(N_growth(t_inv))],
            "t_inv": [float(t_inv)],
        },
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=PRESENT_NE_GROWTH,
        demography=growth_demography(),
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )


# =====================================================================
# The neutral alternative, in closed form
# =====================================================================
# These two functions are the quantitative case against neutrality, and they
# are calibration-free: no mu, no accessibility mask, no absolute diversity.
# Both are exact diffusion results, verified against Wright-Fisher simulation
# (2026-08-07, N = 200/500/1000, 1-2 M replicates each: probabilities matched
# to 3%, conditional times to 0.7-3.0%, converging as N grows -- the residual
# is the usual discrete-to-diffusion correction).

def neutral_reach_probability(p_star: float = P_I_DEFAULT,
                              n_e: float | None = None) -> float:
    """P(a single new neutral inversion ever reaches ``p_star``).

    For a neutral allele the frequency is a martingale, so the probability of
    ever reaching x starting from p0 is simply p0/x (gambler's ruin on the
    diffusion). With p0 = 1/(2 N_e) this is 1/(2 N_e x).
    """
    n_e = PRESENT_NE_GROWTH if n_e is None else n_e
    return float(1.0 / (2.0 * n_e * p_star))


def neutral_hitting_time(p_star: float = P_I_DEFAULT,
                         n_e: float | None = None) -> float:
    """E[generations to reach ``p_star`` | it gets there], neutral, from p0->0.

    Solving  (p(1-p)/4N) T1'' = -p/x  with T1(0) = T1(x) = 0 for the
    probability-weighted mean absorption time, then dividing by the hitting
    probability p/x and taking p -> 0:

        E[t | hit x] = (4 N / x) [ x + (1-x) ln(1-x) ]

    At x = 0.626 the bracket is 0.2583, so this is **1.650 N generations**.

    This is the number that makes neutrality quantitatively untenable, and it
    does not depend on mu. It is also generous to the neutral hypothesis in
    three ways: it conditions on the ~1e-7 event of succeeding at all, it uses
    the *mean* rather than a lower quantile, and evaluating it at the smallest
    Ne on the growth arm (N_ANC) understates the true timescale, since the rise
    would mostly occur at larger N.
    """
    n_e = PRESENT_NE_GROWTH if n_e is None else n_e
    return float((4.0 * n_e / p_star)
                 * (p_star + (1.0 - p_star) * np.log(1.0 - p_star)))


# =====================================================================
# Three-phase family with an EXPLICIT arrival time
# =====================================================================
# WHY THIS EXISTS
# ---------------
# The two-phase family above cannot express "arrived recently". Its rise obeys
# dp/dt = (s_het/p*) p(1-p)(p*-p), whose approach to p* is exponential with rate
# proportional to s_het -- the same s_het that sets the take-off. So a rise slow
# enough to span t_inv is also an approach slow enough to crawl: measured on this
# curve, **70.3% of the rise is spent between 0.90 p* and arrival, and that
# fraction is scale-invariant in s_het**. At the fitted s_het = 3.58e-5 the
# frequency is above 0.563 for 511,600 of its 727,600 generations. "plateau = 0"
# is therefore not "no plateau"; the trajectory has effectively been at
# equilibrium for most of its history no matter what that parameter says.
#
# That is the shape the ANGSD/GL spectrum rejects (NOTES sec 8.5.3): the fitted
# points over-predict the inverted-vs-standard singleton skew (ratio 1.31 against
# an observed 1.211) because they keep the standard arrangement confined to
# 1 - p* = 0.374 for too long. The data want the confinement to be more recent.
#
# THE FAMILY
# ----------
# Three phases, with the arrival time as a free parameter rather than a
# consequence of s_het:
#
#     [t_inv, t_arrive + t_rise]   dormant   p = p_start
#     [t_arrive + t_rise, t_arrive] rise      p_start -> p*   (overdominance ODE)
#     [t_arrive, 0]                plateau   p = p*
#
# so ``t_arrive`` is exactly what it says: generations before the present at
# which the inversion reached its equilibrium frequency. ``t_rise`` is given
# directly and s_het is *derived* from it, which is the inversion of the old
# parameterisation and the whole point -- the rise can now be made fast and late
# instead of slow and early.
#
# Mechanistically this is a soft sweep from standing variation: the inversion
# segregated at low frequency, then became advantageous and swept to a new
# balanced equilibrium. It nests the old family as the special case
# t_arrive = 0, t_rise = t_inv.
#
# WHY IT SHOULD RESOLVE THE TENSION
# ---------------------------------
# The three statistics separate onto the three parameters much more cleanly than
# before, which is what makes this worth doing rather than just re-parameterising
# again:
#   dxy/pi_I          <- t_inv       (between-class lineages cannot coalesce
#                                     before the origin, whatever p did after)
#   pi_I/pi_S         <- p_start and the dormancy length (the squeeze on the
#                                     inverted class while it was rare)
#   SFS I-S contrast  <- t_arrive    (how long the standard class has been
#                                     confined to 0.374)
#
# CAVEATS
# -------
# 1. **Dormancy is held at a constant frequency, not drifting.** A real standing
#    variant would wander. Holding it fixed gives the inverted class a constant
#    coalescent size 2 N(t) p_start during dormancy, which for p_start = 0.025 is
#    a real and possibly strong squeeze -- long dormancy will depress pi_I hard.
#    That is a modelling choice, and it is the parameter the fit is most likely
#    to push against.
# 2. The corners are sharp. msinv's PrecomputedTrajectory interpolates linearly
#    between samples, so ``arrival_curve`` places points densely on the rise and
#    pins both corners exactly; a uniform time grid would round them off and
#    quietly reintroduce the smearing this family exists to remove.
# 3. p* is still asserted at the observed 0.626, not inferred.


def s_het_for_rise(t_rise: float, p_start: float,
                   p_star: float = P_I_DEFAULT,
                   tol: float = ARRIVAL_TOL) -> float:
    """s_het whose overdominance rise from ``p_start`` takes ``t_rise``.

    The inverse of ``rise_time``, which is monotone decreasing in s_het.
    """
    from scipy import optimize
    if t_rise <= 0:
        raise ValueError(f"t_rise must be > 0, got {t_rise!r}")
    return float(optimize.brentq(
        lambda s: rise_time(p_start, s, p_star, tol) - t_rise,
        1e-9, 10.0, rtol=1e-12))


def arrival_curve(t_inv: float, t_arrive: float, t_rise: float,
                  p_start: float, p_star: float = P_I_DEFAULT,
                  n_rise: int = 300, tol: float = ARRIVAL_TOL):
    """(times, freqs) in BACKWARD time on [0, t_inv], three phases.

    ``t_arrive``: generations before present at which p* is reached.
    ``t_rise``:   duration of the sweep from ``p_start`` to p*.
    Requires ``t_arrive + t_rise <= t_inv``; the remainder is dormancy at
    ``p_start``.
    """
    if not 0.0 < p_start < p_star:
        raise ValueError(
            f"p_start ({p_start!r}) must lie in (0, p_star={p_star!r})")
    if t_arrive < 0:
        raise ValueError(f"t_arrive must be >= 0, got {t_arrive!r}")
    t_dorm = t_inv - t_arrive - t_rise
    if t_dorm < 0:
        raise ValueError(
            f"t_arrive ({t_arrive!r}) + t_rise ({t_rise!r}) exceeds t_inv "
            f"({t_inv!r}): the rise cannot start before the inversion exists")

    s_het = s_het_for_rise(t_rise, p_start, p_star, tol)
    p_arrive = p_star * (1.0 - tol)

    # Rise sampled geometrically in p: dense where p is small and moving fast in
    # relative terms, which is where the bottleneck on the inverted class is set.
    p_grid = np.geomspace(p_start, p_arrive, n_rise)
    t_fwd = _time_to_reach(p_grid, p_start, s_het, p_star)   # 0 .. t_rise
    t_back_rise = t_arrive + (t_rise - t_fwd)                # t_arrive .. +t_rise

    times = [0.0, t_arrive]
    freqs = [p_arrive, p_arrive]
    order = np.argsort(t_back_rise)
    times.extend(t_back_rise[order].tolist())
    freqs.extend(p_grid[order].tolist())
    if t_dorm > 0:
        times.append(t_inv)
        freqs.append(p_start)

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    # Strictly increasing times: msinv rejects duplicates, and the corners can
    # collide when t_arrive == 0 or t_rise is tiny.
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, freqs = times[keep], freqs[keep]
    times = np.clip(times, 0.0, t_inv)
    return times.tolist(), freqs.tolist(), s_het


def build_arrival_sim(*, seq_length, t_inv, t_arrive, t_rise,
                      p_start, p_star=P_I_DEFAULT, gamma=1e-15,
                      n_i=100, n_s=100, seed=None, recomb_rate=None,
                      n_rise=300):
    """Growth-arm sim with the three-phase explicit-arrival trajectory."""
    if recomb_rate is None:
        from .slim.config import REC_RATE
        recomb_rate = REC_RATE

    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    tract_length = max(1.0, (bp_right - bp_left) * TRACT_FRACTION)

    times, freqs, _s = arrival_curve(t_inv, t_arrive, t_rise, p_start,
                                     p_star, n_rise=n_rise)
    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        gene_conversion_rate=gamma,
        mean_tract_length=tract_length,
        tract_distribution="geometric",
        trajectory={
            "type": "precomputed",
            "times": times,
            "freqs": [freqs],
            "n_e": [float(N_growth(t_inv))],
            "t_inv": [float(t_inv)],
        },
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=PRESENT_NE_GROWTH,
        demography=growth_demography(),
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )


# =====================================================================
# Drifting dormancy
# =====================================================================
# ``arrival_curve`` holds the dormancy phase at a CONSTANT frequency. That was
# flagged as the strongest assumption in the family, and the scale check says it
# is not a small one: over a 650,000-generation dormancy on the growth arm the
# neutral drift standard deviation at p = 0.32 is **0.275** -- comparable to the
# frequency itself. A real standing variant wanders that far.
#
# It also matters in a specific, one-directional way. The coalescent rate inside
# the inverted class is 1/(2 N(t) p(t)), so what π_I integrates is
# ∫ dt / (2 N p). Because the integrand goes as 1/p it is dominated by whatever
# time the path spends at LOW frequency, and by Jensen's inequality a wandering
# path accumulates strictly more coalescence than a constant path at the same
# mean frequency. So drift should SQUEEZE the inverted class harder, and the fit
# should respond by wanting a higher handoff frequency, a shorter dormancy, or
# both. That prediction is stated here before it is run.
#
# WHAT THIS BUYS BACK: a genuine single origin becomes expressible again. The
# fixed-frequency family could not start from one chromosome, because a constant
# p = 1/(2N) for hundreds of thousands of generations annihilates π_I. A drifting
# path can start at one copy and climb, so ``p_origin`` defaults to 1/(2 N(t_inv))
# and the soft-origin conclusion of NOTES sec 7.5.1 can be re-examined rather
# than assumed.
#
# THE APPROXIMATION, STATED PLAINLY
# ---------------------------------
# This is a **guided (Durham-Gallant modified) diffusion bridge**, not an exact
# Wright-Fisher bridge: the volatility is the WF one, sqrt(p(1-p)/(2N(t))), but
# the drift is the Brownian bridge's linear guiding term (p_hand - p)/(T - tau),
# which is exact for Brownian motion and standard practice for state-dependent
# volatility. It reproduces the right endpoints and the right order of variance;
# it does not reproduce the exact law of the conditioned ancestral frequency
# process. Doing that properly means the WF bridge's true conditioned drift,
# which is not worth it here given that ``p_hand`` and ``t_arrive`` are being
# fitted anyway.
#
# Two further choices worth seeing:
#   * The path is floored at 1/(2 N(t)) -- one copy. Below that the arrangement
#     is lost, and it is not lost, so conditioning on non-loss is correct rather
#     than optional.
#   * Every replicate draws its OWN path, so the simulated ensemble carries the
#     drift variance instead of averaging it away into a single mean trajectory.


def dormancy_bridge(t_inv: float, t_hand: float, p_hand: float,
                    rng, p_origin: float | None = None,
                    n_steps: int = 800):
    """Drifting dormancy path, as (backward times, freqs), oldest last.

    Guided WF diffusion bridge in forward time from ``p_origin`` at ``t_inv``
    to ``p_hand`` at ``t_hand`` (both are BACKWARD times, so
    ``t_inv > t_hand``). See the module notes above for the approximation.

    ``p_origin=None`` means a genuine single origin, 1/(2 N(t_inv)).
    """
    T = float(t_inv) - float(t_hand)
    if T <= 0:
        raise ValueError(
            f"dormancy length must be > 0; t_inv={t_inv!r} t_hand={t_hand!r}")
    if p_origin is None:
        p_origin = 1.0 / (2.0 * float(N_growth(t_inv)))

    k = max(50, int(n_steps))
    dt = T / k
    p = float(p_origin)
    taus = [0.0]
    ps = [p]
    for i in range(k):
        tau = i * dt
        remaining = T - tau
        t_back = t_inv - tau
        n_e = float(N_growth(t_back))
        floor = 1.0 / (2.0 * n_e)
        guide = (p_hand - p) / remaining * dt
        sd = float(np.sqrt(max(p * (1.0 - p), 0.0) / (2.0 * n_e) * dt))
        p = p + guide + sd * float(rng.standard_normal())
        p = min(max(p, floor), 1.0 - floor)
        taus.append(tau + dt)
        ps.append(p)
    ps[-1] = float(p_hand)          # land exactly on the handoff
    # forward tau since t_inv -> backward time before present
    times = [t_inv - x for x in taus]
    return times, ps


def arrival_curve_drift(t_inv: float, t_arrive: float, t_rise: float,
                        p_hand: float, rng, p_star: float = P_I_DEFAULT,
                        p_origin: float | None = None,
                        n_rise: int = 300, n_dorm: int = 800,
                        tol: float = ARRIVAL_TOL):
    """As ``arrival_curve`` but with a DRIFTING dormancy phase.

    ``p_hand`` is the frequency at which the sweep takes over -- the drifting
    analogue of ``arrival_curve``'s ``p_start``. Returns
    ``(times, freqs, s_het)`` with ``times`` increasing from 0 (present) to
    ``t_inv``.
    """
    if not 0.0 < p_hand < p_star:
        raise ValueError(
            f"p_hand ({p_hand!r}) must lie in (0, p_star={p_star!r})")
    t_hand = t_arrive + t_rise
    if t_hand > t_inv:
        raise ValueError(
            f"t_arrive ({t_arrive!r}) + t_rise ({t_rise!r}) exceeds t_inv "
            f"({t_inv!r})")

    s_het = s_het_for_rise(t_rise, p_hand, p_star, tol)
    p_arrive = p_star * (1.0 - tol)

    # plateau, then the rise (geometric in p, as in arrival_curve)
    p_grid = np.geomspace(p_hand, p_arrive, n_rise)
    t_fwd = _time_to_reach(p_grid, p_hand, s_het, p_star)
    t_back_rise = t_arrive + (t_rise - t_fwd)
    order = np.argsort(t_back_rise)

    times = [0.0, t_arrive]
    freqs = [p_arrive, p_arrive]
    times.extend(t_back_rise[order].tolist())
    freqs.extend(p_grid[order].tolist())

    if t_inv - t_hand > 0:
        d_times, d_freqs = dormancy_bridge(
            t_inv, t_hand, p_hand, rng, p_origin=p_origin, n_steps=n_dorm)
        # dormancy_bridge returns oldest-last in backward time; it starts at
        # t_inv and ends at t_hand, so reverse it to keep times increasing.
        times.extend(d_times[::-1])
        freqs.extend(d_freqs[::-1])

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    srt = np.argsort(times, kind="stable")
    times, freqs = times[srt], freqs[srt]
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, freqs = times[keep], freqs[keep]
    times = np.clip(times, 0.0, t_inv)
    return times.tolist(), freqs.tolist(), s_het


def build_arrival_drift_sim(*, seq_length, t_inv, t_arrive, t_rise, p_hand,
                            drift_seed, p_star=P_I_DEFAULT, p_origin=None,
                            gamma=1e-15, n_i=100, n_s=100, seed=None,
                            recomb_rate=None, n_rise=300, n_dorm=800):
    """Three-phase sim with a DRIFTING dormancy phase.

    ``drift_seed`` is separate from ``seed`` on purpose: the coalescent seed and
    the trajectory seed are different sources of randomness, and keeping them
    separate makes it possible to hold one fixed while varying the other.
    """
    if recomb_rate is None:
        from .slim.config import REC_RATE
        recomb_rate = REC_RATE

    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    tract_length = max(1.0, (bp_right - bp_left) * TRACT_FRACTION)

    rng = np.random.default_rng(drift_seed)
    times, freqs, _s = arrival_curve_drift(
        t_inv, t_arrive, t_rise, p_hand, rng, p_star=p_star,
        p_origin=p_origin, n_rise=n_rise, n_dorm=n_dorm)
    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        gene_conversion_rate=gamma,
        mean_tract_length=tract_length,
        tract_distribution="geometric",
        trajectory={
            "type": "precomputed",
            "times": times,
            "freqs": [freqs],
            "n_e": [float(N_growth(t_inv))],
            "t_inv": [float(t_inv)],
        },
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=PRESENT_NE_GROWTH,
        demography=growth_demography(),
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )


# =====================================================================
# Non-neutral dormancy: a balanced polymorphism before the sweep
# =====================================================================
# The drifting dormancy above assumes the inversion was NEUTRAL while standing.
# That was the last live assumption in the family. The biologically natural
# alternative is that it was ALREADY balanced, at a lower equilibrium, and that
# an environmental change moved the equilibrium to 0.626 -- which is what the
# sweep phase then tracks.
#
# This is a clean test because it is a ONE-PARAMETER INTERPOLATION between the
# two arms already run:
#
#     s_dorm -> 0     no restoring force  -> free drift        (NOTES 8.7)
#     s_dorm -> large pinned at p_eq      -> constant frequency (NOTES 8.6)
#
# so both limits are known in advance and the only question is which strength
# the data prefer.
#
# Same functional form as the rise, with the equilibrium at ``p_eq`` instead of
# p*: dp/dt = (s_dorm/p_eq)·p(1-p)(p_eq - p), plus WF noise
# sqrt(p(1-p)/(2N(t))). Near p_eq that linearises to a restoring rate
# k = s_dorm(1 - p_eq) against diffusion D = p_eq(1-p_eq)/(2N), giving a
# stationary standard deviation
#
#     SD = sqrt(p_eq / (4 N s_dorm))
#
# which is the scale to compare against the free-drift SD over the same window
# (0.227 at p_eq = 0.28 over 550 ky). The informative range is therefore
# s_dorm ~ 1e-6 to 1e-4; below that it is indistinguishable from neutral, above
# it from constant.
#
# WHY THIS NEEDS NO BRIDGE, AND IS THEREFORE MORE RIGOROUS THAN THE NEUTRAL ARM
# ---------------------------------------------------------------------------
# With a restoring force the diffusion has a stationary distribution, and every
# 1-D diffusion is reversible with respect to its stationary measure. So the
# backward-time path obeys the SAME SDE, and it can be simulated directly from
# the handoff frequency without any guiding term. The Durham-Gallant
# approximation that the neutral arm needed simply does not arise here.
#
# The ``s_dorm = 0`` case is the exception and is NOT rigorous in this function:
# neutral WF has absorbing boundaries, no stationary measure, and no
# reversibility, so running it backward unconditioned is an approximation. Use
# ``dormancy_bridge`` for the properly conditioned neutral case; ``s_dorm = 0``
# here is only for continuity of the limit.


def dormancy_balanced(t_inv: float, t_hand: float, p_eq: float,
                      s_dorm: float, rng, n_steps: int = 800):
    """Dormancy under balancing selection, as (backward times, freqs).

    Simulated directly backward from ``p_eq`` at ``t_hand`` to ``t_inv``; see
    the module notes for why no bridge is needed when ``s_dorm > 0``.
    Returns oldest-last.
    """
    T = float(t_inv) - float(t_hand)
    if T <= 0:
        raise ValueError(
            f"dormancy length must be > 0; t_inv={t_inv!r} t_hand={t_hand!r}")
    if s_dorm < 0:
        raise ValueError(f"s_dorm must be >= 0, got {s_dorm!r}")

    k = max(50, int(n_steps))
    dt = T / k
    c = (s_dorm / p_eq) if s_dorm > 0 else 0.0
    p = float(p_eq)
    times = [float(t_hand)]
    freqs = [p]
    for i in range(k):
        t_back = t_hand + i * dt
        n_e = float(N_growth(t_back))
        floor = 1.0 / (2.0 * n_e)
        drift = c * p * (1.0 - p) * (p_eq - p) * dt
        sd = float(np.sqrt(max(p * (1.0 - p), 0.0) / (2.0 * n_e) * dt))
        p = p + drift + sd * float(rng.standard_normal())
        p = min(max(p, floor), 1.0 - floor)
        times.append(t_back + dt)
        freqs.append(p)
    return times, freqs


def stationary_sd(p_eq: float, s_dorm: float, n_e: float) -> float:
    """SD of a balanced polymorphism at equilibrium ``p_eq``: sqrt(p/(4 N s))."""
    if s_dorm <= 0:
        return float("inf")
    return float(np.sqrt(p_eq / (4.0 * n_e * s_dorm)))


def arrival_curve_balanced(t_inv: float, t_arrive: float, t_rise: float,
                           p_hand: float, s_dorm: float, rng,
                           p_star: float = P_I_DEFAULT, n_rise: int = 300,
                           n_dorm: int = 800, tol: float = ARRIVAL_TOL):
    """Three-phase curve with dormancy under balancing selection at ``p_hand``."""
    if not 0.0 < p_hand < p_star:
        raise ValueError(
            f"p_hand ({p_hand!r}) must lie in (0, p_star={p_star!r})")
    t_hand = t_arrive + t_rise
    if t_hand > t_inv:
        raise ValueError(
            f"t_arrive + t_rise exceeds t_inv ({t_inv!r})")

    s_het = s_het_for_rise(t_rise, p_hand, p_star, tol)
    p_arrive = p_star * (1.0 - tol)

    p_grid = np.geomspace(p_hand, p_arrive, n_rise)
    t_fwd = _time_to_reach(p_grid, p_hand, s_het, p_star)
    t_back_rise = t_arrive + (t_rise - t_fwd)
    order = np.argsort(t_back_rise)

    times = [0.0, t_arrive]
    freqs = [p_arrive, p_arrive]
    times.extend(t_back_rise[order].tolist())
    freqs.extend(p_grid[order].tolist())

    if t_inv - t_hand > 0:
        d_t, d_f = dormancy_balanced(t_inv, t_hand, p_hand, s_dorm, rng,
                                     n_steps=n_dorm)
        times.extend(d_t)
        freqs.extend(d_f)

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    srt = np.argsort(times, kind="stable")
    times, freqs = times[srt], freqs[srt]
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, freqs = times[keep], freqs[keep]
    return np.clip(times, 0.0, t_inv).tolist(), freqs.tolist(), s_het


def build_arrival_balanced_sim(*, seq_length, t_inv, t_arrive, t_rise, p_hand,
                               s_dorm, drift_seed, p_star=P_I_DEFAULT,
                               gamma=1e-15, n_i=100, n_s=100, seed=None,
                               recomb_rate=None, n_rise=300, n_dorm=800):
    """Three-phase sim with a BALANCED (non-neutral) dormancy phase."""
    if recomb_rate is None:
        from .slim.config import REC_RATE
        recomb_rate = REC_RATE

    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    tract_length = max(1.0, (bp_right - bp_left) * TRACT_FRACTION)

    rng = np.random.default_rng(drift_seed)
    times, freqs, _s = arrival_curve_balanced(
        t_inv, t_arrive, t_rise, p_hand, s_dorm, rng, p_star=p_star,
        n_rise=n_rise, n_dorm=n_dorm)
    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        gene_conversion_rate=gamma,
        mean_tract_length=tract_length,
        tract_distribution="geometric",
        trajectory={
            "type": "precomputed",
            "times": times,
            "freqs": [freqs],
            "n_e": [float(N_growth(t_inv))],
            "t_inv": [float(t_inv)],
        },
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=PRESENT_NE_GROWTH,
        demography=growth_demography(),
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )
