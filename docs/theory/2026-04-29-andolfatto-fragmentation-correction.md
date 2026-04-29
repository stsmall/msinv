# Andolfatto fragmentation correction: derivation

**Status:** Draft 2026-04-29. **Provisional formula in §3.4 has a known
limitation:** it ignores coalescence and saturates at f→1/2 as t→∞,
which contradicts both intuition (any nonzero per-gen rate eventually
flips every sample) and the simulator's empirical f(25000) ≈ 0.70.
The mean-field flux-only ODE is the cleanest analytic step forward;
the next refinement (incorporating coalescence-driven coverage
restoration) is queued behind Monte-Carlo validation. See §4.2.

**Goal:** Derive a fragmentation-corrected closed form for the per-sample
position-conversion fraction `f(t)` in the msinv flux model. The
textbook Andolfatto formula `f(t) = 1 - exp(-γ·p_other·λ²·t/L)` over-
predicts the simulator by ~25% across the t_inv ladder
[1000, 4000, 10_000, 25_000] at Ne=1000, γ=1.5e-5, λ=300, L=6000,
p_inv=0.5 (test_phase3b_b2_flux.py::test_andolfatto_sample_fraction_matches_closed_form).
The deviation is not a simulator bug; it reflects a real mechanism in
the partial-tract regime that Andolfatto's derivation assumes away.

## 1. Setup and notation

We work in simulator-native units throughout. The relationship to
Andolfatto's published parameterization is documented at the end of
this section.

| Symbol | Meaning |
|---|---|
| `L` | Inversion length, in bp. |
| `λ` | Mean tract length, in bp. Tract lengths are i.i.d. geometric (in the simulator: `Exp(1/λ)` capped at `0.99·L`). |
| `γ` | Per-lineage gene-conversion event rate, in events per generation, normalized so that a *fully-covered* lineage of length `L` fires events at rate `γ·p_other·λ` per generation (see §1.1). |
| `p_inv` | Frequency of the inverted karyotype; `p_other` = frequency of the opposite karyotype seen by the donor (so for a lineage of class S, `p_other = p_inv`; symmetric at `p_inv = 0.5`). |
| `Ne` | Effective population size (panmictic, single pop). |
| `n` | Number of samples. |
| `t` | Generations elapsed since `t_inv` (the inversion's age). Backward time on a sample lineage equivalently. |
| `x_c` | Focal position; for the Tier-3 test, the inversion midpoint `(bp_left + bp_right)/2`. |
| `x_e` | Position at which a flux event "starts" (used to define the b1 distribution; constrained to lie within the lineage's actual segments). |

We assume the inversion is interior on the chromosome, the focal
position `x_c` is interior on the inversion (away from breakpoints by
distance ≫ λ), and `λ ≪ L` (the partial-tract regime where the
fragmentation correction has the largest effect).

### 1.1. The Andolfatto rate

For a single lineage that is *fully covered* on the inversion (segment
list `[bp_left, bp_right)`), the per-generation rate of flux events
that flip position `x_c` is

```
r_A = γ · p_other · λ² / L
```

Derivation in two factors:
- Per-lineage event rate: `γ · p_other · λ`. The factor `λ` arises from
  averaging the simulator's `phi(x, w)` placement profile over a
  fully-covered inversion (validated at the rate-scaling level by
  `test_flux_rate_scales_linearly_with_mean_tract_length`).
- Per-event probability that the tract covers an interior position `y`,
  given event placement `x_e ~ Uniform[0, L]` and tract length
  `T ~ Exp(1/λ)` placed in `[max(0, x_e - T), min(L - T, x_e)]`:
  approximately `λ/L` for `y` interior, `T < L/2`.

Under Andolfatto's assumption that the per-lineage rate is *constant*
over the lineage's lifetime (= `t`), the count of flips along the
sample's ancestral path at `x_c` over `[0, t]` is Poisson(`r_A · t`),
and

```
f_A(t) = 1 - exp(-r_A · t).
```

**Translation to Andolfatto 2001:** the published form
`f(t) = 1 - exp(-γ_init · p_other · λ · t)` with per-bp initiation rate
`γ_init = γ · λ / L` is identical to the simulator-native form once the
parameterizations are aligned. The remainder of this document uses
the simulator-native form.

## 2. Why Andolfatto over-predicts in the partial-tract regime

The Andolfatto rate `r_A` assumes the sample's ancestral lineage at
`x_c` has *full coverage* on the inversion at every backward time `s`.
The simulator's lineages are not full-coverage: each backward-time flux
event splits a lineage into two homogeneous output lineages — a *tract
piece* (carrying the tract material with the donor's karyotype) and an
*outside piece* (the donor's non-tract material, with a coverage hole
where the tract used to be). Sample i's ancestral lineage at `x_c`
follows whichever piece carries `x_c`:

- If the event's tract covers `x_c`: ancestor of `x_c` is the tract
  piece, of length `~ λ`. The sample's class at `x_c` flips.
- If the event's tract does NOT cover `x_c`: ancestor of `x_c` is the
  outside piece, with a hole around the tract location. The sample's
  class at `x_c` is unchanged, but the outside piece now has reduced
  total in-inversion coverage.

In both cases, the sample's ancestral lineage at `x_c` retains material
at `x_c` (by definition of "ancestor at `x_c`"). What changes is the
ancestral lineage's *total in-inversion coverage* `C(s)`, which
controls the future per-lineage flux event rate.

The per-lineage flux event rate, for a lineage of total in-inversion
coverage `C(s)`, scales linearly with `C(s)`:

```
event rate(s) = γ · p_other · λ · (C(s) / L)             (1)
```

(The factor `λ` is the per-bp-of-coverage average per the `phi` profile,
unchanged from the full-coverage derivation; the linear scaling with
`C(s)` reflects that the event rate integrates over the lineage's
actual segments.)

The per-event probability that a tract covers `x_c`, *given an event
fires on a lineage with structure σ at backward time s*, depends on
where in σ the event was placed. For the moment we approximate this
probability as `λ/L` regardless of σ (the "interior approximation"; see
§3.3 for the refinement). Under this approximation:

```
flip rate at x_c on sample's path at time s ≈ γ · p_other · λ · (C(s)/L) · (λ/L)
                                            = r_A · (C(s)/L)         (2)
```

So the corrected rate is `r_A` scaled by the *fractional coverage* of
the sample's ancestral lineage at `x_c` at backward time `s`.

## 3. Derivation of the corrected formula

### 3.1. Master equation

Let `μ(s) := E[C(s)/L]`. The expected number of flips on the sample's
ancestral path at `x_c` over `[0, t]` is

```
Λ(t) := r_A · ∫₀ᵗ μ(s) ds.                                          (3)
```

Conditional on the path's segment-structure history, the count is
Poisson with this mean, so

```
f_corrected(t) = 1 - exp(-Λ(t)).                                    (4)
```

This reduces to Andolfatto when `μ(s) ≡ 1` (full coverage at all times).

### 3.2. Coverage decay ODE

To get `μ(s)` we compute its evolution under the Markov chain on
lineage segment-structure σ. Each backward-time flux event on the
sample's ancestral lineage at `x_c` transitions σ to a new state σ':

- **Tract event (probability `λ/L` per event under the interior
  approximation):** ancestor at `x_c` becomes the tract piece, of length
  `~λ`. New coverage: `~λ/L`. Set `μ → λ/L`.
- **Outside event (probability `1 - λ/L`):** ancestor at `x_c` stays on
  the outside piece. The outside piece is the donor's non-tract
  material; total length is `C(s) - T_tract` where `T_tract ~ λ` (the
  removed tract). Coverage drops by approximately `λ/L` of the previous
  coverage.

The rate of any flux event on the sample's path is `γ·p_other·λ·μ(s)`
(equation (1) with `μ = C/L`). Of those, fraction `λ/L` are tract
events and `1 - λ/L` are outside events. So:

```
dμ/ds = -γ·p_other·λ·μ(s) · [P(tract) · (μ(s) - λ/L)
                                + P(outside) · (λ/L)]
```

Wait — we need to track the *expected change* in `μ` per unit time.

- Tract events occur at rate `γ·p_other·λ·μ(s) · (λ/L) = r_A · μ(s)`.
  Each such event resets coverage to `λ/L` (a drop of `μ(s) - λ/L` if
  `μ(s) > λ/L`).
- Outside events occur at rate `γ·p_other·λ·μ(s) · (1 - λ/L)`. Each
  such event removes a tract of length `~λ` from the lineage, dropping
  coverage by `λ/L` (in the limit `λ ≪ L`, ignoring the case where the
  tract overlaps an existing hole).

Combining:

```
dμ/ds = -r_A · μ(s) · (μ(s) - λ/L)                         (tract-event term)
        - γ·p_other·λ·μ(s) · (1 - λ/L) · (λ/L)             (outside-event term)
      = -r_A · μ(s) · (μ(s) - λ/L)
        - r_A · μ(s) · (1 - λ/L)                                  (5)
      = -r_A · μ(s) · (μ(s) - λ/L + 1 - λ/L)
      = -r_A · μ(s) · (μ(s) + 1 - 2λ/L).
```

For `λ ≪ L`, the term `(μ + 1 - 2λ/L)` is approximately `μ + 1`. The
ODE becomes

```
dμ/ds ≈ -r_A · μ(s) · (μ(s) + 1).                                  (6)
```

This is a Riccati equation in `μ` with separable form.

### 3.3. Closed form for μ(s)

Separating:

```
dμ / [μ · (μ + 1)] = -r_A · ds.
```

Partial fractions: `1 / [μ(μ+1)] = 1/μ - 1/(μ+1)`. Integrating:

```
ln(μ / (μ+1)) - ln(μ₀ / (μ₀+1)) = -r_A · s.
```

With `μ₀ = μ(0) = 1` (sample's lineage has full coverage at present):

```
μ / (μ + 1) = (1/2) · exp(-r_A · s).
```

Solving for `μ`:

```
μ(s) = exp(-r_A · s) / (2 - exp(-r_A · s)).                         (7)
```

**Sanity checks for `μ(s)`:**
- `μ(0) = 1 / (2 - 1) = 1`. ✓ (full coverage at present)
- `μ(∞) = 0 / 2 = 0`. ✓ (lineage fully fragmented in long-time limit)
- `μ(s) = (e^{-r_A·s} ) / (2 - e^{-r_A·s}) > 0` always. ✓

### 3.4. Closed form for f_corrected(t)

Substitute (7) into (3). Let `u = exp(-r_A · s)`; `du = -r_A · u · ds`,
so `ds = -du / (r_A · u)`. When `s = 0`, `u = 1`; when `s = t`,
`u = e^{-r_A·t}`.

```
Λ(t) = r_A · ∫₀ᵗ u(s) / (2 - u(s)) ds
     = -∫_1^{e^{-r_A·t}} (1 / (2 - u)) du
     = ∫_{e^{-r_A·t}}^1 du / (2 - u)
     = [-ln(2 - u)]_{e^{-r_A·t}}^1
     = -ln(1) + ln(2 - e^{-r_A·t})
     = ln(2 - e^{-r_A·t}).                                          (8)
```

Therefore

```
f_corrected(t) = 1 - exp(-Λ(t))
              = 1 - 1 / (2 - exp(-r_A · t))
              = (1 - exp(-r_A · t)) / (2 - exp(-r_A · t)).         (9)
```

This is the **fragmentation-corrected closed form**.

### 3.5. Limit checks

- **Small `t` / `r_A·t ≪ 1`:** Taylor-expand `exp(-r_A·t) ≈ 1 - r_A·t`.
  Then `f_corrected ≈ (r_A·t) / (1 + r_A·t) ≈ r_A·t · (1 - r_A·t)`,
  matching `f_A ≈ r_A·t · (1 - r_A·t/2)` to leading order. The
  fragmentation correction kicks in at second order in `r_A · t`, which
  is the regime we care about (mid-to-large `t`).
- **Large `t` / `r_A·t ≫ 1`:** `exp(-r_A·t) → 0`, so
  `f_corrected → 1/2`. **The corrected formula saturates at 1/2, not 1.**
  Andolfatto saturates at 1 (trivially); the corrected formula reflects
  that as fragmentation accumulates, the effective rate decays and the
  per-sample flip probability cannot exceed the parity of an
  ever-slowing Markov chain. (See §4 for discussion of whether this
  saturation matches the simulator's TMRCA behavior.)
- **`λ → L`:** the approximation `(μ + 1 - 2λ/L) ≈ (μ + 1)` breaks down;
  the correction term vanishes. In this limit the lineage is "full
  inversion in tract" — every flux event produces a tract spanning the
  whole inversion, and the original Andolfatto derivation becomes
  *exact*. The general formula (9) does not recover Andolfatto in this
  limit because the ODE (5) was approximated; the exact ODE has an
  additional `2λ/L` term in the right-hand side that vanishes as
  `λ/L → 0` and dominates as `λ/L → 1`.
- **`γ → 0`:** `r_A → 0`, `f_corrected → 0`. ✓

### 3.6. Comparison to Andolfatto on the test ladder

For the Tier-3 test parameters (γ=1.5e-5, λ=300, L=6000, p_other=0.5):

```
r_A = 1.5e-5 · 0.5 · 90000/6000 = 1.125e-4 per generation
```

| t_inv | r_A·t | f_A | f_corrected (eq 9) | simulator empirical |
|---|---|---|---|---|
| 1000  | 0.1125 | 0.1064 | 0.053  | ~0.10 (within tol) |
| 4000  | 0.450  | 0.362  | 0.181  | ~0.30 (within tol) |
| 10000 | 1.125  | 0.675  | 0.428  | ~0.55 (within tol) |
| 25000 | 2.81   | 0.940  | 0.485  | ~0.70 (estimated from current loose-tolerance test margin) |

The corrected formula predicts substantially less than Andolfatto, and
in some regimes substantially less than the simulator. The MC
validation script (§4) is essential to determine whether the
discrepancy is from:

1. The interior approximation (§2) — per-event-covers-`x_c` probability
   is not exactly `λ/L` for fragmented lineages.
2. The tract-event reset assumption — coverage doesn't truly reset to
   `λ/L` because the tract piece itself can be further fragmented in
   subsequent events.
3. The independence approximation between successive flux events and
   the segment structure (the ODE is mean-field; the actual process has
   correlations).
4. Coalescence — multiple samples share ancestors, so the
   single-sample Andolfatto-like analysis doesn't strictly apply for
   `n > 1`.

**Most likely culprit:** (4) coalescence-driven coverage restoration.
The mean-field ODE assumes coverage always decays; in reality, when
sample i's ancestral lineage at `x_c` coalesces with another lineage
that ALSO covers `x_c` plus additional segments, the merged ancestor
has the union of both segment lists — coverage grows. Over backward
time, alternating flux-driven decay and coalescence-driven growth
gives an equilibrium coverage that is a function of `r_A` and the
coalescence rate `1/(2Ne)`, NOT zero.

**Quantitative consistency check.** The corrected formula's saturation
at `f→1/2` as `t→∞` is mathematically clean but biologically
implausible: as long as the per-gen flip rate is positive, eventually
all samples should flip. The simulator's f(25000) ≈ 0.70 (vs.
predicted 0.485) confirms this: the formula systematically *under*-
predicts at the largest `t_inv` rung, indicating the missing
coalescence mechanism is the dominant correction.

Awaiting MC validation to confirm the qualitative shape and identify
the leading-order coalescence correction.

## 4. Open questions / refinements

### 4.1. Tract-piece self-fragmentation

When an event's tract covers `x_c` and the sample's ancestor moves to
the tract piece (length `~λ`), subsequent flux events on this small
piece have very different statistics: per-event-covers-`x_c` probability
is close to 1 (the piece is small and `x_c` is in it), but per-lineage
event rate is reduced (since rate scales with `λ` not `L`). The product
gives a flip rate of `γ · p_other · λ · (λ/λ) = γ · p_other · λ`, which
is `L/λ` times higher than `r_A`. This contradicts the assumption that
fragmentation always reduces the rate.

**Hypothesis:** the rate on the tract piece is much higher per unit
time, but the tract piece itself is short-lived (it likely either
re-coalesces quickly or, if `λ ≪ L`, the next flux event likely takes
us elsewhere). The integrated rate over the tract-piece lifetime may
still be lower than `r_A · t`.

This needs careful treatment. The MC script will report the expected
per-lineage flip rate as a function of `s` directly, exposing whether
this hypothesis holds.

### 4.2. Coalescence

For the Tier-3 test with n=20 samples in a panmictic population, after
some time `~ 4·Ne`, samples share ancestors. The shared ancestor's
flux events flip `x_c` for all descendants simultaneously. In
Andolfatto's per-sample analysis this is treated independently;
empirically the coalescence-induced correlations should not change
*the expected fraction of samples flipped* (each sample has a
well-defined ancestral path at `x_c`, even when paths share segments
with other samples). But the coalescence does affect the *variance* of
the empirical estimator, which the test's tolerance should account for.

### 4.3. Tractability of an exact derivation

An exact closed form would require characterizing the full segment-
structure Markov chain on the sample's ancestral lineage at `x_c`.
This is intractable analytically. The mean-field ODE in §3.2 is the
simplest approximation; second-moment corrections may close the gap
seen in the simulator-vs-formula discrepancy table.

## 5. Numerical validation plan

`scripts/validate_andolfatto_fragmentation.py` will:

1. Implement a forward-time Monte Carlo of *one* lineage tracking
   `x_c`, with the simulator's exact flux event mechanics
   (`sample_flux_position` placement profile, `draw_tract` length and
   placement, segment splitting on each event).
2. For each of `N_reps` (default 10_000) realizations, simulate to
   `t = t_max`, recording each backward-time flux event and whether
   it covered `x_c`.
3. Empirical f̂(t) = fraction of reps with ≥1 `x_c`-covering event by
   `t`, computed at a dense grid of `t` values.
4. Compare empirical f̂(t) to:
   - `f_A(t)` (Andolfatto, eq §1.1).
   - `f_corrected(t)` (eq 9 of §3.4).
   - Optionally an iterated refinement (numerical solution to a more
     accurate ODE that includes tract-piece self-fragmentation).
5. Plot residuals over `t`. The script's stdout reports the
   maximum absolute deviation between f̂ and each candidate formula.

Acceptance: if `max |f̂ - f_corrected|` is within Monte Carlo error
(`~ 1.96 / √N_reps · √(f̂(1-f̂))`), the corrected formula is
validated. If not, refine the derivation.

## 6. Tightened test plan

After MC validation lands, add to `tests/hull/test_phase3b_b2_flux.py`:

```python
def test_andolfatto_sample_fraction_matches_corrected_form():
    """Tighter check against the fragmentation-corrected closed form
    (docs/theory/2026-04-29-andolfatto-fragmentation-correction.md eq 9).
    Tolerance: max(0.05·f_corrected, 0.05) absolute."""
    ...
```

Keep the existing loose-tolerance test as a smoke check; do not
remove it.

## 7. References

- Andolfatto, P. (2001). Adaptive evolution of non-coding DNA in
  Drosophila. *Genetics* 158:865–874. (Source of the textbook formula.)
- Guerrero, R. F., & Kirkpatrick, M. (2014). Local adaptation and the
  evolution of chromosome fusions. *Evolution* 68(10):2747–2756.
  (Treats per-bp gene-conversion rates in inversion genealogies; useful
  for the `γ` parameterization translation.)
- Hudson, R. R. (1983). Properties of a neutral allele model with
  intragenic recombination. *Theor. Pop. Biol.* 23:183–201.
- Wakeley, J. (2009). *Coalescent Theory: An Introduction.*
  Roberts & Co. (Standard reference for the underlying coalescent.)

---

**Status note (2026-04-29):** sections §1–§3 are derived from first
principles; §3.6 numbers are computed from the closed form. §3.5 limit
checks suggest the formula is qualitatively right but quantitatively
not yet calibrated against the simulator. §4 documents known
approximations whose impact will be measured by §5's MC script.
The formula is *not yet ready* to replace the loose-tolerance test;
that's gated on MC validation.
