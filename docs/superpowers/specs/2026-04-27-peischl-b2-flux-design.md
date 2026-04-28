# Peischl 2013 b2 Flux Upgrade — Design

**Date:** 2026-04-27
**Status:** Implemented — landed in commits 89af630..3163d51 (10 commits via subagent-driven development); Tier 3 + cross-feature deferred validation captured in the Deferred Validation Roadmap section. See also `feedback_parity_misnomer.md` for a meta-finding about the "parity" test naming.
**Owner:** stsmall
**Related memory:** `project_msinv_todo.md`, `feedback_inversion_freq_dynamics.md`

## Motivation

The current msinv hull-engine flux model implements the *example numerical
construction* from Peischl et al. 2013 (Heredity) — a single parameter
`flux_window` controls **three** distinct things at once:

1. tract length (`tract_bp = flux_window × inv_length`),
2. spatial flux profile shape (`φ(x) = min(x, 1-x, w) / (1-w)`), and
3. per-bp flip rate (`per-bp rate ≈ γ × w` in the interior).

This conflation has two consequences:

- **Biological mismatch.** With the default `flux_window = 0.05` on a 6 Mb
  3Ra inversion, every flux event converts a 300 kb tract. *Anopheles*/
  *Drosophila* gene-conversion tracts are ~50–500 bp empirically — three
  orders of magnitude smaller.
- **Parameters are entangled.** Cannot vary tract length without also
  rescaling the spatial profile and the per-bp rate, making it impossible
  to disentangle the three effects in ABC inference or in comparing to
  Andolfatto/Guerrero analytic predictions.

This upgrade decouples the three knobs and adds **stochastic tract-length
sampling per event** — the standard biological model assumed by Andolfatto
2001, Guerrero & Kirkpatrick 2014, and msprime's `GeneConversion` event.

## Scope

**In scope:**
- Replace `flux_window` (single conflated parameter) with two clean fields
  on `InversionSpec`: `mean_tract_length` (bp) and `tract_distribution`
  (`'geometric'` | `'fixed'`).
- Sample tract length per event from the configured distribution.
- Migrate every existing test and example to the new API, preserving
  current semantics by using `'fixed'` mode at the equivalent bp value.
- Tier 1 + Tier 2 tests landing in the same change (correctness + smoke
  + spatial profile + per-bp rate calibration).

**Out of scope (deferred to follow-up):**
- Tier 3 theoretical-anchor tests (LD-decay shape comparison, Andolfatto
  fraction-converted) — these add biological-content validation but the
  core API change should land first.
- Additional tract-length distributions (Gamma, Lognormal). The two-field
  API leaves room to extend later if needed.
- SMC-style pending-b2 tracking on lineages. The legacy `msinv2.py`
  memory note about "Peischl b2 algorithm" referred to a feature that
  **does not apply to the ARG-based hull engine** — see the
  Non-Goals section below.

## Non-goals

**No SMC-style pending-b2 tracking.** Peischl 2013's *SMC* algorithm
records pending right-endpoint (b2) events on lineages so they replay
as the chromosome is scanned rightward. The msinv hull engine is ARG-
based — gene-flux events are represented directly as two recombination
events at b1 and b2 in the ARG, no pending-history needed. The
`apply_gene_flux` function at `rust/msinv-core/src/simulator.rs:1914-1982`
already implements this correctly. We do NOT add SMC-style tracking.

## API Design

### New `InversionSpec` fields

```python
@dataclass
class InversionSpec:
    bp_left: float
    bp_right: float
    p_inv: Union[float, Dict[int, float], None] = None
    t_inv: Optional[float] = None
    gene_conversion_rate: float = 1e-9        # γ, per-bp per-gen — UNCHANGED
    mean_tract_length: float = 100.0          # NEW (bp), replaces flux_window
    tract_distribution: str = 'geometric'     # NEW: 'geometric' or 'fixed'
    inv_id: int = -1
    trajectory: Optional[Dict] = None
```

### Removed

`flux_window` is removed entirely. Passing it raises `TypeError` from the
dataclass constructor (default Python behavior — no shim, no
backwards-compatibility alias).

### Validation (`__post_init__`)

- `mean_tract_length >= 0` (error on negative)
- `mean_tract_length == 0` is **legal** — disables flux entirely (per-lineage
  flux rate evaluates to 0, no events fire, the tract sampler is never
  reached). Default value is `100.0`; disabling is opt-in.
- `mean_tract_length > inv_length / 2`: warn (not error) — long tracts
  are physically possible but rare.
- `tract_distribution in {'geometric', 'fixed'}`

### Disabling flux

Two equivalent ways to express "no gene flux":

| Setting | Effect |
|---|---|
| `gene_conversion_rate = 0` | Per-bp rate factor is zero → per-lineage event rate is zero. |
| `mean_tract_length = 0` | Tract size is zero → `phi(x) = 0` everywhere → per-lineage event rate is zero. |

Both produce identical simulator behaviour (no flux events). The
existing test convention of `gene_conversion_rate = NEGLIGIBLE_GAMMA = 1e-15`
remains valid; new code may also use `mean_tract_length = 0` if the
"tract length is zero" framing is more natural for the scientific
context.

### Naming

The API uses `'geometric'` (the literature-standard name for the discrete
gene-conversion model). Internally, with continuous-coordinate genome
representation, we use **Exponential(rate = 1/λ)** as the natural
continuous analog. The continuous limit `Geometric(p) → Exponential(1/λ)`
as bp resolution → 0 makes them equivalent at the `mean_tract_length`
parameterization. This convention is documented in the field docstring.

## Algorithm

### Tract sampling (`draw_tract`)

```pseudocode
fn draw_tract(x_event, inv, rng) -> (f64, f64):
    # Defensive: rate-zero short-circuit elsewhere should prevent us
    # reaching here with mean_tract_length == 0, but guard anyway so
    # we never divide by zero in the Exponential sampler.
    if inv.mean_tract_length == 0.0:
        return (x_event, x_event)              # zero-width tract, no-op

    L = match inv.tract_distribution {
        'fixed':     inv.mean_tract_length
        'geometric': sample_exponential(rng, rate = 1.0 / inv.mean_tract_length)
    }
    L = L.min(inv.length() * 0.99)              # safety cap

    x_rel = x_event - inv.bp_left
    b1_lo = max(0.0, x_rel - L)
    b1_hi = min(inv.length() - L, x_rel)
    b1    = if b1_hi > b1_lo {
        uniform(rng, b1_lo, b1_hi)
    } else {
        clamp(x_rel - L/2, 0, inv.length() - L)
    }
    tract_left  = inv.bp_left + b1
    tract_right = (tract_left + L).min(inv.bp_right)         # clip to inv
    return (tract_left, tract_right)
```

**Boundary handling: clip (not reject-resample, not shift).**
If a sampled `(b1, L)` would put the tract right edge past `bp_right`,
the tract is clipped. For Anopheles biological parameters
(`λ = 100 bp`, `inv = 6e6 bp`), boundary events are < 0.01 % of all
events. The induced bias on the empirical tract-length distribution
near edges is O(λ/inv_length) ≈ 1.7e-5 — far below MC noise of any
test. Clipping matches the existing `'fixed'`-mode behavior, simplifying
the migration.

### Rate calculation (`phi`, `phi_integral`)

**`rust/msinv-core/src/phi.rs` is unchanged.** The existing closed-form
`phi(x) = min(x, 1-x, w) / (1-w)` is reused for both modes by passing
`w = mean_tract_length / inv_length`.

- For `'fixed'`: this is exact (the original Peischl 2013 example formula).
- For `'geometric'`: this is the small-λ_norm interior approximation of
  the true convolution `E_L[min(x, 1-x, L) / (1-L)]`. Accurate to
  O(λ_norm²); for biological parameter ranges (`λ_norm = 1.67e-5` at
  `mean=100 bp` on a 6 Mb inversion), the error is ≪ 10⁻⁴.

The "exact convolution" closed form for Exponential L is integrable but
messy; the approximation is well within MC noise of any test we'd run.

**Per-lineage event rate** stays the same form:

```
rate = γ × p_other × ∫ phi(x) dx  (over the lineage's in-inversion segments)
     × inv_length                  (to express in bp units)
```

For full-coverage lineage with `λ_norm ≪ 1`:
```
rate ≈ γ × p_other × mean_tract_length
```
This is the standard biological formulation: per-bp gene-conversion rate
× mean tract length = expected events per lineage per generation.

### Engine touch points

| File | Change |
|---|---|
| `msinv/hull/inversion.py` | Drop `flux_window`; add `mean_tract_length`, `tract_distribution`; update validation. |
| `rust/msinv-core/src/inversion.rs` | Mirror struct fields. |
| `rust/msinv-py/src/lib.rs` | PyO3 bridge: translate two new Python fields into the Rust struct. |
| `msinv/hull/simulator.py::_draw_tract` | Sample L per event from configured distribution. |
| `rust/msinv-core/src/simulator.rs::draw_tract` | Same. |
| `msinv/hull/_rust_bridge.py` | Serialize new fields across FFI. |
| `rust/msinv-core/src/phi.rs` | **No change.** |
| `rust/msinv-core/src/rate_engine.rs` | **No change in form** — calls `phi_integral` with `w = mean_tract_length / inv_length`. |

## Test plan

Three tiers; **Tier 1 + Tier 2 land alongside the change**, Tier 3 in a
follow-up commit.

### Tier 1 — must pass to land

- **Backward-compat.** All existing `tests/hull/test_phase3_gene_flux.py`
  tests pass after migration. Migrated test sites use
  `mean_tract_length = old_flux_window × inv_length` and
  `tract_distribution = 'fixed'` to preserve original semantics. RNG
  draws differ from the old code, so bit-equivalence is not required;
  distributional equivalence is.
- **Geometric sampling unit test.** Draw N=10⁴ tract lengths from
  `'geometric'` mode in isolation; assert `mean ≈ mean_tract_length`
  within ±2σ (`±2λ/√N`); KS-test CDF against Exponential(1/λ) at
  p > 0.05.
- **Smoke runs.** Simulator produces well-formed tree sequences with
  `'geometric'` mode at biological 3Ra-scale parameters
  (`inv_len=6e6, mean_tract_length=100, γ=1e-6`); doesn't crash.

### Tier 2 — validation, lands alongside Tier 1

- **Spatial profile φ(x) test.** Long sim, single inversion,
  `'geometric'`. Count flux events that touch each bp position. Compare
  empirical φ(x) histogram to the analytic interior approximation
  (≈ λ/inv_length flat in the interior, linear ramp over ~λ at each
  breakpoint). Tolerance ±15 %; NREPS sufficient to drive MC noise
  below tolerance.
- **Per-bp flip-rate calibration.** Single position deep inside an
  inversion, long t_inv, count empirical flips. Assert
  `rate ≈ γ × p_other × λ / inv_length` within MC variance.

### Tier 3 — theoretical anchor (deferred — see Deferred Validation Roadmap)

This launch lands Tier 1+2 only. Tier 3 tests are explicitly deferred to
a follow-up validation pass; see the **Deferred Validation Roadmap**
section below for the full backlog.

## Deferred Validation Roadmap

These items don't block the b2 flux launch but should be picked up as a
batched validation pass once the selection-feature work (`#3` in
`project_msinv_todo.md`) is settled. Tier 3 b2 tests sit alongside
deferred work from other features so the next validation push can
take them all together rather than dribbling them in.

### B2 flux follow-ups (Tier 3, this design)

- **(Q) Tier 3-cheap.** LD-decay shape comparison (full strength) +
  cheap Andolfatto check (monotonicity in t_inv, fixed/geometric
  equivalence in mean fraction at matched λ).
- **(R) Tier 3-full.** Full closed-form Andolfatto fraction-converted
  anchor: assert empirical fraction matches
  `1 − exp(−γ × p_other × λ × t / inv_length)` within MC tolerance.
  Requires simulator instrumentation (per-bp flip-history hook).

### Cross-feature deferred validation

- **T3 (cmig binomial count check).** Carried over from the cmig SFS
  validation work (commit `b90fee2`). Quantitative check that the
  *count* of lineages moved by a `cmig` event matches
  `Binomial(n_eligible, proportion)`. Needs the same per-event hook as
  R's per-bp flip count, so naturally batched with that work.
- **Selection-feature validation suite.** When the proper sweep /
  partial-sweep trajectory model lands (replacing the current
  Hudson-Kaplan endpoint-only operator in `sweep.rs`), it will need
  its own validation tests (sweep age vs s vs p_target regimes;
  comparison to discoal). This is a separate roadmap item (`#3` in
  `project_msinv_todo.md`); design pending.

### When to cash this in

Trigger conditions to do a batched validation pass:
1. Selection feature lands → its tests AND the deferred Tier 3 / T3
   items go in together.
2. ABC pilot starts uncovering posterior anomalies that point at
   under-validated parts of the model — pull the relevant tier-3
   tests forward at that point.
3. New scientific hypothesis requires anchoring against published
   theory (Andolfatto, Guerrero) — pull (R) forward.

Until then, Tier 1+2 + parity is the agreed validation surface for
the b2 flux upgrade.

### Rust ↔ Python parity

The existing parity-test framework iterates over a list of
`InversionSpec`-equivalent fixtures. Extend the fixture list to include
both `'fixed'` and `'geometric'` modes. The parity assertion is
unchanged — both engines must produce identical trees for the same seed.
For `'geometric'`, this depends on the L-sampler being deterministic
under the same RNG state in both languages; both currently use
`Xoshiro256PlusPlus` so parity holds if both languages call
`sample_exponential(rng, 1/λ)` in the same place in the event sequence.

## Migration

Mechanical. For every existing call site of `flux_window=X`:

```python
# before
inv = InversionSpec(..., flux_window=0.05)

# after
inv = InversionSpec(
    ...,
    mean_tract_length=0.05 * (bp_right - bp_left),  # = X * inv_length
    tract_distribution='fixed',
)
```

A grep audit of the repo gives ~12 test files plus a handful of
examples and notebooks. Migration commit is a single mechanical pass.

## Open Questions / Future Work

- **Stochastic tract-length distribution choices beyond geometric.**
  Gamma (with shape parameter) would allow tighter-than-exponential
  variance. Lognormal models the long tail of empirical gene-conversion
  tracts more flexibly. Two-field API (`mean_tract_length` +
  `tract_distribution` enum) leaves room for these. Add only if a
  scientific question requires them.

- **`mean_tract_length` as ABC-inference parameter.** Once the API
  decouples it from γ and the spatial profile, this becomes a
  meaningful free parameter. Future ABC pilot can include it as a
  prior (e.g., `LogUniform[10, 10000]` bp) and infer from data. Not
  blocking on this design; it's a downstream Stage-2 of the Kir/Fol
  roadmap.

- **Per-event spatial heterogeneity.** Some empirical observations
  suggest gene-conversion rate varies along the chromosome
  (recombination hotspots). This isn't part of Peischl 2013 and isn't
  addressed here. The current uniform-b1 sampling is the cleanest
  baseline.

## References

- Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013). A
  sequential coalescent algorithm for chromosomal inversions.
  *Heredity*, 111, 200–209. doi:10.1038/hdy.2013.38
- Andolfatto, P. (2001). Adaptive hitchhiking effects on genome
  variability. *Curr. Opin. Genet. Dev.*, 11(6), 635–641.
- Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). Coalescent
  patterns for chromosomal inversions in divergent populations.
  *Phil. Trans. R. Soc. B*, 367, 430–438.
