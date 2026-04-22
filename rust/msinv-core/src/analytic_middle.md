# Structured-analytic middle phase — prototype

Feature branch: `feature/structured-analytic-middle`. Companion module
`analytic_middle.rs`. Entirely experimental; see
`project_feature_branches_roadmap.md` for motivation.

## What this is

The realistic anopheles Kir/Fol ancestral window (roughly
`t_split ≈ 14 k` → `t_inv ≈ 330 k` generations back, single panmictic
pop, `Ne_anc ≈ 1 M`) spends the vast majority of simulator wall time
churning through partial-coalescence events that keep growing the
active-lineage count by 1 (the "remnant ratchet"). In that window the
demography is uniform in time:

- one effective population
- zero migration
- constant ancestral `Ne`
- constant per-pop inversion frequency `p_inv`
- inversion barrier active for at least one inversion

That is exactly the Kaplan-Hudson-Langley (1989) structured
coalescent regime. In-class coalescence rates are analytic constants:

- S-S pair inside the inversion: `1 / (2·Ne·p_std)`
- I-I pair inside the inversion: `1 / (2·Ne·p_inv)`
- any pair on panmictic positions: `1 / (2·Ne)`

Instead of rebuilding the per-pair rate cache on every event, we draw a
single scalar total rate, sample the next-event time via
`Exp(total_rate)`, pick which event fires by competing rates, and apply
it. This is option (a) from the feature spec — no recombination, no
gene flux during the window.

## Scope landed on this branch

- New module `rust/msinv-core/src/analytic_middle.rs`.
- `HullSimulator::analytic_middle: bool` flag, default `false`.
  Exposed through the PyO3 bridge as a keyword argument on
  `simulate_raw`.
- Gated fast-path at the top of `run_loop` that, when the flag is
  enabled AND the current state matches the gate, fast-forwards to
  the next barrier crossing via Kingman-style exponential draws.
- Fallback: if any gate condition fails the simulator runs the normal
  event loop.

**Gate**:
- single inversion
- single effective population (either `n_pops == 1` or every active
  lineage is already in the same pop)
- zero migration
- no pending demographic event or sweep in `(t_now, t_inv)`
- no exponential-growth rate set for the effective pop
- `≥ 2` active lineages
- barrier active and `t_inv > t_now`

## Benchmark

`rust/msinv-core/tests/analytic_middle_bench.rs`. Run with:

```
cargo test --release -p msinv-core --test analytic_middle_bench -- --ignored --nocapture --test-threads=1
```

Ne=10k, L=1 Mb, inversion spans 180 k–380 k bp. Samples split evenly
between S and I. All runs single-threaded, release profile.

| scenario                                  | event-loop s/rep | analytic s/rep | speedup |
| ----------------------------------------- | ----------------: | -------------: | ------: |
| mini (Ne=1k, t_inv=5k, n=10, rho=4)       |            0.014 |          0.014 |   1.0×  |
| stress (Ne=10k, t_inv=330k, n=30, rho=4)  |            0.012 |          0.014 |   0.8×  |
| medium (Ne=10k, t_inv=50k, n=30, rho=40)  |            0.013 |          0.002 |   6.9×  |
| highn (Ne=10k, t_inv=50k, n=60, rho=40)   |            0.044 |          0.003 |  12.8×  |
| highrho (Ne=10k, t_inv=50k, n=30, rho=200)|            0.113 |          0.003 |  38.9×  |

Speedup grows with `rho` and `n` (i.e. the amount of work the event
loop would have done pre-barrier). The stress case regresses slightly
because with `t_inv = 33 · 2·Ne`, almost every lineage has already
coalesced before the analytic phase starts, so the reclassify-per-event
rebuild pays without any real event-count win.

### Kir/Fol full-scale single-rep bench (2026-04-22 session)

Single-replicate wall time at realistic Kir/Fol demography (3Ra only,
single inversion to fit the prototype's gate):

- Ne_K_current=44 k, Ne_F_current=92 k, Ne_F_postsplit=3 M, Ne_anc=1 M
- t_split=14 k, t_inv=330 k, p_inv(F)=0.73 (p_inv(K)=0)
- samples 74 K + 23 F_S + 23 F_I, L=1 Mb, r=1e-8, γ=1e-9
- bench script: `rust/msinv-core/examples/bench_kirfol_scale.py`

| path            | wall time | nodes   | trees  |
| --------------- | --------: | ------: | -----: |
| event-loop      | 251.1 s   | 175 133 | 44 438 |
| analytic-middle | > 3480 s  | —       | —      |

The analytic-middle side did not complete within ~58 min of wall time
and was killed. Python-level `signal.alarm` did not interrupt the Rust
call (PyO3 holds the GIL), so the "time-box to 600 s" target was
ineffective and a manual SIGTERM was required. At this scale, the
prototype is roughly **≥ 14× slower** than the event loop; the true
ratio is probably much worse because the loop was still progressing
when killed.

A downscaled sweep (same Kir/Fol shape, Ne divided by `scale`) shows
the regression is consistent and gets worse as Ne grows:

| scale | Ne_anc   | event-loop | analytic-middle | analytic / loop |
| ----: | -------: | ---------: | --------------: | --------------: |
| 200   | 5 000    |   0.056 s  |        5.58 s   |          100×   |
| 100   | 10 000   |   0.080 s  |        6.61 s   |           83×   |
|  50   | 20 000   |   0.127 s  |        7.11 s   |           56×   |
|  20   | 50 000   |   0.546 s  |       25.06 s   |           46×   |
|  10   | 100 000  |   2.455 s  |      126.31 s   |           51×   |
|   5   | 200 000  |  11.476 s  |     1181.38 s   |          103×   |
|   1   | 1 000 000|    251 s   |   > 3 480 s     |           ≥ 14× |

Why analytic-middle loses at Kir/Fol scale (root-cause analysis):

1. The Kir/Fol window from t_split=14 k to t_inv=330 k is
   `316 k ≈ 0.16 × 2·Ne_anc` — a *short* window relative to the
   ancestral coalescent timescale. Expected within-class events are
   modest, but the Kingman rate still fires "panmictic any-pair"
   events at rate `n(n-1)/(2·2·Ne)` *on top of* the within-class S-S
   and I-I rates. With `p_inv(ancestral)=0.3`, the in-class rate
   boost is only ~3×; panmictic rate dominates.
2. Each panmictic-tagged coalescence between two inversion-bearing
   lineages triggers `apply_coalescence_partial` in partial mode,
   which produces the merged panmictic block *plus* two remnants
   (a_rem + b_rem). Net: `n` grows by 1 per panmictic event — the
   "remnant ratchet" the prototype was meant to exploit. But the
   Kingman rate does not decay as lineages accumulate (since it is
   keyed on n not on pair-overlap structure), so the loop is pulled
   into an unbounded-n trajectory bounded only by `MAX_ITERS=1e8`.
3. `reclassify_after_coal` is O(n) per event. When the ratchet pushes
   n into the tens of thousands, each event costs tens of thousands
   of ops. 100 M events × 10 k ops ≈ 10^12 ops, which matches the
   hour-scale wall time we observed.
4. Contrast: the main event loop maintains an incremental pair-rate
   cache and per-lineage recomb-clock Fenwick tree, so its per-event
   cost is sub-linear. At these parameters the event-loop path is
   already near-optimal.

The only regime where analytic-middle beats the event loop in the
original micro-benchmarks was short-window (`t_inv = 5·2·Ne`) with
large `rho` and small Ne (≤ 10 k) — i.e. where the event loop is
stalled on pre-barrier recombination work that the analytic path
simply skips. That regime is not representative of Kir/Fol or of any
realistic inversion dating problem in *Anopheles funestus*.

## Correctness test (stats KS comparison)

`rust/msinv-core/examples/validate_analytic_middle.py`. 20 reps at
`scale=100` (Ne_anc=10 k, all other Kir/Fol params unchanged).
Per-rep runtimes: event-loop mean 0.08 s, analytic-middle mean 5.5 s.
Branch-mode summary stats compared with `scipy.stats.ks_2samp`:

|           stat |   loop mean |   am mean |     KS | p-value |
| -------------: | ----------: | --------: | -----: | ------: |
|    `pi_K`      |    1 756    |    1 756  |  0.050 |  1.0000 |
|    `pi_FS`     |    3 237    |    3 231  |  0.050 |  1.0000 |
|    `pi_FI`     |    3 526    |    3 520  |  0.100 |  1.0000 |
|    `dxy_K_FS`  |   6.57 × 10⁴|  6.51 × 10⁴|  0.200 |  0.8320 |
|    `dxy_K_FI`  |   1.93 × 10⁵|  1.94 × 10⁵|  0.250 |  0.5713 |
|    `dxy_FS_FI` |   1.42 × 10⁵|  1.43 × 10⁵|  0.550 |  0.0040 |
|    `Fst_K_FS`  |     0.9268  |    0.9261 |  0.250 |  0.5713 |
|    `Fst_K_FI`  |     0.9730  |    0.9731 |  0.200 |  0.8320 |
|    `Fst_FS_FI` |     0.9533  |    0.9539 |  0.200 |  0.8320 |

Interpretation:

- Within-pop `π` (pi_K, pi_FS, pi_FI): essentially identical means,
  KS p = 1.0. Within-class diversity is preserved exactly — expected,
  since the structured rates within a single pop are an exact
  Kaplan-Hudson-Langley decomposition when recomb is absent.
- Cross-pop `dxy` (K-F): means agree to ~1 %, p > 0.5. Not
  significantly different.
- **`dxy_FS_FI`: p ≈ 0.004** (~1 % mean shift). This is the single
  stat that specifically measures inversion-karyotype S-vs-I divergence
  within the F population. A single p < 0.05 out of 9 tests with n=20
  is at the edge of what multiple-testing correction would flag; the
  1 % mean shift is real but modest.
- All three Fst estimates: p > 0.5. Indistinguishable.

Bottom line on correctness: at scale=100 the analytic-middle path
produces summary-stat distributions that are within ~1 % of the
event-loop reference and only one of nine KS tests is near-significant.
It is not bias-free — the `dxy_FS_FI` shift is consistent with the
double-counted S-S / I-I rate decomposition noted below — but the
error magnitude is small enough for ballpark ABC sims at this scale.
At Kir/Fol full scale the correctness question is moot because the
path doesn't run to completion.

### Known rate-decomposition bug

The analytic loop sums three event rates as though they were
independent, but they are not:

- within-class S-S pairs coalesce at inversion positions at rate
  `1/(2·Ne·p_std)` AND at panmictic positions at rate `1/(2·Ne)`
- "panmictic any-pair" already covers the `1/(2·Ne)` baseline for
  S-S pairs

The current code sums `rate_s = C(ks,2) / (2·Ne·p_std)` and
`rate_p = C(n,2) / (2·Ne)` without subtracting the S-S baseline out of
the panmictic bucket. That inflates total rate (and hence events/time)
and is the likely source of the `dxy_FS_FI` mean shift. The correct
in-inversion decomposition is
`rate_s_inv = C(ks,2) · (1/(2·Ne·p_std) − 1/(2·Ne))` with the
`1/(2·Ne)` baseline left to the panmictic bucket.

## What works / what doesn't

Works:
- Cargo test suite still green (69 lib + 23 integration tests). Six
  new tests in `analytic_middle::tests` exercise gate rejection,
  classifier, and end-to-end run.
- When the flag is on and the gate fires, the run finishes faster for
  the anopheles-scale regime and produces a valid tskit table set.

Does NOT currently work / known limitations:

1. **No recombination during the analytic window.** This is the core
   option-(a) simplification. For `rho > 0` the tree shape inside the
   middle window diverges from the event-loop reference:
   fewer edges, fewer internal nodes, different per-locus tree
   variation. The eventual MRCA distribution and across-population
   summary statistics (π, dxy, Fst) are in the right ballpark but NOT
   ABC-ready. Benchmark output already shows `nodes avg` diverging
   between the two paths (highrho: 5806 vs 376); that's the visible
   signal.
2. **Single inversion only.** The gate rejects multi-inversion specs
   even though the Kir/Fol paper runs 3Ra AND 3Rb together.
3. **Single pop only** (or all lineages already collapsed into one
   pop). Post-Ej single-pop regimes are supported; pre-Ej structured
   multi-pop phases are not.
4. **Panmictic-pair rate counts every pair** regardless of segment
   overlap. That's the Hudson-limit assumption (all lineages fully
   span the sequence). Once fragmentation kicks in the overall
   panmictic rate is an overestimate and coal events fire that produce
   zero edges (panmictic apply_coalescence_partial with
   non-overlapping pair = no-op). The coal wastes a draw but does not
   corrupt the tree.
5. **Pool rebuild is O(n) per event** via `reclassify_after_coal`.
   Fine for the target regime (`n ≈ 100-1000` at middle-phase start)
   but scales poorly if someone pushes `n` into the 10k range with
   this flag on — the per-event cost goes back up linearly.
6. **rate_index panic is fixed on main** (`d441aaa`, pack_ij / unpack_ij
   widened to u64). The feature branch was recut from main after that
   commit, so the full-scale bench is no longer blocked by the panic.
   The analytic-middle path still loses by a wide margin at full scale
   for the reasons in §"Why analytic-middle loses at Kir/Fol scale".

## Recommendation (2026-04-22 session)

**Do not ship the gated `analytic_middle: bool` fast path.** The
prototype is a correctness-calibration hazard (see §"Known
rate-decomposition bug" and the `dxy_FS_FI` KS result) and a
performance regression at every scale we actually care about for
Kir/Fol inversion-age ABC. The 6.9–38.9× micro-benchmark speedups
reported in the original table only materialise for short-window,
high-rho, low-Ne scenarios that do not represent any realistic
*Anopheles funestus* target.

Suggested path forward ("Path 2" in the session summary):

1. Drop the Kingman-style skip-ahead fast path entirely. Keep the
   `analytic_middle.rs` module in the tree for reference.
2. Instead, invest in a compound-rate rewrite of the main event loop:
   replace the separate bitmap/pair-cache/flux/Fenwick scans with a
   single per-event compound-rate draw that shares the O(n) structure
   that already exists in rate_index. That is the work the original
   prototype was trying to approximate, but with analytic shortcuts
   that compromised correctness.
3. If in future someone wants to revisit a structured-middle fast
   path: (i) fix the rate-decomposition bug (subtract the
   `1/(2·Ne)` baseline out of within-class buckets), (ii) decouple
   `rate_p` from `n` so the remnant ratchet does not inflate event
   counts, (iii) add a floor gate on `n_expected · events` that
   predicts when the Kingman loop would be faster than the event
   loop. With those three, the path could be revisited as an opt-in
   for research sandboxes — but not for the default ABC pipeline.

## Next steps, in order

1. **Fix node-count divergence.** The biggest open question is
   whether to promote this path from "speed demo" to "ABC usable".
   Three routes:
   - (b) from the spec: draw per-lineage recomb clocks inside the
     analytic window alongside the coal clocks. Analytic coal is still
     Kingman-style, but whenever a recomb clock wins we apply a
     recombination split and continue. The pool-rebuild cost stays
     `O(n)` per event but stays bounded in the `~few-thousand`
     regime.
   - (c): full per-event exponential-race ARG. Same shape as (b) but
     fold the flux + migration clocks back in so the window can extend
     through non-trivial demography.
   - Accept the divergence and use this path only for *summary-stat*
     sims where the middle-window tree shape is averaged over anyway.
2. **Multi-inversion support.** Extend the classifier to return a
   per-inversion karyotype vector; maintain an S/I pool per inversion
   and race all inversion rates. Important for the 3Ra+3Rb coupled
   inversion design in `project_kir_fol_model_design.md`.
3. **Correctness test.** Compare per-pair branch-mode π + Fst over
   10-20 reps between event-loop and analytic-middle on a scenario
   where recombination is genuinely irrelevant (e.g.
   `recombination_rate = 1e-12`, so rho ≈ 0). With recomb = 0 the two
   paths should produce statistically identical distributions of
   summary stats; that validates the plumbing.
4. **Push back on the O(n) rebuild.** `reclassify_after_coal`
   currently wipes + rebuilds both pools. Maintain them incrementally
   via swap_remove + push for the usual 2-in/3-out pattern of
   `apply_coalescence_partial`.
5. **Revisit the stress-case regression.** The 0.84× at Ne=10k,
   t_inv=330k suggests a minimum-n floor: skip entry if
   `t_inv - t_now < C · 2·Ne` for some `C`. Likely an uninteresting
   tuning knob once `n` comes down.
