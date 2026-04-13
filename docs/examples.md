# Example simulations

All examples in `examples/` are validated against empirical data.
The `_hull.py` examples use the recommended ARG-based simulator;
the older non-suffixed examples use the legacy SMC simulator.

## An. funestus Kiribina / Folonzo

Two An. funestus ecotypes in Burkina Faso, split ~1,300 years ago.
Kiribina (K) is fixed for one homokaryotype of the 3Ra and 3Rb
inversions; Folonzo (Fol) is polymorphic.

**Hull version (recommended):**
```bash
python examples/empirical_kir_fol_hull.py
```

Output: `figures/empirical_kir_fol_hull.pdf` showing dxy and Da
across the chromosome, with the cross-karyotype divergence elevated
inside both inversions and same-karyotype Da flat — matching Small et
al. 2023 Fig S13.

**Legacy SMC version:**
```bash
python examples/sim_kir_fol.py            # 4-scenario figure
python examples/make_kir_fol_figure.py    # publication-style figure
```

## An. gambiae RDL insecticide resistance

The RDL resistance allele arose on the 2L+a background and spread
across the karyotype boundary via gene conversion under strong
insecticide selection (Grau-Bové et al. 2020 MBE).

**Hull version (recommended):**
```bash
python examples/empirical_rdl_sweep_hull.py
```

Output: `figures/empirical_rdl_sweep_hull.pdf` — three scenarios
(neutral / S sweep only / S then I sweep) showing the dxy_SI and
within-class pi signature.

**Legacy SMC version:**
```bash
python examples/sim_rdl_sweep.py
```

## Other applications (legacy SMC simulator)

```bash
python examples/sim_2La.py            # An. gambiae 2La (Fst ~0.53)
python examples/sim_MAPT.py           # Human MAPT H1/H2 (dxy ~0.003)
python examples/replicate_peischl.py  # Peischl 2013 T_SI ∝ 1/phi(x)
```

## Three-way bake-off (msprime ↔ SMC ↔ hull)

```bash
python examples/bakeoff.py
```

Six scenarios, head-to-head comparison. Highlights:

- Panmictic baseline: all three engines agree.
- Single inversion γ=0: SMC ≈ Hull (~1% diff).
- Two-pop split: msprime/Hull agree; **SMC has a known bug** (cross-pop
  dxy is ~half of expected).
- Multi-inv: SMC ≈ Hull.

## Pure SMC ↔ hull comparison

```bash
python examples/compare_smc_vs_hull.py
```

Side-by-side single-inversion comparison with mutations dropped via
msprime on the hull TS so per-bp pi/dxy units are directly
comparable. Demonstrates within ~5% agreement on inv-only single-pop
scenarios.

## Tree topology diagnostic

```bash
python examples/visualize_trees.py
```

Output: `figures/tree_topology_diagnostic.pdf` — tree topology at
five positions (outside, near each breakpoint, centre). Shows that
inside the inversion S samples cluster cleanly with each other and
only meet I samples above t_inv. Outside the inversion samples mix
panmictically.

## Balanced-Ne demonstration

```bash
python examples/demo_kir_fol_balanced_Ne.py
```

Output: `figures/demo_kir_fol_Ne_balance.pdf` — same Kir/Fol scenario
but with constant Ne for both populations. Demonstrates that the dxy
"depression" seen in the original Kir/Fol example with Ne_F=3M is a
known structured-coal artifact at extreme Ne asymmetry, not a bug.
