# Example simulations

All example scripts in `examples/` use the `HullSimulator` and are
validated against empirical data or msprime ground truth.

## An. funestus Kiribina / Folonzo

Two An. funestus ecotypes in Burkina Faso, split ~1,300 years ago.
Kiribina (K) is fixed for one homokaryotype of the 3Ra and 3Rb
inversions; Folonzo (Fol) is polymorphic.

```bash
python examples/empirical_kir_fol.py
```

Output: `figures/empirical_kir_fol.pdf` showing dxy and Da
across the chromosome, with cross-karyotype divergence elevated
inside both inversions and same-karyotype Da flat — matching Small et
al. 2023 Fig S13.

## An. gambiae RDL insecticide resistance

The RDL resistance allele arose on the 2L+a background and spread
across the karyotype boundary via gene conversion under strong
insecticide selection (Grau-Bové et al. 2020 MBE).

```bash
python examples/empirical_rdl_sweep.py
```

Output: `figures/empirical_rdl_sweep.pdf` — three scenarios
(neutral / S sweep only / S then I sweep) showing the dxy_SI and
within-class pi signature.

## Presentation figures (1-8)

```bash
python examples/make_figures.py
```

Produces 8 figures in `figures/` for talks and papers:

1. `fig1_inversion_signal.pdf` — dxy / pi / Da across an inversion
2. `fig2_msprime_comparison.pdf` — msinv ↔ msprime in the no-inv limit
3. `fig3_real_inversions.pdf` — An. funestus 3Ra-like + Human MAPT-like
4. `fig4_multiple_inversions.pdf` — two inversions, two karyotype axes
5. `fig5_trajectories.pdf` — Wright-Fisher origin trajectories + age dist
6. `fig6_phi_profile.pdf` — Peischl phi(x) + cross-class T_MRCA
7. `fig7_performance.pdf` — hull timing across rho with/without inv
8. `fig8_feature_summary.pdf` — feature comparison vs msprime/discoal/SLiM
