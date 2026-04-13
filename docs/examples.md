# Example simulations

All examples in `examples/` are validated against empirical data.

## An. funestus Kiribina / Folonzo (Small et al. 2023)

Two An. funestus ecotypes in Burkina Faso split ~1,300 years ago. Kiribina is fixed for one homokaryotype of the 3Ra and 3Rb inversions; Folonzo is polymorphic.

**Real parameters from Small et al. 2023 Table S8:**
- Ne_K = 70,000 (Kiribina, rice ecotype)
- Ne_F = 3,000,000 (Folonzo, pan-African)
- Ne_Anc = 44,000 (ancestral)
- T_split = 14,000 generations (~1,300 years)
- μ = 3.55e-9 per bp per gen

**Run:**
```bash
python examples/sim_kir_fol.py
```

**Validated**: matches Fst/dxy pattern from Fig S13 (elevated divergence at 3Ra+3Rb, flat collinear).

See: `examples/make_kir_fol_figure.py` for side-by-side comparison with empirical.

## An. gambiae RDL insecticide resistance (Grau-Bové et al. 2020)

The RDL resistance allele (296G) arose on the 2L+a background and spread across karyotype boundaries via gene conversion under strong insecticide selection.

**Run:**
```bash
python examples/sim_rdl_sweep.py
```

**Validated**: reproduces the empirical haplotype asymmetry:
- Longer swept haplotype on the originating background (2L+a)
- Shorter, more eroded haplotype on the receiving background (2La)
- Faster EHH decay on the receiving karyotype

## An. gambiae 2La inversion

**Run:**
```bash
python examples/sim_2La.py
```

**Validated**: Fst = 0.53 (empirical 0.57).

## Human MAPT H1/H2 (chromosome 17q21)

**Run:**
```bash
python examples/sim_MAPT.py
```

**Validated**: dxy/site = 0.0031 (empirical 0.0026).

## Peischl et al. 2013 replication

Tests the T_SI ∝ 1/phi(x) relationship from the original Peischl paper.

**Run:**
```bash
python examples/replicate_peischl.py
```

**Validated**: T_SI ratio breakpoint/center = 29.5x (matches Peischl et al.).

## Three-way comparison: msinv vs msprime

Comparison showing msinv's unique contribution:

| Scenario                    | dxy inv | dxy col | ratio |
|-----------------------------|---------|---------|-------|
| msinv (with flux)           | 1.82    | 0.49    | 3.68  |
| msinv (no flux)             | 1.65    | 0.48    | 3.41  |
| msprime + migration matrix  | 0.51    | 0.50    | 1.02  |

msprime's uniform migration matrix gives flat divergence across the chromosome. Only msinv produces the inversion-specific elevation that is observed empirically.

## Testing gene flux / sweep interactions

The simulator supports testing the "SS bridge" hypothesis for introgression:
- A selected allele (e.g., RDL) arises on one karyotype background
- Strong selection sweeps it to high frequency
- Gene flux transfers it to the opposite background
- Over time, selection spreads it on both backgrounds

```python
sim = MsinvSimulator(
    samples=10, population_size=100_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5, p_inv=0.5, t_inv=10.0,
    sweep=(0.5, 0.1, 'S'),  # sweep at x=0.5, s=0.1, origin=S
    gene_conversion_rate=1e-9,
    seed=42,
)
```

See `examples/sim_rdl_sweep.py` for the full RDL-inspired analysis.
