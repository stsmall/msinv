# SLIM validation for msinv

Forward-time SLIM simulations compared against msinv (hull coalescent)
on three inversion scenarios, with theory lines for reference.

## Scenarios

| # | Setup | Script |
|---|-------|--------|
| 1 | Single inversion, neutral + gene flux | `scenarios/scenario1_single_inv.slim` |
| 2 | Two inversions, neutral + gene flux | `scenarios/scenario2_multi_inv.slim` |
| 3 | Single inversion + hard sweep on S karyotype at x_sel | `scenarios/scenario3_sweep_in_inv.slim` |

## Shared parameters

- Ne = 1000 diploids (WF)
- L = 100 kb
- r = 1e-7 crossover / bp / gen
- γ = 1e-8 non-crossover (gene conversion) / bp / gen, mean tract 100 bp
- μ = 1e-8 / bp / gen (overlaid by `msprime.sim_mutations` post-hoc)
- burn-in = 8·Ne = 8000 gen; t_inv = 4·Ne = 4000 gen
- Balancing selection on inversion marker (s_bal = 0.01 per allele)
  to maintain polymorphism across 4·Ne gen — msinv assumes a
  frequency, SLIM has to defend it against drift.

Scenario 3 extras: sweep allele introduced on 5 random S-karyotype
genomes at t_sweep_factor·Ne gen before sampling, s_coef = 0.05.
Runs are restarted if the sweep is lost.

Sampling: 10 S + 10 I haploid chromosomes, classified by the inv-0
karyotype marker at `bp_left`.

## Running

```bash
.venv/bin/python slim_validation/run_comparison.py --scenario 1 --reps 5
.venv/bin/python slim_validation/run_comparison.py --scenario 2 --reps 5
.venv/bin/python slim_validation/run_comparison.py --scenario 3 --reps 5
.venv/bin/python slim_validation/plot_comparison.py --all
```

Output per scenario:
- `output/scenario{N}_rep{i}_{slim,msinv}.trees` — tree sequences
- `output/scenario{N}_results.npz` — aggregated stats + timing
- `../figures/slim_validation_scenario{N}.pdf` — comparison figure

`.trees` files are git-ignored (size); `.npz` results and figures
are committed for reproducibility.

## Caveats

1. **Gene flux parameterisation differs.** msinv γ is a per-bp
   rate at which a lineage's allele is transferred between
   karyotypes. SLIM models non-crossover recombination events
   whose tracts are short (~100 bp); only tracts crossing the
   inversion boundary would be "visible" as flux, but because the
   recombination callback suppresses crossovers, short GC tracts
   inside the inversion do swap alleles. Effective rate is close
   to γ but not identical.
2. **Inversion frequency maintenance.** SLIM uses balancing
   selection to keep p ≈ 0.5; msinv simply conditions on the
   specified sampling frequencies. The balancing selection adds
   a small amount of extra coalescent time at the marker itself.
3. **Sample classification is by inv-0 only** for scenario 2;
   the inv-1 karyotype is random within each class. Matches
   msinv's default `n_std`/`n_inv` behaviour.
4. **SLIM binary path** defaults to
   `/home/ssmall/miniforge3/envs/popgen/bin/slim` (SLIM 4.2.2).
   Override via `SLIM_BIN=... .venv/bin/python …`.
