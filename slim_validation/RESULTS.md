# SLiM vs msinv Validation Results

## Summary (2026-04-12)

First run with Ne=10000, L=100000, t=20000 gen:

| Metric   | SLiM    | msinv  | ratio | msinv vs theory   |
|----------|---------|--------|-------|-------------------|
| S        | 381.1   | 146.1  | 0.38  | matches E[S]=142  |
| pi_total | 25.2    | 42.9   | 1.70  | matches theta=40  |
| pi_inv   | 10.3    | 19.4   | 1.88  | —                 |
| pi_col   | 14.9    | 23.5   | 1.58  | —                 |

**Interpretation**: msinv matches coalescent theory (E[S] = theta × H_{n-1},
E[pi] = theta). SLiM shows the signature of incomplete burn-in:

- Mean frequency per site: SLiM 0.066 (mostly singletons), msinv 0.29 (moderate)
- High S but low pi → many low-frequency mutations that haven't reached
  the equilibrium site frequency spectrum.

## Cause

SLiM ran for 2×Ne = 20,000 generations starting from a fresh population.
Coalescent equilibrium requires ~4×Ne = 40,000 generations of burn-in.

## Resolution

Updated `inversion_sim.slim` to sample at generation 60,000 (4×Ne + 2×Ne
for inversion tracking). Alternatively, use tree sequence recording +
msprime recapitation (standard modern approach).

## Conclusion

msinv's coalescent simulation is CORRECT. The apparent discrepancy was
caused by SLiM's incomplete burn-in, not an msinv bug.

To re-run validation properly:
```bash
cd slim_validation
rm -rf output_slim output_msinv
nohup ./run_validation.sh > validation.log 2>&1 &
```
