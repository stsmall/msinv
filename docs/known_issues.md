# Known Issues

## Kir/Fol application: reversed dxy pattern with large Ne_F

When simulating Kir/Fol with the real demographic parameters from Small et al. 2023:
- Ne_K = 70,000
- Ne_F = 3,000,000 (exponential growth from Ne_Anc = 44,000)
- T_split = 14,000 gen
- t_inv = 385,000 gen (3Ra age)

The msinv simulation produces **dxy that is LOWER inside the inversion than in collinear regions**, which is the opposite of what empirical data show (Small et al. 2023, Fig. S13).

**Expected empirical pattern:**
- K-Fs (same karyotype): flat across inversion boundary
- K-Fi (different karyotype): highest inside inversion
- Fs-Fi (within Folonzo, different karyotypes): similar to K-Fi, slightly lower

**Observed simulation pattern:**
- All three comparisons show LOWER dxy inside inversions
- Collinear region between inversions shows a large spike (artifact)

### Root cause (unresolved)

Two interacting issues:

1. **Structured coalescent rate scaling with huge N**: Inside the inversion, S-S pairs coalesce at rate `k*(k-1)/2 / p_std`, which is FASTER than panmictic. This correctly models the smaller effective "S population", but gives **shorter** T_MRCA and lower dxy for same-class comparisons. In contrast, panmictic coalescent with the full Ne_F=3M gives much longer T_MRCA. So structured regions end up with LOWER diversity than panmictic regions. This is mathematically correct but produces the "wrong" empirical pattern.

2. **The 4-walk builds independent trees**: Inversion walks use structured coalescent with class-partitioned Ne. Collinear walks use panmictic coalescent with full Ne. The trees at the boundary can have very different heights, creating artifacts.

### What works correctly

- Standard coalescent tests (46/46 pass)
- SLiM validation with proper burn-in: msinv matches forward sim within 5%
- Simple inversion scenarios with constant Ne show expected phi(x) pattern
- Basic demographic inference (stdpopsim) within ~10%

### Workaround

For applications like Kir/Fol, avoid extreme Ne asymmetry. Use either:
- Constant Ne across populations
- Smaller Ne_F (e.g., 100,000 instead of 3,000,000)

### Future work

- Investigate the structured coalescent's behavior under asymmetric Ne
- Consider alternative tree construction at boundaries
- Full forward-time validation with SLiM inversion simulation
