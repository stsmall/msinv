# Validation Infrastructure + Pilot Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `validation/_lib/` infrastructure (stats, equivalence, IO, seeds) for the 5-track msinv validation suite, then run the phase-0 pilot bench to confirm msinv is tractable at L=5 Mb + Ne=1e6 + n=100 reps before committing HPC compute to Tracks 1+2+5.

**Architecture:** Pure-Python validation library in `validation/_lib/` (no msinv source changes). All stats use tskit native + scipy + hand-rolled H-stats (no new heavy deps). Per-rep summary stats persist as `.npz` (consistent with existing `slim_validation/output/` pattern). Pilot bench runs a single msinv scenario at full scale, 3 reps, and measures wall + peak RSS + iters consumed. Pass/fail gates per spec.

**Tech Stack:** Python 3.12, numpy, tskit, msprime, scipy.stats, msinv (HEAD). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md`

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `validation/__init__.py` | Create | Package marker |
| `validation/_lib/__init__.py` | Create | Subpackage marker |
| `validation/_lib/seeds.py` | Create | Deterministic seed generation per (track, scenario, engine, rep) |
| `validation/_lib/stats.py` | Create | tskit-based stats: window stats, SFS, tree-shape, LD, H-stats |
| `validation/_lib/equivalence.py` | Create | KS test + Cohen's D + verdict |
| `validation/_lib/io.py` | Create | save/load per-rep `.npz`; aggregate across reps |
| `validation/pilot/__init__.py` | Create | Package marker |
| `validation/pilot/bench_msinv.py` | Create | Phase-0 pilot: msinv at L=5 Mb + measure timing/RSS |
| `tests/validation/__init__.py` | Create | Test package marker |
| `tests/validation/test_seeds.py` | Create | seed determinism + collision tests |
| `tests/validation/test_stats_windows.py` | Create | window-stat shape + Fst formula tests |
| `tests/validation/test_stats_treeshape.py` | Create | tree-shape stat tests |
| `tests/validation/test_stats_ld.py` | Create | LD r²-decay tests |
| `tests/validation/test_stats_hstats.py` | Create | H1/H12/H2H1 hand-rolled tests |
| `tests/validation/test_equivalence.py` | Create | KS + Cohen's D + verdict tests |
| `tests/validation/test_io.py` | Create | save/load round-trip + aggregate tests |
| `tests/validation/test_pilot_bench.py` | Create | Pilot harness smoke test (scaled-down params) |
| `results/validation/pilot/` | Create dir | Pilot output (gitignored or kept small) |
| `.tmp/pilot_report.md` | Create | Pilot pass/fail report (one-shot, not committed) |

No msinv source modifications. No `slim_validation/` modifications.

---

### Task 1: Create validation/ skeleton + deterministic seeds

**Files:**
- Create: `validation/__init__.py` (empty)
- Create: `validation/_lib/__init__.py` (empty)
- Create: `validation/_lib/seeds.py`
- Create: `tests/validation/__init__.py` (empty)
- Create: `tests/validation/test_seeds.py`

**Why:** Foundation for all subsequent tasks. Seeds need to be reproducible and free of cross-track / cross-engine collisions.

- [ ] **Step 1: Write the failing test**

Create `tests/validation/test_seeds.py`:

```python
"""Tests for validation seed generation."""
import pytest
from validation._lib.seeds import seed_for


def test_same_inputs_give_same_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    assert s1 == s2


def test_different_rep_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="msinv", rep=1)
    assert s1 != s2


def test_different_engine_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="slim", rep=0)
    assert s1 != s2


def test_different_track_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track2", scenario="default", engine="msinv", rep=0)
    assert s1 != s2


def test_seed_in_uint32_range():
    """Seeds must fit in uint32 for SLiM/msprime/discoal compatibility."""
    s = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    assert 1 <= s <= 2**31 - 1


def test_no_collisions_across_10k_combos():
    """Random sample of 10k (track, rep, engine) tuples should give 10k distinct seeds."""
    seeds = set()
    for track_i in range(5):
        for engine in ("msinv", "slim", "msprime", "discoal"):
            for rep in range(500):
                seeds.add(seed_for(track=f"track{track_i}", scenario="default",
                                   engine=engine, rep=rep))
    assert len(seeds) == 5 * 4 * 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/validation/test_seeds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation'`

- [ ] **Step 3: Write the skeleton + seeds module**

Create `validation/__init__.py`:

```python
"""msinv validation suite (publication-grade cross-engine comparison)."""
```

Create `validation/_lib/__init__.py`:

```python
"""Shared validation library: stats, equivalence, IO, seeds."""
```

Create `tests/validation/__init__.py` as an empty file.

Create `validation/_lib/seeds.py`:

```python
"""Deterministic seed generation for the validation suite.

Seeds are derived from a stable hash of (track, scenario, engine, rep) and
clamped to the uint31 range so they round-trip through SLiM, msprime, and
discoal (all of which use 32-bit signed seeds).
"""
import hashlib


def seed_for(*, track: str, scenario: str, engine: str, rep: int) -> int:
    """Return a deterministic uint31 seed for (track, scenario, engine, rep).

    Use kwargs only so call sites are explicit.
    """
    key = f"{track}|{scenario}|{engine}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big")
    return (raw % (2**31 - 1)) + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/validation/test_seeds.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/ tests/validation/
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): scaffold + deterministic seed generation

Foundation for the 5-track validation suite. Seeds derive from a SHA-256
of (track, scenario, engine, rep) clamped to uint31 so they're stable
across reruns and cross-engine compatible.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Window-level stats (π, dxy, Fst, Tajima's D, SFS)

**Files:**
- Create: `validation/_lib/stats.py`
- Create: `tests/validation/test_stats_windows.py`

**Why:** These are the headline stats every track needs. tskit has native implementations; this module wraps them in a uniform per-window interface that matches the spec.

- [ ] **Step 1: Write failing tests**

Create `tests/validation/test_stats_windows.py`:

```python
"""Tests for window-level stats: pi, dxy, Fst, Tajima's D, SFS."""
import numpy as np
import msprime
import pytest

from validation._lib.stats import window_stats, sfs


@pytest.fixture
def two_pop_ts():
    """Small msprime ts with 2 populations for stat testing."""
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=1000)
    demography.add_population(name="B", initial_size=1000)
    demography.set_migration_rate(source="A", dest="B", rate=1e-4)
    demography.set_migration_rate(source="B", dest="A", rate=1e-4)
    ts = msprime.sim_ancestry(
        samples={"A": 10, "B": 10},
        demography=demography,
        sequence_length=100_000,
        recombination_rate=1e-7,
        random_seed=42,
        ploidy=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=43)
    return ts


def test_window_stats_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    pop_b = list(ts.samples(population=1))
    out = window_stats(ts, sample_sets={"A": pop_a, "B": pop_b}, n_windows=40)
    assert out["pi"]["A"].shape == (40,)
    assert out["pi"]["B"].shape == (40,)
    assert out["dxy"]["A_B"].shape == (40,)
    assert out["fst"]["A_B"].shape == (40,)
    assert out["tajimas_d"]["A"].shape == (40,)


def test_window_stats_pi_positive(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    out = window_stats(ts, sample_sets={"A": pop_a}, n_windows=40)
    assert (out["pi"]["A"] >= 0).all()


def test_window_stats_fst_in_range(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    pop_b = list(ts.samples(population=1))
    out = window_stats(ts, sample_sets={"A": pop_a, "B": pop_b}, n_windows=40)
    fst = out["fst"]["A_B"]
    valid = ~np.isnan(fst)
    assert ((fst[valid] >= -0.01) & (fst[valid] <= 1.01)).all()


def test_sfs_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    s = sfs(ts, sample_set=pop_a, folded=True)
    n = len(pop_a)
    assert s.shape == (n // 2 + 1,)
    assert (s >= 0).all()


def test_sfs_unfolded_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    s = sfs(ts, sample_set=pop_a, folded=False)
    n = len(pop_a)
    assert s.shape == (n + 1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_windows.py -v`
Expected: FAIL with `ImportError: cannot import name 'window_stats'`

- [ ] **Step 3: Implement window_stats and sfs**

Create `validation/_lib/stats.py`:

```python
"""Per-rep stats for the validation suite.

All implementations are tskit-native where possible. The H-stats and LD
decay are hand-rolled in later tasks since tskit does not provide them.
"""
from __future__ import annotations

import numpy as np
import tskit


def window_stats(
    ts: tskit.TreeSequence,
    *,
    sample_sets: dict[str, list[int]],
    n_windows: int = 40,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-window pi, dxy, Fst, Tajima's D for each set / set-pair.

    Returns
    -------
    {"pi":         {name: (n_windows,) array, ...},
     "dxy":        {f"{a}_{b}": (n_windows,) array, ...},
     "fst":        {f"{a}_{b}": (n_windows,) array, ...},
     "tajimas_d":  {name: (n_windows,) array, ...}}

    `dxy` and `fst` are computed for every ordered pair of sets (a, b)
    with a < b lexicographically.
    """
    wins = np.linspace(0, ts.sequence_length, n_windows + 1)
    names = sorted(sample_sets)

    pi = {}
    tajd = {}
    for name in names:
        pi[name] = ts.diversity(
            [sample_sets[name]], windows=wins, mode="site"
        ).reshape(-1)
        tajd[name] = ts.Tajimas_D(
            [sample_sets[name]], windows=wins, mode="site"
        ).reshape(-1)

    dxy = {}
    fst = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}_{b}"
            d = ts.divergence(
                [sample_sets[a], sample_sets[b]], windows=wins, mode="site"
            ).reshape(-1)
            dxy[key] = d
            pi_w = (pi[a] + pi[b]) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                fst[key] = np.where(d > 0, 1.0 - pi_w / d, np.nan)

    return {"pi": pi, "dxy": dxy, "fst": fst, "tajimas_d": tajd}


def sfs(
    ts: tskit.TreeSequence,
    *,
    sample_set: list[int],
    folded: bool = True,
) -> np.ndarray:
    """Site frequency spectrum (folded or unfolded).

    Folded shape: (len(sample_set) // 2 + 1,).
    Unfolded shape: (len(sample_set) + 1,).
    """
    return ts.allele_frequency_spectrum(
        [sample_set], polarised=not folded, span_normalise=False
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_windows.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/_lib/stats.py tests/validation/test_stats_windows.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): window-level stats (pi, dxy, Fst, TajD, SFS)

tskit-native wrappers in a uniform per-window interface. Returns nested
dicts keyed by sample-set name / pair name. Fst follows the standard
1 - mean_within / mean_between formula.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Tree-shape, LD, and H-stats

**Files:**
- Modify: `validation/_lib/stats.py` (append: tree_shape_stats, ld_decay, hstats)
- Create: `tests/validation/test_stats_treeshape.py`
- Create: `tests/validation/test_stats_ld.py`
- Create: `tests/validation/test_stats_hstats.py`

**Why:** Tree-shape is the "topology comparison" surrogate; LD and H-stats are required by the spec for sweep tracks 4 + 5. Hand-rolled to avoid pulling scikit-allel.

- [ ] **Step 1: Write failing tests for tree-shape**

Create `tests/validation/test_stats_treeshape.py`:

```python
"""Tests for tree-shape distributions (TMRCA, total branch, Colless)."""
import numpy as np
import msprime
import pytest

from validation._lib.stats import tree_shape_stats


@pytest.fixture
def small_ts():
    ts = msprime.sim_ancestry(
        samples=10, population_size=1000,
        sequence_length=10_000, recombination_rate=1e-7,
        random_seed=11, ploidy=1)
    return ts


def test_tree_shape_returns_three_dists(small_ts):
    out = tree_shape_stats(small_ts, n_samples=50)
    assert "tmrca" in out
    assert "total_branch" in out
    assert "colless" in out
    assert out["tmrca"].shape == (50,)
    assert out["total_branch"].shape == (50,)
    assert out["colless"].shape == (50,)


def test_tmrca_positive(small_ts):
    out = tree_shape_stats(small_ts, n_samples=20)
    assert (out["tmrca"] > 0).all()


def test_total_branch_positive(small_ts):
    out = tree_shape_stats(small_ts, n_samples=20)
    assert (out["total_branch"] > 0).all()


def test_colless_in_range(small_ts):
    """Colless index for n leaves is in [0, (n-1)*(n-2)/2]."""
    out = tree_shape_stats(small_ts, n_samples=20)
    n = small_ts.num_samples
    upper = (n - 1) * (n - 2) // 2
    assert (out["colless"] >= 0).all()
    assert (out["colless"] <= upper).all()
```

- [ ] **Step 2: Verify tree-shape tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_treeshape.py -v`
Expected: FAIL with `ImportError: cannot import name 'tree_shape_stats'`

- [ ] **Step 3: Implement tree_shape_stats**

Append to `validation/_lib/stats.py`:

```python
def tree_shape_stats(
    ts: tskit.TreeSequence, *, n_samples: int = 1000, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Distributions of TMRCA, total branch length, Colless imbalance.

    Sample n_samples random positions across the tree sequence; for each,
    extract the local tree and compute the three statistics.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, ts.sequence_length, size=n_samples)

    tmrca = np.empty(n_samples)
    total_branch = np.empty(n_samples)
    colless = np.empty(n_samples)
    for i, pos in enumerate(positions):
        tree = ts.at(float(pos))
        tmrca[i] = tree.time(tree.root)
        total_branch[i] = tree.total_branch_length
        colless[i] = _colless_imbalance(tree)
    return {"tmrca": tmrca, "total_branch": total_branch, "colless": colless}


def _colless_imbalance(tree) -> int:
    """Sum over internal nodes of |#leaves(left) - #leaves(right)|.

    Defined for binary trees. For multifurcating internal nodes, treat
    children pairwise: for k>=2 children, sum |L_i - L_j| over i<j.
    Multifurcations are rare in coalescent trees; this generalisation
    keeps the statistic finite.
    """
    leaves_below = {}
    total = 0
    for u in tree.nodes(order="postorder"):
        children = tree.children(u)
        if not children:
            leaves_below[u] = 1
        else:
            cnt = sum(leaves_below[c] for c in children)
            leaves_below[u] = cnt
            counts = [leaves_below[c] for c in children]
            for i in range(len(counts)):
                for j in range(i + 1, len(counts)):
                    total += abs(counts[i] - counts[j])
    return total
```

- [ ] **Step 4: Verify tree-shape tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_treeshape.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write failing tests for LD**

Create `tests/validation/test_stats_ld.py`:

```python
"""Tests for LD r²-decay binning."""
import numpy as np
import msprime
import pytest

from validation._lib.stats import ld_decay


@pytest.fixture
def ld_ts():
    ts = msprime.sim_ancestry(
        samples=20, population_size=1000,
        sequence_length=100_000, recombination_rate=1e-7,
        random_seed=21, ploidy=1)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=22)
    return ts


def test_ld_decay_shape(ld_ts):
    bins = np.logspace(2, 5, 11)  # 10 bins from 100 to 1e5 bp
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=2000, seed=0)
    assert out["bin_edges"].shape == (11,)
    assert out["mean_r2"].shape == (10,)
    assert out["count"].shape == (10,)


def test_ld_decay_values_in_unit(ld_ts):
    bins = np.logspace(2, 5, 11)
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=2000, seed=0)
    valid = ~np.isnan(out["mean_r2"])
    assert ((out["mean_r2"][valid] >= 0) & (out["mean_r2"][valid] <= 1)).all()


def test_ld_decay_decreases_with_distance(ld_ts):
    """At a moderate recomb rate, r² should generally decrease with distance.

    Use larger sample for better signal-to-noise; this is a soft check
    (not strictly monotonic at small N, but mean of bins 0-1 > mean of 8-9).
    """
    bins = np.logspace(2, 5, 11)
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=5000, seed=0)
    near = np.nanmean(out["mean_r2"][:2])
    far = np.nanmean(out["mean_r2"][-2:])
    assert near > far
```

- [ ] **Step 6: Verify LD tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_ld.py -v`
Expected: FAIL with `ImportError: cannot import name 'ld_decay'`

- [ ] **Step 7: Implement ld_decay**

Append to `validation/_lib/stats.py`:

```python
def ld_decay(
    ts: tskit.TreeSequence,
    *,
    distance_bins: np.ndarray,
    max_pairs: int = 5000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Mean r² in distance bins, computed over up to `max_pairs` random
    site pairs.

    Parameters
    ----------
    distance_bins : np.ndarray
        Bin edges (length n_bins + 1). Pair distance |pos_a - pos_b| is
        placed in the bin where bin_edges[i] <= d < bin_edges[i+1].

    Returns
    -------
    {"bin_edges":  distance_bins,
     "mean_r2":    (n_bins,) array, NaN if a bin had no pairs,
     "count":      (n_bins,) int array}
    """
    rng = np.random.default_rng(seed)
    n_bins = len(distance_bins) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    # Build per-site genotype matrix at biallelic sites only
    geno = []
    positions = []
    for var in ts.variants():
        if len(var.alleles) != 2:
            continue
        geno.append(var.genotypes.astype(np.int8))
        positions.append(var.site.position)
    if not geno:
        return {"bin_edges": distance_bins,
                "mean_r2": np.full(n_bins, np.nan),
                "count": counts}
    geno = np.array(geno)            # shape (S, n_samples)
    positions = np.array(positions)  # shape (S,)
    n_sites = len(positions)

    # Sample random pairs (i, j) with i != j
    n_pairs = min(max_pairs, n_sites * (n_sites - 1) // 2)
    pairs_seen = 0
    while pairs_seen < n_pairs:
        i = rng.integers(0, n_sites, size=n_pairs * 2)
        j = rng.integers(0, n_sites, size=n_pairs * 2)
        ok = i != j
        i = i[ok]
        j = j[ok]
        for a, b in zip(i.tolist(), j.tolist()):
            if pairs_seen >= n_pairs:
                break
            d = abs(positions[a] - positions[b])
            bin_idx = np.searchsorted(distance_bins, d, side="right") - 1
            if 0 <= bin_idx < n_bins:
                ga = geno[a]
                gb = geno[b]
                pa = ga.mean()
                pb = gb.mean()
                pab = (ga * gb).mean()
                num = (pab - pa * pb) ** 2
                den = pa * (1 - pa) * pb * (1 - pb)
                if den > 0:
                    sums[bin_idx] += num / den
                    counts[bin_idx] += 1
            pairs_seen += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    return {"bin_edges": distance_bins, "mean_r2": mean, "count": counts}
```

- [ ] **Step 8: Verify LD tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_ld.py -v`
Expected: 3 passed.

- [ ] **Step 9: Write failing tests for H-stats**

Create `tests/validation/test_stats_hstats.py`:

```python
"""Tests for Garud H1, H12, H2/H1 hand-rolled implementation."""
import numpy as np
import pytest

from validation._lib.stats import hstats_from_haps


def test_h1_all_identical_is_one():
    """If every haplotype is identical, H1 = 1."""
    haps = np.zeros((10, 5), dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(1.0)
    assert out["H12"] == pytest.approx(1.0)


def test_h1_all_distinct():
    """10 distinct haplotypes: each at frequency 0.1, H1 = 10 * 0.1^2 = 0.1."""
    haps = np.eye(10, dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(0.1)


def test_h12_combines_top_two():
    """5 haps: AAAA, AAAA, BBBB, BBBB, CCCC.
    Frequencies: A=0.4, B=0.4, C=0.2.
    H1 = 0.4^2 + 0.4^2 + 0.2^2 = 0.36
    H12 = (0.4 + 0.4)^2 + 0.2^2 = 0.68
    """
    haps = np.array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,1,1],
                     [1,1,1,1],
                     [2,2,2,2]], dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(0.36)
    assert out["H12"] == pytest.approx(0.68)


def test_h2_over_h1_known_case():
    """Same setup as test_h12_combines_top_two.
    H2 = H1 - 0.4^2 = 0.36 - 0.16 = 0.20
    H2/H1 = 0.20 / 0.36 ≈ 0.5556.
    """
    haps = np.array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,1,1],
                     [1,1,1,1],
                     [2,2,2,2]], dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H2_over_H1"] == pytest.approx(0.20 / 0.36)
```

- [ ] **Step 10: Verify H-stats tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_hstats.py -v`
Expected: FAIL with `ImportError: cannot import name 'hstats_from_haps'`

- [ ] **Step 11: Implement hstats_from_haps and ts wrapper**

Append to `validation/_lib/stats.py`:

```python
def hstats_from_haps(haps: np.ndarray) -> dict[str, float]:
    """Garud et al. 2015 H1, H12, H2, H2/H1 from a haplotype matrix.

    Parameters
    ----------
    haps : np.ndarray, shape (n_haplotypes, n_sites)
        Genotype matrix; each row is a haplotype, each column a SNP.

    Returns
    -------
    {"H1": ..., "H12": ..., "H2": ..., "H2_over_H1": ...}
    """
    n = haps.shape[0]
    if n == 0:
        return {"H1": float("nan"), "H12": float("nan"),
                "H2": float("nan"), "H2_over_H1": float("nan")}
    # Pack each row to a tuple so we can count duplicates
    keys = [tuple(row.tolist()) for row in haps]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    freqs = sorted((c / n for c in counts.values()), reverse=True)
    H1 = float(sum(f * f for f in freqs))
    if len(freqs) >= 2:
        H12 = float((freqs[0] + freqs[1]) ** 2 + sum(f * f for f in freqs[2:]))
        H2 = float(H1 - freqs[0] * freqs[0])
    else:
        H12 = float(freqs[0] ** 2)
        H2 = 0.0
    H2_over_H1 = H2 / H1 if H1 > 0 else float("nan")
    return {"H1": H1, "H12": H12, "H2": H2, "H2_over_H1": H2_over_H1}


def hstats(
    ts: tskit.TreeSequence,
    *,
    sample_set: list[int],
    x_sel: float | None = None,
    window_bp: float | None = None,
) -> dict[str, float]:
    """Garud H-stats from a tskit ts, optionally restricted to a window
    around `x_sel` of width `window_bp`.

    If `x_sel` is None, computes H-stats genome-wide.
    """
    if x_sel is not None and window_bp is None:
        raise ValueError("window_bp required when x_sel is set")
    haps_rows = []
    for var in ts.variants(samples=sample_set):
        if len(var.alleles) != 2:
            continue
        if x_sel is not None:
            if abs(var.site.position - x_sel) > window_bp / 2:
                continue
        haps_rows.append(var.genotypes.astype(np.int8))
    if not haps_rows:
        return hstats_from_haps(np.empty((0, 0), dtype=np.int8))
    haps = np.array(haps_rows).T  # shape (n_samples, n_sites)
    return hstats_from_haps(haps)
```

- [ ] **Step 12: Verify H-stats tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_stats_hstats.py -v`
Expected: 4 passed.

- [ ] **Step 13: Commit**

Run:
```bash
git add validation/_lib/stats.py tests/validation/test_stats_treeshape.py \
        tests/validation/test_stats_ld.py tests/validation/test_stats_hstats.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): tree-shape, LD r²-decay, Garud H-stats

- tree_shape_stats: TMRCA / total branch / Colless distributions
  sampled at random positions (the "topology comparison" surrogate)
- ld_decay: binned r² across random site pairs
- hstats / hstats_from_haps: H1, H12, H2, H2/H1 hand-rolled (no
  scikit-allel dependency)

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Equivalence module (KS test + Cohen's D + verdict)

**Files:**
- Create: `validation/_lib/equivalence.py`
- Create: `tests/validation/test_equivalence.py`

**Why:** Pre-registered equivalence criteria from the spec live in this module so every track applies them identically.

- [ ] **Step 1: Write failing tests**

Create `tests/validation/test_equivalence.py`:

```python
"""Tests for KS test, Cohen's D, and equivalence verdict."""
import numpy as np
import pytest

from validation._lib.equivalence import (
    ks_test, cohens_d, equivalence_verdict,
)


def test_ks_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    stat, p = ks_test(a, b)
    assert 0 <= stat <= 1
    assert p > 0.01  # cannot reject same-distribution null


def test_ks_clearly_different():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=500)
    b = rng.normal(2.0, 1.0, size=500)  # mean shift 2 SD
    _, p = ks_test(a, b)
    assert p < 0.001


def test_cohens_d_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cohens_d(a, b) == pytest.approx(0.0)


def test_cohens_d_unit_for_one_sd_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=10_000)
    b = rng.normal(1.0, 1.0, size=10_000)
    d = cohens_d(a, b)
    assert 0.9 < abs(d) < 1.1


def test_verdict_equivalent_for_identical():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    v = equivalence_verdict(a, b)
    assert v["verdict"] == "equivalent"


def test_verdict_not_equivalent_for_clearly_different():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 500)
    b = rng.normal(2, 1, 500)
    v = equivalence_verdict(a, b)
    assert v["verdict"] == "not_equivalent"


def test_verdict_investigate_high_power_tiny_diff():
    """Large n + tiny mean shift triggers KS rejection but small Cohen's D."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 100_000)
    b = rng.normal(0.05, 1, 100_000)
    v = equivalence_verdict(a, b)
    # Could be 'investigate' (p < 0.01 but D < 0.2) or 'equivalent' if
    # KS happens to not reject — accept either, but never 'not_equivalent'.
    assert v["verdict"] in ("equivalent", "investigate")
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_equivalence.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement equivalence module**

Create `validation/_lib/equivalence.py`:

```python
"""Pre-registered equivalence criteria for the validation suite.

Equivalence is declared when KS p > alpha AND Cohen's D < d_threshold.
Equivalence is rejected when KS p < alpha AND Cohen's D > d_threshold.
The asymmetric cases (one but not both) yield a "investigate" verdict.

Defaults match the spec: alpha=0.01, d_threshold=0.2.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    res = stats.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's D effect size with pooled SD.

    d = (mean(a) - mean(b)) / s_pooled
    where s_pooled = sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) /
                          (n_a + n_b - 2))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0 if float(np.mean(a)) == float(np.mean(b)) else float("inf")
    return float((np.mean(a) - np.mean(b)) / pooled)


def equivalence_verdict(
    a: np.ndarray, b: np.ndarray,
    *, alpha: float = 0.01, d_threshold: float = 0.2,
) -> dict[str, float | Literal["equivalent", "not_equivalent", "investigate"]]:
    """Run KS + Cohen's D and return verdict per the pre-registered rule."""
    stat, p = ks_test(a, b)
    d = cohens_d(a, b)
    if np.isnan(p) or np.isnan(d):
        verdict = "investigate"
    elif p > alpha and abs(d) < d_threshold:
        verdict = "equivalent"
    elif p < alpha and abs(d) > d_threshold:
        verdict = "not_equivalent"
    else:
        verdict = "investigate"
    return {"ks_stat": stat, "ks_p": p, "cohens_d": d, "verdict": verdict}
```

- [ ] **Step 4: Verify tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_equivalence.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/_lib/equivalence.py tests/validation/test_equivalence.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): KS test + Cohen's D equivalence verdict

Pre-registered per the spec: equivalence ⟺ KS p > 0.01 AND |D| < 0.2.
Rejection ⟺ p < 0.01 AND |D| > 0.2. Asymmetric cases → "investigate".

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: IO module (per-rep .npz persistence + aggregation)

**Files:**
- Create: `validation/_lib/io.py`
- Create: `tests/validation/test_io.py`

**Why:** Per-rep stats must persist across reps and across runs so the equivalence test can operate on the full n=100 distribution. `.npz` is consistent with existing `slim_validation/output/` and adds no dependency.

- [ ] **Step 1: Write failing tests**

Create `tests/validation/test_io.py`:

```python
"""Tests for save/load of per-rep stats + aggregation."""
import numpy as np
import pytest

from validation._lib.io import (
    save_rep_stats, load_rep_stats, aggregate_track,
)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "rep_000" / "stats.npz"
    save_rep_stats(
        path,
        pi__A=np.array([1.0, 2.0, 3.0]),
        dxy__A_B=np.array([4.0, 5.0, 6.0]),
        timing_seconds=12.34,
    )
    loaded = load_rep_stats(path)
    np.testing.assert_array_equal(loaded["pi__A"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(loaded["dxy__A_B"], [4.0, 5.0, 6.0])
    assert float(loaded["timing_seconds"]) == pytest.approx(12.34)


def test_aggregate_three_reps(tmp_path):
    track_dir = tmp_path / "track_test"
    for r in range(3):
        path = track_dir / f"rep_{r:03d}" / "stats.npz"
        save_rep_stats(
            path,
            pi__A=np.array([float(r), float(r) + 1.0]),
        )
    agg = aggregate_track(track_dir)
    np.testing.assert_array_equal(agg["pi__A"], [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])


def test_aggregate_skips_missing(tmp_path):
    track_dir = tmp_path / "track_partial"
    save_rep_stats(track_dir / "rep_000" / "stats.npz",
                    pi__A=np.array([1.0]))
    save_rep_stats(track_dir / "rep_002" / "stats.npz",
                    pi__A=np.array([3.0]))
    agg = aggregate_track(track_dir)
    np.testing.assert_array_equal(agg["pi__A"], [[1.0], [3.0]])
    assert agg["__rep_indices__"].tolist() == [0, 2]
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_io.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement io module**

Create `validation/_lib/io.py`:

```python
"""Per-rep .npz persistence + cross-rep aggregation."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REP_RE = re.compile(r"^rep_(\d+)$")


def save_rep_stats(path: str | Path, **stats: np.ndarray | float) -> None:
    """Save a dict of per-rep stats to `.npz`. Creates parent directories.

    Keys may be hierarchical (use `__` as separator, e.g. `pi__A`).
    Scalar values are stored as 0-d arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_rep_stats(path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-rep `.npz` into a dict of numpy arrays."""
    z = np.load(Path(path), allow_pickle=False)
    return {k: z[k] for k in z.files}


def aggregate_track(track_dir: str | Path) -> dict[str, np.ndarray]:
    """Stack per-rep stats from `track_dir/rep_NNN/stats.npz` into
    arrays of shape (n_reps, ...).

    Reps are discovered by scanning subdirs matching `rep_NNN`. Missing
    reps are skipped (their indices recorded under `__rep_indices__`).
    """
    track_dir = Path(track_dir)
    rep_pairs = []
    for sub in sorted(track_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = REP_RE.match(sub.name)
        if not m:
            continue
        npz = sub / "stats.npz"
        if not npz.exists():
            continue
        rep_pairs.append((int(m.group(1)), load_rep_stats(npz)))
    if not rep_pairs:
        return {"__rep_indices__": np.array([], dtype=np.int64)}
    indices = np.array([r for r, _ in rep_pairs], dtype=np.int64)
    keys = set()
    for _, d in rep_pairs:
        keys.update(d.keys())
    out: dict[str, np.ndarray] = {"__rep_indices__": indices}
    for k in keys:
        rows = [d[k] for _, d in rep_pairs if k in d]
        if len(rows) == len(rep_pairs):
            out[k] = np.stack(rows)
    return out
```

- [ ] **Step 4: Verify tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_io.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/_lib/io.py tests/validation/test_io.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): per-rep .npz persistence + cross-rep aggregation

save_rep_stats / load_rep_stats / aggregate_track. Hierarchical keys
via __-separator (e.g. pi__A, fst__A_B). aggregate_track scans rep_NNN
subdirs and stacks into (n_reps, ...) arrays.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Pilot bench harness (msinv at L=5 Mb, Ne=1e6)

**Files:**
- Create: `validation/pilot/__init__.py` (empty)
- Create: `validation/pilot/bench_msinv.py`
- Create: `tests/validation/test_pilot_bench.py`

**Why:** This is the phase-0 gate from the spec. Runs a single msinv simulation under the production-target parameters and measures per-rep wall + peak RSS + iters consumed.

- [ ] **Step 1: Write failing tests (smoke at scaled-down params)**

Create `tests/validation/test_pilot_bench.py`:

```python
"""Smoke tests for the pilot bench harness at SCALED-DOWN params.

The full bench (L=5 Mb, Ne=1e6) is too slow for unit tests; we test
the harness mechanics here and run the real bench manually in Task 7.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.pilot.bench_msinv import run_pilot_rep


def test_smoke_run_creates_outputs(tmp_path):
    """Run the bench at toy params and verify it produces stats + timing."""
    out_dir = tmp_path / "rep_000"
    result = run_pilot_rep(
        out_dir=out_dir,
        rep=0,
        L=10_000,
        Ne=1000,
        n_samples=10,
        inv_bp_left=2_500.0,
        inv_bp_right=7_500.0,
        t_inv=4_000.0,
        mu=1e-7,
        r=1e-7,
        gc_rate=1e-9,
        seed=12345,
    )
    assert (out_dir / "stats.npz").exists()
    assert (out_dir / "timing.json").exists()
    timing = json.loads((out_dir / "timing.json").read_text())
    assert "wall_seconds" in timing
    assert "peak_rss_bytes" in timing
    assert "iters_consumed" in timing
    assert timing["wall_seconds"] > 0
    assert result["wall_seconds"] == timing["wall_seconds"]


def test_smoke_stats_has_expected_keys(tmp_path):
    out_dir = tmp_path / "rep_000"
    run_pilot_rep(
        out_dir=out_dir, rep=0,
        L=10_000, Ne=1000, n_samples=10,
        inv_bp_left=2_500.0, inv_bp_right=7_500.0, t_inv=4_000.0,
        mu=1e-7, r=1e-7, gc_rate=1e-9, seed=12345,
    )
    z = np.load(out_dir / "stats.npz", allow_pickle=False)
    keys = set(z.files)
    # Spot-check a few stats from each module are present
    assert any(k.startswith("pi__") for k in keys)
    assert any(k.startswith("dxy__") for k in keys)
    assert any(k.startswith("fst__") for k in keys)
    assert any(k.startswith("tajimas_d__") for k in keys)
    assert any(k.startswith("tree_") for k in keys)
    assert any(k.startswith("ld_") for k in keys)
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv/bin/python -m pytest tests/validation/test_pilot_bench.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the pilot harness**

Create `validation/pilot/__init__.py` as an empty file.

Create `validation/pilot/bench_msinv.py`:

```python
"""Phase-0 pilot bench: msinv at the validation-suite scale.

Runs a single rep, measures wall + peak RSS + iters consumed, computes
the full validation-suite stats panel, and persists everything to
`out_dir / {stats.npz, timing.json}`.

Used to gate the full n=100 launch: per-rep wall < 4h AND peak RSS < 8GB
must both hold.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import msprime
import tskit

from msinv import HullSimulator, InversionSpec
from validation._lib import io, stats


def run_pilot_rep(
    *,
    out_dir: str | Path,
    rep: int,
    L: float,
    Ne: float,
    n_samples: int,
    inv_bp_left: float,
    inv_bp_right: float,
    t_inv: float,
    mu: float,
    r: float,
    gc_rate: float,
    seed: int,
    iters_max: int = 200_000_000,
) -> dict[str, float]:
    """Run one msinv pilot rep at the given parameters and persist outputs.

    Returns a small dict with the timing info that's also written to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = InversionSpec(
        bp_left=float(inv_bp_left),
        bp_right=float(inv_bp_right),
        p_inv=0.5,
        t_inv=float(t_inv),
        gene_conversion_rate=float(gc_rate),
        inv_id=0,
    )

    n_std = n_samples // 2
    n_inv = n_samples - n_std
    sim = HullSimulator(
        n_std=n_std, n_inv=n_inv,
        population_size=float(Ne),
        sequence_length=float(L),
        recombination_rate=float(r),
        inversions=[inv],
        seed=int(seed),
        iters_max=iters_max,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    ts_raw = sim.simulate()
    wall = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux; report bytes for clarity
    peak_rss = max(rss_before, rss_after) * 1024

    # Overlay neutral mutations
    ts = msprime.sim_mutations(ts_raw, rate=float(mu),
                                random_seed=seed + 1, keep=True)

    # Sample-set partition: first n_std = "S", rest = "I"
    samples = list(ts.samples())
    sset = {"S": samples[:n_std], "I": samples[n_std:]}

    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_S = stats.sfs(ts, sample_set=sset["S"], folded=True)
    sfs_I = stats.sfs(ts, sample_set=sset["I"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(L), 11)
    ld_d = stats.ld_decay(ts, distance_bins=bins, max_pairs=2000, seed=seed + 3)

    flat: dict[str, np.ndarray] = {}
    for sname, arr in win["pi"].items():
        flat[f"pi__{sname}"] = arr
    for pname, arr in win["dxy"].items():
        flat[f"dxy__{pname}"] = arr
    for pname, arr in win["fst"].items():
        flat[f"fst__{pname}"] = arr
    for sname, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{sname}"] = arr
    flat["sfs__S"] = sfs_S
    flat["sfs__I"] = sfs_I
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_bin_edges"] = ld_d["bin_edges"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["ld_count"] = ld_d["count"]
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)

    timing = {
        "wall_seconds": float(wall),
        "peak_rss_bytes": int(peak_rss),
        "iters_consumed": int(getattr(sim, "iters_used", -1)),
        "rep": int(rep),
        "L": float(L),
        "Ne": float(Ne),
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def _cli_main():
    """Run the production-scale pilot: 3 reps at L=5 Mb, Ne=1e6, n=100."""
    import sys
    from validation._lib.seeds import seed_for

    out_root = Path("results/validation/pilot")
    n_reps = 3
    timings = []
    for rep in range(n_reps):
        out_dir = out_root / f"rep_{rep:03d}"
        seed = seed_for(track="pilot", scenario="default",
                          engine="msinv", rep=rep)
        print(f"Pilot rep {rep}: seed={seed}, out={out_dir}", flush=True)
        t = run_pilot_rep(
            out_dir=out_dir, rep=rep,
            L=5_000_000, Ne=1_000_000, n_samples=100,
            inv_bp_left=2_000_000.0, inv_bp_right=3_000_000.0,
            t_inv=4_000_000.0,
            mu=1e-8, r=1e-8, gc_rate=1e-9, seed=seed,
        )
        timings.append(t)
        print(f"  wall={t['wall_seconds']:.1f}s, "
              f"peak_rss={t['peak_rss_bytes'] / 1e9:.2f} GB",
              flush=True)
    walls = [t["wall_seconds"] for t in timings]
    rsses = [t["peak_rss_bytes"] for t in timings]
    print(f"\nPilot summary over {n_reps} reps:")
    print(f"  wall: median={np.median(walls):.1f}s, "
          f"min={min(walls):.1f}s, max={max(walls):.1f}s")
    print(f"  rss : median={np.median(rsses) / 1e9:.2f}GB, "
          f"max={max(rsses) / 1e9:.2f}GB")
    if max(walls) > 4 * 3600:
        print("  GATE: ❌ per-rep wall > 4h — escalate before full launch")
        sys.exit(2)
    if max(rsses) > 8 * 1e9:
        print("  GATE: ⚠️ per-rep RSS > 8GB — discuss before full launch")
        sys.exit(1)
    print("  GATE: ✅ within pilot pass criteria")


if __name__ == "__main__":
    _cli_main()
```

- [ ] **Step 4: Verify tests pass**

Run: `.venv/bin/python -m pytest tests/validation/test_pilot_bench.py -v --timeout=120`
Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/pilot/ tests/validation/test_pilot_bench.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): phase-0 pilot bench harness

run_pilot_rep: one msinv rep + full validation-suite stats panel +
timing/RSS persisted to per-rep dir. CLI driver runs 3 reps at the
production-target params (L=5Mb, Ne=1e6, n=100) and applies the
spec's pass-gate: wall < 4h AND RSS < 8GB.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Run the pilot at production scale + report

**Files:**
- Create: `results/validation/pilot/rep_{000,001,002}/{stats.npz, timing.json}`
- Create: `.tmp/pilot_report.md`

**Why:** This is the actual gating measurement. Determines whether the next plan targets full n=100 or a scaled-down regime.

- [ ] **Step 1: Run the pilot**

Run:
```bash
mkdir -p results/validation/pilot
.venv/bin/python -m validation.pilot.bench_msinv 2>&1 \
  | tee .tmp/pilot_run.log
```
Expected: prints rep-by-rep wall + RSS, ends with `GATE: ✅ within pilot pass criteria` if both gates pass. If gate fails (exit 1 or 2), proceed to Step 3 anyway and write the report.

- [ ] **Step 2: Verify outputs exist**

Run:
```bash
ls -la results/validation/pilot/rep_*/
```
Expected: 3 directories `rep_000/`, `rep_001/`, `rep_002/`, each containing `stats.npz` and `timing.json`.

- [ ] **Step 3: Write the pilot report**

Create `.tmp/pilot_report.md` with the following structure (fill in the actual numbers from `timing.json`):

```markdown
# Pilot bench report — msinv at validation-suite scale

Date: 2026-05-09
Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md
Plan: docs/superpowers/plans/2026-05-09-validation-infra-and-pilot.md

## Parameters

L = 5 Mb, Ne = 1e6, n = 100, μ = 1e-8, r = 1e-8.
Single inversion 2e6–3e6 bp (1 Mb = 20% of L); t_inv = 4·Ne = 4e6 gen.
gc_rate = 1e-9 (msinv default-realistic).

## Per-rep timings

| rep | wall (s) | peak RSS (GB) | iters consumed | num_trees | num_sites |
|---|---|---|---|---|---|
| 0 | <fill> | <fill> | <fill> | <fill> | <fill> |
| 1 | <fill> | <fill> | <fill> | <fill> | <fill> |
| 2 | <fill> | <fill> | <fill> | <fill> | <fill> |

Median wall: <fill> s.
Median RSS: <fill> GB.

## Pass / fail vs spec gates

- ✅/❌/⚠️ Wall < 4 h per rep: <fill>
- ✅/❌/⚠️ RSS < 8 GB per rep: <fill>
- ✅/❌/⚠️ No "barrier era INCOMPLETE" warnings: <fill>

## Recommendation for Plan 2

- "Proceed to full n=100 across Tracks 1, 2, 3, 4, 5 + Q-bias side-track."
- OR: "Scale-down required — drop to L=Xkb / Ne=Y / n=Z; rerun pilot."
- OR: "Escalate to user — pilot exceeds the 4h/8GB envelope."

## Next steps

- If green: write Plan 2 (Tracks 3 + 4 + Q-bias, local; ≤50 cpu parallel)
- If yellow/red: discuss scope change with user before Plan 2

## Total compute estimate (extrapolation from pilot)

n=100 at median wall <fill>s = <fill> CPU-hours per track.
Tracks 3 + 4 + Q-bias on local 50 cpu: ~<fill> hours wall.
Tracks 1, 2, 5 on HPC SLURM at 100 cpu: ~<fill> hours wall.
```

Do not commit the report yet — Step 4 commits it together with results.

- [ ] **Step 4: Commit pilot results + report**

Run:
```bash
git add results/validation/pilot/
git add .tmp/pilot_report.md
# .tmp/ should be gitignored; check first:
git check-ignore .tmp/pilot_report.md && \
  git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test(validation): phase-0 pilot bench results at production scale

3 reps at L=5Mb, Ne=1e6, n=100. Wall/RSS/iters captured per rep.
Pass/fail vs spec gates recorded in .tmp/pilot_report.md (gitignored;
included as scratch artifact).

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
# If .tmp/ isn't gitignored, commit only the results dir:
# git reset HEAD .tmp/pilot_report.md
# git -c commit.gpgsign=false commit -m "..."
```

If `.tmp/` is not in `.gitignore`, drop the `.tmp/pilot_report.md` from staging (the report is local scratch) and commit only `results/validation/pilot/`.

- [ ] **Step 5: Report status to controller**

Output the contents of `.tmp/pilot_report.md` so the controller can decide whether to proceed to Plan 2 or escalate.

---

## Self-Review

**Spec coverage:**
- Spec § "Run ordering" phase 0 (pilot) → Tasks 6 + 7 ✓
- Spec § "Stats panel" → Task 2 (window/SFS), Task 3 (tree-shape, LD, H-stats) ✓
- Spec § "Pre-registered equivalence criteria" → Task 4 ✓
- Spec § "Output / artifacts" subset (`validation/_lib/`, `validation/pilot/`) → Tasks 1-7 ✓
- Spec § "Pilot phase 0" (wall < 4h, RSS < 8 GB gates) → Task 6 CLI driver applies them, Task 7 reports them ✓

Spec items deferred to later plans (out of scope for this plan):
- Track-specific runners (Plan 2: Tracks 3+4+Q-bias; Plan 3: Tracks 1+2+5)
- HPC SLURM scripts (Plan 3)
- Plot module (`plot.py` — deferred to Plan 2 since it has data to plot)
- Methods-section draft (Plan 3 final)

**Placeholder scan:**
- No "TBD"/"TODO" in any task body.
- The pilot report template at Task 7 has `<fill>` placeholders intentionally — those are filled at runtime from `timing.json`. The agent executing Task 7 reads `timing.json` and writes actual numbers.
- All commands shown have explicit expected output.
- All test code shown is complete (not "similar to test X").

**Type / name consistency:**
- `seed_for(track=, scenario=, engine=, rep=)` signature consistent across Tasks 1, 6.
- `window_stats` returns nested dict-of-dicts; flattened in Task 6 with `pi__A` style keys; `aggregate_track` and `equivalence_verdict` consume that key style.
- `ld_decay` returns dict with `bin_edges`/`mean_r2`/`count`; consistent in Tasks 3 + 6.
- `hstats_from_haps` (low-level) vs `hstats` (ts wrapper) — both defined in Task 3, used distinctly.
- All test imports use `from validation._lib.X import Y` consistently.

**Known soft spot:** `iters_used` attribute on `HullSimulator`. Not certain this exists on HEAD; if it doesn't, Task 6 step 3 will write `-1` for `iters_consumed` (defensive `getattr` is in the code). Not blocking — the gate only checks wall + RSS, not iters.
