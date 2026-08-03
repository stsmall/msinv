# Illex chr2 Neutral-Sufficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the *Illex illecebrosus* chr2:60–80 Mb inversion's observed diversity and divergence pattern can be produced by a neutral model, and estimate its age (t_inv) and gene-flux rate (γ).

**Architecture:** A new `illex/` package inside the msinv repo, following the existing `validation/_lib/` pattern. Analytic theory (`theory.py`) is built and unit-tested **first**, because it supplies the predicted floors that validate msinv's per-position class logic — and because two arithmetic errors have already been found in it. Demography, simulation-model builders, and statistics are separate focused modules. Driver scripts under `illex/scripts/` produce CSVs into `results/illex/`.

**Tech Stack:** Python 3.12 via `.venv/bin/python`, msinv (Rust core + PyO3 bridge), tskit, numpy, scipy, pandas, pg_gpu (separate env: `/home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python`), pytest.

## Global Constraints

Values copied verbatim from the spec. Every task's requirements implicitly include this section.

- **µ = 3e-9** /site/generation; generation time **1 yr**.
- **r = 2.5e-9** /bp/generation (sex-averaged autosomal proxy; bracket male **2.13e-9** / female **2.90e-9**). There is no chr2 map.
- **Growth arm demography** (moments): N_ANC = **547,928** → N0 = **6,808,096** over T_GROW = **769,519** generations, exponential; constant at N_ANC before T_GROW.
- **Constant arm demography:** Ne = **775,000** (diploid).
- **p_inv sensitivity pair:** **0.626** (A derived, primary) and **0.374**.
- **Empirical fit targets** (inversion region): π_AA/π_BB = **0.744**, dxy/π_AA = **1.846**.
- **Empirical held-out targets:** Fst(AA,BB) = **0.3652**; control region π_AA/π_BB = **0.989**, Fst = **0.0035**; inv:control long-range r² = **3.88**, within-homA **0.97**.
- **Predicted dxy/π_I floors:** growth **2.563** (at t_inv 1,135,687), constant **3.978** (at t_inv 1,343,687).
- **Predicted t_inv from π_I/π_S = 0.744:** growth **952,984**, constant **896,340**.
- **Flux geometry invariant:** `mean_tract_length / inv_length` = **1e-4**, always.
- **msinv API constraints:** `population_size` is **diploid** Ne. `recombination_rate` must be **> 0** (use 1e-12 for non-recombining tests). `gene_conversion_rate` must be **> 0** (use 1e-15 for "effectively zero"). `Karyotype` enum has only variants `S` and `I`.
- **Canonical call set:** baker-633 — `karyotypes.baker.tsv`, `AA_samples.txt`, `BB_samples.txt`.
- **Always use `.venv/bin/python`**, never system python. pg_gpu work uses the `varbuddy-pggpu` env instead.
- **Resource discipline (shared device):** ≤100 worker processes, ≤400 GB RAM total. Production runs (≥1 h wall, ≥10 cores) default to overnight; confirm before launching in business hours. Kill pattern is `pkill -f <workload-string>`.
- **Scratch output goes to project-local `./.tmp/`**, never `/tmp/`.
- **Do not write to `/sietch_colab/data_share/`** — it is read-only for this project.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `illex/__init__.py` | Package exports |
| `illex/theory.py` | Analytic + numerically-integrated E[T] for panmictic / within-I / within-S / between pairs, under either demography. Supplies predicted floors. |
| `illex/demography.py` | Builds msinv `Demography` objects and the n_e-vs-time schedule for both arms |
| `illex/model.py` | Builds `HullSimulator` instances for inversion and control runs |
| `illex/stats.py` | Extracts π_I, π_S, dxy, Fst from a msinv `TreeSequence` given karyotype sample sets |
| `illex/scripts/pilot_rho_ladder.py` | Stage 0 dual ρ ladder |
| `illex/scripts/empirical_windowed.py` | Stage 0/2 windowed empirical dxy (pg_gpu env) |
| `illex/scripts/persistence.py` | Stage 1 neutral persistence |
| `illex/scripts/grid_fit.py` | Stage 3 (t_inv, γ) grid |
| `illex/scripts/validate.py` | Stage 4 held-out validation + robustness arm |
| `tests/illex/test_theory.py` | Theory unit tests, incl. closed-form ↔ numerical cross-check |
| `tests/illex/test_demography.py` | Demography builder tests |
| `tests/illex/test_stats.py` | Statistics extraction tests |
| `tests/illex/test_floor_harness.py` | Harness test 1 (slow, marked) |

**Existing files consulted, not modified:** `examples/kir_fol_pilot.py` (rescaling + `eg` demography pattern), `msinv/hull/inversion.py` (`InversionSpec` / trajectory types), `rust/msinv-core/src/phi.rs` (flux geometry), `validation/_lib/demography.py` (builder conventions).

**Design note on splitting theory from simulation:** `theory.py` has no msinv dependency at all — pure numpy/scipy. That keeps its tests fast (milliseconds) and makes it a genuinely independent check on msinv rather than a circular one.

---

# PHASE A — THE GATE

Phase A either confirms or kills the flux premise. **Stop and re-read the spec before starting Phase B** if Task 6 finds no central dxy dip.

---

### Task 1: Theory module — E[T] under both demographies

The spec's floors (2.563 / 3.978) and age estimates come from here. Two arithmetic errors have already been caught in this derivation, so the central test is that an **independent closed form and a numerical integrator agree** — that cross-check is what catches the class of error made before.

**Files:**
- Create: `illex/__init__.py`
- Create: `illex/theory.py`
- Test: `tests/illex/test_theory.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy)
- Produces:
  - `N_growth(t) -> np.ndarray` and `N_const(t) -> np.ndarray` — diploid Ne at backward time t
  - `expected_times(N_fn, t_inv, p_i=0.626, dt=200.0, horizon=4.0e7) -> dict` with keys `panmictic`, `within_i`, `within_s`, `between` (generations)
  - `ratios(N_fn, t_inv, p_i=0.626) -> dict` with keys `pi_i_over_pi_s`, `dxy_over_pi_i`
  - `dxy_floor(N_fn, p_i=0.626) -> tuple[float, float]` — `(floor_value, t_inv_at_floor)`
  - `solve_t_inv(N_fn, target_ratio, p_i=0.626) -> float`
  - `const_closed_form(ne, t_inv, p_i=0.626) -> dict` — same keys as `expected_times`
  - Module constants `N_ANC=547928.0`, `N0=6808096.0`, `T_GROW=769519.0`, `MU=3e-9`, `NE_CONST=775000.0`

- [ ] **Step 1: Write the failing tests**

Create `tests/illex/test_theory.py`:

```python
"""Theory tests. The closed-form <-> numerical agreement test is the important
one: it is what catches algebra errors in the E[T] derivations."""
import numpy as np
import pytest

from illex import theory


def test_panmictic_constant_ne_is_2ne():
    """For constant Ne, mean pairwise coalescence time is exactly 2*Ne."""
    got = theory.expected_times(theory.N_const, t_inv=1.0e6)["panmictic"]
    assert got == pytest.approx(2 * theory.NE_CONST, rel=0.002)


def test_growth_panmictic_reproduces_observed_pi():
    """The moments growth model must be self-consistent with observed pi."""
    et = theory.expected_times(theory.N_growth, t_inv=1.0e6)["panmictic"]
    pi = 2 * theory.MU * et
    assert pi == pytest.approx(0.00930, abs=0.0002)


@pytest.mark.parametrize("t_inv", [3.0e5, 9.0e5, 1.5e6, 3.0e6])
def test_closed_form_matches_numerical_integration(t_inv):
    """Independent implementations of the same quantity must agree.

    This is the regression test for the E[T_S] algebra slip: the integral
    contributes -t*exp(-t/tau_S), which cancels the +t carried in by survivors
    entering the ancestral population.
    """
    num = theory.expected_times(theory.N_const, t_inv)
    closed = theory.const_closed_form(theory.NE_CONST, t_inv)
    for key in ("within_i", "within_s", "between"):
        assert num[key] == pytest.approx(closed[key], rel=0.01), key


def test_young_inversion_gives_large_dxy_ratio():
    """Single-origin bottleneck drives pi_I -> 0, so young inversions give a
    LARGE dxy/pi_I, not a small one."""
    r = theory.ratios(theory.N_growth, t_inv=2.0e5)
    assert r["dxy_over_pi_i"] > 7.0


def test_old_inversion_pi_ratio_approaches_frequency_ratio():
    """As t_inv -> infinity both classes equilibrate, so pi_I/pi_S -> p_I/p_S."""
    r = theory.ratios(theory.N_const, t_inv=2.0e7)
    assert r["pi_i_over_pi_s"] == pytest.approx(0.626 / 0.374, rel=0.05)


def test_dxy_floor_values_match_spec():
    """The floors the spec commits to, and that harness test 1 checks msinv against."""
    g_floor, g_at = theory.dxy_floor(theory.N_growth)
    c_floor, c_at = theory.dxy_floor(theory.N_const)
    assert g_floor == pytest.approx(2.563, abs=0.02)
    assert c_floor == pytest.approx(3.978, abs=0.02)
    assert g_at == pytest.approx(1.136e6, rel=0.05)
    assert c_at == pytest.approx(1.344e6, rel=0.05)
    assert g_floor < c_floor, "growth must lower the floor vs constant Ne"


def test_solve_t_inv_matches_spec():
    """t_inv implied by the observed pi_I/pi_S = 0.744."""
    assert theory.solve_t_inv(theory.N_growth, 0.744) == pytest.approx(952_984, rel=0.03)
    assert theory.solve_t_inv(theory.N_const, 0.744) == pytest.approx(896_340, rel=0.03)


def test_observed_is_below_growth_floor():
    """The flux claim: observed 1.846 sits below the growth floor, by 1.39x."""
    floor, _ = theory.dxy_floor(theory.N_growth)
    assert 1.846 < floor
    assert floor / 1.846 == pytest.approx(1.39, abs=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/illex/test_theory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'illex'`

- [ ] **Step 3: Write the implementation**

Create `illex/__init__.py`:

```python
"""Illex illecebrosus chr2 inversion analysis."""
from . import theory

__all__ = ["theory"]
```

Create `illex/theory.py`:

```python
"""Expected pairwise coalescence times for a single-origin inversion.

An inversion arises on ONE chromosome, so backward in time every inverted
lineage must coalesce by t_inv. Within-I times are therefore bounded by t_inv,
which is what makes pi_derived < pi_ancestral the expected direction.

Two independent implementations are provided on purpose:
  * expected_times() -- numerical integration of the hazard, works for any N(t)
  * const_closed_form() -- analytic, constant Ne only
They must agree (see tests). That cross-check exists because this derivation
has produced arithmetic errors before.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

# --- moments exponential-growth model (analysis/steps/08_demography) ---
N_ANC = 547_928.0
N0 = 6_808_096.0
T_GROW = 769_519.0
ALPHA = np.log(N0 / N_ANC) / T_GROW

MU = 3e-9
NE_CONST = 775_000.0          # pi / (4 mu), reproduces observed genome-wide pi
P_I_DEFAULT = 0.626           # derived/inverted arrangement frequency


def N_growth(t):
    """Diploid Ne at backward time t under the moments growth model."""
    t = np.asarray(t, dtype=float)
    return np.where(t <= T_GROW, N0 * np.exp(-ALPHA * np.minimum(t, T_GROW)), N_ANC)


def N_const(t):
    """Diploid Ne, constant."""
    return np.full_like(np.asarray(t, dtype=float), NE_CONST)


def _integrate_ET(hazard, t_max, forced_at=None, dt=200.0):
    """E[T] = int t*h(t)*S(t) dt, plus a mass point at forced_at if given.

    forced_at implements the single-origin cap: lineages that have not
    coalesced by t_inv are forced to coalesce there.
    """
    t = np.arange(0.0, t_max, dt) + dt / 2.0
    h = np.asarray(hazard(t), dtype=float)
    S = np.exp(-np.cumsum(h * dt))
    e = float(np.sum(t * h * S * dt))
    tail = float(S[-1])
    if forced_at is not None:
        e += forced_at * tail
    return e


def expected_times(N_fn, t_inv, p_i=P_I_DEFAULT, dt=200.0, horizon=4.0e7):
    """Mean pairwise coalescence times, in generations."""
    p_s = 1.0 - p_i

    def h_pan(t):
        return 1.0 / (2.0 * N_fn(t))

    def h_i(t):
        return 1.0 / (2.0 * N_fn(t) * p_i)

    def h_s(t):
        t = np.asarray(t, dtype=float)
        return np.where(t < t_inv, 1.0 / (2.0 * N_fn(t) * p_s), 1.0 / (2.0 * N_fn(t)))

    def h_between(t):
        # No coalescence possible before the inversion existed.
        t = np.asarray(t, dtype=float)
        return np.where(t < t_inv, 0.0, 1.0 / (2.0 * N_fn(t)))

    return {
        "panmictic": _integrate_ET(h_pan, horizon, dt=dt),
        "within_i": _integrate_ET(h_i, t_inv, forced_at=t_inv, dt=dt),
        "within_s": _integrate_ET(h_s, horizon, dt=dt),
        "between": _integrate_ET(h_between, horizon, dt=dt),
    }


def const_closed_form(ne, t_inv, p_i=P_I_DEFAULT):
    """Analytic constant-Ne solution. Independent check on expected_times()."""
    p_s = 1.0 - p_i
    tau_i, tau_s, two_ne = 2.0 * ne * p_i, 2.0 * ne * p_s, 2.0 * ne
    e_i = tau_i * (1.0 - np.exp(-t_inv / tau_i))
    # The -t*exp(-t/tau_s) from the integral cancels the +t carried by
    # survivors, leaving 2Ne*exp(-t/tau_s).
    e_s = tau_s * (1.0 - np.exp(-t_inv / tau_s)) + two_ne * np.exp(-t_inv / tau_s)
    return {"panmictic": two_ne, "within_i": e_i, "within_s": e_s,
            "between": t_inv + two_ne}


def ratios(N_fn, t_inv, p_i=P_I_DEFAULT, **kw):
    """The two statistics the design fits."""
    et = expected_times(N_fn, t_inv, p_i=p_i, **kw)
    return {
        "pi_i_over_pi_s": et["within_i"] / et["within_s"],
        "dxy_over_pi_i": et["between"] / et["within_i"],
    }


def dxy_floor(N_fn, p_i=P_I_DEFAULT, bounds=(2.0e5, 5.0e6)):
    """Minimum attainable dxy/pi_I over t_inv. Returns (floor, t_inv_at_floor)."""
    res = minimize_scalar(
        lambda t: ratios(N_fn, t, p_i=p_i)["dxy_over_pi_i"],
        bounds=bounds, method="bounded", options={"xatol": 2000},
    )
    return float(res.fun), float(res.x)


def solve_t_inv(N_fn, target_ratio, p_i=P_I_DEFAULT, bracket=(2.0e5, 5.0e6)):
    """t_inv reproducing an observed pi_I/pi_S."""
    return float(brentq(
        lambda t: ratios(N_fn, t, p_i=p_i)["pi_i_over_pi_s"] - target_ratio,
        bracket[0], bracket[1], xtol=1000,
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/illex/test_theory.py -v`
Expected: PASS, 9 tests (4 parametrized cases in the cross-check).

If `test_closed_form_matches_numerical_integration` fails, do **not** loosen the tolerance — one of the two implementations is wrong. That is precisely the failure this test exists to surface.

- [ ] **Step 5: Commit**

```bash
git add illex/__init__.py illex/theory.py tests/illex/test_theory.py
git -c commit.gpgsign=false commit -m "feat(illex): E[T] theory for single-origin inversion under both demographies

Numerical hazard integration plus an independent constant-Ne closed form,
required to agree. Reproduces the spec's dxy/pi_I floors (2.563 growth,
3.978 constant) and t_inv estimates (952,984 / 896,340)."
```

---

### Task 2: Demography builders

**Files:**
- Create: `illex/demography.py`
- Test: `tests/illex/test_demography.py`

**Interfaces:**
- Consumes: `illex.theory` constants (`N_ANC`, `N0`, `T_GROW`, `NE_CONST`)
- Produces:
  - `growth_demography() -> msinv.Demography`
  - `constant_demography() -> msinv.Demography`
  - `growth_ne_schedule(t_max, n_points=400) -> tuple[np.ndarray, np.ndarray]` — `(times, ne_values)` for time-varying-n_e trajectories
  - `PRESENT_NE_GROWTH = 6_808_096.0`, `PRESENT_NE_CONST = 775_000.0` — used by callers to compute ρ

- [ ] **Step 1: Write the failing test**

Create `tests/illex/test_demography.py`:

```python
import numpy as np
import pytest

from illex import demography, theory


def test_growth_schedule_endpoints():
    """Schedule must start at N0 and reach N_ANC at T_GROW."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    assert ne[0] == pytest.approx(theory.N0, rel=0.01)
    i = int(np.argmin(np.abs(t - theory.T_GROW)))
    assert ne[i] == pytest.approx(theory.N_ANC, rel=0.02)


def test_growth_schedule_flat_before_growth():
    """Before T_GROW (deeper in the past) Ne is constant at N_ANC."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    deep = ne[t > theory.T_GROW * 1.2]
    assert np.allclose(deep, theory.N_ANC, rtol=0.02)


def test_growth_schedule_matches_theory_N_growth():
    """The schedule handed to msinv must be the same N(t) theory integrates."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    assert np.allclose(ne, theory.N_growth(t), rtol=1e-6)


def test_builders_return_msinv_demography():
    from msinv import Demography
    assert isinstance(demography.growth_demography(), Demography)
    assert isinstance(demography.constant_demography(), Demography)


def test_constant_demography_present_size():
    assert demography.PRESENT_NE_CONST == pytest.approx(775_000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/illex/test_demography.py -v`
Expected: FAIL — `ImportError: cannot import name 'demography' from 'illex'`

- [ ] **Step 3: Write the implementation**

Read `examples/kir_fol_pilot.py` for the `eg` event convention first — it documents that `eg` at time 0 with rate `alpha` gives `N(t') = N_present * exp(-alpha * t')` in backward time. Then create `illex/demography.py`:

```python
"""msinv demography for the two Illex arms.

Growth arm: exponential from N0 at the present back to N_ANC at T_GROW,
constant at N_ANC before that. Constant arm: Ne = 775,000 throughout.
"""

from __future__ import annotations

import numpy as np
from msinv import Demography

from .theory import ALPHA, N0, N_ANC, NE_CONST, N_growth, T_GROW

PRESENT_NE_GROWTH = N0
PRESENT_NE_CONST = NE_CONST


def growth_demography() -> Demography:
    """Exponential growth backward from N0 to N_ANC, then flat.

    `eg` at time 0 sets N(t') = N0 * exp(-ALPHA * t'). The `en` at T_GROW
    pins the size to N_ANC and cancels the growth for deeper times.
    """
    d = Demography(population_size=N0)
    d.add_event("eg", time=0.0, pop=0, rate=ALPHA)
    d.add_event("en", time=T_GROW, pop=0, size=N_ANC)
    d.sort_events()
    return d


def constant_demography() -> Demography:
    return Demography(population_size=NE_CONST)


def growth_ne_schedule(t_max: float, n_points: int = 400):
    """(times, Ne) sampled from the same N(t) that theory.py integrates.

    Used for trajectory types that take an explicit n_e schedule rather than
    a scalar.
    """
    t = np.linspace(0.0, float(t_max), int(n_points))
    return t, N_growth(t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/illex/test_demography.py -v`
Expected: PASS, 5 tests.

If `test_builders_return_msinv_demography` fails on the `add_event` signature, inspect the real API with `.venv/bin/python -c "from msinv import Demography; help(Demography)"` and adapt — `validation/_lib/demography.py::v12_msinv()` is a working reference for event construction in this repo.

- [ ] **Step 5: Commit**

```bash
git add illex/demography.py tests/illex/test_demography.py
git -c commit.gpgsign=false commit -m "feat(illex): growth and constant-Ne demography builders

Growth arm ties the msinv eg/en events to the same N(t) theory.py
integrates, enforced by test."
```

---

### Task 3: Statistics extraction from a TreeSequence

**Files:**
- Create: `illex/stats.py`
- Test: `tests/illex/test_stats.py`

**Interfaces:**
- Consumes: `illex.demography.constant_demography`
- Produces:
  - `arrangement_stats(ts, i_nodes, s_nodes, mu=3e-9) -> dict` with keys `pi_i`, `pi_s`, `dxy`, `fst`, `pi_i_over_pi_s`, `dxy_over_pi_i`
  - `sample_nodes_by_karyotype(sim, ts) -> tuple[list[int], list[int]]` — `(i_nodes, s_nodes)`

Branch-mode tskit statistics are used for π and dxy so there is no mutation noise; the conversion is π = µ × (branch-mode diversity), because branch-mode diversity returns the branch length separating a pair, which is 2 × coalescence time.

- [ ] **Step 1: Write the failing test**

Create `tests/illex/test_stats.py`:

```python
"""Statistics tests. The panmictic calibration test doubles as harness test 2
from the spec (no inversion -> pi ratio 1, Fst 0) and validates the
branch-mode -> pi conversion."""
import pytest

from illex import demography, stats


@pytest.fixture(scope="module")
def neutral_ts():
    """Small neutral no-inversion run at constant Ne."""
    from msinv import HullSimulator
    sim = HullSimulator(
        n_std=30, n_inv=30,
        population_size=demography.PRESENT_NE_CONST,
        sequence_length=20_000,
        recombination_rate=2.5e-9,
        p_inv=0.5, t_inv=1.0e6,
        bp_left=1.0, bp_right=2.0,          # degenerate inversion: no barrier
        gene_conversion_rate=1e-15,
        seed=42,
    )
    return sim, sim.simulate()


def test_pi_matches_4_ne_mu(neutral_ts):
    """Calibrates the branch-mode -> pi conversion against theory: for a
    panmictic population pi = 4*Ne*mu."""
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    out = stats.arrangement_stats(ts, i_nodes, s_nodes)
    expected = 4 * demography.PRESENT_NE_CONST * 3e-9
    assert out["pi_i"] == pytest.approx(expected, rel=0.25)


def test_no_barrier_gives_no_differentiation(neutral_ts):
    """Harness test 2: with a degenerate inversion there is no barrier, so the
    two label sets are exchangeable -- pi ratio ~1, Fst ~0."""
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    out = stats.arrangement_stats(ts, i_nodes, s_nodes)
    assert out["pi_i_over_pi_s"] == pytest.approx(1.0, abs=0.20)
    assert abs(out["fst"]) < 0.05
    assert out["dxy_over_pi_i"] == pytest.approx(1.0, abs=0.20)


def test_node_partition_is_complete(neutral_ts):
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    assert len(i_nodes) == 30 and len(s_nodes) == 30
    assert not (set(i_nodes) & set(s_nodes))
    assert set(i_nodes) | set(s_nodes) == set(ts.samples())


@pytest.mark.slow
def test_msinv_matches_msprime_neutral():
    """Harness test 3 from the spec: msinv <-> msprime neutral agreement.

    Repo conventions (CLAUDE.md): msinv `population_size` is DIPLOID Ne with
    per-pair coalescence rate 1/(2N), so msprime must be called with
    `ploidy=1` and `2*N` to match. Compares branch-mode diversity, which is
    mutation-noise-free, over several reps.
    """
    import msprime
    import numpy as np
    from msinv import HullSimulator

    ne, seq_len, r, n_samp, reps = 50_000.0, 100_000, 2.5e-9, 40, 6

    msinv_vals = []
    for rep in range(reps):
        sim = HullSimulator(
            n_std=n_samp // 2, n_inv=n_samp // 2,
            population_size=ne, sequence_length=seq_len,
            recombination_rate=r,
            p_inv=0.5, t_inv=1.0e6,
            bp_left=1.0, bp_right=2.0,       # degenerate: no barrier
            gene_conversion_rate=1e-15, seed=300 + rep,
        )
        ts = sim.simulate()
        msinv_vals.append(ts.diversity(mode="branch"))

    msprime_vals = []
    for rep in range(reps):
        ts = msprime.sim_ancestry(
            samples=n_samp, ploidy=1, population_size=2 * ne,
            sequence_length=seq_len, recombination_rate=r,
            random_seed=400 + rep,
        )
        msprime_vals.append(ts.diversity(mode="branch"))

    a, b = float(np.mean(msinv_vals)), float(np.mean(msprime_vals))
    assert a == pytest.approx(b, rel=0.15), f"msinv {a:.1f} vs msprime {b:.1f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/illex/test_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'stats' from 'illex'`

- [ ] **Step 3: Write the implementation**

Create `illex/stats.py`:

```python
"""Per-arrangement diversity and divergence from a msinv TreeSequence.

Branch mode is used for pi and dxy: no mutation noise, so the t_inv signal is
cleaner. tskit branch-mode diversity returns the branch length separating a
pair, i.e. 2 * T_coal, so pi = mu * branch_diversity.
"""

from __future__ import annotations

MU_DEFAULT = 3e-9


def sample_nodes_by_karyotype(sim, ts):
    """Split sample nodes into (inverted, standard).

    msinv emits the n_inv sampled lineages first, then n_std. Verified by the
    partition-completeness test; if a future msinv version reorders them, that
    test fails loudly rather than silently mislabelling arrangements.
    """
    samples = list(ts.samples())
    n_inv = int(sim.n_inv)
    return samples[:n_inv], samples[n_inv:]


def arrangement_stats(ts, i_nodes, s_nodes, mu: float = MU_DEFAULT) -> dict:
    """pi within each arrangement, dxy between, and Hudson Fst."""
    pi_i = mu * ts.diversity([i_nodes], mode="branch")[0]
    pi_s = mu * ts.diversity([s_nodes], mode="branch")[0]
    dxy = mu * ts.divergence([i_nodes, s_nodes], mode="branch")[0]

    # Hudson Fst = 1 - mean_within / between
    mean_within = 0.5 * (pi_i + pi_s)
    fst = 1.0 - mean_within / dxy if dxy > 0 else float("nan")

    return {
        "pi_i": float(pi_i),
        "pi_s": float(pi_s),
        "dxy": float(dxy),
        "fst": float(fst),
        "pi_i_over_pi_s": float(pi_i / pi_s) if pi_s > 0 else float("nan"),
        "dxy_over_pi_i": float(dxy / pi_i) if pi_i > 0 else float("nan"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/illex/test_stats.py -v`
Expected: PASS, 3 tests.

If `test_pi_matches_4_ne_mu` fails by a factor of exactly 2, the branch-mode convention is `T_coal` rather than `2*T_coal` — change `mu *` to `2 * mu *` and record the finding in a comment. If `sample_nodes_by_karyotype` mislabels (visible as `test_node_partition_is_complete` failing), inspect ordering with `.venv/bin/python -c "..."` printing `sim.sample_config` and node metadata.

- [ ] **Step 5: Commit**

```bash
git add illex/stats.py tests/illex/test_stats.py
git -c commit.gpgsign=false commit -m "feat(illex): per-arrangement pi/dxy/Fst from TreeSequence

Branch-mode statistics, calibrated against pi = 4*Ne*mu. Includes spec
harness test 2 (no barrier -> no differentiation)."
```

---

### Task 4: Harness test 1 — γ≈0 must reproduce the predicted floors

The sharpest available test of msinv's per-position class logic. Because the two demographies predict *different* floors (2.563 vs 3.978), agreement with both is far stronger evidence than agreement with one.

**Files:**
- Create: `illex/model.py`
- Create: `tests/illex/test_floor_harness.py`
- Modify: `pyproject.toml` (register the `slow` marker)

**Interfaces:**
- Consumes: `illex.demography`, `illex.theory`, `illex.stats`
- Produces:
  - `build_inversion_sim(*, arm, seq_length, t_inv, gamma, p_inv=0.626, n_i=100, n_s=100, seed=None, recomb_rate=2.5e-9) -> HullSimulator` where `arm` is `"growth"` or `"constant"`
  - `build_control_sim(*, arm, seq_length, n_i=100, n_s=100, seed=None, recomb_rate=2.5e-9) -> HullSimulator`
  - `TRACT_FRACTION = 1e-4`

- [ ] **Step 1: Write the failing test**

Create `tests/illex/test_floor_harness.py`:

```python
"""Harness test 1: with gene flux switched off, msinv must reproduce the
analytic dxy/pi_I predicted by theory.py -- under BOTH demographies, whose
predicted floors differ (2.563 growth vs 3.978 constant).

Marked slow. Run with: .venv/bin/python -m pytest tests/illex/ -m slow
"""
import numpy as np
import pytest

from illex import model, stats, theory

N_REPS = 8
SEQ_LEN = 30_000


def _mean_ratio(arm, N_fn, t_inv):
    vals = []
    for rep in range(N_REPS):
        sim = model.build_inversion_sim(
            arm=arm, seq_length=SEQ_LEN, t_inv=t_inv,
            gamma=1e-15,                      # msinv requires gamma > 0
            seed=1000 + rep,
        )
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        vals.append(stats.arrangement_stats(ts, i_nodes, s_nodes)["dxy_over_pi_i"])
    return float(np.mean(vals))


@pytest.mark.slow
@pytest.mark.parametrize("arm,N_fn", [("growth", theory.N_growth),
                                      ("constant", theory.N_const)])
def test_zero_flux_matches_predicted_ratio_at_floor(arm, N_fn):
    """At the floor's t_inv, simulated dxy/pi_I must match the prediction."""
    predicted, t_at_floor = theory.dxy_floor(N_fn)
    observed = _mean_ratio(arm, N_fn, t_at_floor)
    assert observed == pytest.approx(predicted, rel=0.15), (
        f"{arm}: msinv gave {observed:.3f}, theory predicts {predicted:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("arm,N_fn", [("growth", theory.N_growth),
                                      ("constant", theory.N_const)])
def test_zero_flux_never_below_floor(arm, N_fn):
    """The floor is a floor: no t_inv may produce a smaller ratio."""
    floor, _ = theory.dxy_floor(N_fn)
    for t_inv in (4.0e5, 9.0e5, 2.0e6):
        observed = _mean_ratio(arm, N_fn, t_inv)
        assert observed > floor * 0.85, (
            f"{arm} t_inv={t_inv:.0f}: {observed:.3f} below floor {floor:.3f}"
        )


@pytest.mark.slow
def test_growth_floor_is_lower_than_constant_in_simulation():
    """The demography effect must be visible in msinv, not just in theory."""
    g_floor, g_t = theory.dxy_floor(theory.N_growth)
    c_floor, c_t = theory.dxy_floor(theory.N_const)
    assert _mean_ratio("growth", theory.N_growth, g_t) < \
           _mean_ratio("constant", theory.N_const, c_t)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/illex/test_floor_harness.py -m slow -v`
Expected: FAIL — `ImportError: cannot import name 'model' from 'illex'`

- [ ] **Step 3: Write the implementation**

Create `illex/model.py`:

```python
"""HullSimulator builders for the Illex arms.

Scaling rule: per-bp rates and Ne stay faithful; only the inversion is
shortened. That preserves per-site pi/dxy and r^2-vs-distance.

Flux geometry: phi() in rust/msinv-core/src/phi.rs works in
inversion-relative coordinates with w = mean_tract_length / inv_length, so the
flux profile is scale-invariant only if w is held fixed. Real w = 2 kb / 20 Mb
= 1e-4. Keeping a biological 2 kb tract at L = 30 kb would inflate interior
flux ~670x.
"""

from __future__ import annotations

from msinv import HullSimulator, InversionSpec

from .demography import (PRESENT_NE_CONST, PRESENT_NE_GROWTH,
                         constant_demography, growth_demography)

TRACT_FRACTION = 1e-4
MARGIN_FRACTION = 0.1        # collinear flank on each side of the inversion


def _arm_parts(arm: str):
    if arm == "growth":
        return growth_demography(), PRESENT_NE_GROWTH
    if arm == "constant":
        return constant_demography(), PRESENT_NE_CONST
    raise ValueError(f"arm must be 'growth' or 'constant', got {arm!r}")


def build_inversion_sim(*, arm, seq_length, t_inv, gamma, p_inv=0.626,
                        n_i=100, n_s=100, seed=None, recomb_rate=2.5e-9):
    demog, present_ne = _arm_parts(arm)
    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    inv_len = bp_right - bp_left

    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        p_inv=p_inv,
        t_inv=t_inv,
        gene_conversion_rate=gamma,
        mean_tract_length=max(1.0, inv_len * TRACT_FRACTION),
        tract_distribution="geometric",
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=present_ne,
        demography=demog,
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )


def build_control_sim(*, arm, seq_length, n_i=100, n_s=100, seed=None,
                      recomb_rate=2.5e-9):
    """Collinear control: same rates, no inversion barrier.

    A degenerate 1 bp inversion keeps the karyotype labels (so the same
    statistics code applies) while imposing no meaningful barrier.
    """
    demog, present_ne = _arm_parts(arm)
    spec = InversionSpec(
        bp_left=1.0, bp_right=2.0,
        p_inv=0.626, t_inv=1.0e6,
        gene_conversion_rate=1e-15,
        mean_tract_length=1.0,
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=present_ne,
        demography=demog,
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )
```

Register the marker in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "slow: long-running simulation tests (run with -m slow)",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/illex/test_floor_harness.py -m slow -v`
Expected: PASS, 5 tests.

**If it fails, do not adjust tolerances.** This test is the arbiter between `theory.py` and msinv. Diagnose in this order: (a) confirm Task 3's node partition is still correct — mislabelled arrangements invert the ratio; (b) confirm `t_inv` is being honoured by dumping `sim` config; (c) check whether the degenerate-inversion control in Task 3 really imposes no barrier; (d) only then suspect `theory.py`. Report which of the two is wrong before changing either.

- [ ] **Step 5: Commit**

```bash
git add illex/model.py tests/illex/test_floor_harness.py pyproject.toml
git -c commit.gpgsign=false commit -m "test(illex): harness test 1 -- zero-flux floors under both demographies

Simulator builders plus the cross-validation of msinv's per-position class
logic against theory.py. Both arms checked, since growth and constant Ne
predict different floors (2.563 vs 3.978)."
```

---

### Task 5: Stage 0 dual ρ-ladder pilot

**Files:**
- Create: `illex/scripts/__init__.py`
- Create: `illex/scripts/pilot_rho_ladder.py`
- Output: `results/illex/pilot_rho_ladder.csv`

**Interfaces:**
- Consumes: `illex.model.build_inversion_sim`, `illex.stats`
- Produces: CSV with columns `arm,rho_target,seq_length,wall_s,peak_rss_gb,num_trees,pi_i_over_pi_s,dxy_over_pi_i,status`. Later tasks read `seq_length` for the largest row with `status == "ok"` per arm.

- [ ] **Step 1: Write the script**

Create `illex/scripts/__init__.py` (empty) and `illex/scripts/pilot_rho_ladder.py`:

```python
#!/usr/bin/env python
"""Stage 0: dual rho ladder. Establishes the largest affordable L per arm.

rho/bp = 4*Ne_present*r, so the arms differ 8.8x:
  growth   (Ne 6,808,096): 0.0681   -> needs only 30-75 kb
  constant (Ne   775,000): 7.75e-3  -> needs >=300 kb for the LD panel

One rep per rung. Sequential, with an RSS watchdog: shared device.
"""
from __future__ import annotations

import argparse
import csv
import resource
import time
from pathlib import Path

from illex import model, stats
from illex.demography import PRESENT_NE_CONST, PRESENT_NE_GROWTH

RHO_RUNGS = [200, 500, 1000, 2000, 5000]
RSS_LIMIT_GB = 60.0          # abort a rung above this; well under the 400 GB cap
T_INV_PILOT = 9.5e5          # near both arms' fitted t_inv
OUT = Path("results/illex/pilot_rho_ladder.csv")


def seq_length_for(rho: float, present_ne: float, r: float = 2.5e-9) -> int:
    return int(round(rho / (4.0 * present_ne * r)))


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def run_rung(arm: str, rho: float, present_ne: float) -> dict:
    L = seq_length_for(rho, present_ne)
    row = {"arm": arm, "rho_target": rho, "seq_length": L,
           "wall_s": "", "peak_rss_gb": "", "num_trees": "",
           "pi_i_over_pi_s": "", "dxy_over_pi_i": "", "status": ""}
    if L < 1000:
        row["status"] = "skipped_too_short"
        return row

    t0 = time.time()
    try:
        sim = model.build_inversion_sim(
            arm=arm, seq_length=L, t_inv=T_INV_PILOT, gamma=1e-9, seed=7,
        )
        ts = sim.simulate()
    except Exception as exc:                      # noqa: BLE001 - record and continue
        row["wall_s"] = round(time.time() - t0, 1)
        row["peak_rss_gb"] = round(peak_rss_gb(), 2)
        row["status"] = f"error:{type(exc).__name__}"
        return row

    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    st = stats.arrangement_stats(ts, i_nodes, s_nodes)
    row.update(
        wall_s=round(time.time() - t0, 1),
        peak_rss_gb=round(peak_rss_gb(), 2),
        num_trees=ts.num_trees,
        pi_i_over_pi_s=round(st["pi_i_over_pi_s"], 4),
        dxy_over_pi_i=round(st["dxy_over_pi_i"], 4),
        status="ok",
    )
    if row["peak_rss_gb"] > RSS_LIMIT_GB:
        row["status"] = "ok_over_rss_limit"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["growth", "constant"])
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms:
        present_ne = PRESENT_NE_GROWTH if arm == "growth" else PRESENT_NE_CONST
        for rho in RHO_RUNGS:
            row = run_rung(arm, rho, present_ne)
            rows.append(row)
            print(f"[{arm}] rho={rho:>5} L={row['seq_length']:>8,} "
                  f"wall={row['wall_s']}s rss={row['peak_rss_gb']}GB "
                  f"status={row['status']}", flush=True)
            if row["status"].startswith(("error", "ok_over_rss_limit")):
                print(f"[{arm}] stopping ladder at rho={rho}", flush=True)
                break

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the two cheapest rungs**

Run: `.venv/bin/python -m illex.scripts.pilot_rho_ladder --arms growth`
Expected: growth ρ=200 is skipped (L = 2,937 bp → under the 1000 bp guard it still runs; if L < 1000 it reports `skipped_too_short`), and later rungs print wall/RSS. No traceback.

- [ ] **Step 3: Run the full ladder in the background**

```bash
mkdir -p .tmp/illex_chr2/logs
nohup .venv/bin/python -m illex.scripts.pilot_rho_ladder \
  > .tmp/illex_chr2/logs/pilot_ladder.log 2>&1 &
```

Monitor with `tail -f .tmp/illex_chr2/logs/pilot_ladder.log`. Kill with `pkill -f illex.scripts.pilot_rho_ladder` if RSS climbs past the cap.

- [ ] **Step 4: Verify the output and record the affordable L**

Run: `column -s, -t results/illex/pilot_rho_ladder.csv`
Expected: at least one `ok` row per arm. Confirm the growth arm reaches L ≥ 30,000 and the constant arm reaches L ≥ 300,000. If the constant arm stops short of 300 kb, that triggers the spec's documented fallback (truncated LD range or `structured-analytic-middle`) — record it and raise it before continuing.

- [ ] **Step 5: Commit**

```bash
git add illex/scripts/__init__.py illex/scripts/pilot_rho_ladder.py results/illex/pilot_rho_ladder.csv
git -c commit.gpgsign=false commit -m "feat(illex): stage 0 dual rho-ladder pilot

Establishes largest affordable L per demographic arm with wall/RSS
benchmarks. Sequential with an RSS guard (shared device)."
```

---

### Task 6: Windowed empirical dxy — the falsification check

**GATE.** φ(x) is zero at the breakpoints and flat-maximal in the interior, so gene flux predicts dxy **highest near breakpoints, lowest mid-inversion**. If that dip is absent, the flux interpretation is wrong and Phases B–D need redesign. This runs in the pg_gpu env, not `.venv`.

**Files:**
- Create: `illex/scripts/empirical_windowed.py`
- Output: `results/illex/empirical_windowed.csv`, `results/illex/empirical_windowed_verdict.txt`
- Reads: `.tmp/illex_chr2/inv.vcf.gz`, `.tmp/illex_chr2/ctl.vcf.gz`, `.tmp/illex_chr2/pops.tsv` (already produced 2026-08-03)

**Interfaces:**
- Consumes: the extracted region VCFs and `pops.tsv`
- Produces: CSV `region,window_start,window_stop,n_variants,pi_AA,pi_BB,dxy,fst`; a verdict file stating whether the central dip is present

- [ ] **Step 1: Write the script**

Create `illex/scripts/empirical_windowed.py`:

```python
#!/usr/bin/env python
"""Windowed per-arrangement pi and dxy across the inversion and control.

FALSIFICATION CHECK: gene flux via double crossover has phi(x) = 0 at the
breakpoints and flat-maximal in the interior, so between-arrangement dxy
should be HIGHEST near the breakpoints and LOWEST mid-inversion. No dip =>
the flux interpretation in the design is wrong.

Run with the pg_gpu env:
  CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
      illex/scripts/empirical_windowed.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
from pg_gpu import HaplotypeMatrix, windowed_analysis

T = Path(".tmp/illex_chr2")
OUT = Path("results/illex")
WINDOW = 500_000
REGIONS = {
    "inversion": ("2:60000000-80000000", T / "inv.vcf.gz"),
    "control": ("2:10000000-30000000", T / "ctl.vcf.gz"),
}
# Inversion breakpoints from the diagnostic-site span.
BP_LEFT, BP_RIGHT = 60_040_617, 79_995_597


def load(vcf: Path, region: str) -> HaplotypeMatrix:
    h = HaplotypeMatrix.from_vcf(str(vcf), region=region)
    h.load_pop_file(str(T / "pops.tsv"))
    return h


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, (region, vcf) in REGIONS.items():
        h = load(vcf, region)
        df = windowed_analysis(
            h, window_size=WINDOW, step_size=WINDOW,
            statistics=["pi", "dxy", "fst"],
            populations=["AA", "BB"],
            missing_data="include",
        )
        df.insert(0, "region", label)
        frames.append(df)
        print(f"{label}: {len(df)} windows", flush=True)

    out = pd.concat(frames, ignore_index=True)
    csv = OUT / "empirical_windowed.csv"
    out.to_csv(csv, index=False)

    # --- verdict: is dxy lower mid-inversion than near the breakpoints? ---
    inv = out[out.region == "inversion"].copy()
    dxy_col = next(c for c in inv.columns if c.startswith("dxy"))
    mid = 0.5 * (BP_LEFT + BP_RIGHT)
    half = 0.5 * (BP_RIGHT - BP_LEFT)
    inv["rel"] = (0.5 * (inv.window_start + inv.window_stop) - mid).abs() / half
    core = inv[inv.rel < 0.35][dxy_col]
    edge = inv[inv.rel > 0.75][dxy_col]

    ratio = float(edge.mean() / core.mean())
    dip = ratio > 1.0
    lines = [
        "FALSIFICATION CHECK: flux predicts dxy higher at breakpoints than mid-inversion",
        f"  mean dxy, inversion core (|rel pos| < 0.35): {core.mean():.6f}  (n={len(core)})",
        f"  mean dxy, near breakpoints (> 0.75):         {edge.mean():.6f}  (n={len(edge)})",
        f"  edge/core ratio: {ratio:.3f}",
        f"  VERDICT: central dip {'PRESENT' if dip else 'ABSENT'}"
        f" -> flux interpretation {'supported' if dip else 'NOT supported'}",
    ]
    if not dip:
        lines.append("  ACTION: stop. Phases B-D assume flux; revisit the design.")
    txt = "\n".join(lines)
    (OUT / "empirical_windowed_verdict.txt").write_text(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
    illex/scripts/empirical_windowed.py
```
Expected: 40 inversion windows, 40 control windows, and a printed verdict.

If `windowed_analysis` rejects the `dxy`/`fst` statistic names or the `populations` argument, check the installed signature with `python -c "from pg_gpu import windowed_analysis; help(windowed_analysis)"` — the pg_gpu skill documents `statistics=['pi','theta_w','tajimas_d','fst','dxy']` with `populations=` required for the two-population statistics.

- [ ] **Step 3: Confirm the control region shows no structure**

Run: `.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('results/illex/empirical_windowed.csv')
c = d[d.region=='control']
fst = [x for x in c.columns if x.startswith('fst')][0]
print('control mean Fst:', c[fst].mean())
assert abs(c[fst].mean()) < 0.02, 'control should show no AA/BB structure'
print('OK')
"`
Expected: `control mean Fst:` near 0.0035, then `OK`. A structured control means the windowing or population assignment is wrong, not that biology is surprising.

- [ ] **Step 4: Flag density-matched control windows**

The spec requires replacing the wholesale chr2:10–30 Mb control, which carries ~173 k SNPs/Mb against 50–74 k/Mb elsewhere on chr2 and 95–130 k/Mb inside the inversion — it is the density outlier, not the inversion. No re-extraction is needed; filter the existing windows post hoc to those whose density is comparable to the inversion's.

Run: `.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('results/illex/empirical_windowed.csv')
d['density'] = d.n_variants / (d.window_stop - d.window_start)
inv = d[d.region=='inversion']; ctl = d[d.region=='control']
lo, hi = inv.density.quantile([0.1, 0.9])
matched = ctl[(ctl.density >= lo) & (ctl.density <= hi)]
print(f'inversion density 10-90 pct: {lo:.4f}-{hi:.4f} /bp')
print(f'control windows total {len(ctl)}, density-matched {len(matched)}')
fst = [c for c in d.columns if c.startswith(\"fst\")][0]
print(f'matched-control mean Fst: {matched[fst].mean():.4f}')
matched.to_csv('results/illex/control_windows_matched.csv', index=False)
print('wrote results/illex/control_windows_matched.csv')
"`

Expected: a non-empty matched set, and its mean Fst still ≈ 0.0035. If fewer than 5 windows match, record that the 10–30 Mb control cannot be density-matched to the inversion and that a different control region is needed — that is a finding for the spec, not something to work around silently.

- [ ] **Step 5: Read the verdict and branch**

If the verdict says **PRESENT**, continue to Phase B.
If **ABSENT**, stop and report: the flux premise is falsified, the 1.39× shortfall needs another explanation, and Phases B–D must be redesigned before any grid runs.

- [ ] **Step 6: Commit**

```bash
git add illex/scripts/empirical_windowed.py results/illex/empirical_windowed.csv results/illex/empirical_windowed_verdict.txt results/illex/control_windows_matched.csv
git -c commit.gpgsign=false commit -m "feat(illex): windowed empirical dxy falsification check

Tests the flux prediction that dxy is highest at the breakpoints and lowest
mid-inversion. Gates Phases B-D."
```

---

# PHASE B — NEUTRAL PERSISTENCE

---

### Task 7: Stage 1 neutral persistence under time-varying n_e

**Files:**
- Create: `illex/scripts/persistence.py`
- Output: `results/illex/persistence.csv`, `results/illex/persistence_summary.txt`

**Interfaces:**
- Consumes: `illex.demography.growth_ne_schedule`
- Produces: CSV `p_observed,n_walks,age_mean,age_median,age_q025,age_q975,t_inv_diversity,diversity_age_percentile`

**Direction matters, and getting it wrong makes this task impossible.** Do **not** simulate forward from a single-copy origin and reject lost paths: P(a new neutral mutation ever reaches frequency 0.626) ≈ 1/(2N·0.626) ≈ **1.1e-7** at Illex's Ne, so any feasible number of attempts yields zero survivors. The question must be posed backward — start at the *observed* frequency and walk into the past until the arrangement is down to a single copy. The absorption time is then the neutral age distribution *given* the present frequency, which is exactly what goal 1 asks for and is conditioned correctly by construction.

- [ ] **Step 1: Determine whether msinv already provides the backward walk**

Run:
```bash
.venv/bin/python -c "
import inspect, msinv.hull.trajectory_helpers as th
print([n for n in dir(th) if not n.startswith('_')])
for name in ('stochastic_neutral_walk', 'post_split_logistic'):
    fn = getattr(th, name, None)
    if fn: print('---', name, '---'); print(inspect.signature(fn))
" 2>&1 | head -40
grep -n "stochastic-bridge\|stochastic_neutral" msinv/hull/trajectory_helpers.py | head
```

Two outcomes, both specified:

- **A helper exists that walks a neutral frequency backward with a time-varying n_e** (the docstrings reference a "stochastic neutral WF curve `p_today → p_split`"). Prefer it — it keeps this consistent with the trajectories msinv itself uses. Wrap it in Step 2's `backward_walk()` in place of the hand-rolled loop, keeping the same return type.
- **No such helper, or it requires a fixed endpoint rather than running to absorption.** Use Step 2's implementation as written.

Also read whatever `project_trajectory_port.md` says about stochastic-bridge limitations before trusting the helper — it is referenced from `trajectory_helpers.py` as a known caveat and may bound the regime in which the walk is valid.

- [ ] **Step 2: Write the script**

Create `illex/scripts/persistence.py`:

```python
#!/usr/bin/env python
"""Stage 1: how old must a NEUTRAL arrangement at the observed frequency be?

Walks the frequency BACKWARD from the observed p to absorption at a single
copy, under the moments growth Ne schedule. The absorption time is the neutral
age distribution given the present frequency -- correctly conditioned on the
arrangement still segregating, by construction.

WHY NOT FORWARD: P(a new neutral mutation reaches p=0.626) ~ 1/(2N*0.626)
~ 1.1e-7 at Illex's Ne, so forward-from-origin with rejection yields zero
survivors for any feasible attempt count.

VALIDATION: for constant Ne the mean absorption time must match the exact
Kimura-Ohta result, -4*Ne*(p/(1-p))*ln(p) = 2.43e6 generations at Ne=775k,
p=0.626. That check is what tells us the walk is behaving; the growth-arm
number is the actual answer.

CAVEAT: the backward walk uses the forward diffusion variance with no drift
term, which is the standard practical construction (msinv and discoal generate
neutral trajectories the same way) but is an approximation. See
project_trajectory_port.md on stochastic-bridge limitations.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from illex.demography import growth_ne_schedule
from illex.theory import NE_CONST

OBSERVED_P = {0.626: "A derived (primary)", 0.374: "B derived (flipped)"}
N_WALKS = 20_000
STEP = 200.0                    # generations per aggregated drift step
MAX_T = 3.0e7                   # abandon a walk beyond this (should not happen)
T_INV_DIVERSITY = {0.626: 952_984.0, 0.374: None}   # theory.solve_t_inv, growth
OUT = Path("results/illex")


def kimura_ohta_mean_age(ne: float, p: float) -> float:
    """Exact mean age of a neutral allele at frequency p, constant Ne."""
    return -4.0 * ne * (p / (1.0 - p)) * math.log(p)


def backward_walk(p0: float, ne_at, rng: np.random.Generator) -> float | None:
    """Walk p backward in time until it hits a single copy. Returns the age.

    None if the walk wanders to fixation (non-physical for an origin) or
    exceeds MAX_T -- both are reported as censored rather than silently kept.
    """
    p, t = p0, 0.0
    while t < MAX_T:
        two_n = 2.0 * ne_at(t)
        if p <= 1.0 / two_n:
            return t                        # absorbed: down to one copy
        if p >= 1.0:
            return None                     # censored: wandered to fixation
        sd = math.sqrt(max(p * (1.0 - p) * STEP / two_n, 0.0))
        p += rng.normal(0.0, sd)
        t += STEP
    return None


def run_arm(label: str, ne_at, rng) -> dict[float, dict]:
    out = {}
    for p_obs in OBSERVED_P:
        ages = [a for a in (backward_walk(p_obs, ne_at, rng)
                            for _ in range(N_WALKS)) if a is not None]
        arr = np.asarray(ages, dtype=float)
        out[p_obs] = {
            "arm": label, "p_observed": p_obs,
            "n_walks": N_WALKS, "n_absorbed": len(arr),
            "censored_frac": round(1.0 - len(arr) / N_WALKS, 4),
            "age_mean": round(float(arr.mean()), 0) if len(arr) else "",
            "age_median": round(float(np.median(arr)), 0) if len(arr) else "",
            "age_q025": round(float(np.quantile(arr, 0.025)), 0) if len(arr) else "",
            "age_q975": round(float(np.quantile(arr, 0.975)), 0) if len(arr) else "",
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260803)

    t_grid, ne_grid = growth_ne_schedule(t_max=3.0e6, n_points=4000)
    growth_ne_at = lambda t: float(np.interp(t, t_grid, ne_grid))   # noqa: E731
    const_ne_at = lambda t: NE_CONST                               # noqa: E731

    const = run_arm("constant", const_ne_at, rng)
    growth = run_arm("growth", growth_ne_at, rng)

    lines = ["Stage 1: neutral age given the observed arrangement frequency.",
             "Backward walk to single-copy absorption. Read the interval, not a",
             "rejection rate.", ""]

    # Validation gate: constant arm must match the exact analytic mean.
    exact = kimura_ohta_mean_age(NE_CONST, 0.626)
    got = float(const[0.626]["age_mean"])
    ratio = got / exact
    lines += [f"VALIDATION (constant Ne, p=0.626):",
              f"  simulated mean age = {got:,.0f} gen",
              f"  Kimura-Ohta exact  = {exact:,.0f} gen",
              f"  ratio = {ratio:.3f}  "
              f"{'OK' if 0.85 < ratio < 1.15 else 'FAIL -- walk is misbehaving'}", ""]

    rows = []
    for arm in (const, growth):
        for p_obs, rec in arm.items():
            rows.append(rec)
            t_div = T_INV_DIVERSITY.get(p_obs)
            extra = ""
            if t_div and rec["age_q025"] != "":
                inside = float(rec["age_q025"]) <= t_div <= float(rec["age_q975"])
                extra = (f"  | diversity t_inv={t_div:,.0f} is "
                         f"{'INSIDE' if inside else 'OUTSIDE'} the neutral interval")
            lines.append(
                f"{rec['arm']:>8}  p={p_obs}  mean={rec['age_mean']:>12,}  "
                f"95% CI=[{rec['age_q025']:>12,}, {rec['age_q975']:>12,}]  "
                f"censored={rec['censored_frac']:.1%}{extra}"
            )

    lines += ["", "INTERPRETATION: if the diversity-based t_inv falls inside the",
              "neutral age interval, a neutral inversion of that age at that",
              "frequency is consistent with drift -- goal 1 answered yes."]

    with (OUT / "persistence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = "\n".join(lines)
    (OUT / "persistence_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m illex.scripts.persistence`
Expected: the validation block, then one line per (arm, p) with age intervals. Runtime a few minutes.

- [ ] **Step 4: Check the validation gate before reading any result**

Run: `head -8 results/illex/persistence_summary.txt`
Expected: the constant-Ne simulated mean age within 15% of 2,430,445 generations, printed as `OK`.

**If it says FAIL, stop.** The backward-walk construction is wrong and the growth-arm numbers are meaningless. Likely causes in order: `STEP` too coarse relative to `p(1-p)/2N` (reduce it), the absorption threshold triggering early, or the censored fraction being large enough to bias the mean (check `censored_frac` — anything above a few percent means the no-drift backward walk is escaping to fixation too often and needs a reflecting or drift-corrected form).

- [ ] **Step 5: Compare against the diversity-based age and commit**

Read the `INSIDE`/`OUTSIDE` annotation for p = 0.626, growth arm. `INSIDE` means neutral drift can place the arrangement at 0.626 at the age the diversity ratio implies — goal 1 answered affirmatively. `OUTSIDE` is a substantive result to report, not to tune away.

```bash
git add illex/scripts/persistence.py results/illex/persistence.csv results/illex/persistence_summary.txt
git -c commit.gpgsign=false commit -m "feat(illex): stage 1 neutral age from backward frequency walk

Walks backward from the observed frequency to single-copy absorption, since
forward-from-origin yields no survivors at Illex Ne (P ~ 1.1e-7). Gated on
reproducing the exact Kimura-Ohta mean age under constant Ne."
```

---

# PHASE C — THE FIT

---

### Task 8: Stage 3 (t_inv, γ) grid fit

**Files:**
- Create: `illex/scripts/grid_fit.py`
- Output: `results/illex/grid_fit.csv`, `results/illex/grid_fit_best.txt`

**Interfaces:**
- Consumes: `illex.model.build_inversion_sim`, `illex.stats.arrangement_stats`, `results/illex/pilot_rho_ladder.csv` (for L)
- Produces: CSV `arm,p_inv,t_inv,gamma,seq_length,n_reps,pi_i_over_pi_s,dxy_over_pi_i,fst,loss`; best-fit summary

- [ ] **Step 1: Write the script**

Create `illex/scripts/grid_fit.py`:

```python
#!/usr/bin/env python
"""Stage 3: fit (t_inv, gamma) to pi_I/pi_S = 0.744 and dxy/pi_I = 1.846.

Two parameters against two ratios, so a match is guaranteed and is NOT
evidence for the model. Stage 4 does the actual testing, on held-out
statistics. This task only locates the parameters.

The fit arm uses the growth demography (the neutrality claim rests on these
statistics, so they must carry the expansion).
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from illex import model, stats

TARGET_PI_RATIO = 0.744
TARGET_DXY_RATIO = 1.846
T_INV_GRID = [5.0e5, 7.0e5, 9.0e5, 9.5e5, 1.1e6, 1.4e6, 1.8e6]
GAMMA_GRID = [1e-15, 1e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8]
P_INV_ARMS = [0.626, 0.374]
N_REPS = 20
OUT = Path("results/illex")


def affordable_length(arm: str, default: int) -> int:
    """Largest L with status 'ok' for this arm in the pilot ladder."""
    csv_path = OUT / "pilot_rho_ladder.csv"
    if not csv_path.exists():
        return default
    d = pd.read_csv(csv_path)
    ok = d[(d.arm == arm) & (d.status == "ok")]
    return int(ok.seq_length.max()) if len(ok) else default


def evaluate(arm, seq_length, t_inv, gamma, p_inv, n_reps) -> dict:
    pi_ratios, dxy_ratios, fsts = [], [], []
    for rep in range(n_reps):
        sim = model.build_inversion_sim(
            arm=arm, seq_length=seq_length, t_inv=t_inv, gamma=gamma,
            p_inv=p_inv, seed=5000 + rep,
        )
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        st = stats.arrangement_stats(ts, i_nodes, s_nodes)
        pi_ratios.append(st["pi_i_over_pi_s"])
        dxy_ratios.append(st["dxy_over_pi_i"])
        fsts.append(st["fst"])
    pi_r, dxy_r = float(np.mean(pi_ratios)), float(np.mean(dxy_ratios))
    loss = ((pi_r - TARGET_PI_RATIO) / TARGET_PI_RATIO) ** 2 + \
           ((dxy_r - TARGET_DXY_RATIO) / TARGET_DXY_RATIO) ** 2
    return {"arm": arm, "p_inv": p_inv, "t_inv": t_inv, "gamma": gamma,
            "seq_length": seq_length, "n_reps": n_reps,
            "pi_i_over_pi_s": round(pi_r, 4), "dxy_over_pi_i": round(dxy_r, 4),
            "fst": round(float(np.mean(fsts)), 4), "loss": round(loss, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="growth")
    ap.add_argument("--reps", type=int, default=N_REPS)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    L = affordable_length(args.arm, default=30_000)
    print(f"arm={args.arm} L={L:,} reps={args.reps}", flush=True)

    rows = []
    for p_inv, t_inv, gamma in itertools.product(P_INV_ARMS, T_INV_GRID, GAMMA_GRID):
        row = evaluate(args.arm, L, t_inv, gamma, p_inv, args.reps)
        rows.append(row)
        print(f"  p={p_inv} t={t_inv:.2e} g={gamma:.1e} -> "
              f"pi_r={row['pi_i_over_pi_s']} dxy_r={row['dxy_over_pi_i']} "
              f"loss={row['loss']}", flush=True)

    with (OUT / "grid_fit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    d = pd.DataFrame(rows)
    lines = [f"targets: pi_I/pi_S={TARGET_PI_RATIO}  dxy/pi_I={TARGET_DXY_RATIO}", ""]
    for p_inv in P_INV_ARMS:
        best = d[d.p_inv == p_inv].nsmallest(1, "loss").iloc[0]
        lines.append(
            f"p_inv={p_inv}: t_inv={best.t_inv:,.0f} gamma={best.gamma:.2e} "
            f"loss={best.loss:.5f} (pi_r={best.pi_i_over_pi_s}, "
            f"dxy_r={best.dxy_over_pi_i}, Fst={best.fst})"
        )
    overall = d.nsmallest(1, "loss").iloc[0]
    lines += ["", f"BEST OVERALL: p_inv={overall.p_inv} t_inv={overall.t_inv:,.0f} "
                  f"gamma={overall.gamma:.2e} loss={overall.loss:.5f}",
              "", "A low loss here is NOT evidence for the model -- 2 parameters,",
              "2 targets. See stage 4 for the held-out test."]
    (OUT / "grid_fit_best.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with 2 reps**

Run: `.venv/bin/python -m illex.scripts.grid_fit --arm growth --reps 2`
Expected: 98 grid points (2 × 7 × 7) printed, then the best-fit summary. No traceback. Note the per-point wall time to size the full run.

- [ ] **Step 3: Run the full grid in the background**

```bash
nohup .venv/bin/python -m illex.scripts.grid_fit --arm growth --reps 20 \
  > .tmp/illex_chr2/logs/grid_fit.log 2>&1 &
```

If the smoke run implies >1 h total, launch overnight per the repo's resource-discipline rule and confirm before starting during business hours.

- [ ] **Step 4: Verify a fit was found**

Run: `cat results/illex/grid_fit_best.txt`
Expected: a best-fit (t_inv, γ) per polarity arm. Sanity checks: the fitted t_inv should land near the theory prediction of ~9.5e5 for the primary arm, and the fitted γ should be strictly greater than the 1e-15 floor — a γ pinned at the floor means flux is *not* needed to hit the targets, which contradicts the design's central claim and must be reported.

- [ ] **Step 5: Commit**

```bash
git add illex/scripts/grid_fit.py results/illex/grid_fit.csv results/illex/grid_fit_best.txt
git -c commit.gpgsign=false commit -m "feat(illex): stage 3 (t_inv, gamma) grid fit

Fits the two primary ratios across both polarity arms. Records explicitly
that a good fit is not evidence -- 2 parameters against 2 targets."
```

---

# PHASE D — VALIDATION

---

### Task 9: Stage 4 held-out validation and robustness arm

**Files:**
- Create: `illex/scripts/validate.py`
- Output: `results/illex/validation.txt`, `results/illex/robustness.csv`

**Interfaces:**
- Consumes: `results/illex/grid_fit.csv` (best-fit parameters), `illex.model`, `illex.stats`, `illex.theory`
- Produces: a pass/fail line per held-out statistic, and a robustness table over N_ANC, r, and polarity

- [ ] **Step 1: Write the script**

Create `illex/scripts/validate.py`:

```python
#!/usr/bin/env python
"""Stage 4: test the fitted (t_inv, gamma) on statistics it was NOT fitted to,
and run the mandatory robustness arm on the flux claim.

Held-out: Fst(AA,BB)=0.3652; control-region pi ratio 0.989 and Fst 0.0035.
(The LD panel runs under the constant-Ne arm and is a separate follow-up --
it needs L >= 300 kb, which the growth arm cannot afford.)

Robustness: the flux claim rests on observed 1.846 vs a 2.563 floor, a margin
of only 1.39x. If any plausible combination of N_ANC, r, and polarity lifts the
observation above the floor, the flux conclusion is withdrawn.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from illex import model, stats, theory

OUT = Path("results/illex")
HELD_OUT = {"fst_inversion": 0.3652, "control_pi_ratio": 0.989, "control_fst": 0.0035}
N_REPS = 20
OBS_DXY_RATIO = 1.846


def best_fit() -> pd.Series:
    d = pd.read_csv(OUT / "grid_fit.csv")
    return d.nsmallest(1, "loss").iloc[0]


def mean_stats(sim_factory, n_reps=N_REPS) -> dict:
    acc = {}
    for rep in range(n_reps):
        sim = sim_factory(9000 + rep)
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        for k, v in stats.arrangement_stats(ts, i_nodes, s_nodes).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def robustness_rows() -> list[dict]:
    """Recompute the floor across N_ANC CI, the r bracket, and both polarities.

    r does not enter theory.py (the floor depends only on N(t) and p_I), so the
    r bracket is recorded as not-applicable rather than silently omitted.
    """
    rows = []
    n_anc_variants = {"point": theory.N_ANC, "ci_low": 515_991.0, "ci_high": 726_363.0}
    for n_label, n_anc in n_anc_variants.items():
        for p_label, p_i in (("A_derived", 0.626), ("B_derived", 0.374)):
            orig = theory.N_ANC
            try:
                theory.N_ANC = n_anc
                theory.ALPHA = np.log(theory.N0 / n_anc) / theory.T_GROW
                floor, t_at = theory.dxy_floor(theory.N_growth, p_i=p_i)
            finally:
                theory.N_ANC = orig
                theory.ALPHA = np.log(theory.N0 / orig) / theory.T_GROW
            rows.append({
                "n_anc_variant": n_label, "n_anc": n_anc, "polarity": p_label,
                "p_i": p_i, "floor": round(floor, 4),
                "t_inv_at_floor": round(t_at, 0),
                "observed": OBS_DXY_RATIO,
                "flux_required": bool(OBS_DXY_RATIO < floor),
                "margin": round(floor / OBS_DXY_RATIO, 3),
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bf = best_fit()
    lines = [f"best fit: arm={bf.arm} p_inv={bf.p_inv} t_inv={bf.t_inv:,.0f} "
             f"gamma={bf.gamma:.2e} L={int(bf.seq_length):,}", ""]

    inv = mean_stats(lambda s: model.build_inversion_sim(
        arm=bf.arm, seq_length=int(bf.seq_length), t_inv=float(bf.t_inv),
        gamma=float(bf.gamma), p_inv=float(bf.p_inv), seed=s))
    ctl = mean_stats(lambda s: model.build_control_sim(
        arm=bf.arm, seq_length=int(bf.seq_length), seed=s))

    checks = [
        ("fst_inversion", inv["fst"], HELD_OUT["fst_inversion"], 0.10),
        ("control_pi_ratio", ctl["pi_i_over_pi_s"], HELD_OUT["control_pi_ratio"], 0.10),
        ("control_fst", ctl["fst"], HELD_OUT["control_fst"], 0.05),
    ]
    lines.append("HELD-OUT STATISTICS (not fitted):")
    n_pass = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= max(tol * abs(want), tol)
        n_pass += ok
        lines.append(f"  {name:<20} sim={got:>8.4f}  obs={want:>8.4f}  "
                     f"{'PASS' if ok else 'FAIL'}")
    lines += ["", f"{n_pass}/{len(checks)} held-out statistics reproduced.",
              "", "VERDICT: " + ("neutral model reproduces the held-out set "
                                 "-> neutrality sufficient"
                                 if n_pass == len(checks) else
                                 "neutral model fails held-out statistics "
                                 "-> neutrality insufficient")]

    rob = robustness_rows()
    with (OUT / "robustness.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rob[0]))
        w.writeheader()
        w.writerows(rob)
    still = all(r["flux_required"] for r in rob)
    lines += ["", "ROBUSTNESS OF THE FLUX CLAIM:",
              f"  flux required under all {len(rob)} N_ANC x polarity combinations: {still}",
              f"  smallest margin: {min(r['margin'] for r in rob):.3f}x",
              "  NOTE: r does not enter the floor (it depends only on N(t) and p_I),",
              "        so the male/female bracket is not applicable here.",
              ("  -> flux claim holds" if still else
               "  -> flux claim WITHDRAWN: some plausible parameters remove the need")]

    txt = "\n".join(lines)
    (OUT / "validation.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m illex.scripts.validate`
Expected: the held-out PASS/FAIL table, a verdict line, and the robustness summary.

- [ ] **Step 3: Verify the robustness table is complete**

Run: `column -s, -t results/illex/robustness.csv`
Expected: 6 rows (3 N_ANC variants × 2 polarities), each with `floor`, `observed`, `flux_required`, `margin`. If any row has `flux_required == False`, the flux claim is withdrawn per the spec's risk table — report it, do not suppress it.

- [ ] **Step 4: Commit**

```bash
git add illex/scripts/validate.py results/illex/validation.txt results/illex/robustness.csv
git -c commit.gpgsign=false commit -m "feat(illex): stage 4 held-out validation and flux robustness arm

Tests the fitted parameters on Fst and the control-region null, and
recomputes the dxy/pi_I floor across the N_ANC CI and both polarities."
```

- [ ] **Step 5: Run the full test suite and report**

Run: `.venv/bin/python -m pytest tests/illex/ -v` then `.venv/bin/python -m pytest tests/illex/ -m slow -v`
Expected: all fast tests pass; all slow harness tests pass. Report the final verdict from `results/illex/validation.txt` together with the Task 6 falsification verdict and the Task 7 persistence percentiles — those three together answer the spec's three goals.

---

## Deferred, with reasons

- **LD panel (constant-Ne arm, L ≥ 300 kb).** Needs the pilot to confirm 300 kb is affordable, and is validation-only. Split out because it uses a different demographic arm and a different statistic pipeline (mutation-overlaid, MAF ≥ 0.05, Rogers–Huff composite r² on pseudo-diploids) than everything above.
- ***I. argentinus* presence/absence** for the independent t_inv bracket — a data question on `analysis/steps/08_argentinus`, not a simulation task.
- **chr2 accessibility mask** — would convert the ratio-only targets into absolute per-bp ones. Out of scope; the design is deliberately ratio-based.
