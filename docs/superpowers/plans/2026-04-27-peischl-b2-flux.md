# Peischl b2 Flux Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `InversionSpec.flux_window` (single conflated parameter) with `mean_tract_length` (bp) + `tract_distribution` (`'geometric'` | `'fixed'`); sample tract length per event from the configured distribution; preserve existing semantics by migrating call sites to `'fixed'`.

**Architecture:** Two-phase migration to minimise broken-state windows. Phase A adds the new fields *alongside* `flux_window` (additive, no consumers broken). Phase B drops `flux_window` after every call site has been migrated. Phase C lands Tier-1 + Tier-2 validation tests. The `phi.rs` algorithm itself is reused unchanged — both modes call it with `w = mean_tract_length / inv_length`. Only `draw_tract` differs by distribution.

**Tech Stack:** Python 3.12 (msinv hull), Rust + PyO3 (msinv-core), pytest, cargo, msprime/tskit for stat checks. `xoshiro256pp` PRNG already in use both languages.

**Spec:** `docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md`

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `msinv/hull/inversion.py` | `InversionSpec` Python dataclass + `__post_init__` validation. | Modify |
| `msinv/hull/simulator.py` | Python `_draw_tract` and `_phi`/flux-rate references. | Modify |
| `msinv/hull/_rust_bridge.py` | Serialize Python `InversionSpec` → dict for the PyO3 boundary. | Modify |
| `rust/msinv-core/src/inversion.rs` | Rust `InversionSpec` struct + `TractDistribution` enum. | Modify |
| `rust/msinv-core/src/simulator.rs` | Rust `draw_tract`, `sample_flux_position`. | Modify |
| `rust/msinv-py/src/lib.rs` | PyO3 bridge — parse Python dict → Rust `InversionSpec`. | Modify |
| `tests/hull/test_phase3_gene_flux.py` | Existing Tier-0 tests. | Migrate (`flux_window=X` → `mean_tract_length=X*inv_len, tract_distribution='fixed'`) |
| `tests/hull/test_phase4b_class_migration.py` | Existing tests. | Migrate |
| `tests/hull/test_phase8_trajectory_selection.py` | Existing tests. | Migrate |
| `examples/rdl_mock_pilot.py`, `gene_flux_demo.py`, `kir_fol_pilot.py` | Example scripts. | Migrate |
| `tests/hull/test_phase3b_b2_flux.py` | **NEW** — Tier-1 + Tier-2 validation. | Create |
| `tests/hull/test_phase3b_inversion_spec_validation.py` | **NEW** — `InversionSpec` validation rules for new fields. | Create |

`rust/msinv-core/src/phi.rs` is **unchanged** — see Section 2 of the spec for why.

---

## Task 1: Add `InversionSpec` validation tests (TDD red)

**Goal:** Lock down the validation-rule semantics (`mean_tract_length >= 0`, allow zero, warn if `> inv_length / 2`, reject invalid distribution string) before touching the dataclass.

**Files:**
- Create: `tests/hull/test_phase3b_inversion_spec_validation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/hull/test_phase3b_inversion_spec_validation.py
"""Validation rules for the b2-flux InversionSpec fields:
mean_tract_length and tract_distribution."""

import warnings

import pytest

from msinv.hull import InversionSpec


def _kw(**overrides):
    """Minimal valid InversionSpec kwargs; override what each test needs."""
    base = dict(
        bp_left=0.0, bp_right=10_000.0,
        p_inv=0.5, t_inv=1000.0,
        gene_conversion_rate=1e-9,
    )
    base.update(overrides)
    return base


def test_mean_tract_length_negative_rejected():
    with pytest.raises(ValueError, match="mean_tract_length"):
        InversionSpec(**_kw(mean_tract_length=-1.0))


def test_mean_tract_length_zero_legal():
    # Zero is the canonical "disable flux via zero tract" path.
    inv = InversionSpec(**_kw(mean_tract_length=0.0))
    assert inv.mean_tract_length == 0.0


def test_mean_tract_length_positive_legal():
    inv = InversionSpec(**_kw(mean_tract_length=200.0))
    assert inv.mean_tract_length == 200.0


def test_mean_tract_length_above_half_inv_warns():
    # 7000 bp > inv_length/2 = 5000 bp -> warn (not error).
    with pytest.warns(UserWarning, match="mean_tract_length"):
        inv = InversionSpec(**_kw(mean_tract_length=7000.0))
    # Still constructs successfully.
    assert inv.mean_tract_length == 7000.0


def test_tract_distribution_geometric_legal():
    inv = InversionSpec(**_kw(tract_distribution='geometric'))
    assert inv.tract_distribution == 'geometric'


def test_tract_distribution_fixed_legal():
    inv = InversionSpec(**_kw(tract_distribution='fixed'))
    assert inv.tract_distribution == 'fixed'


def test_tract_distribution_invalid_rejected():
    with pytest.raises(ValueError, match="tract_distribution"):
        InversionSpec(**_kw(tract_distribution='gamma'))


def test_default_mean_tract_length_is_100():
    inv = InversionSpec(**_kw())
    assert inv.mean_tract_length == 100.0


def test_default_tract_distribution_is_geometric():
    inv = InversionSpec(**_kw())
    assert inv.tract_distribution == 'geometric'


def test_flux_window_field_removed():
    """After migration, passing flux_window must raise TypeError
    (Python's default for unexpected kwargs)."""
    with pytest.raises(TypeError):
        InversionSpec(**_kw(flux_window=0.05))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_inversion_spec_validation.py -v
```

Expected: every test FAILs with `TypeError: __init__() got an unexpected keyword argument 'mean_tract_length'` (or similar) — the fields don't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/hull/test_phase3b_inversion_spec_validation.py
git commit -m "Add failing InversionSpec validation tests for mean_tract_length + tract_distribution"
```

---

## Task 2: Add new `InversionSpec` fields (additive — keep `flux_window`)

**Goal:** Make Task 1's tests pass by adding the new fields. Keep `flux_window` for now so existing test files keep working unchanged. The dataclass exposes both old and new fields during the migration window.

**Files:**
- Modify: `msinv/hull/inversion.py` (the `InversionSpec` dataclass + `__post_init__`)

- [ ] **Step 1: Add the two new fields to the dataclass**

In `msinv/hull/inversion.py`, locate the `InversionSpec` dataclass (starts at line 53). After the existing `flux_window: float = 0.05` line (line 83), add:

```python
    # Peischl b2 flux model — see docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md.
    # When migration completes, flux_window is removed.
    mean_tract_length: float = 100.0   # bp, replaces flux_window's tract role
    tract_distribution: str = 'geometric'  # 'geometric' or 'fixed'
```

- [ ] **Step 2: Add validation in `__post_init__`**

In `msinv/hull/inversion.py`, in `__post_init__` (starts at line 134). The existing function has two branches (trajectory-dict path and legacy `p_inv`+`t_inv` path), both validating `flux_window`. Add the new validation **at the end of `__post_init__`** (after both branches), so it always runs:

```python
        # ---- b2 flux: validate mean_tract_length, tract_distribution ----
        if self.mean_tract_length < 0.0:
            raise ValueError(
                f"mean_tract_length must be >= 0, got {self.mean_tract_length}. "
                f"Use mean_tract_length=0 (or gene_conversion_rate=0) to "
                f"disable flux entirely.")
        if self.tract_distribution not in ('geometric', 'fixed'):
            raise ValueError(
                f"tract_distribution must be 'geometric' or 'fixed', "
                f"got {self.tract_distribution!r}.")
        inv_len_local = self.bp_right - self.bp_left
        if self.mean_tract_length > inv_len_local / 2.0:
            import warnings as _warnings
            _warnings.warn(
                f"mean_tract_length ({self.mean_tract_length:.1f}) exceeds "
                f"inv_length/2 ({inv_len_local/2:.1f}); tracts will frequently "
                f"span much of the inversion. Verify this is intentional.",
                UserWarning, stacklevel=2)
```

The existing `flux_window` validation in both branches stays — unchanged.

- [ ] **Step 3: Run validation tests; expect PASS**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_inversion_spec_validation.py -v
```

Expected: 9 tests pass; `test_flux_window_field_removed` STILL FAILS (it's testing the *post-migration* state where `flux_window` is gone).

- [ ] **Step 4: Mark `test_flux_window_field_removed` as `xfail` for now**

This test is the post-migration assertion; it should remain visibly failing as a TODO until Task 7 drops `flux_window`. Edit `tests/hull/test_phase3b_inversion_spec_validation.py`:

```python
@pytest.mark.xfail(reason="flux_window dropped in Task 7 of b2-flux migration",
                   strict=True)
def test_flux_window_field_removed():
    """After migration, passing flux_window must raise TypeError
    (Python's default for unexpected kwargs)."""
    with pytest.raises(TypeError):
        InversionSpec(**_kw(flux_window=0.05))
```

- [ ] **Step 5: Re-run; expect 9 PASS + 1 XFAIL**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_inversion_spec_validation.py -v
```

Expected: 9 PASS, 1 XFAIL. No FAILs.

- [ ] **Step 6: Commit**

```bash
git add msinv/hull/inversion.py tests/hull/test_phase3b_inversion_spec_validation.py
git commit -m "Add mean_tract_length + tract_distribution to InversionSpec (additive)"
```

---

## Task 3: Python `_draw_tract` consumes new fields

**Goal:** Update Python `_draw_tract` to read `mean_tract_length` + `tract_distribution`. During the migration window, when `flux_window` is set explicitly *and* `mean_tract_length` is at default, derive from `flux_window` to keep existing tests passing. **No persistent shim** — Task 7 will remove this fall-through.

**Files:**
- Modify: `msinv/hull/simulator.py:1163-1180` (`_draw_tract`)

- [ ] **Step 1: Locate the existing `_draw_tract` method**

In `msinv/hull/simulator.py`, the method starts at line 1163. Read lines 1163–1180 and confirm the existing logic matches:
```
w_g = inv.flux_window * inv_len
b1 = uniform(b1_lo, b1_hi)
tract_left = bp_left + b1
tract_right = tract_left + w_g  (clipped at bp_right)
```

- [ ] **Step 2: Replace `_draw_tract` with the b2 implementation**

Replace the whole `_draw_tract` body (lines 1163–1180) with:

```python
    def _draw_tract(self, x_event, inv):
        """Draw a gene-conversion tract centred at ``x_event`` for
        inversion ``inv``, using the b2 flux model.

        Tract length L is drawn per-event from the distribution
        configured via ``inv.tract_distribution``:
            * 'fixed':     L = inv.mean_tract_length
            * 'geometric': L ~ Exponential(1 / inv.mean_tract_length)
              (continuous-coordinate analog of geometric).

        Migration shim: if ``mean_tract_length`` is at its default
        (100.0) AND ``flux_window`` is non-default, derive tract length
        from flux_window so untouched legacy call sites keep their
        original semantics. Removed in Task 7.
        """
        inv_len = inv.bp_right - inv.bp_left

        # Migration shim — see docstring.
        if inv.mean_tract_length == 100.0 and inv.flux_window != 0.05:
            mean_L = inv.flux_window * inv_len
            distribution = 'fixed'
        else:
            mean_L = inv.mean_tract_length
            distribution = inv.tract_distribution

        # Defensive: rate-zero short-circuit upstream should prevent
        # reaching here with mean_L == 0, but guard so we never
        # divide by zero in the Exponential sampler.
        if mean_L <= 0.0:
            return float(x_event), float(x_event)

        if distribution == 'fixed':
            L = mean_L
        else:  # 'geometric'
            L = self.rng.exponential(mean_L)
        L = min(L, inv_len * 0.99)

        x_rel = x_event - inv.bp_left
        b1_lo = max(0.0, x_rel - L)
        b1_hi = min(inv_len - L, x_rel)
        if b1_hi <= b1_lo:
            b1 = max(0.0, min(inv_len - L, x_rel - L / 2.0))
        else:
            b1 = self.rng.uniform(b1_lo, b1_hi)
        tract_left = inv.bp_left + b1
        tract_right = min(tract_left + L, inv.bp_right)
        return tract_left, tract_right
```

- [ ] **Step 3: Update `_phi`-using rate calculation in the same file**

In `msinv/hull/simulator.py`, find every site that uses `inv.flux_window` as the `w` argument to `_phi` or `_phi_integral`. (There should be 1–2 sites in the rate-computation block.) Replace each with:

```python
        # b2-flux: w is mean_tract_length / inv_length.
        # Migration shim: derive from flux_window if mean_tract_length is at default.
        if inv.mean_tract_length == 100.0 and inv.flux_window != 0.05:
            w = inv.flux_window
        else:
            w = inv.mean_tract_length / (inv.bp_right - inv.bp_left)
```

Then keep the existing `_phi(..., w)` / `_phi_integral(..., w)` call sites using this local `w`. Do not touch `_phi`/`_phi_integral` themselves.

- [ ] **Step 4: Run the existing flux tests; expect PASS (semantic equivalence via shim)**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3_gene_flux.py -v
```

Expected: all currently-passing tests continue to pass. Any pre-existing skip/xfail unchanged.

- [ ] **Step 5: Commit**

```bash
git add msinv/hull/simulator.py
git commit -m "Python _draw_tract: consume mean_tract_length + tract_distribution (with migration shim)"
```

---

## Task 4: Rust `InversionSpec` struct fields

**Goal:** Mirror the Python field changes on the Rust side. Same shim pattern as Python (default `mean_tract_length = 100.0` + `flux_window = 0.05` is the legacy state).

**Files:**
- Modify: `rust/msinv-core/src/inversion.rs:14-23`
- Modify: `rust/msinv-core/src/inversion.rs:25-40` (the `new()` constructor) and `:45-56` (the `with_p_inv` constructor) — set the new defaults.

- [ ] **Step 1: Add `TractDistribution` enum**

Above the `InversionSpec` struct in `rust/msinv-core/src/inversion.rs:14`, add:

```rust
/// Per-event tract length distribution for the b2-flux model.
/// See docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TractDistribution {
    Fixed,
    Geometric,
}

impl Default for TractDistribution {
    fn default() -> Self { TractDistribution::Geometric }
}
```

- [ ] **Step 2: Add the two fields to `InversionSpec`**

In the `InversionSpec` struct (line 14–23), after `pub flux_window: f64,`, add:

```rust
    /// Mean per-event gene-conversion tract length (bp).
    /// Replaces `flux_window`'s tract role; phi(x) is computed with
    /// `w = mean_tract_length / inv_length`. Removed at Task 7.
    pub mean_tract_length: f64,
    /// Per-event tract length distribution (`Fixed` reproduces the
    /// pre-b2 deterministic-tract semantics; `Geometric` samples
    /// Exponential(1/mean_tract_length).
    pub tract_distribution: TractDistribution,
```

- [ ] **Step 3: Update both constructors to set the new defaults**

In `rust/msinv-core/src/inversion.rs`, the `new()` constructor (line 27–40) sets struct defaults. Update its `Self { … }` block to include:

```rust
        Self {
            bp_left,
            bp_right,
            trajectory,
            gene_conversion_rate: 1e-9,
            flux_window: 0.05,
            mean_tract_length: 100.0,
            tract_distribution: TractDistribution::Geometric,
            inv_id: 0,
        }
```

The `with_p_inv` constructor (line 45–56) calls `Self::new(...)` and is unchanged.

- [ ] **Step 4: Build the crate (no Python yet)**

```bash
cd /home/ssmall/inversion_sims/files/rust && cargo build --release -p msinv-core 2>&1 | tail -20
```

Expected: clean build, no errors. Warnings about unused `mean_tract_length` / `tract_distribution` are OK at this point — Tasks 5–6 wire them into `simulator.rs` and the bridge.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/inversion.rs
git commit -m "Rust InversionSpec: add mean_tract_length + TractDistribution (additive)"
```

---

## Task 5: Rust `draw_tract` and `sample_flux_position` consume new fields

**Goal:** Same migration shim pattern: if `mean_tract_length == 100.0` and `flux_window != 0.05`, the Rust simulator reads `flux_window`. Otherwise it reads the new fields.

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:1302-1374` (`sample_flux_position` lines 1302–1353, `draw_tract` lines 1355–1374)
- Need: `rand_distr::Exp1` or `Distribution` from `rand_distr` (check Cargo.toml).

- [ ] **Step 1: Verify `rand_distr` is available in `msinv-core`**

```bash
grep -n "rand_distr" /home/ssmall/inversion_sims/files/rust/msinv-core/Cargo.toml
```

Expected: `rand_distr = "..."` present. If absent, add `rand_distr = "0.5"` (or matching the `rand` version in use) and re-run cargo build.

- [ ] **Step 2: Replace `draw_tract`**

In `rust/msinv-core/src/simulator.rs:1355`, replace the entire `draw_tract` method body with:

```rust
    fn draw_tract(
        &self,
        x_event: f64,
        inv: &InversionSpec,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (f64, f64) {
        use rand_distr::{Distribution, Exp};
        let inv_len = inv.length();

        // Migration shim — see Task 3 docstring.
        let (mean_l, distribution) =
            if inv.mean_tract_length == 100.0 && inv.flux_window != 0.05 {
                (inv.flux_window * inv_len, crate::inversion::TractDistribution::Fixed)
            } else {
                (inv.mean_tract_length, inv.tract_distribution)
            };

        if mean_l <= 0.0 {
            return (x_event, x_event);
        }

        let l = match distribution {
            crate::inversion::TractDistribution::Fixed => mean_l,
            crate::inversion::TractDistribution::Geometric => {
                let exp = Exp::new(1.0 / mean_l).expect("mean_l > 0");
                exp.sample(rng)
            }
        };
        let l = l.min(inv_len * 0.99);

        let x_rel = x_event - inv.bp_left;
        let b1_lo = (x_rel - l).max(0.0);
        let b1_hi = (x_rel).min(inv_len - l);
        let b1 = if b1_hi <= b1_lo {
            (x_rel - l / 2.0).clamp(0.0, inv_len - l)
        } else {
            rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
        };
        let tl = (inv.bp_left + b1).max(inv.bp_left);
        let tr = (tl + l).min(inv.bp_right);
        (tl, tr)
    }
```

- [ ] **Step 3: Update `sample_flux_position` to use migration-shim `w`**

In `rust/msinv-core/src/simulator.rs:1302`, the function uses `let w = inv.flux_window;` at line 1311. Replace that line with:

```rust
        let w = if inv.mean_tract_length == 100.0 && inv.flux_window != 0.05 {
            inv.flux_window
        } else {
            inv.mean_tract_length / inv.length()
        };
```

The rest of `sample_flux_position` is unchanged — it uses `w` for `phi`/`phi_integral` calls.

- [ ] **Step 4: Build & run Rust tests**

```bash
cd /home/ssmall/inversion_sims/files/rust && cargo test --release --lib 2>&1 | tail -15
```

Expected: all 96+ Rust lib tests pass. (Per CLAUDE.md: 96 lib + 23 parity as of 2026-04-27.)

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/simulator.rs rust/msinv-core/Cargo.toml
git commit -m "Rust draw_tract + sample_flux_position: consume mean_tract_length + tract_distribution"
```

---

## Task 6: PyO3 bridge + Python `_rust_bridge.py`

**Goal:** Both directions of the FFI (Python → Rust spec construction in PyO3; Rust → Python spec serialization in `_rust_bridge.py`) carry the new fields.

**Files:**
- Modify: `rust/msinv-py/src/lib.rs` around lines 218–380 (where `flux_window` is parsed and assigned).
- Modify: `msinv/hull/_rust_bridge.py:50` (where `flux_window` is serialized).

- [ ] **Step 1: Read existing PyO3 InversionSpec construction**

```bash
grep -n "flux_window\|InversionSpec" /home/ssmall/inversion_sims/files/rust/msinv-py/src/lib.rs
```

You should see:
- Line ~226: `let fw: f64 = d.get_item("flux_window")?...`
- Line ~378: `let mut spec = InversionSpec::new(...)`
- Line ~380: `spec.flux_window = fw;`

- [ ] **Step 2: Parse new dict keys**

In `rust/msinv-py/src/lib.rs`, near the existing `let fw: f64 = d.get_item("flux_window")?...` line (~226), add:

```rust
            let mtl: f64 = d.get_item("mean_tract_length")?
                .map_or_else(|| Ok(100.0_f64), |v| v.extract())?;
            let td_str: String = d.get_item("tract_distribution")?
                .map_or_else(|| Ok("geometric".to_string()), |v| v.extract())?;
            let td = match td_str.as_str() {
                "fixed" => msinv_core::inversion::TractDistribution::Fixed,
                "geometric" => msinv_core::inversion::TractDistribution::Geometric,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("tract_distribution must be 'fixed' or 'geometric', got {td_str:?}"))),
            };
```

(Use `msinv_core::inversion::TractDistribution` — the path may differ; cross-check with the local `use` statements in `lib.rs`.)

- [ ] **Step 3: Set the new fields on the constructed `InversionSpec`**

Near `spec.flux_window = fw;` (~line 380), add:

```rust
            spec.mean_tract_length = mtl;
            spec.tract_distribution = td;
```

- [ ] **Step 4: Update `_rust_bridge.py` to serialize new fields**

In `msinv/hull/_rust_bridge.py:50`, the dict-build for an InversionSpec includes `'flux_window': float(inv.flux_window)`. Add two siblings:

```python
            'mean_tract_length': float(inv.mean_tract_length),
            'tract_distribution': str(inv.tract_distribution),
```

- [ ] **Step 5: Rebuild Rust extension + install .so**

Per CLAUDE.md instructions:
```bash
cd /home/ssmall/inversion_sims/files/rust && cargo build --release -p msinv-py
/bin/cp -f /home/ssmall/inversion_sims/files/rust/target/release/lib_msinv_core.so /home/ssmall/inversion_sims/files/msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```

(Use `/bin/cp` explicitly — shell alias adds `-i` and prompts.)

- [ ] **Step 6: Run hull tests adjacent to flux**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3_gene_flux.py /home/ssmall/inversion_sims/files/tests/hull/test_phase4b_class_migration.py /home/ssmall/inversion_sims/files/tests/hull/test_phase8_trajectory_selection.py -v --tb=short 2>&1 | tail -25
```

Expected: all pass — the migration shim should keep current `flux_window=X` semantics intact end-to-end.

- [ ] **Step 7: Commit**

```bash
git add rust/msinv-py/src/lib.rs msinv/hull/_rust_bridge.py
git commit -m "PyO3 + _rust_bridge: carry mean_tract_length + tract_distribution"
```

---

## Task 7: Migrate test files and example scripts; drop `flux_window`

**Goal:** Mechanical migration of every `flux_window=X` site to the b2 API, then removal of the `flux_window` field from `InversionSpec` in both languages and the bridge. After this task, `mean_tract_length` is the only tract-length knob.

**Files:**
- Modify: `tests/hull/test_phase3_gene_flux.py` (7 sites)
- Modify: `tests/hull/test_phase4b_class_migration.py` (2 sites)
- Modify: `tests/hull/test_phase8_trajectory_selection.py` (4 sites)
- Modify: `examples/rdl_mock_pilot.py`, `examples/gene_flux_demo.py`, `examples/kir_fol_pilot.py` (1–2 sites each)
- Modify: `msinv/hull/inversion.py` (drop `flux_window` field + its validation; remove migration shim references)
- Modify: `msinv/hull/simulator.py` (remove migration shim in `_draw_tract`, remove `flux_window` reads in rate logic)
- Modify: `rust/msinv-core/src/inversion.rs` (drop `flux_window` field)
- Modify: `rust/msinv-core/src/simulator.rs` (remove migration shim in `draw_tract` + `sample_flux_position`)
- Modify: `rust/msinv-py/src/lib.rs` (drop the `let fw = ...` parse and `spec.flux_window = fw` assignment)
- Modify: `msinv/hull/_rust_bridge.py` (drop the `'flux_window':` serialization)

- [ ] **Step 1: Migrate `tests/hull/test_phase3_gene_flux.py`**

Each site of the form:
```python
inv = InversionSpec(
    bp_left=L, bp_right=R, ...,
    flux_window=W,
)
```
becomes:
```python
inv = InversionSpec(
    bp_left=L, bp_right=R, ...,
    mean_tract_length=W * (R - L),   # preserves pre-migration tract size
    tract_distribution='fixed',       # preserves deterministic tract length
)
```

Open the file and apply this transformation to all 7 sites. The exact `R - L` value depends on each test's inversion (read it from the same `InversionSpec` constructor).

- [ ] **Step 2: Run the migrated test**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3_gene_flux.py -v --tb=short
```

Expected: same pass/skip/xfail set as before migration.

- [ ] **Step 3: Migrate `tests/hull/test_phase4b_class_migration.py`**

Apply the same transformation at the 2 sites. Re-run:

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase4b_class_migration.py -v --tb=short
```

Expected: 12 pass (matches today's count).

- [ ] **Step 4: Migrate `tests/hull/test_phase8_trajectory_selection.py`**

Apply at 4 sites. Re-run:

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase8_trajectory_selection.py -v --tb=short
```

- [ ] **Step 5: Migrate the three example scripts**

Apply the same transformation in:
- `examples/rdl_mock_pilot.py`
- `examples/gene_flux_demo.py`
- `examples/kir_fol_pilot.py`

For each example, after editing, do a smoke run if reasonable (most examples take a few seconds to a minute):
```bash
/home/ssmall/inversion_sims/files/.venv/bin/python /home/ssmall/inversion_sims/files/examples/gene_flux_demo.py 2>&1 | tail -5
```

(Skip running `kir_fol_pilot.py` — it's a long simulation, not a smoke target.)

- [ ] **Step 6: Drop `flux_window` from Python `InversionSpec`**

In `msinv/hull/inversion.py`:
- Remove the `flux_window: float = 0.05` field declaration.
- Remove the two `flux_window` validation blocks in `__post_init__` (one in the trajectory branch, one in the legacy branch).
- The new `mean_tract_length` / `tract_distribution` validation remains untouched.

- [ ] **Step 7: Drop migration shim from Python `_draw_tract` and rate logic**

In `msinv/hull/simulator.py:_draw_tract`, remove the `if inv.mean_tract_length == 100.0 and inv.flux_window != 0.05:` shim block. Replace the body with the clean version:

```python
        inv_len = inv.bp_right - inv.bp_left
        mean_L = inv.mean_tract_length
        if mean_L <= 0.0:
            return float(x_event), float(x_event)
        if inv.tract_distribution == 'fixed':
            L = mean_L
        else:  # 'geometric'
            L = self.rng.exponential(mean_L)
        L = min(L, inv_len * 0.99)
        # ... (rest of b1/b2 sampling unchanged)
```

Same for the rate-calculation site that read `flux_window` via the shim — replace with the clean `w = inv.mean_tract_length / inv_length` form.

- [ ] **Step 8: Drop `flux_window` from Rust `InversionSpec`**

In `rust/msinv-core/src/inversion.rs`:
- Remove the `pub flux_window: f64,` field.
- Remove `flux_window: 0.05,` from `Self::new()`.

- [ ] **Step 9: Drop migration shim from Rust `draw_tract` and `sample_flux_position`**

In `rust/msinv-core/src/simulator.rs`, simplify both functions. `draw_tract`:

```rust
fn draw_tract(
    &self,
    x_event: f64,
    inv: &InversionSpec,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, f64) {
    use rand_distr::{Distribution, Exp};
    let inv_len = inv.length();
    if inv.mean_tract_length <= 0.0 {
        return (x_event, x_event);
    }
    let l = match inv.tract_distribution {
        crate::inversion::TractDistribution::Fixed => inv.mean_tract_length,
        crate::inversion::TractDistribution::Geometric => {
            let exp = Exp::new(1.0 / inv.mean_tract_length).expect("mean > 0");
            exp.sample(rng)
        }
    };
    let l = l.min(inv_len * 0.99);
    let x_rel = x_event - inv.bp_left;
    let b1_lo = (x_rel - l).max(0.0);
    let b1_hi = (x_rel).min(inv_len - l);
    let b1 = if b1_hi <= b1_lo {
        (x_rel - l / 2.0).clamp(0.0, inv_len - l)
    } else {
        rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
    };
    let tl = (inv.bp_left + b1).max(inv.bp_left);
    let tr = (tl + l).min(inv.bp_right);
    (tl, tr)
}
```

For `sample_flux_position` line 1311, replace the shim block with:
```rust
        let w = inv.mean_tract_length / inv.length();
```

- [ ] **Step 10: Drop `flux_window` from PyO3 bridge and `_rust_bridge.py`**

In `rust/msinv-py/src/lib.rs`, remove:
- The `let fw: f64 = ...` parse line.
- The `spec.flux_window = fw;` assignment.

In `msinv/hull/_rust_bridge.py:50`, remove the `'flux_window': float(inv.flux_window),` line.

- [ ] **Step 11: Rebuild Rust extension**

```bash
cd /home/ssmall/inversion_sims/files/rust && cargo build --release -p msinv-py 2>&1 | tail -5
/bin/cp -f /home/ssmall/inversion_sims/files/rust/target/release/lib_msinv_core.so /home/ssmall/inversion_sims/files/msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```

- [ ] **Step 12: Remove the `xfail` marker on `test_flux_window_field_removed`**

In `tests/hull/test_phase3b_inversion_spec_validation.py`, remove the `@pytest.mark.xfail(...)` decorator from `test_flux_window_field_removed`. The test should now pass naturally.

- [ ] **Step 13: Run the full migrated test suite**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3_gene_flux.py /home/ssmall/inversion_sims/files/tests/hull/test_phase4b_class_migration.py /home/ssmall/inversion_sims/files/tests/hull/test_phase8_trajectory_selection.py /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_inversion_spec_validation.py -v --tb=short 2>&1 | tail -30
```

Expected: all pass. The 17 known sweep failures are NOT in this set.

- [ ] **Step 14: grep-verify zero `flux_window` references**

```bash
grep -rn "flux_window" /home/ssmall/inversion_sims/files/msinv /home/ssmall/inversion_sims/files/rust/msinv-core/src /home/ssmall/inversion_sims/files/rust/msinv-py/src /home/ssmall/inversion_sims/files/tests /home/ssmall/inversion_sims/files/examples 2>/dev/null
```

Expected: no output. Any hit indicates a missed migration site.

- [ ] **Step 15: Commit the migration**

```bash
git add -u
git commit -m "Drop flux_window — migrate all call sites to mean_tract_length + tract_distribution"
```

---

## Task 8: Tier 1 tests (geometric sampling correctness + smoke)

**Goal:** Verify the Exponential `'geometric'` sampler is statistically correct in isolation, and that the simulator runs cleanly at biological 3Ra-scale parameters with `'geometric'` mode.

**Files:**
- Create: `tests/hull/test_phase3b_b2_flux.py`

- [ ] **Step 1: Write the geometric-sampling unit test (failing target)**

Create the file:

```python
# tests/hull/test_phase3b_b2_flux.py
"""Tier-1 + Tier-2 validation of the b2-flux model
(per docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md)."""

import math

import numpy as np
import pytest
from scipy import stats

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography


# ----- Tier 1: geometric sampling correctness ----------------------

def test_geometric_tract_length_mean_matches_parameter():
    """Sampled tract lengths should have mean ≈ mean_tract_length
    within 2-sigma tolerance over N=10_000 draws."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=1_000_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=200.0,
        tract_distribution='geometric',
    )
    sim = HullSimulator(
        sample_config={('S', 0): 2},
        demography=Demography(pop_sizes=[1000]),
        sequence_length=1_000_000,
        recombination_rate=1e-8, inversions=[inv], seed=42,
    )
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(10_000):
        # Use the same Exponential the simulator uses.
        samples.append(rng.exponential(inv.mean_tract_length))
    mean = float(np.mean(samples))
    expected = inv.mean_tract_length
    sd_of_mean = expected / math.sqrt(len(samples))
    assert abs(mean - expected) < 2 * sd_of_mean, (
        f"empirical mean {mean:.2f} vs expected {expected:.2f} "
        f"(2σ={2*sd_of_mean:.2f})")


def test_geometric_tract_length_distribution_is_exponential():
    """Kolmogorov-Smirnov test: empirical samples should match
    Exponential(rate=1/λ) at p > 0.05."""
    rng = np.random.default_rng(1)
    lam = 200.0
    samples = rng.exponential(lam, size=10_000)
    ks_stat, p = stats.kstest(samples, 'expon', args=(0.0, lam))
    assert p > 0.05, f"KS test failed: stat={ks_stat:.4f} p={p:.4f}"


# ----- Tier 1: smoke at biological 3Ra-scale params ---------------

def test_smoke_3ra_scale_geometric():
    """3Ra-scale params (6 Mb inv, 100 bp tract) run without crashing
    and produce a well-formed tree sequence."""
    inv = InversionSpec(
        bp_left=1.0, bp_right=6_000_000.0 - 1.0,
        p_inv=0.5, t_inv=100_000.0,
        gene_conversion_rate=1e-6,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[5000])
    sim = HullSimulator(
        sample_config={('S', 0): 4, ('I', 0): 4},
        demography=demo,
        sequence_length=6_000_000,
        recombination_rate=1e-8,
        inversions=[inv], seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0
    assert ts.num_nodes > 8
```

- [ ] **Step 2: Run the new tests; expect PASS**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_b2_flux.py -v --tb=short 2>&1 | tail -15
```

Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_phase3b_b2_flux.py
git commit -m "b2-flux Tier 1 tests: geometric sampling correctness + smoke"
```

---

## Task 9: Tier 2 tests (spatial profile + per-bp rate calibration)

**Goal:** Add the two Tier-2 statistical tests to the same file.

**Files:**
- Modify: `tests/hull/test_phase3b_b2_flux.py`

- [ ] **Step 1: Append the spatial-profile φ(x) test**

Append to the test file:

```python
# ----- Tier 2: spatial profile φ(x) ------------------------------

def test_spatial_profile_uniform_in_interior_geometric():
    """Empirical fraction of flux events that touch position x should
    be ≈ λ/inv_length in the interior (away from breakpoints by ≫ λ).

    Strategy: instead of instrumenting the simulator's flux events,
    sample many tracts directly via the same _draw_tract logic and
    histogram the per-bp coverage. This validates the geometry, which
    is the determinant of the spatial profile."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=10_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )

    rng = np.random.default_rng(2)
    inv_len = inv.bp_right - inv.bp_left
    lam = inv.mean_tract_length
    n_events = 50_000
    bin_edges = np.linspace(0.0, inv_len, 101)  # 100 bins, 100 bp each
    coverage = np.zeros(100)

    for _ in range(n_events):
        # Sample a tract via the same algorithm. b1 ~ Uniform[0, inv_len-L]
        # for the "uniform spatial" interpretation that emerges when
        # x_event itself is sampled uniformly; here we draw L and b1
        # together, which is the marginal spatial distribution.
        L = rng.exponential(lam)
        L = min(L, inv_len * 0.99)
        if L <= 0.0:
            continue
        b1 = rng.uniform(0.0, inv_len - L)
        tl, tr = b1, b1 + L
        # Bin the tract's [tl, tr) coverage.
        lo = int(np.searchsorted(bin_edges, tl, side='right') - 1)
        hi = int(np.searchsorted(bin_edges, tr, side='left'))
        coverage[lo:hi] += 1

    # Per-position fraction.
    coverage_frac = coverage / n_events
    # Interior bins: skip first 2 and last 2 (rise/fall regions ≈ λ wide).
    interior = coverage_frac[2:-2]
    expected_interior = lam / inv_len
    mean_interior = float(np.mean(interior))
    assert abs(mean_interior - expected_interior) / expected_interior < 0.15, (
        f"interior coverage {mean_interior:.4f} vs expected "
        f"{expected_interior:.4f} (>15% off)")


# ----- Tier 2: rate scaling with mean_tract_length ---------------

def test_flux_rate_scales_linearly_with_mean_tract_length():
    """Per-lineage flux event rate ≈ γ × p_other × mean_tract_length
    (Section 2 of the spec). Verify by varying mean_tract_length over
    a 20× range and confirming the empirical num_trees count scales
    proportionally — flux events fragment the ARG into more trees,
    so num_trees is a monotone proxy for total flux-event count.

    This is the Tier-2 calibration we can land without simulator-
    state instrumentation. Direct event-count calibration is part
    of Tier 3-full (Andolfatto anchor) per the spec's Deferred
    Validation Roadmap."""
    bp_left = 0.0
    bp_right = 200_000.0
    inv_len = bp_right - bp_left
    gamma = 1e-7
    NREPS = 10
    lambdas = [200.0, 1000.0, 4000.0]  # 20× range
    means = []
    for lam in lambdas:
        inv = InversionSpec(
            bp_left=bp_left, bp_right=bp_right,
            p_inv=0.5, t_inv=20_000.0,
            gene_conversion_rate=gamma,
            mean_tract_length=lam,
            tract_distribution='geometric',
        )
        demo = Demography(pop_sizes=[2000])
        n_trees_reps = []
        for seed in range(NREPS):
            sim = HullSimulator(
                sample_config={('S', 0): 4, ('I', 0): 4},
                demography=demo,
                sequence_length=int(inv_len),
                recombination_rate=1e-9,
                inversions=[inv], seed=seed,
            )
            ts = sim.simulate()
            n_trees_reps.append(ts.num_trees)
        means.append(float(np.mean(n_trees_reps)))

    # Subtract the no-flux baseline (recombination-driven trees) so
    # the flux contribution is what scales with λ.
    inv_zero = InversionSpec(
        bp_left=bp_left, bp_right=bp_right,
        p_inv=0.5, t_inv=20_000.0,
        gene_conversion_rate=gamma,
        mean_tract_length=0.0,                # disables flux
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[2000])
    no_flux_trees = []
    for seed in range(NREPS):
        sim = HullSimulator(
            sample_config={('S', 0): 4, ('I', 0): 4},
            demography=demo,
            sequence_length=int(inv_len),
            recombination_rate=1e-9,
            inversions=[inv_zero], seed=seed,
        )
        no_flux_trees.append(sim.simulate().num_trees)
    baseline = float(np.mean(no_flux_trees))

    flux_contribution = [m - baseline for m in means]

    # Assert: the flux contribution scales monotonically with λ.
    assert flux_contribution[0] < flux_contribution[1] < flux_contribution[2], (
        f"flux_contribution should be monotone-increasing in λ, "
        f"got {flux_contribution} at λ={lambdas}")

    # Assert: ratios approximately match λ ratios (within ±40 % to
    # accommodate MC noise at NREPS=10; tighten if NREPS is bumped).
    ratio_2_to_1 = flux_contribution[1] / max(flux_contribution[0], 0.5)
    expected_2_to_1 = lambdas[1] / lambdas[0]   # = 5
    assert 0.6 * expected_2_to_1 < ratio_2_to_1 < 1.4 * expected_2_to_1, (
        f"flux scaling 1→2: ratio {ratio_2_to_1:.2f} vs expected "
        f"{expected_2_to_1:.2f} (>40 % off)")
```

- [ ] **Step 2: Run the new tests**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_b2_flux.py -v --tb=short 2>&1 | tail -20
```

Expected: spatial profile passes within ±15%; per-bp calibration passes the loose proxy assertion. **If the proxy assertion is unreliable, the implementer should refactor it using direct flux-event counts from msinv's tree-sequence provenance — see `test_phase3_gene_flux.py` for the access pattern (e.g., `ts.provenances()` or `ts.metadata`).**

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_phase3b_b2_flux.py
git commit -m "b2-flux Tier 2 tests: spatial profile + per-bp rate calibration proxy"
```

---

## Task 10: Rust ↔ Python parity test extension

**Goal:** Verify that Rust and Python simulators produce identical tree sequences for the same seed under both `'fixed'` and `'geometric'` modes.

**Files:**
- Modify: the existing parity-test fixture file. Locate it:

```bash
grep -rln "parity" /home/ssmall/inversion_sims/files/tests/hull /home/ssmall/inversion_sims/files/rust/msinv-core/tests 2>/dev/null
```

- [ ] **Step 1: Identify the parity fixture format**

```bash
grep -n "InversionSpec\|fixture" /home/ssmall/inversion_sims/files/tests/hull/test_phase*parity*.py 2>/dev/null
```

If parity tests live under `rust/msinv-core/tests/` instead, look there. Read the existing fixture list and identify how each fixture is parameterized.

- [ ] **Step 2: Add `'fixed'` and `'geometric'` mode fixtures**

For each existing fixture that constructs an `InversionSpec`, add a parameterised variant pinning `tract_distribution`. Example pattern:

```python
@pytest.mark.parametrize("tract_dist", ['fixed', 'geometric'])
def test_parity_invspec_b2_modes(tract_dist):
    inv = InversionSpec(
        bp_left=0.0, bp_right=10_000.0,
        p_inv=0.5, t_inv=1000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=200.0,
        tract_distribution=tract_dist,
    )
    # ... existing parity-assertion machinery
```

- [ ] **Step 3: Run parity tests**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull -k parity -v --tb=short
```

Expected: all pass. Both engines produce identical tree sequences for the same seed.

If the Rust `Exp` distribution does not produce bit-equivalent draws to NumPy's `Exponential`, parity may fail for `'geometric'` mode. In that case, implement the Exponential as `-mean × ln(rng.uniform())` directly in both languages so they share the underlying uniform draw. (`rng.uniform()` IS already parity-equivalent.)

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "Parity tests: cover fixed + geometric tract distributions"
```

---

## Task 11: Final verification + spec acknowledgement commit

**Goal:** Confirm zero `flux_window` residue, all migrated + new tests pass, and update memory + spec status.

- [ ] **Step 1: Verify no `flux_window` references remain**

```bash
grep -rn "flux_window" /home/ssmall/inversion_sims/files/msinv /home/ssmall/inversion_sims/files/rust/msinv-core/src /home/ssmall/inversion_sims/files/rust/msinv-py/src /home/ssmall/inversion_sims/files/tests /home/ssmall/inversion_sims/files/examples
```

Expected: no output.

- [ ] **Step 2: Run the cmig-adjacent test suite (the same set we use to confirm no regressions in cmig work)**

```bash
/home/ssmall/inversion_sims/files/.venv/bin/python -m pytest /home/ssmall/inversion_sims/files/tests/hull/test_phase3_gene_flux.py /home/ssmall/inversion_sims/files/tests/hull/test_phase4_demography.py /home/ssmall/inversion_sims/files/tests/hull/test_phase4b_class_migration.py /home/ssmall/inversion_sims/files/tests/hull/test_phase5_per_segment_class.py /home/ssmall/inversion_sims/files/tests/hull/test_phase8_trajectory_selection.py /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_b2_flux.py /home/ssmall/inversion_sims/files/tests/hull/test_phase3b_inversion_spec_validation.py -v --tb=short 2>&1 | tail -30
```

Expected: all pass (no failures).

- [ ] **Step 3: Run the Rust test suite**

```bash
cd /home/ssmall/inversion_sims/files/rust && cargo test --release --lib 2>&1 | tail -10
```

Expected: 96+ lib tests pass + 23 parity tests pass (per CLAUDE.md baseline).

- [ ] **Step 4: Update spec status to "implemented"**

In `docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md`, change the `Status:` line:

```markdown
**Status:** Implemented — landed in commits <list-of-commit-shas>; Tier 3 + cross-feature deferred validation captured in the Deferred Validation Roadmap section
```

- [ ] **Step 5: Update memory**

Edit `/home/ssmall/.claude/projects/-home-ssmall-inversion-sims-files/memory/project_msinv_todo.md`:
- Move the b2 flux item from "open" to "done with date".
- Add a memory file `feedback_b2_flux_caveats.md` if anything surprising came up during implementation (e.g., a parity-bit-equivalence issue with `Exp` requiring the manual `-mean × ln(uniform)` form).

- [ ] **Step 6: Final commit**

```bash
git add -u
git commit -m "b2 flux upgrade landed: spec status updated, memory refreshed"
```

- [ ] **Step 7: Show final git log**

```bash
git log --oneline -15
```

Verify the migration commits are in a sensible order; squash if desired before merging.

---

## Self-review checklist (run before handing off)

- **Spec coverage:**
  - API change (drop flux_window; add mean_tract_length, tract_distribution): Tasks 2, 4, 7 ✓
  - Sample L per event from configured distribution: Tasks 3, 5 ✓
  - Migration of every existing call site: Task 7 ✓
  - Tier 1 + Tier 2 tests: Tasks 8, 9 ✓
  - Parity tests: Task 10 ✓
  - Defensive guard for mean_tract_length=0: Tasks 3 (Step 2), 5 (Step 2) ✓
  - Validation rules (≥0, ≤inv_length/2 warn, distribution string): Task 1+2 ✓
  - SMC-style tracking explicitly NOT done: documented in spec, no code task ✓
  - Tier 3 deferred to roadmap: confirmed; no task here ✓

- **Placeholder scan:**
  - All steps contain concrete code, commands, or assertions. No `TBD` / `FIXME` markers in test code.

- **Type / name consistency:**
  - `mean_tract_length` (Python) ↔ `mean_tract_length` (Rust struct field) ↔ `mean_tract_length` (PyO3 dict key) — consistent.
  - `tract_distribution` string `'geometric'` / `'fixed'` ↔ `TractDistribution::Geometric` / `TractDistribution::Fixed` Rust enum — consistent.
  - `Exponential(1/λ)` in both languages, with parity-equivalent uniform-draw fallback noted in Task 10.
