"""Tests for v12 demography builder."""

import pytest

from msinv import Demography
from validation._lib.demography import (
    v12_msinv,
    NE_K_PRESENT,
    NE_F_PRESENT,
    NE_F_AT_SPLIT,
    NE_KF_AT_MERGE,
    NE_ANC_DEEP,
    T_KF_SPLIT,
    T_ANC_RENAME,
    T_INV_3RA,
    P_INV_F_3RA,
    P_INV_K_3RA,
    P_INV_ANC_3RA,
    GAMMA_3RA,
)


def test_v12_returns_demography():
    d = v12_msinv()
    assert isinstance(d, Demography)


def test_v12_two_populations():
    """v12 has exactly K (pop 0) and F (pop 1) as named pops."""
    d = v12_msinv()
    assert len(d.pop_sizes) == 2
    assert d.pop_sizes[0] == NE_K_PRESENT
    assert d.pop_sizes[1] == NE_F_PRESENT


def test_v12_constants():
    """Sanity-check the v12 constants against Small 2023 / v11 file."""
    assert NE_K_PRESENT == 126_772
    assert NE_F_PRESENT == 2_496_632
    assert NE_F_AT_SPLIT == 158_711
    assert NE_KF_AT_MERGE == 86_000
    assert NE_ANC_DEEP == 450_000
    assert T_KF_SPLIT == 9_194
    assert T_ANC_RENAME == 87_163
    assert T_INV_3RA == 330_000
    assert P_INV_F_3RA == 0.73
    assert P_INV_K_3RA == 0.0


def test_v12_has_kf_split_event():
    """An 'ej' event at T_KF_SPLIT joining F (pop 1) into K (pop 0)."""
    d = v12_msinv()
    events = list(d.events)
    ej_events = [e for e in events if e[0] == "ej"]
    assert any(e[1] == T_KF_SPLIT and e[2] == 1 and e[3] == 0 for e in ej_events), (
        f"expected ('ej', {T_KF_SPLIT}, 1, 0) — got ej events: {ej_events}"
    )


def test_v12_has_anc_deep_size_change():
    """An 'en' event at T_ANC_RENAME setting pop 0 to NE_ANC_DEEP."""
    d = v12_msinv()
    events = list(d.events)
    en_events = [e for e in events if e[0] == "en"]
    assert any(
        e[1] == T_ANC_RENAME and e[2] == 0 and e[3] == NE_ANC_DEEP for e in en_events
    ), f"expected ('en', {T_ANC_RENAME}, 0, {NE_ANC_DEEP}) — got: {en_events}"


def test_v12_no_migration_events():
    """v12 has zero migration events (the agreed K↔F simplification)."""
    d = v12_msinv()
    events = list(d.events)
    mig_events = [e for e in events if e[0] in ("em", "eM")]
    assert mig_events == []


# --- v12 msprime + discoal builders -------------------------------

import msprime
import pytest


def test_v12_msprime_returns_demography():
    from validation._lib.demography import v12_msprime
    d = v12_msprime()
    assert isinstance(d, msprime.Demography)


def test_v12_msprime_has_two_pops():
    from validation._lib.demography import v12_msprime, NE_K_PRESENT, NE_F_PRESENT
    d = v12_msprime()
    names = {p.name for p in d.populations}
    assert "K" in names
    assert "F" in names
    sizes = {p.name: p.initial_size for p in d.populations
             if p.name in ("K", "F")}
    assert sizes["K"] == NE_K_PRESENT
    assert sizes["F"] == NE_F_PRESENT


def test_v12_msprime_has_kf_split():
    from validation._lib.demography import v12_msprime, T_KF_SPLIT
    d = v12_msprime()
    # PopulationSplit lives in msprime.demography, not the top-level namespace
    PopulationSplit = msprime.demography.PopulationSplit
    splits = [e for e in d.events if isinstance(e, PopulationSplit)]
    assert any(e.time == T_KF_SPLIT for e in splits), (
        f"expected a PopulationSplit at t={T_KF_SPLIT}; got {splits}")


def test_v12_msprime_no_migration():
    from validation._lib.demography import v12_msprime
    d = v12_msprime()
    # Migration matrix should be all-zero
    import numpy as np
    M = d.migration_matrix
    assert np.allclose(M, 0.0), f"expected zero migration; got {M}"


def test_v12_discoal_events_returns_list():
    from validation._lib.demography import v12_discoal_events
    args = v12_discoal_events()
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


def test_v12_discoal_events_has_ed_at_split():
    """K-F split: -ed t/(4N0) src dst in discoal's CLI."""
    from validation._lib.demography import (
        v12_discoal_events, V12_DISCOAL_N0, T_KF_SPLIT,
    )
    args = v12_discoal_events()
    # Find a -ed token followed by the scaled split time.
    expected_t = T_KF_SPLIT / (4.0 * V12_DISCOAL_N0)
    found = False
    for i, a in enumerate(args):
        if a == "-ed" and i + 3 < len(args):
            t = float(args[i + 1])
            if abs(t - expected_t) < 1e-9:
                found = True
                break
    assert found, f"expected '-ed {expected_t} ...' in args; got {args[:20]}"


def test_v12_discoal_n0_is_k_present():
    from validation._lib.demography import V12_DISCOAL_N0, NE_K_PRESENT
    assert V12_DISCOAL_N0 == NE_K_PRESENT
