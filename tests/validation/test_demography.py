"""Tests for v12 demography builder."""
import pytest

from msinv import Demography
from validation._lib.demography import (
    v12_msinv,
    NE_K_PRESENT, NE_F_PRESENT, NE_F_AT_SPLIT,
    NE_KF_AT_MERGE, NE_ANC_DEEP,
    T_KF_SPLIT, T_ANC_RENAME,
    T_INV_3RA, P_INV_F_3RA, P_INV_K_3RA, P_INV_ANC_3RA,
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
    assert any(e[1] == T_KF_SPLIT and e[2] == 1 and e[3] == 0
               for e in ej_events), (
        f"expected ('ej', {T_KF_SPLIT}, 1, 0) — got ej events: {ej_events}")


def test_v12_has_anc_deep_size_change():
    """An 'en' event at T_ANC_RENAME setting pop 0 to NE_ANC_DEEP."""
    d = v12_msinv()
    events = list(d.events)
    en_events = [e for e in events if e[0] == "en"]
    assert any(e[1] == T_ANC_RENAME and e[2] == 0 and e[3] == NE_ANC_DEEP
               for e in en_events), (
        f"expected ('en', {T_ANC_RENAME}, 0, {NE_ANC_DEEP}) — got: {en_events}")


def test_v12_no_migration_events():
    """v12 has zero migration events (the agreed K↔F simplification)."""
    d = v12_msinv()
    events = list(d.events)
    mig_events = [e for e in events if e[0] in ("em", "eM")]
    assert mig_events == []
