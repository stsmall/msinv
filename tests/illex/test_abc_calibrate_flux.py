import numpy as np

from illex.abc.calibrate_flux import c_star, gamma_max

CS = [1e-9, 1e-8, 1e-7, 1e-6]
BASE = np.array([1.0, 1.0, 1.0])


def _arr(*m):
    return [np.full(3, v) for v in m]


def test_first_exceedance():
    assert c_star(CS, _arr(1.0, 1.02, 1.2, 1.5), BASE, 0.05) == 1e-7


def test_none():
    assert c_star(CS, _arr(1.0, 1.01, 0.97, 1.04), BASE, 0.05) is None
    assert gamma_max(None) == 1e-5


def test_noise_nonmonotone_takes_first_and_negative():
    assert c_star(CS, _arr(1.0, 1.0, 1.3, 1.0), BASE, 0.05) == 1e-7      # later return to baseline ignored
    assert c_star(CS, _arr(1.0, 0.9, 1.0, 1.0), BASE, 0.05) == 1e-8      # decrease counts
    assert c_star(CS, _arr(1.0, 1.04, 1.0, 1.0), BASE, 0.05) is None


def test_gamma_max():
    assert abs(gamma_max(1e-9) - 1e-7) < 1e-20
    assert gamma_max(1e-6) == 1e-5


def test_gamma_from_c_gives_per_site_c_for_any_tract():
    import pytest
    from illex.abc import constants as C
    from illex.abc import simulate as SIM

    for L in (188.0, 300.0, 1000.0):
        orig = SIM.TRACT_BP
        try:
            SIM.TRACT_BP = L
            w = L / C.BODY_BP
            g = SIM.gamma_from_c(1e-6)
            assert g * w / (1 - w) * L == pytest.approx(1e-6, rel=1e-9)   # interior initiation × tract = c
        finally:
            SIM.TRACT_BP = orig
