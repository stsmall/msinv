import numpy as np

from illex.abc import constants as C
from illex.slim import config


def test_constants_match_spec():
    assert C.P_NOW == 0.626 and C.P_TOL == 0.02
    assert (C.BP_LEFT, C.BP_RIGHT) == (60_550_000, 79_520_000)
    assert (C.N_I_CHROM, C.N_S_CHROM, C.N_IND) == (508, 190, 633)
    assert C.PRIOR == dict(t_inv=(1e5, 1.5e6), p_hard=0.5, p0=(1e-3, 0.10), s=(1e-6, 1e-2), p_star=(0.3, 0.95))
    assert C.TARGETS == dict(pi_ratio=0.7377, dxy_pi=1.8763, f1_ratio=1.2307)
    assert C.REC_RATE is config.REC_RATE
    assert float(f"{C.REC_RATE:.4g}") == 2.158e-9


def test_rec_rate_matches_male_map():
    import os
    import pytest
    if not os.path.exists(config.CHR2_RMAP_MALE_UF):
        pytest.skip("male chr2 map not on this machine")
    assert np.isclose(config.rec_rate_for_inversion(config.CHR2_RMAP_MALE_UF), C.REC_RATE, rtol=2e-3)
