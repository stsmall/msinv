import pytest

from msinv._msinv_core import InvTrajectory
from msinv.hull import InversionSpec
from msinv.hull._rust_bridge import inversion_to_dict


def test_constant_trajectory_eval():
    inv = InversionSpec(
        bp_left=10.0, bp_right=20.0, p_inv=0.3, t_inv=100.0, mean_tract_length=0.0
    )
    tr = InvTrajectory(inversion_to_dict(inv, 1))
    assert tr.p_inv([0.0, 50.0, 100.0, 150.0], 0) == [0.3, 0.3, 0.0, 0.0]
    assert tr.t_inv_max() == 100.0
    assert tr.knots() == []


def test_precomputed_trajectory_is_linear_and_exposes_knots():
    traj = {
        "type": "precomputed",
        "times": [0.0, 100.0, 200.0],
        "freqs": [[0.6, 0.4, 0.2]],
        "n_e": [1000.0],
        "t_inv": [200.0],
    }
    inv = InversionSpec(
        bp_left=10.0, bp_right=20.0, trajectory=traj, mean_tract_length=0.0
    )
    tr = InvTrajectory(inversion_to_dict(inv, 1))
    assert tr.p_inv([50.0], 0)[0] == pytest.approx(0.5)
    assert tr.knots() == [0.0, 100.0, 200.0]
