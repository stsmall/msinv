"""Guards for I2: build_inversion_sim must reject p_start values that
silently break the requested t_inv, rather than passing them through to
msinv's untested/clamping fallbacks (see task-final-fixes-report.md)."""
import pytest

from illex import model


def test_p_start_above_p_inv_raises():
    """p_start >= p_inv yields a non-positive s, entering msinv's
    untested s<=0 fallback in DeterministicTrajectory -- must be rejected
    before it ever reaches msinv."""
    with pytest.raises(ValueError, match="p_start"):
        model.build_inversion_sim(
            arm="constant", seq_length=30_000, t_inv=5.0e5, gamma=1e-15,
            p_inv=0.626, p_start=0.7, seed=1,
        )


def test_p_start_equal_to_p_inv_raises():
    with pytest.raises(ValueError, match="p_start"):
        model.build_inversion_sim(
            arm="constant", seq_length=30_000, t_inv=5.0e5, gamma=1e-15,
            p_inv=0.626, p_start=0.626, seed=1,
        )


def test_p_start_below_clamp_floor_raises():
    """A p_start below msinv's 1/(2*n_e) clamp floor would silently be
    clamped by msinv and re-derive its own t_inv (measured up to a 32%
    error against the requested t_inv) -- must be rejected up front
    instead."""
    with pytest.raises(ValueError, match="clamp floor"):
        model.build_inversion_sim(
            arm="growth", seq_length=30_000, t_inv=5.0e5, gamma=1e-15,
            p_inv=0.626, p_start=1e-9, seed=1,
        )


def test_p_start_at_exact_hard_sweep_limit_does_not_raise():
    """The documented single-founder limit itself (p_start == 1/(2*n_e)
    exactly) must remain valid -- the clamp-floor guard is a strict
    less-than, not less-than-or-equal."""
    ne_traj = model.trajectory_ne("constant")
    sim = model.build_inversion_sim(
        arm="constant", seq_length=30_000, t_inv=5.0e5, gamma=1e-15,
        p_inv=0.626, p_start=1.0 / (2.0 * ne_traj), seed=1,
    )
    assert sim is not None
