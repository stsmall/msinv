"""Phase 6b — joint forward WF tests (sweep_trajectory specifics).

Tests features that don't depend on full simulator integration: the
trajectory shape itself.
"""

import pytest

from msinv.hull.sweep import Sweep


def test_j1_no_flux_locks_a_to_origin_kary():
    """γ=0, A originated on I → (S,A) stays exactly 0; (I,A) rises."""
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.001,
        partial_sweep_final_freq=0.99, gamma_flux=0.0,
    )
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.3], pop_size=10_000.0)
    final = rust_sw.trajectory_samples()[-1][1][0]
    # final[0]=(S,a), final[1]=(S,A), final[2]=(I,a), final[3]=(I,A)
    assert final[1] < 1e-9, f"(S,A) should stay 0, got {final[1]}"
    assert final[3] > 0.1, f"(I,A) should rise, got {final[3]}"


def test_j2_rdl_lifecycle_post_flux_mixing():
    """γ>0, origin on I → (S,A) grows over time; total A → near partial freq."""
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=2_000.0, f0=0.001,
        partial_sweep_final_freq=0.95,
        gamma_flux=1e-3, mean_tract_length=1000.0,
    )
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.3], pop_size=10_000.0)
    final = rust_sw.trajectory_samples()[-1][1][0]
    total_a = final[1] + final[3]
    assert total_a >= 0.90, f"total A should reach ~partial_sweep_final_freq, got {total_a}"
    assert final[1] > 1e-3, f"(S,A) should accumulate via flux, got {final[1]}"


def test_j3_origin_symmetry():
    """Origin on S vs origin on I should produce mirror trajectories at p_inv=0.5."""
    base_kwargs = dict(
        x_sel=50_000.0, tau=0.0, origin_pop=0, target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.001,
        partial_sweep_final_freq=0.99,
    )
    sw_s = Sweep(origin_kary="S", **base_kwargs)
    sw_i = Sweep(origin_kary="I", **base_kwargs)
    rs = sw_s.to_rust(); rs.build_trajectory(n_pops=1, p_inv_init=[0.5], pop_size=10_000.0)
    ri = sw_i.to_rust(); ri.build_trajectory(n_pops=1, p_inv_init=[0.5], pop_size=10_000.0)
    fs = rs.trajectory_samples()[-1][1][0]
    fi = ri.trajectory_samples()[-1][1][0]
    # (S,A) for origin=S should equal (I,A) for origin=I (mirror via the
    # symmetric p_inv=0.5 init + symmetric class structure)
    assert abs(fs[1] - fi[3]) < 1e-3, f"S-mirror={fs[1]}, I-mirror={fi[3]}"


@pytest.mark.skip(reason="requires demography accessor wiring (Task 12 follow-up)")
def test_j4_bottleneck_through_sweep():
    """Pop size change during sweep window should affect trajectory speed."""
    pass


@pytest.mark.skip(reason="requires simulator-side sweep dispatch (Task 13 follow-up)")
def test_j5_backward_flux_consistent_with_trajectory():
    """Backward-time flux events fire at right rate during sweep window."""
    pass


@pytest.mark.skip(reason="requires multi-pop pop_size accessor (PyO3 build_trajectory limitation)")
def test_j6_migration_spreads_sweep():
    """2-pop, m(1,0)>0, origin in pop 0 → A appears in pop 1."""
    pass


@pytest.mark.skip(reason="requires multi-pop pop_size accessor (PyO3 build_trajectory limitation)")
def test_j7_no_migration_keeps_pops_independent():
    """m=0, 2-pop → pop 1 stays unaffected by sweep in pop 0."""
    pass


@pytest.mark.skip(reason="requires simulator-side sweep dispatch (Task 13 follow-up)")
def test_j8_soft_sweep_seeds_K_founders():
    """f0=0.05 → K≈ceil(2N·p_kary·f0) origins seeded across distinct lineages."""
    pass


@pytest.mark.skip(reason="requires simulator-side sweep dispatch (Task 13 follow-up)")
def test_j9_recurrent_de_novo_count():
    """uA>0 → Poisson(uA·2N·duration) origins fire across the sweep window."""
    pass
