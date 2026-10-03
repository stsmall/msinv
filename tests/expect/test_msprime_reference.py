import msprime
import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.hull import Demography, InversionSpec

pytestmark = pytest.mark.slow
REPS = 4000


def _branch_sfs(dem, samples, n):
    out = []
    for ts in msprime.sim_ancestry(
        samples=samples,
        demography=dem,
        ploidy=1,
        sequence_length=1,
        num_replicates=REPS,
        random_seed=7,
    ):
        out.append(ts.allele_frequency_spectrum(mode="branch", polarised=True)[1:n])
    a = np.asarray(out)
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(REPS)


def test_one_population_flank_site():
    n_e, p, t_inv, r, delta = 1000.0, 0.3, 3000.0, 1e-6, 300.0
    c = r * delta
    dem = msprime.Demography()
    dem.add_population(name="S", initial_size=2 * n_e * (1 - p))
    dem.add_population(name="I", initial_size=2 * n_e * p)
    dem.add_population(name="P", initial_size=2 * n_e)
    dem.set_migration_rate(source="I", dest="S", rate=c * (1 - p))
    dem.set_migration_rate(source="S", dest="I", rate=c * p)
    dem.add_population_split(time=t_inv, derived=["S", "I"], ancestral="P")
    mean, se = _branch_sfs(dem, {"I": 6}, 6)
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv, mean_tract_length=0.0
    )
    model = ExpectModel(Demography([n_e]), inv, r)
    got = model.expected_sfs(10_000.0 - delta, 0, "I", 6)
    assert np.all(np.abs(got - mean) < 4 * se)


def test_two_populations_conditional_migration():
    n_e, p0, p1, t_inv, m = 1000.0, 0.2, 0.6, 3000.0, 5e-4
    dem = msprime.Demography()
    for name, size in (
        ("S0", 2 * n_e * (1 - p0)),
        ("I0", 2 * n_e * p0),
        ("S1", 2 * n_e * (1 - p1)),
        ("I1", 2 * n_e * p1),
        ("P0", 2 * n_e),
        ("P1", 2 * n_e),
    ):
        dem.add_population(name=name, initial_size=size)
    # Conditional rates m * f_dest(K) / f_src(K) (spec §3.4 rule 3).
    dem.set_migration_rate(source="I0", dest="I1", rate=m * p1 / p0)
    dem.set_migration_rate(source="I1", dest="I0", rate=m * p0 / p1)
    dem.set_migration_rate(source="S0", dest="S1", rate=m * (1 - p1) / (1 - p0))
    dem.set_migration_rate(source="S1", dest="S0", rate=m * (1 - p0) / (1 - p1))
    dem.set_migration_rate(source="P0", dest="P1", rate=m)
    dem.set_migration_rate(source="P1", dest="P0", rate=m)
    dem.add_population_split(time=t_inv, derived=["S0", "I0"], ancestral="P0")
    dem.add_population_split(time=t_inv, derived=["S1", "I1"], ancestral="P1")
    mean, se = _branch_sfs(dem, {"I0": 1, "I1": 1}, 2)
    inv = InversionSpec(
        bp_left=10_000.0,
        bp_right=20_000.0,
        p_inv={0: p0, 1: p1},
        t_inv=t_inv,
        mean_tract_length=0.0,
    )
    model = ExpectModel(
        Demography([n_e, n_e], migration_matrix=[[0, m], [m, 0]]), inv, 1e-8
    )
    got = model.pairwise_times(15_000.0, (0, "I"), (1, "I")).mean
    assert abs(got - mean[0] / 2) < 4 * se[0] / 2


def test_size_change_epoch():
    n_e, p, t_inv = 1000.0, 0.4, 3000.0
    dem = msprime.Demography()
    dem.add_population(name="S", initial_size=2 * n_e * (1 - p))
    dem.add_population(name="I", initial_size=2 * n_e * p)
    dem.add_population(name="P", initial_size=n_e)  # 2 * (N / 2) after 800
    dem.add_population_parameters_change(
        time=800.0, population="I", initial_size=n_e * p
    )
    dem.add_population_parameters_change(
        time=800.0, population="S", initial_size=n_e * (1 - p)
    )
    dem.add_population_split(time=t_inv, derived=["S", "I"], ancestral="P")
    dem.sort_events()
    mean, se = _branch_sfs(dem, {"I": 4}, 4)
    demo = Demography([n_e])
    demo.add_population_size_change(800.0, population=0, new_size=n_e / 2)
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv, mean_tract_length=0.0
    )
    got = ExpectModel(demo, inv, 1e-8).expected_sfs(15_000.0, 0, "I", 4)
    assert np.all(np.abs(got - mean) < 4 * se)
