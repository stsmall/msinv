import numpy as np
import pytest

from msinv.expect._chain import I, P, S, Chain, canon, cell


def _chain(sample, n_pops=1, mig=(), joins=(), switching=True, origin="barrier",
           origin_pop=None, max_states=None):
    return Chain(sample, n_pops, set(mig), list(joins), switching, origin,
                 origin_pop, max_states)


def test_pair_chain_states_one_population():
    ch = _chain([(0, "I"), (0, "I")])
    # Before t_inv: II, IS, SS. After: PP. Plus the absorbed state.
    assert ch.size == 5


def test_sfs_chain_size_n20():
    # Bicoloured partitions of 20 with >= 2 parts (24,842 - 2), partitions of
    # 20 with >= 2 parts after t_inv (627 - 1), plus the absorbed state.
    assert _chain([(0, "I")] * 20).size == 24840 + 626 + 1
    assert _chain([(0, "I")] * 20, switching=False).size == 626 + 626 + 1


def test_max_states_guard():
    with pytest.raises(ValueError, match="max_states"):
        _chain([(0, "I")] * 20, max_states=1000)


def test_generators_conserve_probability():
    ch = _chain([(0, "I"), (0, "S"), (1, "I")], n_pops=2, mig=[(0, 1), (1, 0)],
                joins=[(1, 0)])
    for at in ch.AT.values():
        assert np.allclose(np.asarray(at.sum(axis=0)).ravel(), 0.0)
    for bt in ch.BT.values():
        assert np.allclose(np.asarray(bt.sum(axis=0)).ravel(), 1.0)


def test_single_origin_merges_inverted_lineages():
    ch = _chain([(0, "I"), (0, "I"), (0, "S")], switching=False, origin="single",
                origin_pop=0)
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 1), (cell(0, S), 1)])]
    j = ch.BT["t_inv"][:, i].nonzero()[0][0]
    assert ch.states[j] == canon([(cell(0, P), 1), (cell(0, P), 2)])


def test_rewards_count_blocks():
    ch = _chain([(0, "I")] * 3, switching=False)
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 2)])]
    assert ch.rewards[:, i].tolist() == [1.0, 1.0]
