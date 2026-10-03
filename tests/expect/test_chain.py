import numpy as np
import pytest

from msinv.expect._chain import I, P, S, Chain, canon, cell


def _chain(
    sample,
    n_pops=1,
    mig=(),
    joins=(),
    switching=True,
    origin="barrier",
    origin_pop=None,
    max_states=None,
):
    return Chain(
        sample, n_pops, set(mig), list(joins), switching, origin, origin_pop, max_states
    )


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
    ch = _chain(
        [(0, "I"), (0, "S"), (1, "I")], n_pops=2, mig=[(0, 1), (1, 0)], joins=[(1, 0)]
    )
    for at in ch.AT.values():
        assert np.allclose(np.asarray(at.sum(axis=0)).ravel(), 0.0)
    for bt in ch.BT.values():
        assert np.allclose(np.asarray(bt.sum(axis=0)).ravel(), 1.0)


def test_single_origin_merges_inverted_lineages():
    ch = _chain(
        [(0, "I"), (0, "I"), (0, "S")], switching=False, origin="single", origin_pop=0
    )
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 1), (cell(0, S), 1)])]
    j = ch.BT["t_inv"][:, i].nonzero()[0][0]
    assert ch.states[j] == canon([(cell(0, P), 1), (cell(0, P), 2)])


def test_rewards_count_blocks():
    ch = _chain([(0, "I")] * 3, switching=False)
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 2)])]
    assert ch.rewards[:, i].tolist() == [1.0, 1.0]


def _rate(ch, key, src, dst):
    return ch.AT[key][ch.index[dst], ch.index[src]]


def _st(*lin):
    return canon(lin)


def test_coal_rates_same_type():
    ch = _chain([(0, "I")] * 3, switching=False)
    a = ch.AT[("coal", 0, I)]
    i = ch.init
    j = ch.index[_st((cell(0, I), 1), (cell(0, I), 2))]
    assert a[j, i] == 3
    assert a[i, i] == -3


def test_coal_rates_distinct_block_types():
    ch = _chain([(0, "I")] * 4, switching=False)
    c = cell(0, I)
    s = _st((c, 1), (c, 1), (c, 2))
    assert _rate(ch, ("coal", 0, I), s, _st((c, 2), (c, 2))) == 1
    assert _rate(ch, ("coal", 0, I), s, _st((c, 1), (c, 3))) == 2


def test_migration_rate():
    ch = _chain([(0, "I"), (0, "I")], n_pops=2, mig=[(0, 1)], switching=False)
    s = _st((cell(0, I), 1), (cell(0, I), 1))
    t = _st((cell(0, I), 1), (cell(1, I), 1))
    assert _rate(ch, ("mig", 0, 1, I), s, t) == 2
    assert ("mig", 1, 0, I) not in ch.AT


def test_switching_rate():
    ch = _chain([(0, "I"), (0, "I")])
    s = _st((cell(0, I), 1), (cell(0, I), 1))
    t = _st((cell(0, I), 1), (cell(0, S), 1))
    assert _rate(ch, ("sw", 0, I), s, t) == 2


def test_join_map():
    ch = _chain(
        [(0, "I"), (1, "I"), (0, "S")], n_pops=2, joins=[(1, 0)], switching=False
    )
    bt = ch.BT[("join", 1, 0)]
    s = _st((cell(0, I), 1), (cell(1, I), 1), (cell(0, S), 1))
    t = _st((cell(0, I), 1), (cell(0, I), 1), (cell(0, S), 1))
    assert ch.states[bt[:, ch.index[s]].nonzero()[0][0]] == t


def test_join_map_identity_state():
    # Migration 1 -> 0 lets the pop-1 lineage leave; that state is a fixed point.
    ch = _chain(
        [(0, "I"), (0, "I"), (1, "I")],
        n_pops=2,
        mig=[(1, 0)],
        joins=[(1, 0)],
        switching=False,
    )
    bt = ch.BT[("join", 1, 0)]
    s = _st((cell(0, I), 1), (cell(0, I), 1), (cell(0, I), 1))
    assert ch.states[bt[:, ch.index[s]].nonzero()[0][0]] == s


def test_barrier_t_inv():
    ch = _chain([(0, "I"), (1, "S")], n_pops=2, switching=False)
    s = _st((cell(0, I), 1), (cell(1, S), 1))
    t = _st((cell(0, P), 1), (cell(1, P), 1))
    assert ch.states[ch.BT["t_inv"][:, ch.index[s]].nonzero()[0][0]] == t


def test_single_origin_two_populations():
    ch = _chain(
        [(0, "I"), (1, "I"), (1, "S")],
        n_pops=2,
        switching=False,
        origin="single",
        origin_pop=0,
    )
    s = _st((cell(0, I), 1), (cell(1, I), 1), (cell(1, S), 1))
    t = _st((cell(0, P), 2), (cell(1, P), 1))
    assert ch.states[ch.BT["t_inv"][:, ch.index[s]].nonzero()[0][0]] == t


def test_origin_validation():
    with pytest.raises(ValueError, match="origin"):
        _chain([(0, "I")] * 2, origin="bogus")
    with pytest.raises(ValueError, match="origin_pop"):
        _chain([(0, "I")] * 2, origin="single", origin_pop=None)
