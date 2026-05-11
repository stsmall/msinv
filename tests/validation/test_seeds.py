"""Tests for validation seed generation."""

import pytest
from validation._lib.seeds import seed_for


def test_same_inputs_give_same_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    assert s1 == s2


def test_different_rep_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="msinv", rep=1)
    assert s1 != s2


def test_different_engine_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track1", scenario="default", engine="slim", rep=0)
    assert s1 != s2


def test_different_track_gives_different_seed():
    s1 = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    s2 = seed_for(track="track2", scenario="default", engine="msinv", rep=0)
    assert s1 != s2


def test_seed_in_uint32_range():
    """Seeds must fit in uint32 for SLiM/msprime/discoal compatibility."""
    s = seed_for(track="track1", scenario="default", engine="msinv", rep=0)
    assert 1 <= s <= 2**31 - 1


def test_no_collisions_across_10k_combos():
    """Random sample of 10k (track, rep, engine) tuples should give 10k distinct seeds."""
    seeds = set()
    for track_i in range(5):
        for engine in ("msinv", "slim", "msprime", "discoal"):
            for rep in range(500):
                seeds.add(
                    seed_for(
                        track=f"track{track_i}",
                        scenario="default",
                        engine=engine,
                        rep=rep,
                    )
                )
    assert len(seeds) == 5 * 4 * 500
