"""Deterministic seed generation for the validation suite.

Seeds are derived from a stable hash of (track, scenario, engine, rep) and
clamped to the uint31 range so they round-trip through SLiM, msprime, and
discoal (all of which use 32-bit signed seeds).
"""
import hashlib


def seed_for(*, track: str, scenario: str, engine: str, rep: int) -> int:
    """Return a deterministic uint31 seed for (track, scenario, engine, rep).

    Use kwargs only so call sites are explicit.
    """
    key = f"{track}|{scenario}|{engine}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big")
    return (raw % (2**31 - 1)) + 1
