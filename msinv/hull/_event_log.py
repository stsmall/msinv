"""Helpers for analyzing simulator event logs.

The event log is a list of dicts (or None), populated when a
HullSimulator was constructed with record_events=True. Each dict has a
'kind' field of 'cmig' or 'flux' plus the variant's typed fields.
"""

from __future__ import annotations

import numpy as np


def _require_log(log):
    if log is None:
        raise ValueError(
            "event log not recorded; pass record_events=True to HullSimulator"
        )
    return log


def filter_cmig(log):
    """Return only the cmig records from the log.

    Raises ValueError if the log is None.
    """
    return [r for r in _require_log(log) if r["kind"] == "cmig"]


def filter_flux(log, inv_id=None):
    """Return only the flux records from the log.

    Optionally filter to a specific inv_id.
    Raises ValueError if the log is None.
    """
    out = [r for r in _require_log(log) if r["kind"] == "flux"]
    if inv_id is not None:
        out = [r for r in out if r["inv_id"] == inv_id]
    return out


def tract_lengths(flux_records):
    """Return the tract lengths (right - left) for each flux record."""
    return np.array([r["tract_right"] - r["tract_left"] for r in flux_records])


def survival_curve(values, ds):
    """S(d) = fraction of values >= d, evaluated at each d in ds."""
    values = np.asarray(values)
    return np.array([float(np.mean(values >= d)) for d in ds])


def coverage_count(flux_records, position):
    """How many flux events have tract_left <= position <= tract_right."""
    return sum(
        1 for r in flux_records if r["tract_left"] <= position <= r["tract_right"]
    )


def samples_converted_at(flux_records, ts, position):
    """Fraction of samples whose ancestry at `position` was hit by ≥1
    flux event.

    For each flux record whose tract covers `position`, find the
    segment in `tract_segments` that contains `position`, take its
    `node_id`, and union the descendant samples (in the marginal tree
    at `position`) into a converted set.

    Parameters
    ----------
    flux_records : iterable of dicts
        Filtered flux records (from `filter_flux`). Each record must
        contain `tract_left`, `tract_right`, and `tract_segments`
        (a list of dicts with keys `seg_left`, `seg_right`, `node_id`
        spanning the tract).
    ts : tskit.TreeSequence
    position : float

    Returns
    -------
    fraction : float in [0.0, 1.0]
        len(converted) / ts.num_samples; 0.0 if num_samples == 0.
    """
    if ts.num_samples == 0:
        return 0.0
    tree = ts.at(position)
    converted: set[int] = set()
    for rec in flux_records:
        if not (rec["tract_left"] <= position <= rec["tract_right"]):
            continue
        segs = rec.get("tract_segments")
        if not segs:
            continue
        node_id = -1
        for seg in segs:
            if seg["seg_left"] <= position < seg["seg_right"]:
                node_id = int(seg["node_id"])
                break
        if node_id < 0 and segs:
            last = segs[-1]
            if last["seg_left"] <= position <= last["seg_right"]:
                node_id = int(last["node_id"])
        if node_id < 0 or node_id >= ts.num_nodes:
            continue
        for s in tree.samples(node_id):
            converted.add(int(s))
    return len(converted) / ts.num_samples
