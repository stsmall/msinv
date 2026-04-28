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
            "event log not recorded; pass record_events=True to "
            "HullSimulator")
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
    return np.array(
        [r["tract_right"] - r["tract_left"] for r in flux_records])


def survival_curve(values, ds):
    """S(d) = fraction of values >= d, evaluated at each d in ds."""
    values = np.asarray(values)
    return np.array([float(np.mean(values >= d)) for d in ds])


def coverage_count(flux_records, position):
    """How many flux events have tract_left <= position <= tract_right."""
    return sum(1 for r in flux_records
                 if r["tract_left"] <= position <= r["tract_right"])
