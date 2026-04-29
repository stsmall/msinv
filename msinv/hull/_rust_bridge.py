"""Rust-accelerated HullSimulator bridge.

Wraps ``_msinv_core.simulate_raw()`` to present the same API as the
pure-Python ``HullSimulator``. The Python ``msinv.hull.__init__``
conditionally imports from here when the Rust extension is available.
"""

import tskit

try:
    from msinv._msinv_core import simulate_raw as _simulate_raw
    RUST_AVAILABLE = True
except ImportError:
    try:
        from _msinv_core import simulate_raw as _simulate_raw
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False


def rust_simulate(simulator) -> 'tuple[tskit.TreeSequence, list | None]':
    """Run a simulation using the Rust core.

    Parameters
    ----------
    simulator : HullSimulator
        A Python HullSimulator instance — its attributes are read and
        passed to the Rust ``simulate_raw`` function.

    Returns
    -------
    (tskit.TreeSequence, list | None)
        - First element: the tree sequence (with simplify already applied).
        - Second element: the event log as a list of dicts when the
          simulator was constructed with ``record_events=True``,
          otherwise ``None``. Each dict has a ``"kind"`` key of
          ``"cmig"`` or ``"flux"`` plus the variant fields. See
          :mod:`msinv.hull._event_log` for parsing helpers.
    """
    # --- Sample config ---
    # Convert sample_config dict → list of (kary_str, pop, count) tuples.
    sample_list = []
    n_inv = len(simulator.inversions)
    for (karyotype, pop), count in simulator.sample_config.items():
        if karyotype is None:
            kary_str = 'P'
        elif isinstance(karyotype, str) and len(karyotype) == 1:
            kary_str = karyotype
        elif hasattr(karyotype, '__iter__'):
            kary_str = ''.join(k if k else 'P' for k in karyotype)
        else:
            kary_str = str(karyotype)
        sample_list.append((kary_str, int(pop), int(count)))

    # --- Inversions ---
    n_pops = simulator.demography.n_pops
    inv_dicts = []
    for inv in simulator.inversions:
        d = {
            'bp_left': float(inv.bp_left),
            'bp_right': float(inv.bp_right),
            'gene_conversion_rate': float(inv.gene_conversion_rate),
            'mean_tract_length': float(inv.mean_tract_length),
            'tract_distribution': str(inv.tract_distribution),
        }
        if getattr(inv, 'trajectory', None) is not None:
            # New trajectory path — pass dict straight to Rust.
            d['trajectory'] = dict(inv.trajectory)
        else:
            # Back-compat: constant p_inv/t_inv
            d['p_inv'] = inv._p_inv_as_list(n_pops)
            d['t_inv'] = float(inv.t_inv)
        inv_dicts.append(d)

    # --- Sweeps ---
    sweep_specs = [sw.to_rust() for sw in simulator.sweeps]

    # --- Demography ---
    demo = simulator.demography
    pop_sizes = [float(n) for n in demo.pop_sizes]
    mig = [[float(demo.migration_matrix[i][j])
             for j in range(demo.n_pops)]
            for i in range(demo.n_pops)]

    # Convert events to tuples.
    demo_events = []
    for ev in demo.events:
        demo_events.append(ev)  # already tuples like ('ej', t, src, dst)

    # --- Call Rust ---
    raw, event_log = _simulate_raw(
        sample_config=sample_list,
        pop_sizes=pop_sizes,
        sequence_length=float(simulator.L),
        recombination_rate=float(simulator.r),
        inversions=inv_dicts if inv_dicts else None,
        sweeps=sweep_specs if sweep_specs else None,
        demo_events=demo_events if demo_events else None,
        migration_matrix=mig,
        seed=int(simulator.rng.integers(0, 2**63))
            if hasattr(simulator, 'rng') else 42,
        stop_at=float(getattr(simulator, 'stop_at', float('inf'))),
        compound_rate=bool(getattr(simulator, 'compound_rate', False)),
        iters_max=int(getattr(simulator, 'iters_max', 10_000_000)),
        gc_stride=int(getattr(simulator, 'gc_stride', 160)),
        record_events=bool(getattr(simulator, '_record_events', False)),
    )

    # --- Convert to tskit TreeSequence ---
    tc = tskit.TableCollection(raw['sequence_length'])
    for _ in range(int(raw['num_populations'])):
        tc.populations.add_row()
    tc.nodes.set_columns(
        flags=raw['node_flags'],
        time=raw['node_time'],
        population=raw['node_population'],
    )
    tc.edges.set_columns(
        left=raw['edge_left'],
        right=raw['edge_right'],
        parent=raw['edge_parent'],
        child=raw['edge_child'],
    )
    # Rust emits edges pre-sorted in tskit canonical order
    # (time[parent] asc, parent, child, left); skip tc.sort().
    ts = tc.tree_sequence()
    # event_log is None when record_events=False, or a list of dicts when True.
    return ts.simplify(), event_log, int(raw.get("sweep_a_count", 0))
