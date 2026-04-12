"""
ctypes bridge to libmsinv.so — full structured coalescent with inversions.

Handles trajectory pre-computation in Python, passes flat arrays to C.
"""
import ctypes
import numpy as np
import os

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    libpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'libmsinv.so')
    if not os.path.exists(libpath):
        return None
    _lib = ctypes.CDLL(libpath)

    # msinv_seed
    _lib.msinv_seed.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    _lib.msinv_seed.restype = None

    # msinv_simulate_flat
    _lib.msinv_simulate_flat.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,          # nsam, n_std, n_inv
        ctypes.c_double, ctypes.c_double, ctypes.c_int,    # theta, rho, nsites
        # Inversion 1
        ctypes.c_double, ctypes.c_double, ctypes.c_double, # bp_l, bp_r, gamma
        ctypes.c_double, ctypes.c_double,                   # flux_w, t_inv
        ctypes.POINTER(ctypes.c_double),                    # traj_times
        ctypes.POINTER(ctypes.c_double),                    # traj_freqs
        ctypes.c_int, ctypes.c_int,                         # n_steps, n_pops
        # Inversion 2
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        # Demography
        ctypes.c_int, ctypes.c_double,                     # n_pops, mig_rate
        ctypes.c_char_p,                                    # demo_types
        ctypes.POINTER(ctypes.c_double),                    # demo_times
        ctypes.POINTER(ctypes.c_int),                       # demo_pop_i
        ctypes.POINTER(ctypes.c_int),                       # demo_pop_j
        ctypes.POINTER(ctypes.c_double),                    # demo_values
        ctypes.c_int,                                       # n_demo_events
        ctypes.POINTER(ctypes.c_double),                    # pop_sizes
        ctypes.POINTER(ctypes.c_int),                       # sample_config
        # Output
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib.msinv_simulate_flat.restype = ctypes.c_int

    return _lib


def is_available():
    return _load() is not None


def seed(s):
    lib = _load()
    lib.msinv_seed(ctypes.c_uint64(s), ctypes.c_uint64(s ^ 0xDEADBEEF))


def _make_traj_arrays(p_inv_func, t_inv=None):
    """Pre-compute trajectory as flat arrays for C."""
    if p_inv_func is None:
        # Constant zero
        times = np.array([0.0, 100.0])
        freqs = np.array([0.0, 0.0])
        return times, freqs, 2, 1, 0.0

    n_pops = getattr(p_inv_func, 'n_pops', 1)
    t_inv_val = getattr(p_inv_func, 't_inv', t_inv or 100.0)

    # Sample trajectory at fine resolution
    n_steps = min(10000, max(100, int(t_inv_val * 100)))
    times = np.linspace(0, t_inv_val * 1.01, n_steps)

    if n_pops > 1:
        freqs = np.zeros(n_pops * n_steps)
        for i, t in enumerate(times):
            for p in range(n_pops):
                freqs[p * n_steps + i] = p_inv_func(t, p)
    else:
        freqs = np.array([p_inv_func(t) for t in times])

    return times, freqs, n_steps, n_pops, t_inv_val


def simulate_one(nsam, n_std, n_inv, theta, rho, nsites,
                 p_inv=0.5, gamma=0.05, flux_w=0.3,
                 bp_left=0.3, bp_right=0.7,
                 t_inv=10.0, p_inv_func=None,
                 inversions=None,
                 n_pops=1, mig_rate=0.0,
                 demography=None, sample_config=None,
                 max_sites=100000):
    """
    Run one replicate via C. Returns (positions, haplotypes).

    Parameters match MsinvSimulator but are passed flat to C.
    Trajectories are pre-computed in Python, passed as arrays.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("libmsinv.so not found")

    # Build trajectory arrays for up to 2 inversions
    if inversions and len(inversions) >= 1:
        inv = inversions[0]
        bp_l1 = inv.bp_left
        bp_r1 = inv.bp_right
        g1 = inv.gamma if inv.gamma is not None else gamma
        fw1 = inv.flux_w
        t1, f1, ns1, np1, ti1 = _make_traj_arrays(
            getattr(inv, 'p_inv_func', p_inv_func),
            getattr(inv, 't_inv', t_inv))
    elif p_inv > 0:
        bp_l1 = bp_left
        bp_r1 = bp_right
        g1 = gamma
        fw1 = flux_w
        t1, f1, ns1, np1, ti1 = _make_traj_arrays(p_inv_func, t_inv)
    else:
        bp_l1 = -1.0  # no inversion
        bp_r1 = bp_l1
        g1 = fw1 = ti1 = 0.0
        t1 = np.zeros(2)
        f1 = np.zeros(2)
        ns1 = 2
        np1 = 1

    if inversions and len(inversions) >= 2:
        inv2 = inversions[1]
        bp_l2 = inv2.bp_left
        bp_r2 = inv2.bp_right
        g2 = inv2.gamma if inv2.gamma is not None else gamma
        fw2 = inv2.flux_w
        t2, f2, ns2, np2, ti2 = _make_traj_arrays(
            getattr(inv2, 'p_inv_func', p_inv_func),
            getattr(inv2, 't_inv', t_inv))
    else:
        bp_l2 = -1.0
        bp_r2 = bp_l2
        g2 = fw2 = ti2 = 0.0
        t2 = np.zeros(2)
        f2 = np.zeros(2)
        ns2 = 2
        np2 = 1

    # Demography events
    if demography is not None and hasattr(demography, 'events'):
        events = sorted(demography.events, key=lambda e: e[1])
        n_events = len(events)
        d_types = b''.join(e[0][0:1].encode() if isinstance(e[0], str)
                          else bytes([e[0]]) for e in events)
        d_times = np.array([e[1] for e in events], dtype=np.float64)
        d_pop_i = np.array([e[2] if len(e) > 2 else 0 for e in events],
                           dtype=np.int32)
        d_pop_j = np.array([e[3] if len(e) > 3 else 0 for e in events],
                           dtype=np.int32)
        d_values = np.array([e[-1] if len(e) > 2 else 0.0 for e in events],
                            dtype=np.float64)
    else:
        n_events = 0
        d_types = b''
        d_times = np.zeros(1, dtype=np.float64)
        d_pop_i = np.zeros(1, dtype=np.int32)
        d_pop_j = np.zeros(1, dtype=np.int32)
        d_values = np.zeros(1, dtype=np.float64)

    # Pop sizes
    if demography is not None and hasattr(demography, 'state'):
        ps = np.array(demography.state.pop_sizes[:n_pops], dtype=np.float64)
    else:
        ps = np.ones(max(1, n_pops), dtype=np.float64)

    # Sample config
    if sample_config is not None and isinstance(sample_config, dict):
        sc = np.zeros(2 * n_pops, dtype=np.int32)
        for (cls, pop), count in sample_config.items():
            idx = pop if cls == 'S' else n_pops + pop
            sc[idx] = count
        sc_ptr = sc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        sc_ptr = None

    # Output arrays
    out_haps = np.zeros(nsam * max_sites, dtype=np.int8)
    out_pos = np.zeros(max_sites, dtype=np.float64)

    # Ensure contiguous arrays
    t1 = np.ascontiguousarray(t1, dtype=np.float64)
    f1 = np.ascontiguousarray(f1, dtype=np.float64)
    t2 = np.ascontiguousarray(t2, dtype=np.float64)
    f2 = np.ascontiguousarray(f2, dtype=np.float64)

    n_muts = lib.msinv_simulate_flat(
        nsam, n_std, n_inv, theta, rho, nsites,
        bp_l1, bp_r1, g1, fw1, ti1,
        t1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ns1, np1,
        bp_l2, bp_r2, g2, fw2, ti2,
        t2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ns2, np2,
        n_pops, mig_rate,
        d_types if n_events > 0 else None,
        d_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        d_pop_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        d_pop_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        d_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_events,
        ps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sc_ptr,
        out_haps.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        out_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        max_sites)

    if n_muts <= 0:
        return [], np.zeros((nsam, 0), dtype=np.int8)

    positions = list(out_pos[:n_muts])
    haplotypes = out_haps.reshape(nsam, max_sites)[:, :n_muts].copy()
    return positions, haplotypes
