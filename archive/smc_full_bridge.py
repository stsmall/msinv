"""ctypes bridge to libsmc_full.so — one C call per replicate."""
import ctypes
import numpy as np
import os

_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib
    libpath = os.path.join(os.path.dirname(__file__), 'libsmc_full.so')
    if not os.path.exists(libpath):
        return None
    _lib = ctypes.CDLL(libpath)

    _lib.smc_full_seed.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    _lib.smc_full_seed.restype = None

    _lib.smc_full_simulate.argtypes = [
        ctypes.c_int,      # n_std
        ctypes.c_int,      # n_inv
        ctypes.c_double,   # theta
        ctypes.c_double,   # rho
        ctypes.c_int,      # nsites
        ctypes.c_double,   # p_inv
        ctypes.c_double,   # c_flux
        ctypes.c_double,   # t_inv
        ctypes.c_double,   # bp_left
        ctypes.c_double,   # bp_right
        ctypes.c_double,   # flux_w
        ctypes.POINTER(ctypes.c_int8),    # out_haps
        ctypes.POINTER(ctypes.c_double),  # out_positions
        ctypes.c_int,      # max_sites
    ]
    _lib.smc_full_simulate.restype = ctypes.c_int
    return _lib


def is_available():
    return _load() is not None


def seed(s):
    lib = _load()
    lib.smc_full_seed(ctypes.c_uint64(s), ctypes.c_uint64(s ^ 0xDEADBEEF))


def simulate_one(n_std, n_inv, theta, rho, nsites,
                  p_inv=0.5, c_flux=0.01, t_inv=10.0,
                  bp_left=0.3, bp_right=0.7, flux_w=0.3,
                  max_sites=50000):
    """Run one replicate via C. Returns (positions, haplotypes)."""
    lib = _load()
    nsam = n_std + n_inv

    out_haps = np.zeros(nsam * max_sites, dtype=np.int8)
    out_positions = np.zeros(max_sites, dtype=np.float64)

    n_muts = lib.smc_full_simulate(
        n_std, n_inv, theta, rho, nsites,
        p_inv, c_flux, t_inv,
        bp_left, bp_right, flux_w,
        out_haps.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        out_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        max_sites)

    if n_muts <= 0:
        return [], np.zeros((nsam, 0), dtype=np.int8)

    positions = list(out_positions[:n_muts])
    haplotypes = out_haps.reshape(nsam, max_sites)[:, :n_muts].copy()
    return positions, haplotypes
