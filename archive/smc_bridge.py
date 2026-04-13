"""ctypes bridge to libsmc_core.so for C-accelerated SMC operations."""
import ctypes
import numpy as np
import os

_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib
    libpath = os.path.join(os.path.dirname(__file__), 'libsmc_core.so')
    if not os.path.exists(libpath):
        return None
    _lib = ctypes.CDLL(libpath)

    # void smc_seed(uint64_t s)
    _lib.smc_seed.argtypes = [ctypes.c_uint64]
    _lib.smc_seed.restype = None

    # void smc_branch_lengths(double*, int*, char*, int, double*, double*, double*)
    _lib.smc_branch_lengths.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # time
        ctypes.POINTER(ctypes.c_int),     # parent
        ctypes.POINTER(ctypes.c_byte),    # klass
        ctypes.c_int,                     # n
        ctypes.POINTER(ctypes.c_double),  # l_s
        ctypes.POINTER(ctypes.c_double),  # l_i
        ctypes.POINTER(ctypes.c_double),  # t_max
    ]
    _lib.smc_branch_lengths.restype = None

    # int smc_panmictic_pr(double*, int*, int*, int*, char*, char*, int*, int, int)
    _lib.smc_panmictic_pr.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # time
        ctypes.POINTER(ctypes.c_int),     # parent
        ctypes.POINTER(ctypes.c_int),     # left_child
        ctypes.POINTER(ctypes.c_int),     # right_sib
        ctypes.POINTER(ctypes.c_byte),    # klass
        ctypes.POINTER(ctypes.c_byte),    # population
        ctypes.POINTER(ctypes.c_int),     # sample_id
        ctypes.c_int,                     # n_nodes
        ctypes.c_int,                     # root
    ]
    _lib.smc_panmictic_pr.restype = ctypes.c_int

    # int smc_structured_reattach(...)
    _lib.smc_structured_reattach.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # time
        ctypes.POINTER(ctypes.c_int),     # parent
        ctypes.POINTER(ctypes.c_int),     # left_child
        ctypes.POINTER(ctypes.c_int),     # right_sib
        ctypes.POINTER(ctypes.c_byte),    # klass
        ctypes.POINTER(ctypes.c_byte),    # population
        ctypes.POINTER(ctypes.c_int),     # sample_id
        ctypes.c_int,                     # n_nodes
        ctypes.c_int,                     # root
        ctypes.c_int,                     # target
        ctypes.c_byte,                    # fclass
        ctypes.c_double,                  # t_cut
        ctypes.c_double,                  # p_inv
        ctypes.c_double,                  # c
        ctypes.c_double,                  # rho
        ctypes.c_double,                  # phi_x
        ctypes.c_double,                  # t_inv
    ]
    _lib.smc_structured_reattach.restype = ctypes.c_int

    return _lib


def is_available():
    return _load() is not None


def seed(s):
    lib = _load()
    if lib:
        lib.smc_seed(ctypes.c_uint64(s))


def branch_lengths(time, parent, klass, n):
    """Returns (L_S, L_I, t_max)."""
    lib = _load()
    l_s = ctypes.c_double()
    l_i = ctypes.c_double()
    t_max = ctypes.c_double()
    lib.smc_branch_lengths(
        time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        klass.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        n, ctypes.byref(l_s), ctypes.byref(l_i), ctypes.byref(t_max))
    return l_s.value, l_i.value, t_max.value


def panmictic_pr(tree):
    """Panmictic prune-and-reattach. Modifies tree in place, returns new root."""
    lib = _load()
    new_root = lib.smc_panmictic_pr(
        tree.time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tree.parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.left_child.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.right_sib.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.klass.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        tree.population.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        tree.sample_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.n, tree.root)
    # The C function adds at most 1 node
    tree.n += 1
    tree.root = new_root
    return new_root


def structured_reattach(tree, target, fclass, t_cut, p_inv, c, rho, phi_x, t_inv):
    """Structured reattach. Returns new root."""
    lib = _load()
    new_root = lib.smc_structured_reattach(
        tree.time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tree.parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.left_child.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.right_sib.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.klass.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        tree.population.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        tree.sample_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tree.n, tree.root,
        target, ctypes.c_byte(fclass), t_cut,
        p_inv, c, rho, phi_x, t_inv)
    # C function may add several nodes (flux events + coal)
    # Count actual nodes used
    while tree.n < tree.max_nodes and tree.time[tree.n] != 0 and tree.parent[tree.n] != -1:
        tree.n += 1
    # Actually, safer: scan for max used node
    for i in range(tree.n, min(tree.n + 100, tree.max_nodes)):
        if tree.parent[i] != -1 or tree.left_child[i] != -1:
            tree.n = i + 1
    tree.root = new_root
    return new_root
