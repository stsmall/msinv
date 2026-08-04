"""Per-arrangement diversity and divergence from a msinv TreeSequence.

Branch mode is used for pi and dxy: no mutation noise, so the t_inv signal is
cleaner. tskit branch-mode diversity returns the branch length separating a
pair, i.e. 2 * T_coal, so pi = mu * branch_diversity.
"""

from __future__ import annotations

MU_DEFAULT = 3e-9


def sample_nodes_by_karyotype(sim, ts):
    """Split sample nodes into (inverted, standard).

    msinv's ``HullSimulator`` builds ``sample_config`` as an ordered dict
    with the "S" (standard) entry inserted before the "I" (inverted) entry
    whenever the simpler ``n_std``/``n_inv`` constructor API is used (see
    ``msinv/hull/simulator.py``, ~line 425-429). Both the pure-Python
    fallback (``_initial_lineages``) and the Rust backend
    (``make_initial_lineages`` in ``rust/msinv-core/src/simulator.rs``,
    fed via ``rust/msinv-py/src/lib.rs``) create sample nodes by iterating
    that ordered list in order and calling ``tables.add_sample`` -- which
    assigns tskit node IDs sequentially starting at 0. So node IDs
    ``0 .. n_std-1`` are the standard karyotype and node IDs
    ``n_std .. n_std+n_inv-1`` are the inverted karyotype -- standard
    samples first, then inverted, the reverse of a naive "inverted first"
    guess. Verified two ways: (1) tracing the S-before-I dict insertion
    order through the Python constructor, the Rust bridge's sample_list
    conversion, and Rust's ``make_initial_lineages``; (2) empirically, with
    a strongly asymmetric (n_std=55, n_inv=5) run under an artificial
    strong inversion barrier, where the *last* 5 sample nodes form a
    monophyletic clade distinct from the other 55 in every marginal tree,
    consistent with them being the small n_inv=5 class.

    The partition-completeness test guards only sizes/disjointness/coverage
    and would pass either way; test_sample_nodes_by_karyotype_direction (in
    test_stats.py) is the regression guard for *direction* -- it fails if a
    future msinv version ever swaps which node-ID block is standard vs.
    inverted.

    Unverified: this ordering (and the ``sim.n_std`` split point used
    below) was only traced/tested for the ``n_std=``/``n_inv=`` constructor
    API. The raw ``sample_config=`` constructor path may insert "I" before
    "S" for a given call and has not been checked.
    """
    samples = list(ts.samples())
    n_std = int(sim.n_std)
    return samples[n_std:], samples[:n_std]


def arrangement_stats(ts, i_nodes, s_nodes, mu: float = MU_DEFAULT) -> dict:
    """pi within each arrangement, dxy between, and Hudson Fst."""
    pi_i = mu * ts.diversity([i_nodes], mode="branch")[0]
    pi_s = mu * ts.diversity([s_nodes], mode="branch")[0]
    # tskit 1.0.2 returns a bare numpy scalar (not a length-1 array) from
    # divergence() when given exactly two sample sets and no explicit
    # `indexes`, unlike diversity()'s always-an-array return -- so no [0].
    dxy = mu * ts.divergence([i_nodes, s_nodes], mode="branch")

    # Hudson Fst = 1 - mean_within / between
    mean_within = 0.5 * (pi_i + pi_s)
    fst = 1.0 - mean_within / dxy if dxy > 0 else float("nan")

    return {
        "pi_i": float(pi_i),
        "pi_s": float(pi_s),
        "dxy": float(dxy),
        "fst": float(fst),
        "pi_i_over_pi_s": float(pi_i / pi_s) if pi_s > 0 else float("nan"),
        "dxy_over_pi_i": float(dxy / pi_i) if pi_i > 0 else float("nan"),
    }
