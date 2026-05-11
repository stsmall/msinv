#!/usr/bin/env python3
"""Mock RDL / 2La pilot setup — NOT RUNNABLE AT REAL SCALE.

Captures the An. gambiae 2La biology and the young-RDL-sweep-inside-
old-inversion model that the pan-African ABC targets. Documented to
illustrate WHY we need Path 2 (compound per-pair rate main-path
rewrite) before this can run at realistic Ne.

The mock uses real-biology parameters. Running at those parameters
with the current event loop will hit the remnant-ratchet explosion
documented in `project_panic_kirfol_en.md`:

  Phase 1 (0 -> ~1000 gen):  multi-pop + migration, cheap
  Phase 2 (sweep at t_sweep): force-coalesce RDL carriers, cheap
  Phase 3 (1000 -> t_2La):    multi-pop + migration + active 2La
                              barrier. n grows unboundedly via
                              class-mismatch partial coalescences.
                              *This phase is the bottleneck*
  Phase 4 (post-barrier):     huge active-n Hudson cleanup.

Path 1 (analytic-middle fast path) does NOT apply here — it requires
single-pop and zero migration during the barrier. Path 2 fixes the
ratchet for ALL demographies.

Use `SCALE=tiny` for a quick syntax / config test that actually runs.
Use `SCALE=real` to document the full biology; currently hangs.

When Path 2 lands, flip SCALE=real and run as the RDL ABC driver.
"""

from __future__ import annotations

import argparse
import sys
import time

from msinv import HullSimulator, InversionSpec, Demography
from msinv.hull.sweep import Sweep


# ------------------------------------------------------------------
# An. gambiae 2La + RDL real-biology parameters (mock; refine with
# literature + your data).
# ------------------------------------------------------------------

# 2La is a ~22 Mb inversion on 2L, estimated age ~500k gen (Cheng et al.
# 2012 / Fouet et al.). Large, old, carries the RDL locus at its distal
# end. Frequencies vary by population and habitat.
REAL_2LA = dict(
    bp_left=20_500_000,
    bp_right=42_200_000,
    # p_inv per pop — gambiae examples (adjust to your sampling):
    #   pop 0 (savanna): high 2La+ (inverted) frequency
    #   pop 1 (forest):  low 2La+ frequency
    p_inv={0: 0.85, 1: 0.10},
    t_inv=500_000,
    gene_conversion_rate=1e-9,
    mean_tract_length=1_085_000.0,
    tract_distribution="fixed",
)

# RDL (Rdl gene) is near the distal end of 2La. Selected allele arose
# during the organochlorine era and rose rapidly under dieldrin/HCH
# pressure. Sweep started ~50-200 gen ago depending on locale.
REAL_RDL_POS = 40_000_000  # inside 2La, toward distal end
REAL_SWEEP = dict(
    x_sel=REAL_RDL_POS,
    t_event=200,  # backward gen; refine from data
    # target: sweep on I-class at 2La (inv_id=0) since RDL is in the inv
    target=(0, "I"),
    population=None,  # global sweep; or restrict to one
    sweep_window=50_000,  # ~50 kb window around RDL
    selection_coefficient=0.1,  # ABC prior will cover this
    starting_frequency=0.0,  # hard sweep; ABC prior covers soft
)

# Gambiae demography (toy 2-pop version; full pan-African ABC will
# use 4-6 pops per Small + Neafsey etc.)
REAL_DEMO = dict(
    pop_sizes=[1_000_000, 2_000_000],  # savanna, forest; order-of-mag
    events=[
        # Recent bottleneck in pop 1 (example)
        ("en", 5_000, 1, 500_000),
        # Deep ancestral merge
        ("ej", 100_000, 1, 0),
        ("en", 100_000, 0, 3_000_000),
    ],
    # Symmetric migration while pops are separate
    migration_rate=1e-4,
)

# Sample composition — 2La karyotype × population
REAL_SAMPLES = {
    ("II", 0): 40,  # 2La+ homokaryotes in savanna
    ("SS", 0): 10,  # 2L+ (standard) homokaryotes in savanna
    ("II", 1): 5,  # rare inverted in forest
    ("SS", 1): 45,  # standard dominates forest
}

# Chromosome scale. Real 2L arm is ~48 Mb; the 2La inv is 22 Mb of that.
# Running the full 2L at Ne=1e6+ is where we hit the remnant ratchet.
REAL_L = 48_000_000
REAL_R = 1e-8

# ------------------------------------------------------------------
# Tiny-scale mirror (runs in seconds; same SHAPE, not same biology)
# ------------------------------------------------------------------
TINY_2LA = dict(
    bp_left=200_000,
    bp_right=420_000,
    p_inv={0: 0.85, 1: 0.10},
    t_inv=1_000,  # compressed barrier
    gene_conversion_rate=1e-9,
    mean_tract_length=11_000.0,
    tract_distribution="fixed",
)
TINY_RDL_POS = 400_000
TINY_SWEEP = dict(
    x_sel=TINY_RDL_POS,
    t_event=50,
    target=(0, "I"),
    population=None,
    sweep_window=5_000,
    selection_coefficient=0.1,
    starting_frequency=0.0,
)
TINY_DEMO = dict(
    pop_sizes=[2_000, 4_000],
    events=[
        ("en", 50, 1, 1_000),
        ("ej", 500, 1, 0),
        ("en", 500, 0, 6_000),
    ],
    migration_rate=1e-4,
)
TINY_SAMPLES = {
    ("II", 0): 8,
    ("SS", 0): 2,
    ("II", 1): 1,
    ("SS", 1): 9,
}
TINY_L = 500_000
TINY_R = 1e-8


SCALES = {
    "real": dict(
        inv=REAL_2LA,
        sweep=REAL_SWEEP,
        demo=REAL_DEMO,
        samples=REAL_SAMPLES,
        L=REAL_L,
        r=REAL_R,
    ),
    "tiny": dict(
        inv=TINY_2LA,
        sweep=TINY_SWEEP,
        demo=TINY_DEMO,
        samples=TINY_SAMPLES,
        L=TINY_L,
        r=TINY_R,
    ),
}


def build_sim(cfg, seed):
    demo = Demography(pop_sizes=cfg["demo"]["pop_sizes"])
    n_pops = len(cfg["demo"]["pop_sizes"])
    m = cfg["demo"]["migration_rate"]
    for i in range(n_pops):
        for j in range(n_pops):
            if i != j:
                demo.migration_matrix[i][j] = m
    for ev in cfg["demo"]["events"]:
        demo.add_event(ev)

    inv = InversionSpec(**cfg["inv"])
    sweep = Sweep(
        x_sel=cfg["sweep"]["x_sel"],
        t_event=cfg["sweep"]["t_event"],
        target_class="I",  # for Python API; bridge maps
        population=cfg["sweep"]["population"],
        sweep_window=cfg["sweep"]["sweep_window"],
        selection_coefficient=cfg["sweep"]["selection_coefficient"],
        starting_frequency=cfg["sweep"]["starting_frequency"],
    )

    return HullSimulator(
        sample_config=cfg["samples"],
        demography=demo,
        sequence_length=cfg["L"],
        recombination_rate=cfg["r"],
        inversions=[inv],
        sweeps=[sweep],
        seed=seed,
    )


def estimate_cost(cfg):
    """Rough wall-time estimate given scale and perf table."""
    rho = 4.0 * max(cfg["demo"]["pop_sizes"]) * cfg["r"] * cfg["L"]
    has_ratchet = cfg["inv"]["t_inv"] > 5_000 and max(cfg["demo"]["pop_sizes"]) > 1e5
    note = f"rho ≈ {rho:.0f}; "
    if has_ratchet:
        note += "RATCHET EXPECTED — Path 2 required to run realistically"
    else:
        note += "tractable on event loop"
    return rho, note


def abc_priors():
    """Prior ranges for the selection+timing ABC once Path 2 lands."""
    return dict(
        selection_coefficient=(0.001, 0.5),
        t_event=(10, 500),  # gen, young sweep
        starting_frequency=(0.0, 0.1),  # hard to mild soft
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALES), default="tiny")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build sim + report cost estimate but don't run.",
    )
    args = p.parse_args()

    cfg = SCALES[args.scale]
    rho, note = estimate_cost(cfg)
    print(f"Scale: {args.scale}")
    print(f"  L = {cfg['L']:,} bp")
    print(
        f"  inv = [{cfg['inv']['bp_left']:,}, {cfg['inv']['bp_right']:,}) "
        f"age={cfg['inv']['t_inv']:,} gen"
    )
    print(
        f"  sweep at bp={cfg['sweep']['x_sel']:,} "
        f"t={cfg['sweep']['t_event']} gen  s={cfg['sweep']['selection_coefficient']}"
    )
    print(f"  pops: {cfg['demo']['pop_sizes']}  m={cfg['demo']['migration_rate']}")
    print(f"  {note}")
    print(f"  ABC priors: {abc_priors()}")

    if args.dry_run:
        return

    if args.scale == "real":
        print("\n[WARN] scale=real will hang under current event loop.")
        print("       Wait for Path 2 (compound per-pair rate rewrite).")
        print("       Use --dry-run to skip the actual simulation.")
        sys.exit(2)

    sim = build_sim(cfg, args.seed)
    t0 = time.time()
    ts = sim.simulate()
    print(
        f"\nsim done: nodes={ts.num_nodes} edges={ts.num_edges} "
        f"trees={ts.num_trees} dt={time.time() - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
