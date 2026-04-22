#!/usr/bin/env python3
"""Single-rep Kir/Fol-scale benchmark: event-loop vs analytic-middle.

Target biology (Small et al. 2023; 3Ra only to fit analytic-middle gate):
  Ne_anc = 1e6, Ne_K_current = 44_000, Ne_F_current = 92_000,
  Ne_F_postsplit = 3_000_000
  t_split = 14_000, t_inv = 330_000
  3Ra: bp_left=180_000, bp_right=380_000 in 1 Mb
  p_inv = {0: 0.0, 1: 0.73}
  Samples: 74 K (S), 23 F_S, 23 F_I
  r = 1e-8, gamma = 1e-9

Analytic-middle gate limitations for this case:
  - Single inversion only: the full Kir/Fol has 3Ra + 3Rb; we run 3Ra only
  - Gate rejects when a demo event sits strictly between t_now and t_inv.
    The Kir/Fol model puts an `en` at T_NE_ANC_STEP=100_000 before
    t_inv=330_000, which would cause the analytic-middle loop to not
    engage at all. To exercise the fast path we OMIT the 100k step here;
    this leaves the window between t_split (14k) and t_inv (330k) as a
    clean single-pop constant-Ne phase.

Time-box each side to 600 s; print wall time, num_nodes, num_trees.
"""
from __future__ import annotations

import argparse
import signal
import time

from msinv import HullSimulator, InversionSpec, Demography


NE_K_CURRENT = 44_000
NE_F_CURRENT = 92_000
NE_F_POSTSPLIT = 3_000_000
NE_ANC_1 = 1_000_000
T_SPLIT = 14_000
T_INV = 330_000
P_INV_3RA_F = 0.73
R = 1.0e-8
GAMMA = 1.0e-9
FLUX_WINDOW = 0.05

N_K = 74
N_F_S = 23
N_F_I = 23


def build_sim(seed, analytic_middle):
    L = 1_000_000
    demo = Demography(pop_sizes=[NE_K_CURRENT, NE_F_CURRENT])
    demo.add_event(('en', T_SPLIT, 1, NE_F_POSTSPLIT))  # F resized at t_split
    demo.add_event(('ej', T_SPLIT, 1, 0))               # merge F into K
    demo.add_event(('en', T_SPLIT, 0, NE_ANC_1))        # anc Ne constant to t_inv
    demo.add_inversion_freq_change(T_SPLIT, 0, inv_id=0, p_inv=0.30)

    inv_3ra = InversionSpec(
        bp_left=180_000, bp_right=380_000,
        p_inv={0: 0.0, 1: P_INV_3RA_F},
        t_inv=T_INV, gene_conversion_rate=GAMMA, flux_window=FLUX_WINDOW,
    )

    return HullSimulator(
        sample_config={('S', 0): N_K, ('S', 1): N_F_S, ('I', 1): N_F_I},
        demography=demo,
        sequence_length=L,
        recombination_rate=R,
        inversions=[inv_3ra],
        sweeps=[],
        seed=seed,
        analytic_middle=analytic_middle,
    )


class Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise Timeout()


def run_one(analytic_middle, seed=4242, timeout_s=600):
    sim = build_sim(seed, analytic_middle)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_s))
    t0 = time.time()
    try:
        ts = sim.simulate()
        dt = time.time() - t0
        signal.alarm(0)
        return dict(
            ok=True,
            wall_s=dt,
            num_nodes=int(ts.num_nodes),
            num_trees=int(ts.num_trees),
            num_edges=int(ts.num_edges),
        )
    except Timeout:
        dt = time.time() - t0
        signal.alarm(0)
        return dict(ok=False, wall_s=dt, reason='timeout')
    except Exception as exc:
        dt = time.time() - t0
        signal.alarm(0)
        return dict(ok=False, wall_s=dt, reason=f'{type(exc).__name__}: {exc}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=4242)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--mode', choices=['both', 'loop', 'analytic'],
                   default='both')
    args = p.parse_args()

    print(f"Kir/Fol single-rep bench (3Ra only), seed={args.seed}, "
          f"timeout={args.timeout}s per side")
    print(f"  samples: K={N_K}  F_S={N_F_S}  F_I={N_F_I}")
    print(f"  t_split={T_SPLIT}  t_inv={T_INV}  "
          f"Ne_anc={NE_ANC_1}  Ne_F_postsplit={NE_F_POSTSPLIT}")

    if args.mode in ('both', 'loop'):
        print("\n-- event-loop --")
        r1 = run_one(analytic_middle=False, seed=args.seed,
                     timeout_s=args.timeout)
        print(f"  {r1}")

    if args.mode in ('both', 'analytic'):
        print("\n-- analytic-middle --")
        r2 = run_one(analytic_middle=True, seed=args.seed,
                     timeout_s=args.timeout)
        print(f"  {r2}")


if __name__ == '__main__':
    main()
