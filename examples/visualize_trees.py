#!/usr/bin/env python3
"""
Visualize tree topologies from msinv at multiple positions across an
inversion, for a simple 1-population constant-size scenario.

Diagnostic to verify that not just the diversity statistics but the
underlying tree topologies match the structured-coalescent expectation:
- Inside the inversion: lineages should cluster strictly by karyotype
  (S samples merge with each other, I with each other; S and I only
  merge above t_inv).
- Outside the inversion: panmictic — no karyotype structure.
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import msinv
from msinv.simulator import build_structured_tree, ConstantFrequency


# ---- Parameters: 1 population, constant Ne, single inversion ----
Ne = 10_000
mu = 1e-8
r = 1e-8
L_bp = 100_000
theta = 4 * Ne * mu * L_bp     # ≈ 40
rho = 4 * Ne * r * L_bp        # ≈ 40
nsites = 1000
bp_left = 0.30                 # fractional positions
bp_right = 0.70
p_inv = 0.5
t_inv_gen = 80_000             # 4 Ne gen — old enough for clean structure
t_inv = t_inv_gen / (2 * Ne)   # 4.0 coal units

n_std = 5  # 5 S samples (homokaryotypes for standard arrangement)
n_inv = 5  # 5 I samples

SEED = 42


# ---- Helper: layout & draw a tree ----
def assign_layout(root, leaf_x, x_map):
    """Assign x-coordinates to all nodes by recursive postorder.
    x_map: dict id(node) -> x. Node uses __slots__ so we can't attach.
    """
    if not root.children:
        x = leaf_x[root.sample_id]
        x_map[id(root)] = x
        return x
    xs = [assign_layout(c, leaf_x, x_map) for c in root.children]
    x = sum(xs) / len(xs)
    x_map[id(root)] = x
    return x


def collect_nodes(root):
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def draw_tree(ax, root, leaf_order, t_inv_v=None, title=''):
    """Draw the tree on the given axes.

    Branch color = class:
      S = #1976D2 (blue)
      I = #C2185B (red)
      mixed/post-t_inv = #555555 (grey)

    Sample labels colored by class.
    """
    # leaf_order: list of sample_ids defining x-position order
    leaf_x = {sid: i for i, sid in enumerate(leaf_order)}
    x_map = {}
    assign_layout(root, leaf_x, x_map)
    nodes = collect_nodes(root)

    color = {'S': '#1976D2', 'I': '#C2185B', 'mixed': '#555555'}

    # Draw branches: from each non-root node up to its parent
    for n in nodes:
        if n.parent is not None:
            x_n, y_n = x_map[id(n)], n.time
            x_p, y_p = x_map[id(n.parent)], n.parent.time
            cls = n.branch_class
            c = color.get(cls, '#555555')
            # Vertical segment from n to parent's height, then horizontal
            ax.plot([x_n, x_n], [y_n, y_p], color=c, lw=1.6, zorder=2)
            ax.plot([x_n, x_p], [y_p, y_p], color=c, lw=1.6, zorder=2)

    # Mark t_inv
    if t_inv_v is not None:
        ax.axhline(t_inv_v, color='#FF9800', ls='--', lw=1, alpha=0.7,
                   zorder=1)
        ax.text(len(leaf_order) - 0.5, t_inv_v, '$t_{inv}$',
                color='#E65100', fontsize=8, va='bottom', ha='right')

    # Sample labels
    for n in nodes:
        if not n.children:
            sid = n.sample_id
            cls = n.branch_class
            label = ('S' if sid < n_std else 'I') + str(sid)
            ax.text(x_map[id(n)], -0.05, label, ha='center', va='top',
                    fontsize=9, color=color[cls], fontweight='bold')

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(leaf_order) - 0.5)
    # Cap y-axis at twice t_inv for readability if root.time isn't crazy
    ymax = max(n.time for n in nodes)
    ax.set_ylim(-0.4, max(ymax * 1.1, t_inv_v * 1.3 if t_inv_v else 1))
    ax.set_ylabel('time (coal units)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---- Build trees at multiple positions ----
positions = [
    ('outside left', 0.10),
    ('inside (left bp)', 0.32),
    ('inside (center)', 0.50),
    ('inside (right bp)', 0.68),
    ('outside right', 0.85),
]

print(f"Generating trees at {len(positions)} positions...")
print(f"Inversion: bp_left={bp_left}, bp_right={bp_right}, "
      f"t_inv={t_inv:.2f} coal units")
print(f"Samples: {n_std} S + {n_inv} I = {n_std + n_inv} total\n")

# Sample config: S samples first, then I samples
sc = {('S', 0): n_std, ('I', 0): n_inv}

trees = {}
for name, x in positions:
    rng = np.random.default_rng(SEED)
    # Inside inv -> structured; outside -> panmictic-style fresh build
    if bp_left <= x < bp_right:
        # Structured at this position: phi(x) is the in-inv flux probability
        from msinv.simulator import GeneFluxModel
        fm = GeneFluxModel(w=0.3)
        rel = (x - bp_left) / (bp_right - bp_left)
        rel = max(0.02, min(0.98, rel))
        phi_x = fm.phi(rel)
        root, leaves = build_structured_tree(
            n_std, n_inv, p_inv, 0.0, 2.0, phi_x, rng,
            p_inv_func=ConstantFrequency(p_inv, t_inv=t_inv),
            sample_config=sc, n_pops=1, mig_rate=0.0)
    else:
        # Outside: panmictic (effective p_inv=0)
        root, leaves = build_structured_tree(
            n_std + n_inv, 0, 0.0, 0.0, 2.0, 0.0, rng,
            p_inv_func=ConstantFrequency(0.0, t_inv=0.0),
            sample_config=sc, n_pops=1, mig_rate=0.0)
    trees[name] = (root, leaves)
    # Quick T_MRCA report
    sample_leaves = sorted([n for n in leaves], key=lambda n: n.sample_id)

    def mrca_time(a_idx, b_idx):
        anc = set(); n = sample_leaves[a_idx]
        while n: anc.add(id(n)); n = n.parent
        n = sample_leaves[b_idx]
        while n:
            if id(n) in anc:
                return n.time
            n = n.parent
        return float('nan')
    SS = mrca_time(0, 1)               # S0, S1 (within S)
    II = mrca_time(n_std, n_std + 1)   # I0, I1 (within I)
    SI = mrca_time(0, n_std)           # S0, I0 (cross)
    print(f"  {name:<22} x={x:.2f}: T_MRCA SS={SS:.2f}  II={II:.2f}  SI={SI:.2f}")


# ---- Plot ----
fig = plt.figure(figsize=(20, 5.5))
gs = GridSpec(1, len(positions), wspace=0.35)

# Decide a stable leaf order: S samples on left, I samples on right
leaf_order = list(range(n_std)) + list(range(n_std, n_std + n_inv))

for i, (name, x) in enumerate(positions):
    ax = fig.add_subplot(gs[i])
    root, _ = trees[name]
    in_inv = bp_left <= x < bp_right
    region = 'INSIDE inversion' if in_inv else 'outside (collinear)'
    title = f'{name}\nx={x:.2f}  ({region})'
    draw_tree(ax, root, leaf_order, t_inv_v=t_inv, title=title)

# Legend
legend_elems = [
    Line2D([0], [0], color='#1976D2', lw=2, label='S (standard)'),
    Line2D([0], [0], color='#C2185B', lw=2, label='I (inverted)'),
    Line2D([0], [0], color='#FF9800', lw=1, ls='--',
           label=f'$t_{{inv}}$ = {t_inv:.2f}'),
]
fig.legend(handles=legend_elems, loc='upper right',
           bbox_to_anchor=(0.99, 0.97), fontsize=9, framealpha=0.9)

fig.suptitle(
    f'Tree topology across an inversion '
    f'(1 pop, Ne={Ne:,}, p_inv={p_inv}, t_inv={t_inv:.1f} coal units)\n'
    f'Samples cluster strictly by karyotype INSIDE the inversion '
    f'(merge only above $t_{{inv}}$); '
    f'panmictic OUTSIDE (no class structure)',
    fontsize=11, fontweight='bold', y=1.02)

fig.savefig('figures/tree_topology_diagnostic.pdf',
            bbox_inches='tight', dpi=150)
print(f"\nFigure saved: figures/tree_topology_diagnostic.pdf")
