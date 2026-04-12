/*
 * libmsinv.c — SMC inner loop + demography + inversion model for msinv.
 *
 * Extends smc_full.c with:
 *   - Trajectory interpolation and gene flux model
 *   - Multi-population demography (size changes, growth, migration, splits)
 *   - InversionSpec / SimParams master structs
 *
 * Compile:
 *   gcc -O3 -shared -fPIC -o libmsinv.so libmsinv.c -lm
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define NULL_NODE -1
#define CLASS_S 0
#define CLASS_I 1
#define MAX_NODES 16384
#define MAX_MUTATIONS 100000
#define MAX_PENDING 256

/* ================================================================
 * RNG (xorshift128+)
 * ================================================================ */

static uint64_t rng_s[2] = {12345678901234ULL, 98765432109876ULL};

void smc_full_seed(uint64_t s0, uint64_t s1) {
    rng_s[0] = s0 ? s0 : 1;
    rng_s[1] = s1 ? s1 : 1;
}

static inline double rng_uniform(void) {
    uint64_t s0 = rng_s[0];
    uint64_t s1 = rng_s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    rng_s[1] = (s1 << 36) | (s1 >> 28);
    return (double)(result >> 11) / (double)(1ULL << 53);
}

static inline double rng_exponential(double rate) {
    if (rate <= 0) return 1e30;
    double u = rng_uniform();
    if (u <= 0) u = 1e-15;
    return -log(u) / rate;
}

static inline int rng_poisson(double lambda) {
    if (lambda <= 0) return 0;
    if (lambda > 500) {
        /* Normal approximation for large lambda */
        double u1 = rng_uniform();
        double u2 = rng_uniform();
        double z = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
        int k = (int)(lambda + sqrt(lambda) * z + 0.5);
        return k > 0 ? k : 0;
    }
    double L = exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
        k++;
        p *= rng_uniform();
    } while (p > L);
    return k - 1;
}

/* ================================================================
 * Tree data (flat arrays, allocated by caller)
 * ================================================================ */

typedef struct {
    double time[MAX_NODES];
    int parent[MAX_NODES];
    int left_child[MAX_NODES];
    int right_sib[MAX_NODES];
    int8_t klass[MAX_NODES];
    int8_t population[MAX_NODES];
    int sample_id[MAX_NODES];
    int8_t active[MAX_NODES];  /* 1 if node is in the tree, 0 if orphaned */
    int node_id[MAX_NODES];    /* monotonically increasing node ID */
    int n;       /* number of nodes allocated */
    int root;
    int nsam;    /* number of sample leaves */
    int next_node_id;  /* counter for assigning unique node IDs */
    /* Free list for node recycling */
    int free_list[MAX_NODES];
    int free_count;
} Tree;

/* ================================================================
 * Tree operations
 * ================================================================ */

static void tree_init(Tree *t) {
    t->n = 0;
    t->root = NULL_NODE;
    t->nsam = 0;
    t->next_node_id = 0;
    t->free_count = 0;
    memset(t->parent, 0xFF, sizeof(t->parent));
    memset(t->left_child, 0xFF, sizeof(t->left_child));
    memset(t->right_sib, 0xFF, sizeof(t->right_sib));
    memset(t->active, 0, sizeof(t->active));
    memset(t->node_id, 0xFF, sizeof(t->node_id));
}

static int tree_add_node(Tree *t, double time, int8_t klass, int8_t pop, int sid) {
    int i;
    if (t->free_count > 0) {
        i = t->free_list[--t->free_count];
    } else {
        i = t->n++;
    }
    t->time[i] = time;
    t->parent[i] = NULL_NODE;
    t->left_child[i] = NULL_NODE;
    t->right_sib[i] = NULL_NODE;
    t->klass[i] = klass;
    t->population[i] = pop;
    t->sample_id[i] = sid;
    t->active[i] = 1;
    t->node_id[i] = t->next_node_id++;
    return i;
}

static void tree_free_node(Tree *t, int i) {
    t->active[i] = 0;
    t->parent[i] = NULL_NODE;
    t->left_child[i] = NULL_NODE;
    t->right_sib[i] = NULL_NODE;
    if (t->free_count < MAX_NODES)
        t->free_list[t->free_count++] = i;
}

static void tree_add_child(Tree *t, int p, int c) {
    t->parent[c] = p;
    t->right_sib[c] = t->left_child[p];
    t->left_child[p] = c;
}

static void tree_remove_child(Tree *t, int p, int c) {
    int prev = NULL_NODE;
    int cur = t->left_child[p];
    while (cur != NULL_NODE) {
        if (cur == c) {
            if (prev == NULL_NODE)
                t->left_child[p] = t->right_sib[cur];
            else
                t->right_sib[prev] = t->right_sib[cur];
            t->parent[c] = NULL_NODE;
            t->right_sib[c] = NULL_NODE;
            return;
        }
        prev = cur;
        cur = t->right_sib[cur];
    }
}

static int tree_get_sibling(Tree *t, int node) {
    int p = t->parent[node];
    if (p == NULL_NODE) return NULL_NODE;
    int c = t->left_child[p];
    while (c != NULL_NODE) {
        if (c != node) return c;
        c = t->right_sib[c];
    }
    return NULL_NODE;
}

static int tree_num_children(Tree *t, int node) {
    int count = 0;
    int c = t->left_child[node];
    while (c != NULL_NODE) { count++; c = t->right_sib[c]; }
    return count;
}

/* ================================================================
 * Branch length computation
 * ================================================================ */

static void tree_branch_lengths(Tree *t, double *L_S, double *L_I, double *t_max) {
    double ls = 0, li = 0, tm = 0;
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        if (t->time[i] > tm) tm = t->time[i];
        int p = t->parent[i];
        if (p >= 0) {
            double bl = t->time[p] - t->time[i];
            if (bl > 0) {
                if (t->klass[i] == CLASS_S) ls += bl;
                else li += bl;
            }
        }
    }
    *L_S = ls; *L_I = li; *t_max = tm;
}

/* ================================================================
 * Build panmictic coalescent tree
 * ================================================================ */

static void build_panmictic(Tree *t, int n_std, int n_inv) {
    tree_init(t);
    int nsam = n_std + n_inv;
    t->nsam = nsam;

    /* Add sample leaves */
    int active[MAX_NODES];
    for (int i = 0; i < n_std; i++)
        active[i] = tree_add_node(t, 0.0, CLASS_S, 0, i);
    for (int i = 0; i < n_inv; i++)
        active[n_std + i] = tree_add_node(t, 0.0, CLASS_I, 0, n_std + i);

    int k = nsam;
    double tc = 0.0;
    while (k > 1) {
        double rate = (double)(k * (k - 1)) / 2.0;
        tc += rng_exponential(rate);

        /* Pick two random lineages */
        int i1 = (int)(rng_uniform() * k);
        int i2;
        do { i2 = (int)(rng_uniform() * k); } while (i2 == i1);
        if (i1 >= k) i1 = k - 1;
        if (i2 >= k) i2 = k - 1;

        int coal = tree_add_node(t, tc, CLASS_S, 0, NULL_NODE);
        tree_add_child(t, coal, active[i1]);
        tree_add_child(t, coal, active[i2]);

        /* Remove i1, i2 from active, add coal */
        if (i1 > i2) { int tmp = i1; i1 = i2; i2 = tmp; }
        active[i2] = active[k - 1];
        active[i1] = active[k - 2];
        active[k - 2] = coal;
        k--;
    }
    t->root = active[0];
}

/* ================================================================
 * Panmictic prune-and-reattach (coalescent-based)
 * ================================================================ */

static void smc_prune_reattach(Tree *t) {
    /* Get branches */
    int br_idx[MAX_NODES];
    double br_len[MAX_NODES];
    int br_count = 0;
    double total_L = 0;

    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int p = t->parent[i];
        if (p >= 0) {
            double bl = t->time[p] - t->time[i];
            if (bl > 0) {
                br_idx[br_count] = i;
                br_len[br_count] = bl;
                total_L += bl;
                br_count++;
            }
        }
    }
    if (br_count == 0 || total_L <= 0) return;

    /* Choose branch weighted by length */
    double r = rng_uniform() * total_L;
    double cum = 0;
    int bi = br_count - 1;
    for (int i = 0; i < br_count; i++) {
        cum += br_len[i];
        if (r < cum) { bi = i; break; }
    }
    int target = br_idx[bi];
    double t_cut = t->time[target] + rng_uniform() * br_len[bi];

    /* Prune */
    int p = t->parent[target];
    if (p < 0 || tree_num_children(t, p) != 2) return;
    int sib = tree_get_sibling(t, target);
    if (sib < 0) return;

    int gp = t->parent[p];
    tree_remove_child(t, p, target);
    tree_remove_child(t, p, sib);
    if (gp >= 0) {
        tree_remove_child(t, gp, p);
        tree_add_child(t, gp, sib);
    } else {
        t->parent[sib] = NULL_NODE;
    }
    if (t->root == p) t->root = sib;
    t->parent[target] = NULL_NODE;
    tree_free_node(t, p);  /* recycle the pruned coalescence node */

    /* Coalescent-based reattach */
    double t_now = t_cut;

    /* Collect sorted unique times above t_cut */
    double times_above[MAX_NODES];
    int n_times = 0;
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        if (t->time[i] > t_now && i != target && i != p) {
            /* Simple insertion sort */
            int j = n_times;
            while (j > 0 && times_above[j-1] > t->time[i]) {
                times_above[j] = times_above[j-1]; j--;
            }
            times_above[j] = t->time[i];
            n_times++;
        }
    }
    /* Remove duplicates */
    if (n_times > 1) {
        int w = 1;
        for (int i = 1; i < n_times; i++)
            if (times_above[i] != times_above[w-1])
                times_above[w++] = times_above[i];
        n_times = w;
    }

    int reattached = 0;
    for (int ti = 0; ti < n_times && !reattached; ti++) {
        double t_next = times_above[ti];
        /* Count branches alive at t_now */
        int k = 0;
        int candidates[MAX_NODES];
        for (int i = 0; i < t->n; i++) {
            if (!t->active[i]) continue;
            if (i == target || i == p) continue;
            int pi = t->parent[i];
            if (pi >= 0 && t->time[i] <= t_now && t->time[pi] > t_now) {
                candidates[k++] = i;
            }
        }
        if (k <= 0) { t_now = t_next; continue; }

        double dt = rng_exponential((double)k);
        if (t_now + dt < t_next) {
            double t_a = t_now + dt;
            int ci = (int)(rng_uniform() * k);
            if (ci >= k) ci = k - 1;
            int attach = candidates[ci];
            int ap = t->parent[attach];

            int coal = tree_add_node(t, t_a, t->klass[attach], t->population[attach], NULL_NODE);
            if (ap >= 0) {
                tree_remove_child(t, ap, attach);
                tree_add_child(t, ap, coal);
            }
            tree_add_child(t, coal, attach);
            tree_add_child(t, coal, target);
            if (ap < 0) t->root = coal;
            reattached = 1;
        } else {
            t_now = t_next;
        }
    }

    if (!reattached) {
        /* Above root */
        double dt = rng_exponential(1.0);
        double t_c = (t_now > t->time[t->root]) ? t_now : t->time[t->root];
        t_c += dt;
        int coal = tree_add_node(t, t_c, CLASS_S, 0, NULL_NODE);
        tree_add_child(t, coal, t->root);
        tree_add_child(t, coal, target);
        t->root = coal;
    }

    /* Update root */
    int node = 0;
    while (t->parent[node] != NULL_NODE) node = t->parent[node];
    t->root = node;
}

/* ================================================================
 * Drop mutations on current tree for a segment
 * ================================================================ */

typedef struct {
    double position;
    int leaf_bits[MAX_NODES / 32 + 1]; /* bitmap of sample IDs */
} Mutation;

static int drop_mutations_segment(Tree *t, double left, double right,
                                    double theta, Mutation *muts, int mut_count,
                                    int max_muts) {
    double seg_len = right - left;
    if (seg_len <= 0) return mut_count;

    /* Get branches and total length */
    int br_idx[MAX_NODES];
    double br_len[MAX_NODES];
    int br_count = 0;
    double total_L = 0;

    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int p = t->parent[i];
        if (p >= 0) {
            double bl = t->time[p] - t->time[i];
            if (bl > 0) {
                br_idx[br_count] = i;
                br_len[br_count] = bl;
                total_L += bl;
                br_count++;
            }
        }
    }
    if (total_L <= 0 || br_count == 0) return mut_count;

    int n_muts = rng_poisson((theta / 2.0) * total_L * seg_len);
    if (n_muts <= 0) return mut_count;
    if (mut_count + n_muts > max_muts) n_muts = max_muts - mut_count;

    /* Cumulative branch lengths for selection */
    double cum_bl[MAX_NODES];
    cum_bl[0] = br_len[0];
    for (int i = 1; i < br_count; i++)
        cum_bl[i] = cum_bl[i-1] + br_len[i];

    for (int m = 0; m < n_muts; m++) {
        double pos = left + rng_uniform() * seg_len;

        /* Choose branch */
        double r = rng_uniform() * total_L;
        int bi = br_count - 1;
        for (int i = 0; i < br_count; i++) {
            if (r < cum_bl[i]) { bi = i; break; }
        }
        int node = br_idx[bi];

        /* Find leaves below this node */
        Mutation *mut = &muts[mut_count];
        mut->position = pos;
        memset(mut->leaf_bits, 0, sizeof(mut->leaf_bits));

        /* DFS to find leaves */
        int stack[MAX_NODES];
        int sp = 0;
        stack[sp++] = node;
        while (sp > 0) {
            int cur = stack[--sp];
            if (t->sample_id[cur] >= 0) {
                int sid = t->sample_id[cur];
                mut->leaf_bits[sid / 32] |= (1 << (sid % 32));
            }
            int c = t->left_child[cur];
            while (c != NULL_NODE) {
                stack[sp++] = c;
                c = t->right_sib[c];
            }
        }
        mut_count++;
    }
    return mut_count;
}

/* ================================================================
 * Main simulation function (exported)
 * ================================================================ */

int smc_full_simulate(
    int n_std, int n_inv, double theta, double rho, int nsites,
    double p_inv, double c_flux, double t_inv,
    double bp_left, double bp_right, double flux_w,
    /* Output: haplotype matrix (nsam * max_sites), positions */
    int8_t *out_haps, double *out_positions, int max_sites)
{
    int nsam = n_std + n_inv;
    double inv_len = bp_right - bp_left;

    Tree tree;
    build_panmictic(&tree, n_std, n_inv);

    Mutation *mutations = (Mutation *)calloc(max_sites, sizeof(Mutation));
    if (!mutations) return -1;
    int mut_count = 0;

    double pos = 0.0;

    for (int iter = 0; iter < 500000 && pos < 1.0; iter++) {
        int in_inv = (bp_left <= pos && pos < bp_right);

        /* Branch lengths */
        double L_S, L_I, t_max;
        tree_branch_lengths(&tree, &L_S, &L_I, &t_max);
        double L_total = L_S + L_I;

        double weighted_L, next_boundary;
        if (in_inv && p_inv > 0) {
            double p_std = 1.0 - p_inv;
            weighted_L = L_S * p_std + L_I * p_inv;
            next_boundary = bp_right;
        } else {
            weighted_L = L_total;
            next_boundary = (pos < bp_left) ? bp_left : 1.0;
        }

        if (weighted_L <= 0) {
            mut_count = drop_mutations_segment(&tree, pos, next_boundary,
                                                theta, mutations, mut_count, max_sites);
            pos = next_boundary;
            continue;
        }

        double rate = (rho / 2.0) * weighted_L;
        double dx = rng_exponential(rate);
        double extent = dx;
        if (extent > next_boundary - pos) extent = next_boundary - pos;
        if (extent > 1.0 - pos) extent = 1.0 - pos;
        if (extent <= 0) extent = 1e-10;

        double new_pos = pos + extent;

        /* Drop mutations */
        mut_count = drop_mutations_segment(&tree, pos, new_pos,
                                            theta, mutations, mut_count, max_sites);

        if (dx < (next_boundary - pos) && dx < (1.0 - pos)) {
            new_pos = pos + dx;
            /* Recombination: panmictic prune-and-reattach */
            /* (simplified: uses panmictic for all regions) */
            smc_prune_reattach(&tree);
        }
        /* Boundary handling: just continue (tree persists) */
        /* TODO: structured reattach, boundary rebuilds */

        pos = new_pos;
    }

    /* Build output: sort mutations, fill haplotype matrix */
    /* Simple bubble sort (ok for small mutation counts) */
    for (int i = 0; i < mut_count - 1; i++)
        for (int j = 0; j < mut_count - 1 - i; j++)
            if (mutations[j].position > mutations[j+1].position) {
                Mutation tmp = mutations[j];
                mutations[j] = mutations[j+1];
                mutations[j+1] = tmp;
            }

    int n_out = (mut_count < max_sites) ? mut_count : max_sites;
    memset(out_haps, 0, nsam * max_sites);

    for (int j = 0; j < n_out; j++) {
        out_positions[j] = mutations[j].position;
        for (int s = 0; s < nsam; s++) {
            if (mutations[j].leaf_bits[s / 32] & (1 << (s % 32)))
                out_haps[s * max_sites + j] = 1;
        }
    }

    free(mutations);
    return n_out;
}

/* ================================================================
 * Normal RNG (Box-Muller transform)
 * ================================================================ */

static double rng_normal(double mu, double sigma) {
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-30) u1 = 1e-30;
    double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    return mu + sigma * z;
}

/* ================================================================
 * tree_find_root — walk up from any active leaf to find the root
 * ================================================================ */

static int tree_find_root(Tree *t) {
    /* Find any active leaf and walk up */
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int node = i;
        while (t->parent[node] != NULL_NODE)
            node = t->parent[node];
        return node;
    }
    return NULL_NODE;
}

/* ================================================================
 * Trajectory interpolation
 * ================================================================ */

#define MAX_TRAJ_STEPS 100000

typedef struct {
    double *times;      /* pre-computed time array (length n_steps) */
    double *freqs;      /* pre-computed freq array (n_pops * n_steps for multi-pop) */
    int n_steps;
    int n_pops;         /* 1 for single-pop trajectories */
    double t_inv;       /* inversion age */
} Trajectory;

/* Linear interpolation of trajectory frequency at time t for a given pop.
 * times[] is assumed sorted ascending.  Returns 0 for t >= t_inv. */
static double traj_interp(Trajectory *traj, double t, int pop) {
    if (!traj || !traj->times || !traj->freqs || traj->n_steps <= 0)
        return 0.0;
    if (t >= traj->t_inv) return 0.0;
    if (t <= traj->times[0])
        return traj->freqs[pop * traj->n_steps + 0];

    int lo = 0, hi = traj->n_steps - 1;
    /* Binary search for the interval containing t */
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (traj->times[mid] <= t) lo = mid;
        else hi = mid;
    }
    double t0 = traj->times[lo];
    double t1 = traj->times[hi];
    double f0 = traj->freqs[pop * traj->n_steps + lo];
    double f1 = traj->freqs[pop * traj->n_steps + hi];
    if (t1 <= t0) return f0;
    double frac = (t - t0) / (t1 - t0);
    return f0 + frac * (f1 - f0);
}

/* Global inversion frequency = max across all pops */
static double traj_p_inv_global(Trajectory *traj, double t) {
    if (!traj || traj->n_steps <= 0) return 0.0;
    double mx = 0.0;
    for (int p = 0; p < traj->n_pops; p++) {
        double f = traj_interp(traj, t, p);
        if (f > mx) mx = f;
    }
    return mx;
}

/* ================================================================
 * tree_weighted_branch_length — per-branch weighted_L using trajectory
 * ================================================================ */

typedef double (*traj_fn_t)(Trajectory *traj, double t, int pop);

static double tree_weighted_branch_length(Tree *t, Trajectory *traj,
                                          traj_fn_t interp_fn) {
    double wL = 0.0;
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int p = t->parent[i];
        if (p < 0) continue;
        double bl = t->time[p] - t->time[i];
        if (bl <= 0) continue;

        /* Midpoint approximation: evaluate trajectory at branch midpoint */
        double t_mid = t->time[i] + bl * 0.5;
        int pop = (int)t->population[i];
        double p_inv = interp_fn(traj, t_mid, pop);

        double weight;
        if (t->klass[i] == CLASS_I) {
            weight = p_inv;
        } else {
            weight = 1.0 - p_inv;
        }
        if (weight < 0) weight = 0;
        wL += bl * weight;
    }
    return wL;
}

/* ================================================================
 * Gene flux model
 * ================================================================ */

/* phi_at: gene flux rate modifier.  phi(x, w) = min(x, 1-x, w) / (1-w)
 * x = position within inversion (0..1 normalized), w = flux window width */
static double phi_at(double x, double w) {
    if (w <= 0.0 || w >= 1.0) return 0.0;
    double d = x;
    if (1.0 - x < d) d = 1.0 - x;
    if (w < d) d = w;
    return d / (1.0 - w);
}

/* draw_b2: Peischl b2 boundary for gene flux tract.
 * Returns uniformly-drawn b2 in [x, x + (1-w)] clamped to [0,1]. */
static double draw_b2(double x, double w) {
    double lo = x;
    double hi = x + (1.0 - w);
    if (hi > 1.0) hi = 1.0;
    if (lo < 0.0) lo = 0.0;
    return lo + rng_uniform() * (hi - lo);
}

/* ================================================================
 * InversionSpec
 * ================================================================ */

#define MAX_INVERSIONS 4

typedef struct {
    double bp_left, bp_right;   /* breakpoints in [0,1] */
    double gamma;               /* absolute gene flux rate */
    double flux_w;              /* flux window width (fraction of inv length) */
    Trajectory traj;            /* pre-computed frequency trajectory */
} InversionSpec;

/* ================================================================
 * Demography system
 * ================================================================ */

#define MAX_POPS 8
#define MAX_DEMO_EVENTS 64

typedef struct {
    int n_pops;
    double pop_sizes[MAX_POPS];
    double growth_rates[MAX_POPS];
    double growth_start[MAX_POPS];
    double mig_matrix[MAX_POPS][MAX_POPS];
} DemoState;

typedef struct {
    char type;          /* 'N' = set size, 'n' = set one pop size,
                           'G' = set all growth, 'g' = set pop growth,
                           'M' = set sym mig, 'm' = set pairwise mig,
                           'j' = mass migration (join), 's' = pop split */
    double time;
    int pop_i, pop_j;
    double value;
} DemoEvent;

typedef struct {
    DemoState state;
    DemoEvent events[MAX_DEMO_EVENTS];
    int n_events;
    int next_event;     /* index of next unconsumed event */
} Demography;

/* Initialize demography: n_pops populations, all N=1, uniform migration */
static void demo_init(Demography *d, int n_pops, double mig_rate) {
    memset(d, 0, sizeof(Demography));
    d->state.n_pops = n_pops;
    for (int i = 0; i < n_pops; i++) {
        d->state.pop_sizes[i] = 1.0;
        d->state.growth_rates[i] = 0.0;
        d->state.growth_start[i] = 0.0;
    }
    for (int i = 0; i < n_pops; i++)
        for (int j = 0; j < n_pops; j++)
            d->state.mig_matrix[i][j] = (i == j) ? 0.0 : mig_rate;
    d->n_events = 0;
    d->next_event = 0;
}

/* Deep copy demography (no heap allocs in DemoState so memcpy suffices) */
static void demo_copy(Demography *dst, const Demography *src) {
    memcpy(dst, src, sizeof(Demography));
}

/* Effective pop size at time t, accounting for exponential growth.
 * N(t) = N_0 * exp(-alpha * (t - t_start))  [backwards in time, so
 * positive alpha = growth forward = shrinkage backward]. */
static double demo_pop_size_at(DemoState *s, int pop, double t) {
    double N0 = s->pop_sizes[pop];
    double alpha = s->growth_rates[pop];
    if (alpha == 0.0) return N0;
    double dt = t - s->growth_start[pop];
    if (dt < 0) dt = 0;
    return N0 * exp(-alpha * dt);
}

/* Coalescent rate factor = 1 / N(t) for the given population */
static double demo_coal_rate_factor(Demography *d, int pop, double t) {
    double N = demo_pop_size_at(&d->state, pop, t);
    if (N <= 0) N = 1e-30;
    return 1.0 / N;
}

/* Time of next unconsumed event, or 1e30 if none remain */
static double demo_next_event_time(Demography *d) {
    if (d->next_event >= d->n_events) return 1e30;
    return d->events[d->next_event].time;
}

/* Apply all events with time <= t, modifying DemoState in place.
 * Events must be pre-sorted by time (ascending). */
static void demo_apply_events(Demography *d, double t) {
    while (d->next_event < d->n_events &&
           d->events[d->next_event].time <= t) {
        DemoEvent *e = &d->events[d->next_event];
        DemoState *s = &d->state;
        switch (e->type) {
        case 'N':  /* Set all pop sizes */
            for (int i = 0; i < s->n_pops; i++) {
                s->pop_sizes[i] = e->value;
                s->growth_rates[i] = 0.0;
            }
            break;
        case 'n':  /* Set one pop size */
            if (e->pop_i >= 0 && e->pop_i < s->n_pops) {
                s->pop_sizes[e->pop_i] = e->value;
                s->growth_rates[e->pop_i] = 0.0;
                s->growth_start[e->pop_i] = e->time;
            }
            break;
        case 'G':  /* Set growth rate for all pops */
            for (int i = 0; i < s->n_pops; i++) {
                /* Freeze current size before changing growth */
                s->pop_sizes[i] = demo_pop_size_at(s, i, e->time);
                s->growth_rates[i] = e->value;
                s->growth_start[i] = e->time;
            }
            break;
        case 'g':  /* Set growth rate for one pop */
            if (e->pop_i >= 0 && e->pop_i < s->n_pops) {
                s->pop_sizes[e->pop_i] = demo_pop_size_at(s, e->pop_i, e->time);
                s->growth_rates[e->pop_i] = e->value;
                s->growth_start[e->pop_i] = e->time;
            }
            break;
        case 'M':  /* Set symmetric migration rate (all pairs) */
            for (int i = 0; i < s->n_pops; i++)
                for (int j = 0; j < s->n_pops; j++)
                    s->mig_matrix[i][j] = (i == j) ? 0.0 : e->value;
            break;
        case 'm':  /* Set pairwise migration rate */
            if (e->pop_i >= 0 && e->pop_i < s->n_pops &&
                e->pop_j >= 0 && e->pop_j < s->n_pops)
                s->mig_matrix[e->pop_i][e->pop_j] = e->value;
            break;
        case 'j':  /* Mass migration: move all lineages from pop_i to pop_j */
            /* This is a join: pop_i merges into pop_j (backward in time) */
            if (e->pop_i >= 0 && e->pop_i < s->n_pops &&
                e->pop_j >= 0 && e->pop_j < s->n_pops) {
                /* Zero out migration to/from the source */
                for (int k = 0; k < s->n_pops; k++) {
                    s->mig_matrix[e->pop_i][k] = 0.0;
                    s->mig_matrix[k][e->pop_i] = 0.0;
                }
            }
            break;
        case 's':  /* Pop split: pop_i derived from pop_j at this time */
            if (e->pop_i >= 0 && e->pop_i < s->n_pops &&
                e->pop_j >= 0 && e->pop_j < s->n_pops) {
                for (int k = 0; k < s->n_pops; k++) {
                    s->mig_matrix[e->pop_i][k] = 0.0;
                    s->mig_matrix[k][e->pop_i] = 0.0;
                }
            }
            break;
        default:
            break;
        }
        d->next_event++;
    }
}

/* ================================================================
 * SimParams — master parameter struct
 * ================================================================ */

typedef struct {
    int nsam, n_std, n_inv;
    double theta, rho;
    int nsites;
    InversionSpec inversions[MAX_INVERSIONS];
    int n_inversions;
    Demography demo;
    int sample_counts[2][MAX_POPS];  /* [CLASS_S/CLASS_I][pop] */
    int use_sample_config;
    int record_edges;
    int use_4walk;
} SimParams;

/* ================================================================
 * Phase 3: Structured coalescent with karyotype classes
 * ================================================================ */

/* ----------------------------------------------------------------
 * ActiveList — tracks lineages during structured coalescent
 * ---------------------------------------------------------------- */

typedef struct {
    int node[MAX_NODES];
    int8_t klass[MAX_NODES];
    int8_t pop[MAX_NODES];
    int count;
} ActiveList;

static void active_init(ActiveList *a) {
    a->count = 0;
}

static void active_add(ActiveList *a, int node, int8_t cls, int8_t pop) {
    int i = a->count++;
    a->node[i] = node;
    a->klass[i] = cls;
    a->pop[i] = pop;
}

/* Count lineages matching given class and population */
static int active_count_class_pop(ActiveList *a, int8_t cls, int8_t pop) {
    int c = 0;
    for (int i = 0; i < a->count; i++)
        if (a->klass[i] == cls && a->pop[i] == pop) c++;
    return c;
}

/* Pick a random lineage matching class+pop; return its index in active list.
 * Returns -1 if none found. */
static int active_pick_random(ActiveList *a, int8_t cls, int8_t pop) {
    int indices[MAX_NODES];
    int n = 0;
    for (int i = 0; i < a->count; i++)
        if (a->klass[i] == cls && a->pop[i] == pop)
            indices[n++] = i;
    if (n == 0) return -1;
    return indices[(int)(rng_uniform() * n) % n];
}

/* Coalesce two random lineages of given class+pop.
 * Creates a new coalescent node in tree, updates active list. */
static void active_coalesce(ActiveList *a, Tree *t, int8_t cls, int8_t pop,
                            double time) {
    int indices[MAX_NODES];
    int n = 0;
    for (int i = 0; i < a->count; i++)
        if (a->klass[i] == cls && a->pop[i] == pop)
            indices[n++] = i;
    if (n < 2) return;

    /* Pick two distinct random indices */
    int p1 = (int)(rng_uniform() * n);
    int p2;
    do { p2 = (int)(rng_uniform() * n); } while (p2 == p1);
    if (p1 >= n) p1 = n - 1;
    if (p2 >= n) p2 = n - 1;
    int i1 = indices[p1], i2 = indices[p2];

    int n1 = a->node[i1], n2 = a->node[i2];
    int coal = tree_add_node(t, time, cls, pop, NULL_NODE);
    tree_add_child(t, coal, n1);
    tree_add_child(t, coal, n2);

    /* Remove i1, i2 (remove higher index first) */
    if (i1 > i2) { int tmp = i1; i1 = i2; i2 = tmp; }
    a->node[i2] = a->node[a->count - 1];
    a->klass[i2] = a->klass[a->count - 1];
    a->pop[i2] = a->pop[a->count - 1];
    a->count--;
    a->node[i1] = a->node[a->count - 1];
    a->klass[i1] = a->klass[a->count - 1];
    a->pop[i1] = a->pop[a->count - 1];
    a->count--;

    /* Add the coalescent node */
    active_add(a, coal, cls, pop);
}

/* Gene flux: pick a random lineage of class+pop, flip its class.
 * Creates a degree-2 flux node in the tree. */
static void active_flux(ActiveList *a, Tree *t, int8_t cls, int8_t pop,
                        double time) {
    int idx = active_pick_random(a, cls, pop);
    if (idx < 0) return;

    int8_t new_cls = (cls == CLASS_S) ? CLASS_I : CLASS_S;
    int old_node = a->node[idx];

    int flux_node = tree_add_node(t, time, new_cls, pop, NULL_NODE);
    tree_add_child(t, flux_node, old_node);

    a->node[idx] = flux_node;
    a->klass[idx] = new_cls;
}

/* Migration: pick a random lineage of class+pop, move to another pop.
 * No new tree node — just changes population label. */
static void active_migrate(ActiveList *a, Tree *t, int8_t cls, int8_t pop,
                           int n_pops) {
    int idx = active_pick_random(a, cls, pop);
    if (idx < 0) return;

    /* Choose destination: uniform among other pops */
    if (n_pops <= 1) return;
    int to_pop;
    do { to_pop = (int)(rng_uniform() * n_pops); } while (to_pop == pop);
    if (to_pop >= n_pops) to_pop = n_pops - 1;

    a->pop[idx] = (int8_t)to_pop;
    t->population[a->node[idx]] = (int8_t)to_pop;
}

/* Apply mass migration ('j') events: move all lineages from pop_i to pop_j */
static void active_apply_join(ActiveList *a, Tree *t, int src, int dst) {
    for (int i = 0; i < a->count; i++) {
        if (a->pop[i] == src) {
            a->pop[i] = (int8_t)dst;
            t->population[a->node[i]] = (int8_t)dst;
        }
    }
}

/* Set all lineages to class S (for panmictic transition) */
static void active_set_all_standard(ActiveList *a) {
    for (int i = 0; i < a->count; i++)
        a->klass[i] = CLASS_S;
}

/* Count distinct populations present among active lineages */
static int active_count_pops(ActiveList *a, int n_pops) {
    int present[MAX_POPS];
    memset(present, 0, sizeof(int) * n_pops);
    for (int i = 0; i < a->count; i++) {
        int p = a->pop[i];
        if (p >= 0 && p < n_pops) present[p] = 1;
    }
    int c = 0;
    for (int i = 0; i < n_pops; i++) c += present[i];
    return c;
}

/* Get the single population present (assumes count_pops == 1) */
static int active_single_pop(ActiveList *a) {
    if (a->count > 0) return (int)a->pop[0];
    return 0;
}

/* ----------------------------------------------------------------
 * Rate table for structured coalescent events
 * ---------------------------------------------------------------- */

#define MAX_RATE_ENTRIES 256

typedef struct {
    char type;      /* 'c' = coal, 'f' = flux, 'm' = migrate */
    int8_t cls;
    int8_t pop;
    double rate;
} RateEntry;

/* ----------------------------------------------------------------
 * build_structured — structured coalescent tree builder
 *
 * Port of Python build_structured_tree.
 * Returns root node index, leaves are 0..nsam-1.
 * ---------------------------------------------------------------- */

static int build_structured(Tree *t, SimParams *sp, InversionSpec *inv,
                            double phi_x) {
    tree_init(t);

    int n_pops = sp->demo.state.n_pops;
    if (n_pops < 1) n_pops = 1;

    /* Create leaves from sample_config or n_std/n_inv */
    ActiveList active;
    active_init(&active);
    int sid = 0;

    if (sp->use_sample_config) {
        for (int cls = 0; cls <= 1; cls++) {
            for (int pop = 0; pop < n_pops; pop++) {
                int cnt = sp->sample_counts[cls][pop];
                for (int j = 0; j < cnt; j++) {
                    int nd = tree_add_node(t, 0.0, (int8_t)cls, (int8_t)pop, sid);
                    active_add(&active, nd, (int8_t)cls, (int8_t)pop);
                    sid++;
                }
            }
        }
    } else {
        for (int i = 0; i < sp->n_std; i++) {
            int nd = tree_add_node(t, 0.0, CLASS_S, 0, sid);
            active_add(&active, nd, CLASS_S, 0);
            sid++;
        }
        for (int i = 0; i < sp->n_inv; i++) {
            int nd = tree_add_node(t, 0.0, CLASS_I, 0, sid);
            active_add(&active, nd, CLASS_I, 0);
            sid++;
        }
    }
    t->nsam = sid;

    /* Make a working copy of demography so events can be consumed */
    Demography demo;
    demo_copy(&demo, &sp->demo);

    double tc = 0.0;
    int panmictic_mode = 0;

    while (active.count > 1) {
        /* Apply demographic events at current time */
        int old_next = demo.next_event;
        demo_apply_events(&demo, tc);
        /* Handle join events: move active lineages */
        for (int ei = old_next; ei < demo.next_event; ei++) {
            DemoEvent *ev = &demo.events[ei];
            if (ev->type == 'j' || ev->type == 's') {
                active_apply_join(&active, t, ev->pop_i, ev->pop_j);
            }
        }
        n_pops = demo.state.n_pops;

        /* Check inversion frequency */
        double p_inv_global = 0.0;
        if (!panmictic_mode && inv) {
            p_inv_global = traj_p_inv_global(&inv->traj, tc);
        }

        /* Transition to panmictic if inversion frequency reaches 0 */
        if (!panmictic_mode && p_inv_global <= 0.0) {
            active_set_all_standard(&active);

            /* If single population: fast panmictic finish */
            if (active_count_pops(&active, n_pops) == 1) {
                int pop0 = active_single_pop(&active);
                while (active.count > 1) {
                    int k = active.count;
                    double sf = demo_coal_rate_factor(&demo, pop0, tc);
                    double rate = (double)(k * (k - 1)) / 2.0 * sf;
                    if (rate <= 0) {
                        double nt = demo_next_event_time(&demo);
                        if (nt < 1e30) { tc = nt; demo_apply_events(&demo, tc); continue; }
                        break;
                    }
                    double dt = rng_exponential(rate);
                    double nt = demo_next_event_time(&demo);
                    if (tc + dt >= nt) { tc = nt; demo_apply_events(&demo, tc); continue; }
                    tc += dt;
                    active_coalesce(&active, t, CLASS_S, pop0, tc);
                }
                break;
            }
            /* Multiple pops: continue rate-based loop with panmictic_mode */
            panmictic_mode = 1;
        }

        /* Build rate table */
        RateEntry rates[MAX_RATE_ENTRIES];
        int n_rates = 0;
        double total_rate = 0.0;

        for (int cls = 0; cls <= 1; cls++) {
            for (int pop = 0; pop < n_pops; pop++) {
                int k = active_count_class_pop(&active, (int8_t)cls, (int8_t)pop);
                if (k == 0) continue;

                /* Per-population inversion frequency */
                double p_inv_t = 0.0;
                if (!panmictic_mode && inv) {
                    p_inv_t = traj_interp(&inv->traj, tc, pop);
                    if (p_inv_t < 0) p_inv_t = 0;
                }
                double p_std_t = 1.0 - p_inv_t;

                double f;
                if (p_inv_t > 0) {
                    f = (cls == CLASS_S) ? p_std_t : p_inv_t;
                } else {
                    f = 1.0; /* panmictic */
                }

                /* Coalescence */
                if (k >= 2 && f > 0) {
                    double sf = demo_coal_rate_factor(&demo, pop, tc);
                    double r = (double)(k * (k - 1)) / 2.0 / f * sf;
                    if (r > 0 && n_rates < MAX_RATE_ENTRIES) {
                        rates[n_rates].type = 'c';
                        rates[n_rates].cls = (int8_t)cls;
                        rates[n_rates].pop = (int8_t)pop;
                        rates[n_rates].rate = r;
                        total_rate += r;
                        n_rates++;
                    }
                }

                /* Gene flux (within population, only if inversion exists) */
                if (k > 0 && phi_x > 0 && p_inv_t > 0 && inv) {
                    double f_other = (cls == CLASS_S) ? p_inv_t : p_std_t;
                    double rf = (double)k * inv->gamma * f_other * phi_x;
                    if (rf > 0 && n_rates < MAX_RATE_ENTRIES) {
                        rates[n_rates].type = 'f';
                        rates[n_rates].cls = (int8_t)cls;
                        rates[n_rates].pop = (int8_t)pop;
                        rates[n_rates].rate = rf;
                        total_rate += rf;
                        n_rates++;
                    }
                }

                /* Migration */
                if (k > 0) {
                    for (int op = 0; op < n_pops; op++) {
                        if (op == pop) continue;
                        double m = demo.state.mig_matrix[op][pop];
                        if (m > 0) {
                            double rm = (double)k * m / 2.0;
                            if (rm > 0 && n_rates < MAX_RATE_ENTRIES) {
                                rates[n_rates].type = 'm';
                                rates[n_rates].cls = (int8_t)cls;
                                rates[n_rates].pop = (int8_t)pop;
                                rates[n_rates].rate = rm;
                                total_rate += rm;
                                n_rates++;
                            }
                        }
                    }
                }
            }
        }

        if (total_rate <= 0) {
            /* Try jumping to t_inv */
            if (inv && tc < inv->traj.t_inv) {
                tc = inv->traj.t_inv;
                continue;
            }
            /* Try next demographic event */
            double nt = demo_next_event_time(&demo);
            if (nt < 1e30) { tc = nt; continue; }
            break; /* stuck — shouldn't happen */
        }

        double dt = rng_exponential(total_rate);

        /* Check t_inv boundary */
        if (!panmictic_mode && inv && tc + dt >= inv->traj.t_inv) {
            tc = inv->traj.t_inv;
            continue;
        }

        /* Check next demographic event */
        double nt = demo_next_event_time(&demo);
        if (tc + dt >= nt) {
            tc = nt;
            continue;
        }

        tc += dt;

        /* Choose event */
        double u = rng_uniform() * total_rate;
        double cum = 0.0;
        int chosen = n_rates - 1;
        for (int i = 0; i < n_rates; i++) {
            cum += rates[i].rate;
            if (u < cum) { chosen = i; break; }
        }

        char etype = rates[chosen].type;
        int8_t ecls = rates[chosen].cls;
        int8_t epop = rates[chosen].pop;

        if (etype == 'c') {
            active_coalesce(&active, t, ecls, epop, tc);
        } else if (etype == 'f') {
            active_flux(&active, t, ecls, epop, tc);
        } else if (etype == 'm') {
            active_migrate(&active, t, ecls, epop, n_pops);
        }
    }

    t->root = (active.count > 0) ? active.node[0] : 0;
    return t->root;
}

/* ----------------------------------------------------------------
 * coalesce_above_root_structured
 *
 * Two lineages (floating + root) above the tree.
 * Compete: coalescence (if same class) vs flux (both can flip).
 * At t >= t_inv: force panmictic coalescence.
 * ---------------------------------------------------------------- */

static int coalesce_above_root_structured(Tree *t, int root_node,
                                          int floating, int8_t fclass,
                                          double t_start,
                                          InversionSpec *inv, double phi_x,
                                          Demography *demo, int pop) {
    int8_t rclass = t->klass[root_node];
    double tc = t_start;

    for (int iter = 0; iter < 100000; iter++) {
        double p_inv_t = inv ? traj_interp(&inv->traj, tc, pop) : 0.0;

        /* Beyond inversion age: panmictic coalescence */
        if (p_inv_t <= 0) {
            double dt = rng_exponential(1.0);
            tc += dt;
            int coal = tree_add_node(t, tc, CLASS_S, (int8_t)pop, NULL_NODE);
            tree_add_child(t, coal, root_node);
            tree_add_child(t, coal, floating);
            t->klass[floating] = CLASS_S;
            return coal;
        }

        double p_std_t = 1.0 - p_inv_t;

        /* Flux rate for floating lineage */
        double p_other_f = (fclass == CLASS_S) ? p_inv_t : p_std_t;
        double rf_floating = inv ? inv->gamma * p_other_f * phi_x : 0.0;

        /* Flux rate for root lineage */
        double p_other_r = (rclass == CLASS_S) ? p_inv_t : p_std_t;
        double rf_root = inv ? inv->gamma * p_other_r * phi_x : 0.0;

        /* Coalescence rate (only if same class) */
        double rc = 0.0;
        if (fclass == rclass) {
            double p_same = (fclass == CLASS_S) ? p_std_t : p_inv_t;
            rc = (p_same > 0) ? 1.0 / p_same : 0.0;
        }

        /* Scale coalescence by demography */
        if (demo) rc *= demo_coal_rate_factor(demo, pop, tc);

        double total = rc + rf_floating + rf_root;
        if (total <= 0) {
            if (inv && tc < inv->traj.t_inv) { tc = inv->traj.t_inv; continue; }
            /* Fallback */
            tc += 50.0;
            int coal = tree_add_node(t, tc, fclass, (int8_t)pop, NULL_NODE);
            tree_add_child(t, coal, root_node);
            tree_add_child(t, coal, floating);
            t->klass[floating] = fclass;
            return coal;
        }

        double dt = rng_exponential(total);

        /* Check t_inv */
        if (inv && tc + dt >= inv->traj.t_inv) {
            tc = inv->traj.t_inv;
            continue;
        }

        tc += dt;
        double u = rng_uniform() * total;

        if (u < rc) {
            /* Coalescence */
            int coal = tree_add_node(t, tc, fclass, (int8_t)pop, NULL_NODE);
            tree_add_child(t, coal, root_node);
            tree_add_child(t, coal, floating);
            t->klass[floating] = fclass;
            return coal;
        } else if (u < rc + rf_floating) {
            /* Flux on floating lineage */
            int8_t new_cls = (fclass == CLASS_S) ? CLASS_I : CLASS_S;
            int fn = tree_add_node(t, tc, new_cls, (int8_t)pop, NULL_NODE);
            tree_add_child(t, fn, floating);
            t->klass[floating] = fclass;
            floating = fn;
            fclass = new_cls;
        } else {
            /* Flux on root lineage */
            int8_t new_cls = (rclass == CLASS_S) ? CLASS_I : CLASS_S;
            int fn = tree_add_node(t, tc, new_cls, (int8_t)pop, NULL_NODE);
            tree_add_child(t, fn, root_node);
            t->klass[root_node] = rclass;
            root_node = fn;
            rclass = new_cls;
        }
    }

    /* Absolute fallback */
    tc += 1.0;
    int coal = tree_add_node(t, tc, fclass, (int8_t)pop, NULL_NODE);
    tree_add_child(t, coal, root_node);
    tree_add_child(t, coal, floating);
    t->klass[floating] = fclass;
    return coal;
}

/* ----------------------------------------------------------------
 * reattach_structured
 *
 * Walk backward in time through tree intervals, attempting
 * coalescence or gene flux for the floating lineage.
 * ---------------------------------------------------------------- */

static int reattach_structured(Tree *t, int floating, int8_t fclass,
                               double t_start, InversionSpec *inv,
                               double phi_x, Demography *demo, int pop) {
    double tc = t_start;

    for (int safety = 0; safety < 50000; safety++) {
        double p_inv_t = inv ? traj_interp(&inv->traj, tc, pop) : 0.0;

        /* Beyond inversion age: panmictic — coalesce above root */
        if (p_inv_t <= 0) {
            fclass = CLASS_S;
            return coalesce_above_root_structured(t, t->root, floating,
                                                  CLASS_S, tc, NULL, phi_x,
                                                  demo, pop);
        }

        double p_std_t = 1.0 - p_inv_t;

        /* Collect sorted unique node times above tc */
        double times_above[MAX_NODES];
        int n_times = 0;
        for (int i = 0; i < t->n; i++) {
            if (!t->active[i]) continue;
            if (i == floating) continue;
            if (t->time[i] > tc) {
                /* Insertion sort */
                int j = n_times;
                while (j > 0 && times_above[j-1] > t->time[i]) {
                    times_above[j] = times_above[j-1]; j--;
                }
                times_above[j] = t->time[i];
                n_times++;
            }
        }
        /* Deduplicate */
        if (n_times > 1) {
            int w = 1;
            for (int i = 1; i < n_times; i++)
                if (times_above[i] != times_above[w-1])
                    times_above[w++] = times_above[i];
            n_times = w;
        }

        int went_above = 1;
        for (int ti = 0; ti < n_times; ti++) {
            double t_next = times_above[ti];

            p_inv_t = inv ? traj_interp(&inv->traj, tc, pop) : 0.0;
            if (p_inv_t <= 0) { went_above = 1; break; }
            p_std_t = 1.0 - p_inv_t;

            /* Count same-class branches alive at tc */
            int same[MAX_NODES];
            int k_same = 0;
            for (int i = 0; i < t->n; i++) {
                if (!t->active[i] || i == floating) continue;
                int pi = t->parent[i];
                if (pi >= 0 && t->time[i] <= tc && t->time[pi] > tc
                    && t->klass[i] == fclass) {
                    same[k_same++] = i;
                }
            }

            double p_same = (fclass == CLASS_S) ? p_std_t : p_inv_t;
            double p_other = (fclass == CLASS_S) ? p_inv_t : p_std_t;

            double rate_coal = (k_same > 0 && p_same > 0)
                               ? (double)k_same / p_same : 0.0;
            /* Scale coalescence by demography */
            if (demo) rate_coal *= demo_coal_rate_factor(demo, pop, tc);

            double rate_flux = (phi_x > 0 && inv)
                               ? inv->gamma * p_other * phi_x : 0.0;
            double total = rate_coal + rate_flux;

            if (total <= 0) {
                /* Jump to t_inv if available */
                if (inv && inv->traj.t_inv < t_next && tc < inv->traj.t_inv) {
                    tc = inv->traj.t_inv;
                    went_above = 0;
                    break;
                }
                tc = t_next;
                continue;
            }

            double dt = rng_exponential(total);
            double t_event = tc + dt;

            /* Check t_inv */
            if (inv && t_event >= inv->traj.t_inv) {
                tc = inv->traj.t_inv;
                went_above = 0;
                break;
            }

            if (t_event < t_next) {
                if (rng_uniform() * total < rate_coal && k_same > 0) {
                    /* Coalescence: attach to a random same-class branch */
                    int ci = (int)(rng_uniform() * k_same);
                    if (ci >= k_same) ci = k_same - 1;
                    int attach_to = same[ci];
                    int old_p = t->parent[attach_to];

                    int coal = tree_add_node(t, t_event, fclass, (int8_t)pop,
                                             NULL_NODE);
                    if (old_p >= 0) {
                        tree_remove_child(t, old_p, attach_to);
                        tree_add_child(t, old_p, coal);
                    }
                    tree_add_child(t, coal, attach_to);
                    tree_add_child(t, coal, floating);
                    t->klass[floating] = fclass;

                    if (old_p < 0) {
                        /* attach_to was root */
                        t->root = coal;
                    }
                    return t->root;
                } else {
                    /* Flux: flip class of floating lineage */
                    int8_t new_cls = (fclass == CLASS_S) ? CLASS_I : CLASS_S;
                    int fn = tree_add_node(t, t_event, new_cls, (int8_t)pop,
                                           NULL_NODE);
                    tree_add_child(t, fn, floating);
                    t->klass[floating] = fclass;
                    floating = fn;
                    fclass = new_cls;
                    tc = t_event;
                    went_above = 0;
                    break;
                }
            } else {
                tc = t_next;
            }
        }

        if (went_above) {
            int new_root = coalesce_above_root_structured(
                t, t->root, floating, fclass, tc, inv, phi_x, demo, pop);
            t->root = new_root;
            return t->root;
        }
    }

    /* Safety fallback: coalesce above root */
    int new_root = coalesce_above_root_structured(
        t, t->root, floating, fclass, tc, inv, phi_x, demo, pop);
    t->root = new_root;
    return t->root;
}

/* ----------------------------------------------------------------
 * smc_prune_reattach_structured
 *
 * One SMC update: pick a branch of recomb_class weighted by length,
 * prune above it, reattach under structured coalescent.
 * ---------------------------------------------------------------- */

static void smc_prune_reattach_structured(Tree *t, int8_t recomb_class,
                                          InversionSpec *inv, double phi_x,
                                          Demography *demo) {
    /* Collect branches of the specified recomb_class */
    int br_idx[MAX_NODES];
    double br_len[MAX_NODES];
    int br_count = 0;
    double total_L = 0;

    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int p = t->parent[i];
        if (p < 0) continue;
        double bl = t->time[p] - t->time[i];
        if (bl <= 0) continue;
        if (t->klass[i] == recomb_class) {
            br_idx[br_count] = i;
            br_len[br_count] = bl;
            total_L += bl;
            br_count++;
        }
    }
    if (br_count == 0 || total_L <= 0) return;

    /* Choose branch weighted by length */
    double r = rng_uniform() * total_L;
    double cum = 0;
    int bi = br_count - 1;
    for (int i = 0; i < br_count; i++) {
        cum += br_len[i];
        if (r < cum) { bi = i; break; }
    }
    int target = br_idx[bi];
    double t_cut = t->time[target] + rng_uniform() * br_len[bi];

    /* Prune: walk up from target past degree-2 nodes */
    int p = t->parent[target];
    if (p < 0 || tree_num_children(t, p) != 2) return;
    int sib = tree_get_sibling(t, target);
    if (sib < 0) return;

    int gp = t->parent[p];
    tree_remove_child(t, p, target);
    tree_remove_child(t, p, sib);
    if (gp >= 0) {
        tree_remove_child(t, gp, p);
        tree_add_child(t, gp, sib);
    } else {
        t->parent[sib] = NULL_NODE;
    }
    if (t->root == p) t->root = sib;
    t->parent[target] = NULL_NODE;
    tree_free_node(t, p);

    /* Reattach under structured coalescent */
    int target_pop = (int)t->population[target];
    reattach_structured(t, target, recomb_class, t_cut, inv, phi_x,
                        demo, target_pop);

    /* Ensure root is correct */
    t->root = tree_find_root(t);
}

/* ----------------------------------------------------------------
 * Boundary transition helpers
 * ---------------------------------------------------------------- */

/* Collect all leaf node indices (sample_id >= 0) */
static int tree_collect_leaves(Tree *t, int *leaves, int max_leaves) {
    int n = 0;
    for (int i = 0; i < t->n; i++) {
        if (t->active[i] && t->sample_id[i] >= 0 && n < max_leaves)
            leaves[n++] = i;
    }
    return n;
}

/* rebuild_panmictic_from_leaves: collect leaves, clear internal nodes,
 * build a fresh panmictic coalescent tree. */
static void rebuild_panmictic_from_leaves(Tree *t) {
    int leaves[MAX_NODES];
    int n_leaves = tree_collect_leaves(t, leaves, MAX_NODES);
    if (n_leaves <= 1) return;

    /* Save leaf info, then reinit tree */
    double leaf_times[MAX_NODES];
    int8_t leaf_klass[MAX_NODES];
    int8_t leaf_pop[MAX_NODES];
    int leaf_sid[MAX_NODES];
    for (int i = 0; i < n_leaves; i++) {
        int nd = leaves[i];
        leaf_times[i] = t->time[nd];
        leaf_klass[i] = t->klass[nd];
        leaf_pop[i] = t->population[nd];
        leaf_sid[i] = t->sample_id[nd];
    }

    tree_init(t);

    /* Re-create leaves */
    int active_arr[MAX_NODES];
    for (int i = 0; i < n_leaves; i++) {
        active_arr[i] = tree_add_node(t, leaf_times[i], CLASS_S,
                                       leaf_pop[i], leaf_sid[i]);
    }
    t->nsam = n_leaves;

    /* Panmictic coalescent */
    int k = n_leaves;
    double tc = 0.0;
    while (k > 1) {
        double rate = (double)(k * (k - 1)) / 2.0;
        tc += rng_exponential(rate);

        int i1 = (int)(rng_uniform() * k);
        int i2;
        do { i2 = (int)(rng_uniform() * k); } while (i2 == i1);
        if (i1 >= k) i1 = k - 1;
        if (i2 >= k) i2 = k - 1;

        int coal = tree_add_node(t, tc, CLASS_S, 0, NULL_NODE);
        tree_add_child(t, coal, active_arr[i1]);
        tree_add_child(t, coal, active_arr[i2]);

        if (i1 > i2) { int tmp = i1; i1 = i2; i2 = tmp; }
        active_arr[i2] = active_arr[k - 1];
        active_arr[i1] = active_arr[k - 2];
        active_arr[k - 2] = coal;
        k--;
    }
    t->root = active_arr[0];
}

/* rebuild_structured_from_leaves: collect leaves, reassign classes
 * based on current inversion frequency, build structured tree. */
static void rebuild_structured_from_leaves(Tree *t, SimParams *sp,
                                           InversionSpec *inv,
                                           double phi_x) {
    int leaves[MAX_NODES];
    int n_leaves = tree_collect_leaves(t, leaves, MAX_NODES);
    if (n_leaves <= 1) return;

    /* Save leaf info */
    double leaf_times[MAX_NODES];
    int8_t leaf_pop[MAX_NODES];
    int leaf_sid[MAX_NODES];
    for (int i = 0; i < n_leaves; i++) {
        int nd = leaves[i];
        leaf_times[i] = t->time[nd];
        leaf_pop[i] = t->population[nd];
        leaf_sid[i] = t->sample_id[nd];
    }

    /* Reassign classes based on sample config or original assignments.
     * Preserve the original class if it was set from sample_config. */
    int8_t leaf_klass[MAX_NODES];
    if (sp->use_sample_config) {
        /* Reconstruct from sample_counts: first S samples, then I samples */
        int total_s = 0;
        for (int p = 0; p < sp->demo.state.n_pops; p++)
            total_s += sp->sample_counts[CLASS_S][p];
        for (int i = 0; i < n_leaves; i++) {
            leaf_klass[i] = (leaf_sid[i] < total_s) ? CLASS_S : CLASS_I;
        }
    } else {
        for (int i = 0; i < n_leaves; i++) {
            leaf_klass[i] = (leaf_sid[i] < sp->n_std) ? CLASS_S : CLASS_I;
        }
    }

    tree_init(t);

    /* Re-create leaves and populate sample_counts for build_structured */
    SimParams sp_copy;
    memcpy(&sp_copy, sp, sizeof(SimParams));
    sp_copy.use_sample_config = 1;
    memset(sp_copy.sample_counts, 0, sizeof(sp_copy.sample_counts));

    for (int i = 0; i < n_leaves; i++) {
        sp_copy.sample_counts[(int)leaf_klass[i]][(int)leaf_pop[i]]++;
    }

    /* Use build_structured to create the tree */
    build_structured(t, &sp_copy, inv, phi_x);
}

/* ================================================================
 * Phase 4: SMC Walk Engine
 * ================================================================ */

/* ----------------------------------------------------------------
 * Flux heap (min-heap for pending Peischl b2 flux reversions)
 * ---------------------------------------------------------------- */

typedef struct {
    double b2_abs;     /* absolute chromosome position where flux reverts */
    int flux_node;     /* node index of the flux event */
} PendingFlux;

typedef struct {
    PendingFlux items[MAX_PENDING];
    int count;
} FluxHeap;

static void flux_heap_init(FluxHeap *h) {
    h->count = 0;
}

static void flux_heap_push(FluxHeap *h, double b2_abs, int flux_node) {
    if (h->count >= MAX_PENDING) return;
    int i = h->count++;
    h->items[i].b2_abs = b2_abs;
    h->items[i].flux_node = flux_node;
    /* Sift up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->items[parent].b2_abs <= h->items[i].b2_abs) break;
        PendingFlux tmp = h->items[parent];
        h->items[parent] = h->items[i];
        h->items[i] = tmp;
        i = parent;
    }
}

static PendingFlux *flux_heap_peek(FluxHeap *h) {
    if (h->count <= 0) return NULL;
    return &h->items[0];
}

static PendingFlux flux_heap_pop(FluxHeap *h) {
    PendingFlux top = h->items[0];
    h->count--;
    if (h->count > 0) {
        h->items[0] = h->items[h->count];
        /* Sift down */
        int i = 0;
        for (;;) {
            int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
            if (left < h->count &&
                h->items[left].b2_abs < h->items[smallest].b2_abs)
                smallest = left;
            if (right < h->count &&
                h->items[right].b2_abs < h->items[smallest].b2_abs)
                smallest = right;
            if (smallest == i) break;
            PendingFlux tmp = h->items[smallest];
            h->items[smallest] = h->items[i];
            h->items[i] = tmp;
            i = smallest;
        }
    }
    return top;
}

/* ----------------------------------------------------------------
 * Edge recorder (for tskit output)
 * ---------------------------------------------------------------- */

#define MAX_EDGES 500000

typedef struct {
    double left[MAX_EDGES], right[MAX_EDGES];
    int parent_id[MAX_EDGES], child_id[MAX_EDGES];
    int n_edges;
    /* Active intervals: one per tree node index */
    int active_parent[MAX_NODES], active_child[MAX_NODES];
    double active_left[MAX_NODES];
    int n_active;
} EdgeRecorder;

static void edge_init(EdgeRecorder *er) {
    er->n_edges = 0;
    er->n_active = 0;
    memset(er->active_parent, 0xFF, sizeof(er->active_parent));
    memset(er->active_child, 0xFF, sizeof(er->active_child));
}

/* Open active intervals for all edges in the current tree. */
static void edge_open_all(EdgeRecorder *er, Tree *t, double pos) {
    er->n_active = 0;
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        int p = t->parent[i];
        if (p < 0) continue;
        int idx = er->n_active++;
        er->active_parent[idx] = t->node_id[p];
        er->active_child[idx] = t->node_id[i];
        er->active_left[idx] = pos;
    }
}

/* Compare old tree edges to new tree edges; close removed, open added. */
static void edge_update(EdgeRecorder *er, Tree *old_t, Tree *new_t,
                        double pos) {
    /* Close all old active intervals */
    for (int i = 0; i < er->n_active; i++) {
        if (er->n_edges < MAX_EDGES) {
            int e = er->n_edges++;
            er->left[e] = er->active_left[i];
            er->right[e] = pos;
            er->parent_id[e] = er->active_parent[i];
            er->child_id[e] = er->active_child[i];
        }
    }
    /* Open new intervals */
    edge_open_all(er, new_t, pos);
    (void)old_t; /* old_t is unused since we close all and reopen */
}

/* Close all remaining active intervals at the end of the chromosome. */
static void edge_close_all(EdgeRecorder *er, double pos) {
    for (int i = 0; i < er->n_active; i++) {
        if (er->n_edges < MAX_EDGES) {
            int e = er->n_edges++;
            er->left[e] = er->active_left[i];
            er->right[e] = pos;
            er->parent_id[e] = er->active_parent[i];
            er->child_id[e] = er->active_child[i];
        }
    }
    er->n_active = 0;
}

/* ----------------------------------------------------------------
 * tree_weighted_branch_length_inv — compute weighted branch length
 * inside inversion region, using per-branch population frequency.
 * Also computes weighted S and I lengths separately.
 * ---------------------------------------------------------------- */

static void tree_weighted_branch_lengths_inv(Tree *t, Trajectory *traj,
                                             double *wL_S_out,
                                             double *wL_I_out,
                                             double *t_max_out) {
    double wL_S = 0.0, wL_I = 0.0, t_max = 0.0;
    for (int i = 0; i < t->n; i++) {
        if (!t->active[i]) continue;
        if (t->time[i] > t_max) t_max = t->time[i];
        int p = t->parent[i];
        if (p < 0) continue;
        double bl = t->time[p] - t->time[i];
        if (bl <= 0) continue;

        int pop = (int)t->population[i];
        double p_inv_t = traj_interp(traj, 0.5 * t_max, pop);
        double w;
        if (p_inv_t > 0) {
            w = (t->klass[i] == CLASS_S) ? (1.0 - p_inv_t) : p_inv_t;
        } else {
            w = 1.0;
        }
        if (t->klass[i] == CLASS_S)
            wL_S += bl * w;
        else
            wL_I += bl * w;
    }
    *wL_S_out = wL_S;
    *wL_I_out = wL_I;
    if (t_max_out) *t_max_out = t_max;
}

/* ----------------------------------------------------------------
 * walk_segment — Single left-to-right SMC walk from start_pos to
 * end_pos within one region (inversion or collinear).
 *
 * Parameters:
 *   t          — tree (already built for the starting state)
 *   sp         — simulation parameters
 *   inv        — inversion spec (NULL for collinear walks)
 *   bp_l, bp_r — inversion breakpoints (0, 0 for collinear)
 *   inv_len    — bp_r - bp_l
 *   start_pos  — starting position
 *   end_pos    — ending position
 *   muts       — mutation array to fill
 *   mut_count  — current mutation count (in/out)
 *   max_muts   — maximum mutations
 *   er         — edge recorder (NULL if not recording)
 *
 * Returns updated mut_count.
 * ---------------------------------------------------------------- */

static int walk_segment(Tree *t, SimParams *sp, InversionSpec *inv,
                        double bp_l, double bp_r, double inv_len,
                        double start_pos, double end_pos,
                        Mutation *muts, int mut_count, int max_muts,
                        EdgeRecorder *er) {
    double pos = start_pos;
    FluxHeap fheap;
    flux_heap_init(&fheap);

    /* Make a working copy of demography */
    Demography demo;
    demo_copy(&demo, &sp->demo);

    for (int iter = 0; iter < 500000 && pos < end_pos; iter++) {
        int in_inv = (inv != NULL && inv_len > 0 &&
                      bp_l <= pos && pos < bp_r);

        /* Branch lengths */
        double L_S, L_I, t_max;
        tree_branch_lengths(t, &L_S, &L_I, &t_max);
        double L_total = L_S + L_I;

        double weighted_L, next_boundary;

        if (in_inv) {
            /* Weighted branch length using per-pop frequencies */
            double wL_S, wL_I;
            tree_weighted_branch_lengths_inv(t, &inv->traj,
                                            &wL_S, &wL_I, &t_max);
            weighted_L = wL_S + wL_I;
            next_boundary = bp_r < end_pos ? bp_r : end_pos;
        } else {
            weighted_L = L_total;
            if (pos < bp_l && bp_l < end_pos && inv != NULL && inv_len > 0)
                next_boundary = bp_l < end_pos ? bp_l : end_pos;
            else
                next_boundary = end_pos;
        }

        if (weighted_L <= 0) {
            /* Drop mutations, skip to boundary */
            mut_count = drop_mutations_segment(t, pos, next_boundary,
                                               sp->theta, muts, mut_count,
                                               max_muts);
            pos = next_boundary;

            /* Boundary transitions */
            if (pos < end_pos) {
                int entering_inv = (!in_inv && inv != NULL && inv_len > 0 &&
                                    pos >= bp_l && pos < bp_r);
                int leaving_inv = (in_inv && pos >= bp_r);
                if (entering_inv) {
                    double inv_pos = (pos - bp_l) / inv_len;
                    if (inv_pos < 0.02) inv_pos = 0.02;
                    if (inv_pos > 0.98) inv_pos = 0.98;
                    double phi = phi_at(inv_pos, inv->flux_w);
                    if (er) edge_update(er, t, t, pos);
                    rebuild_structured_from_leaves(t, sp, inv, phi);
                    if (er) edge_open_all(er, t, pos);
                } else if (leaving_inv) {
                    if (er) edge_update(er, t, t, pos);
                    rebuild_panmictic_from_leaves(t);
                    if (er) edge_open_all(er, t, pos);
                }
            }
            continue;
        }

        double rate = (sp->rho / 2.0) * weighted_L;
        double dx = rng_exponential(rate);
        double extent = dx;
        if (extent > next_boundary - pos) extent = next_boundary - pos;
        if (extent > end_pos - pos) extent = end_pos - pos;
        if (extent <= 0) extent = 1e-10;

        double new_pos = pos + extent;

        /* Check pending flux reversions before this position */
        while (fheap.count > 0) {
            PendingFlux *pf = flux_heap_peek(&fheap);
            if (pf->b2_abs > new_pos) break;
            /* Flux reversion: just pop it (the tree was already updated) */
            flux_heap_pop(&fheap);
        }

        /* Drop mutations on this segment */
        mut_count = drop_mutations_segment(t, pos, new_pos,
                                           sp->theta, muts, mut_count,
                                           max_muts);

        if (dx < (next_boundary - pos) && dx < (end_pos - pos)) {
            /* Recombination event */
            new_pos = pos + dx;
            int new_in_inv = (inv != NULL && inv_len > 0 &&
                              bp_l <= new_pos && new_pos < bp_r);

            /* Save old tree state for edge recording */
            Tree old_tree;
            if (er) memcpy(&old_tree, t, sizeof(Tree));

            if (new_in_inv) {
                /* Per-branch per-pop weighted class selection */
                double wL_S = 0.0, wL_I = 0.0;
                tree_weighted_branch_lengths_inv(t, &inv->traj,
                                                &wL_S, &wL_I, NULL);
                double wL = wL_S + wL_I;
                int8_t recomb_class;
                if (wL > 0) {
                    recomb_class = (rng_uniform() * wL < wL_S)
                                   ? CLASS_S : CLASS_I;
                } else {
                    recomb_class = CLASS_S;
                }

                double inv_pos = (new_pos - bp_l) / inv_len;
                if (inv_pos < 0.02) inv_pos = 0.02;
                if (inv_pos > 0.98) inv_pos = 0.98;
                double phi = phi_at(inv_pos, inv->flux_w);

                smc_prune_reattach_structured(t, recomb_class, inv, phi,
                                              &demo);
                t->root = tree_find_root(t);

                /* Check for new flux nodes and push b2 to heap.
                 * (In the C version flux nodes are created during
                 *  reattach_structured; we don't need explicit b2 tracking
                 *  since the tree already encodes the class changes.) */
            } else {
                /* Collinear recombination: panmictic prune-and-reattach */
                smc_prune_reattach(t);
                t->root = tree_find_root(t);
            }

            /* Edge recording */
            if (er) edge_update(er, &old_tree, t, new_pos);
        } else {
            /* Boundary hit */
            int was_in_inv = in_inv;
            int entering = (!in_inv && inv != NULL && inv_len > 0 &&
                            new_pos >= bp_l && new_pos < bp_r);
            int leaving = (was_in_inv && new_pos >= bp_r);

            if (entering) {
                double inv_pos = (new_pos - bp_l) / inv_len;
                if (inv_pos < 0.02) inv_pos = 0.02;
                if (inv_pos > 0.98) inv_pos = 0.98;
                double phi = phi_at(inv_pos, inv->flux_w);

                Tree old_tree;
                if (er) memcpy(&old_tree, t, sizeof(Tree));

                rebuild_structured_from_leaves(t, sp, inv, phi);

                if (er) edge_update(er, &old_tree, t, new_pos);
            } else if (leaving) {
                Tree old_tree;
                if (er) memcpy(&old_tree, t, sizeof(Tree));

                rebuild_panmictic_from_leaves(t);

                if (er) edge_update(er, &old_tree, t, new_pos);
            }
        }
        pos = new_pos;
    }

    return mut_count;
}

/* ================================================================
 * Phase 5: Main Entry Points (EXPORTED)
 * ================================================================ */

/* ----------------------------------------------------------------
 * Mutation comparison function for qsort
 * ---------------------------------------------------------------- */

static int mutation_cmp(const void *a, const void *b) {
    const Mutation *ma = (const Mutation *)a;
    const Mutation *mb = (const Mutation *)b;
    if (ma->position < mb->position) return -1;
    if (ma->position > mb->position) return 1;
    return 0;
}

/* ----------------------------------------------------------------
 * run_single_walk — helper to set up a tree and run walk_segment
 *
 * If inv is non-NULL and gamma > 0, builds a structured tree;
 * otherwise builds a panmictic tree.
 * ---------------------------------------------------------------- */

static int run_single_walk(SimParams *sp, InversionSpec *inv,
                           double bp_l, double bp_r, double inv_len,
                           double start_pos, double end_pos,
                           double phi_init,
                           Mutation *muts, int mut_count, int max_muts,
                           EdgeRecorder *er) {
    Tree tree;

    if (inv != NULL && inv->gamma > 0 && inv_len > 0) {
        /* Structured tree */
        build_structured(&tree, sp, inv, phi_init);
    } else {
        /* Panmictic tree — treat all samples as standard */
        int nsam = sp->n_std + sp->n_inv;
        build_panmictic(&tree, nsam, 0);
    }

    if (er) edge_open_all(er, &tree, start_pos);

    mut_count = walk_segment(&tree, sp, inv, bp_l, bp_r, inv_len,
                             start_pos, end_pos,
                             muts, mut_count, max_muts, er);

    if (er) edge_close_all(er, end_pos);

    return mut_count;
}

/* ----------------------------------------------------------------
 * build_haplotype_matrix — sort mutations and fill output arrays
 * ---------------------------------------------------------------- */

static int build_haplotype_matrix(Mutation *muts, int mut_count, int nsam,
                                  int8_t *out_haps, double *out_positions,
                                  int max_sites) {
    if (mut_count <= 0) return 0;

    /* Sort by position using qsort */
    qsort(muts, (size_t)mut_count, sizeof(Mutation), mutation_cmp);

    int n_out = (mut_count < max_sites) ? mut_count : max_sites;
    memset(out_haps, 0, (size_t)nsam * (size_t)max_sites);

    for (int j = 0; j < n_out; j++) {
        out_positions[j] = muts[j].position;
        for (int s = 0; s < nsam; s++) {
            if (muts[j].leaf_bits[s / 32] & (1 << (s % 32)))
                out_haps[s * max_sites + j] = 1;
        }
    }
    return n_out;
}

/* ----------------------------------------------------------------
 * msinv_seed — exported RNG seed function
 * ---------------------------------------------------------------- */

void msinv_seed(uint64_t s0, uint64_t s1) {
    smc_full_seed(s0, s1);
}

/* ----------------------------------------------------------------
 * msinv_simulate_one — Main exported function for ms-format output.
 *
 * Strategy:
 *   n_inversions == 0: simple panmictic walk
 *   n_inversions == 1: 4-walk strategy
 *   n_inversions > 1:  multi-inversion 4-walk
 * ---------------------------------------------------------------- */

int msinv_simulate_one(SimParams *sp, int8_t *out_haps,
                       double *out_positions, int max_sites) {
    int nsam = sp->nsam;

    Mutation *mutations = (Mutation *)calloc((size_t)max_sites,
                                             sizeof(Mutation));
    if (!mutations) return -1;
    int mut_count = 0;

    if (sp->n_inversions == 0) {
        /* ============================================================
         * No inversions: simple panmictic walk 0 -> 1
         * ============================================================ */
        Tree tree;
        build_panmictic(&tree, nsam, 0);

        mut_count = walk_segment(&tree, sp, NULL, 0, 0, 0,
                                 0.0, 1.0,
                                 mutations, 0, max_sites, NULL);

    } else if (sp->n_inversions == 1) {
        /* ============================================================
         * Single inversion: 4-walk strategy
         *
         * Walk 1: bp_left -> center (structured, rightward)
         * Walk 2: bp_right -> center (structured, mirrored)
         * Walk 3: bp_left -> 0 (collinear, mirrored)
         * Walk 4: bp_right -> 1 (collinear, rightward)
         * ============================================================ */
        InversionSpec *inv = &sp->inversions[0];
        double bp_l = inv->bp_left;
        double bp_r = inv->bp_right;
        double inv_len = bp_r - bp_l;
        double center = (bp_l + bp_r) / 2.0;

        /* -- Walk 1: bp_left -> center (inversion, structured) -- */
        {
            double inv_pos_init = 0.02;
            double phi_init = phi_at(inv_pos_init, inv->flux_w);
            int pre = mut_count;
            mut_count = run_single_walk(sp, inv, bp_l, bp_r, inv_len,
                                        bp_l, center, phi_init,
                                        mutations, mut_count, max_sites,
                                        NULL);
            /* Filter: keep only mutations in [bp_l, center) */
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                if (mutations[j].position >= bp_l &&
                    mutations[j].position < center) {
                    if (write != j)
                        mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* -- Walk 2: bp_right -> center (inversion, mirrored) -- */
        {
            double m_bp_l = 1.0 - bp_r;
            double m_bp_r = 1.0 - bp_l;
            double m_start = m_bp_l;
            double m_end = 1.0 - center;

            double inv_pos_init = 0.02;
            double phi_init = phi_at(inv_pos_init, inv->flux_w);

            /* Build a mirrored SimParams for the walk */
            InversionSpec inv_mirror;
            memcpy(&inv_mirror, inv, sizeof(InversionSpec));
            inv_mirror.bp_left = m_bp_l;
            inv_mirror.bp_right = m_bp_r;

            int pre = mut_count;
            mut_count = run_single_walk(sp, &inv_mirror,
                                        m_bp_l, m_bp_r, inv_len,
                                        m_start, m_end, phi_init,
                                        mutations, mut_count, max_sites,
                                        NULL);
            /* Un-mirror and filter: keep only mutations in [center, bp_r) */
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                double real_pos = 1.0 - mutations[j].position;
                if (real_pos >= center && real_pos < bp_r) {
                    mutations[j].position = real_pos;
                    if (write != j)
                        mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* -- Walk 3: bp_left -> 0 (collinear, mirrored) -- */
        {
            double m_start = 1.0 - bp_l;
            double m_end = 1.0;

            /* Dummy inversion spec for collinear (gamma=0) */
            int pre = mut_count;
            mut_count = run_single_walk(sp, NULL, 0, 0, 0,
                                        m_start, m_end, 0.0,
                                        mutations, mut_count, max_sites,
                                        NULL);
            /* Un-mirror and filter: keep only mutations in [0, bp_l) */
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                double real_pos = 1.0 - mutations[j].position;
                if (real_pos >= 0.0 && real_pos < bp_l) {
                    mutations[j].position = real_pos;
                    if (write != j)
                        mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* -- Walk 4: bp_right -> 1 (collinear, rightward) -- */
        {
            int pre = mut_count;
            mut_count = run_single_walk(sp, NULL, 0, 0, 0,
                                        bp_r, 1.0, 0.0,
                                        mutations, mut_count, max_sites,
                                        NULL);
            /* Filter: keep only mutations in [bp_r, 1.0] */
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                if (mutations[j].position >= bp_r &&
                    mutations[j].position <= 1.0) {
                    if (write != j)
                        mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

    } else {
        /* ============================================================
         * Multiple inversions: 4-walk per inversion + collinear gaps
         * ============================================================ */

        /* Sort inversions by bp_left (simple insertion sort) */
        int inv_order[MAX_INVERSIONS];
        for (int i = 0; i < sp->n_inversions; i++) inv_order[i] = i;
        for (int i = 1; i < sp->n_inversions; i++) {
            int key = inv_order[i];
            double key_bp = sp->inversions[key].bp_left;
            int j = i - 1;
            while (j >= 0 && sp->inversions[inv_order[j]].bp_left > key_bp) {
                inv_order[j + 1] = inv_order[j];
                j--;
            }
            inv_order[j + 1] = key;
        }

        /* Process each inversion: walk from both breakpoints to center */
        for (int ii = 0; ii < sp->n_inversions; ii++) {
            int idx = inv_order[ii];
            InversionSpec *inv = &sp->inversions[idx];
            double bp_l = inv->bp_left;
            double bp_r = inv->bp_right;
            double inv_len = bp_r - bp_l;
            double center = (bp_l + bp_r) / 2.0;

            /* Walk 1: bp_left -> center (structured, rightward) */
            {
                double phi_init = phi_at(0.02, inv->flux_w);
                int pre = mut_count;
                mut_count = run_single_walk(sp, inv, bp_l, bp_r, inv_len,
                                            bp_l, center, phi_init,
                                            mutations, mut_count, max_sites,
                                            NULL);
                int write = pre;
                for (int j = pre; j < mut_count; j++) {
                    if (mutations[j].position >= bp_l &&
                        mutations[j].position < center) {
                        if (write != j) mutations[write] = mutations[j];
                        write++;
                    }
                }
                mut_count = write;
            }

            /* Walk 2: bp_right -> center (mirrored) */
            {
                double m_bp_l = 1.0 - bp_r;
                double m_bp_r = 1.0 - bp_l;
                double m_start = m_bp_l;
                double m_end = 1.0 - center;
                double phi_init = phi_at(0.02, inv->flux_w);

                InversionSpec inv_mirror;
                memcpy(&inv_mirror, inv, sizeof(InversionSpec));
                inv_mirror.bp_left = m_bp_l;
                inv_mirror.bp_right = m_bp_r;

                int pre = mut_count;
                mut_count = run_single_walk(sp, &inv_mirror,
                                            m_bp_l, m_bp_r, inv_len,
                                            m_start, m_end, phi_init,
                                            mutations, mut_count, max_sites,
                                            NULL);
                int write = pre;
                for (int j = pre; j < mut_count; j++) {
                    double real_pos = 1.0 - mutations[j].position;
                    if (real_pos >= center && real_pos < bp_r) {
                        mutations[j].position = real_pos;
                        if (write != j) mutations[write] = mutations[j];
                        write++;
                    }
                }
                mut_count = write;
            }
        }

        /* Process collinear gaps between inversions */
        double col_regions[MAX_INVERSIONS + 1][2];
        int n_col = 0;
        double prev_right = 0.0;
        for (int ii = 0; ii < sp->n_inversions; ii++) {
            int idx = inv_order[ii];
            double bl = sp->inversions[idx].bp_left;
            if (bl > prev_right) {
                col_regions[n_col][0] = prev_right;
                col_regions[n_col][1] = bl;
                n_col++;
            }
            double br = sp->inversions[idx].bp_right;
            if (br > prev_right) prev_right = br;
        }
        if (prev_right < 1.0) {
            col_regions[n_col][0] = prev_right;
            col_regions[n_col][1] = 1.0;
            n_col++;
        }

        for (int ci = 0; ci < n_col; ci++) {
            double col_left = col_regions[ci][0];
            double col_right = col_regions[ci][1];
            double col_center = (col_left + col_right) / 2.0;

            /* Right half: col_center -> col_right (rightward) */
            {
                int pre = mut_count;
                mut_count = run_single_walk(sp, NULL, 0, 0, 0,
                                            col_center, col_right, 0.0,
                                            mutations, mut_count, max_sites,
                                            NULL);
                int write = pre;
                for (int j = pre; j < mut_count; j++) {
                    if (mutations[j].position >= col_center &&
                        mutations[j].position < col_right) {
                        if (write != j) mutations[write] = mutations[j];
                        write++;
                    }
                }
                mut_count = write;
            }

            /* Left half: col_left -> col_center (mirrored) */
            {
                double m_start = 1.0 - col_center;
                double m_end = 1.0 - col_left;
                int pre = mut_count;
                mut_count = run_single_walk(sp, NULL, 0, 0, 0,
                                            m_start, m_end, 0.0,
                                            mutations, mut_count, max_sites,
                                            NULL);
                int write = pre;
                for (int j = pre; j < mut_count; j++) {
                    double real_pos = 1.0 - mutations[j].position;
                    if (real_pos >= col_left &&
                        real_pos < col_center) {
                        mutations[j].position = real_pos;
                        if (write != j) mutations[write] = mutations[j];
                        write++;
                    }
                }
                mut_count = write;
            }
        }
    }

    /* Build sorted haplotype matrix */
    int n_out = build_haplotype_matrix(mutations, mut_count, nsam,
                                       out_haps, out_positions, max_sites);
    free(mutations);
    return n_out;
}

/* ----------------------------------------------------------------
 * msinv_simulate_one_with_edges — Exported function for tskit output.
 *
 * Same as msinv_simulate_one but also records edges and node info
 * for tree sequence construction.
 * ---------------------------------------------------------------- */

int msinv_simulate_one_with_edges(SimParams *sp,
    int8_t *out_haps, double *out_positions, int max_sites,
    double *edge_left, double *edge_right,
    int *edge_parent, int *edge_child,
    double *node_time, int8_t *node_pop, int max_edges, int max_nodes_out,
    int *n_edges_out, int *n_nodes_out)
{
    int nsam = sp->nsam;

    Mutation *mutations = (Mutation *)calloc((size_t)max_sites,
                                             sizeof(Mutation));
    if (!mutations) return -1;

    EdgeRecorder *er = (EdgeRecorder *)calloc(1, sizeof(EdgeRecorder));
    if (!er) { free(mutations); return -1; }
    edge_init(er);

    int mut_count = 0;

    if (sp->n_inversions == 0) {
        /* Simple panmictic walk with edge recording */
        Tree tree;
        build_panmictic(&tree, nsam, 0);
        edge_open_all(er, &tree, 0.0);

        mut_count = walk_segment(&tree, sp, NULL, 0, 0, 0,
                                 0.0, 1.0,
                                 mutations, 0, max_sites, er);
        edge_close_all(er, 1.0);

        /* Export node info from last tree state */
        int n_nodes = tree.next_node_id;
        if (n_nodes > max_nodes_out) n_nodes = max_nodes_out;
        for (int i = 0; i < tree.n && i < max_nodes_out; i++) {
            int nid = tree.node_id[i];
            if (nid >= 0 && nid < max_nodes_out) {
                node_time[nid] = tree.time[i];
                node_pop[nid] = tree.population[i];
            }
        }
        if (n_nodes_out) *n_nodes_out = n_nodes;

    } else if (sp->n_inversions == 1) {
        /* 4-walk with edge recording */
        InversionSpec *inv = &sp->inversions[0];
        double bp_l = inv->bp_left;
        double bp_r = inv->bp_right;
        double inv_len = bp_r - bp_l;
        double center = (bp_l + bp_r) / 2.0;

        /* Walk 1: bp_left -> center */
        {
            double phi_init = phi_at(0.02, inv->flux_w);
            Tree tree;
            build_structured(&tree, sp, inv, phi_init);
            edge_open_all(er, &tree, bp_l);
            int pre = mut_count;
            mut_count = walk_segment(&tree, sp, inv, bp_l, bp_r, inv_len,
                                     bp_l, center,
                                     mutations, mut_count, max_sites, er);
            edge_close_all(er, center);
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                if (mutations[j].position >= bp_l &&
                    mutations[j].position < center) {
                    if (write != j) mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* Walk 2: bp_right -> center (mirrored) */
        {
            double m_bp_l = 1.0 - bp_r;
            double m_bp_r = 1.0 - bp_l;
            double phi_init = phi_at(0.02, inv->flux_w);
            InversionSpec inv_mirror;
            memcpy(&inv_mirror, inv, sizeof(InversionSpec));
            inv_mirror.bp_left = m_bp_l;
            inv_mirror.bp_right = m_bp_r;

            Tree tree;
            build_structured(&tree, sp, &inv_mirror, phi_init);
            edge_open_all(er, &tree, 1.0 - bp_r);
            int pre = mut_count;
            mut_count = walk_segment(&tree, sp, &inv_mirror,
                                     m_bp_l, m_bp_r, inv_len,
                                     m_bp_l, 1.0 - center,
                                     mutations, mut_count, max_sites, er);
            edge_close_all(er, 1.0 - center);
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                double real_pos = 1.0 - mutations[j].position;
                if (real_pos >= center && real_pos < bp_r) {
                    mutations[j].position = real_pos;
                    if (write != j) mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* Walk 3: bp_left -> 0 (collinear, mirrored) */
        {
            Tree tree;
            build_panmictic(&tree, nsam, 0);
            double m_start = 1.0 - bp_l;
            edge_open_all(er, &tree, m_start);
            int pre = mut_count;
            mut_count = walk_segment(&tree, sp, NULL, 0, 0, 0,
                                     m_start, 1.0,
                                     mutations, mut_count, max_sites, er);
            edge_close_all(er, 1.0);
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                double real_pos = 1.0 - mutations[j].position;
                if (real_pos >= 0.0 && real_pos < bp_l) {
                    mutations[j].position = real_pos;
                    if (write != j) mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        /* Walk 4: bp_right -> 1 (collinear, rightward) */
        {
            Tree tree;
            build_panmictic(&tree, nsam, 0);
            edge_open_all(er, &tree, bp_r);
            int pre = mut_count;
            mut_count = walk_segment(&tree, sp, NULL, 0, 0, 0,
                                     bp_r, 1.0,
                                     mutations, mut_count, max_sites, er);
            edge_close_all(er, 1.0);
            int write = pre;
            for (int j = pre; j < mut_count; j++) {
                if (mutations[j].position >= bp_r &&
                    mutations[j].position <= 1.0) {
                    if (write != j) mutations[write] = mutations[j];
                    write++;
                }
            }
            mut_count = write;
        }

        if (n_nodes_out) *n_nodes_out = 0; /* placeholder */

    } else {
        /* Multi-inversion with edges: not yet implemented, fall through
         * to non-edge version for mutations */
        free(er);
        free(mutations);
        /* Delegate to non-edge version */
        int result = msinv_simulate_one(sp, out_haps, out_positions,
                                         max_sites);
        if (n_edges_out) *n_edges_out = 0;
        if (n_nodes_out) *n_nodes_out = 0;
        return result;
    }

    /* Copy edge data to output arrays */
    int ne = er->n_edges;
    if (ne > max_edges) ne = max_edges;
    for (int i = 0; i < ne; i++) {
        edge_left[i] = er->left[i];
        edge_right[i] = er->right[i];
        edge_parent[i] = er->parent_id[i];
        edge_child[i] = er->child_id[i];
    }
    if (n_edges_out) *n_edges_out = ne;

    free(er);

    /* Build sorted haplotype matrix */
    int n_out = build_haplotype_matrix(mutations, mut_count, nsam,
                                       out_haps, out_positions, max_sites);
    free(mutations);
    return n_out;
}

/* ================================================================
 * Flat-argument wrapper for Python ctypes bridge.
 * Builds SimParams internally from individual arguments.
 * ================================================================ */

int msinv_simulate_flat(
    int nsam, int n_std, int n_inv,
    double theta, double rho, int nsites,
    double bp_left_1, double bp_right_1, double gamma_1, double flux_w_1,
    double t_inv_1,
    double *traj_times_1, double *traj_freqs_1,
    int traj_n_steps_1, int traj_n_pops_1,
    double bp_left_2, double bp_right_2, double gamma_2, double flux_w_2,
    double t_inv_2,
    double *traj_times_2, double *traj_freqs_2,
    int traj_n_steps_2, int traj_n_pops_2,
    int n_pops, double mig_rate,
    char *demo_types, double *demo_times,
    int *demo_pop_i, int *demo_pop_j, double *demo_values,
    int n_demo_events,
    double *pop_sizes,
    int *sample_config,
    int8_t *out_haps, double *out_positions, int max_sites)
{
    SimParams sp;
    memset(&sp, 0, sizeof(sp));

    sp.nsam = nsam;
    sp.n_std = n_std;
    sp.n_inv = n_inv;
    sp.theta = theta;
    sp.rho = rho;
    sp.nsites = nsites;
    sp.use_4walk = 1;

    demo_init(&sp.demo, n_pops, mig_rate);
    if (pop_sizes) {
        for (int i = 0; i < n_pops && i < MAX_POPS; i++)
            sp.demo.state.pop_sizes[i] = pop_sizes[i];
    }
    for (int i = 0; i < n_demo_events && i < MAX_DEMO_EVENTS; i++) {
        DemoEvent *e = &sp.demo.events[sp.demo.n_events++];
        e->type = demo_types[i];
        e->time = demo_times[i];
        e->pop_i = demo_pop_i[i];
        e->pop_j = demo_pop_j[i];
        e->value = demo_values[i];
    }

    if (sample_config) {
        sp.use_sample_config = 1;
        for (int p = 0; p < n_pops && p < MAX_POPS; p++) {
            sp.sample_counts[CLASS_S][p] = sample_config[p];
            sp.sample_counts[CLASS_I][p] = sample_config[n_pops + p];
        }
    }

    sp.n_inversions = 0;
    if (bp_left_1 >= 0) {
        InversionSpec *inv = &sp.inversions[0];
        inv->bp_left = bp_left_1;
        inv->bp_right = bp_right_1;
        inv->gamma = gamma_1;
        inv->flux_w = flux_w_1;
        inv->traj.times = traj_times_1;
        inv->traj.freqs = traj_freqs_1;
        inv->traj.n_steps = traj_n_steps_1;
        inv->traj.n_pops = traj_n_pops_1;
        inv->traj.t_inv = t_inv_1;
        sp.n_inversions = 1;
    }
    if (bp_left_2 >= 0) {
        InversionSpec *inv = &sp.inversions[1];
        inv->bp_left = bp_left_2;
        inv->bp_right = bp_right_2;
        inv->gamma = gamma_2;
        inv->flux_w = flux_w_2;
        inv->traj.times = traj_times_2;
        inv->traj.freqs = traj_freqs_2;
        inv->traj.n_steps = traj_n_steps_2;
        inv->traj.n_pops = traj_n_pops_2;
        inv->traj.t_inv = t_inv_2;
        sp.n_inversions = 2;
    }

    return msinv_simulate_one(&sp, out_haps, out_positions, max_sites);
}
