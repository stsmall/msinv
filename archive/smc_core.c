/*
 * smc_core.c — C implementation of the SMC inner loop for msinv.
 *
 * Operates on flat numpy-compatible arrays:
 *   double time[max_nodes]
 *   int parent[max_nodes]
 *   int left_child[max_nodes]
 *   int right_sib[max_nodes]
 *   signed char klass[max_nodes]
 *   signed char population[max_nodes]
 *   int sample_id[max_nodes]
 *
 * Compile:
 *   gcc -O3 -shared -fPIC -o libsmc_core.so smc_core.c -lm
 */

#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define NULL_NODE -1
#define CLASS_S 0
#define CLASS_I 1

/* ================================================================
 * Simple xorshift64 RNG
 * ================================================================ */

static uint64_t rng_state = 123456789ULL;

void smc_seed(uint64_t s) { rng_state = s ? s : 1; }

static double rng_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0x1FFFFFFFFFFFFF) / (double)0x20000000000000;
}

static double rng_exponential(double rate) {
    if (rate <= 0) return 1e30;
    double u = rng_uniform();
    if (u <= 0) u = 1e-15;
    return -log(u) / rate;
}

/* ================================================================
 * Tree helpers
 * ================================================================ */

static int get_sibling(int *parent, int *left_child, int *right_sib, int node) {
    int p = parent[node];
    if (p == NULL_NODE) return NULL_NODE;
    int c = left_child[p];
    while (c != NULL_NODE) {
        if (c != node) return c;
        c = right_sib[c];
    }
    return NULL_NODE;
}

static int num_children(int *left_child, int *right_sib, int node) {
    int count = 0;
    int c = left_child[node];
    while (c != NULL_NODE) { count++; c = right_sib[c]; }
    return count;
}

static void add_child(int *parent, int *left_child, int *right_sib, int p, int ch) {
    parent[ch] = p;
    right_sib[ch] = left_child[p];
    left_child[p] = ch;
}

static void remove_child(int *parent, int *left_child, int *right_sib, int p, int ch) {
    int prev = NULL_NODE;
    int c = left_child[p];
    while (c != NULL_NODE) {
        if (c == ch) {
            if (prev == NULL_NODE) left_child[p] = right_sib[c];
            else right_sib[prev] = right_sib[c];
            parent[ch] = NULL_NODE;
            right_sib[ch] = NULL_NODE;
            return;
        }
        prev = c;
        c = right_sib[c];
    }
}

/* ================================================================
 * Branch lengths (exported)
 * ================================================================ */

void smc_branch_lengths(double *time, int *parent, signed char *klass,
                         int n, double *l_s, double *l_i, double *t_max) {
    double ls = 0, li = 0, tm = 0;
    for (int i = 0; i < n; i++) {
        if (time[i] > tm) tm = time[i];
        int p = parent[i];
        if (p >= 0) {
            double bl = time[p] - time[i];
            if (bl > 0) {
                if (klass[i] == CLASS_S) ls += bl; else li += bl;
            }
        }
    }
    *l_s = ls; *l_i = li; *t_max = tm;
}

/* ================================================================
 * Panmictic prune-and-reattach (exported)
 * ================================================================ */

int smc_panmictic_pr(double *time, int *parent, int *left_child,
                      int *right_sib, signed char *klass,
                      signed char *population, int *sample_id,
                      int n_nodes, int root) {
    /* Get branches */
    int indices[8192], count = 0;
    double lengths[8192], total_L = 0;
    for (int i = 0; i < n_nodes; i++) {
        int p = parent[i];
        if (p >= 0) {
            double bl = time[p] - time[i];
            if (bl > 0) {
                indices[count] = i;
                lengths[count] = bl;
                total_L += bl;
                count++;
            }
        }
    }
    if (count == 0) return root;

    /* Choose branch */
    double r = rng_uniform() * total_L;
    double cum = 0; int bi = count - 1;
    for (int i = 0; i < count; i++) {
        cum += lengths[i];
        if (r < cum) { bi = i; break; }
    }
    int target = indices[bi];
    double t_cut = time[target] + rng_uniform() * lengths[bi];

    /* Prune */
    int p = parent[target];
    if (p < 0) return root;
    if (num_children(left_child, right_sib, p) != 2) return root;
    int sib = get_sibling(parent, left_child, right_sib, target);
    if (sib < 0) return root;

    int gp = parent[p];
    remove_child(parent, left_child, right_sib, p, target);
    remove_child(parent, left_child, right_sib, p, sib);
    if (gp >= 0) {
        remove_child(parent, left_child, right_sib, gp, p);
        add_child(parent, left_child, right_sib, gp, sib);
    } else {
        parent[sib] = NULL_NODE;
    }
    right_sib[p] = NULL_NODE;
    int new_root = (root == p) ? sib : root;

    /* Reattach: find branches above t_cut */
    int above_idx[8192]; double above_len[8192];
    int above_count = 0; double total_a = 0;
    for (int i = 0; i < n_nodes; i++) {
        if (i == target || i == p) continue;
        int pi = parent[i];
        if (pi >= 0 && time[i] <= t_cut && t_cut < time[pi]) {
            above_idx[above_count] = i;
            above_len[above_count] = time[pi] - t_cut;
            total_a += above_len[above_count];
            above_count++;
        }
    }

    int coal = n_nodes;
    if (above_count > 0) {
        double r2 = rng_uniform() * total_a;
        double cum2 = 0; int ai = above_count - 1;
        for (int i = 0; i < above_count; i++) {
            cum2 += above_len[i];
            if (r2 < cum2) { ai = i; break; }
        }
        int attach = above_idx[ai];
        int ap = parent[attach];
        double t_a = t_cut + rng_uniform() * (time[ap] - t_cut);

        time[coal] = t_a;
        klass[coal] = klass[attach];
        population[coal] = population[attach];
        left_child[coal] = NULL_NODE;
        right_sib[coal] = NULL_NODE;
        parent[coal] = NULL_NODE;
        sample_id[coal] = NULL_NODE;

        /* Insert coal between attach and ap */
        remove_child(parent, left_child, right_sib, ap, attach);
        add_child(parent, left_child, right_sib, ap, coal);
        add_child(parent, left_child, right_sib, coal, attach);
        add_child(parent, left_child, right_sib, coal, target);
        return new_root;
    } else {
        double t_c = (t_cut > time[new_root]) ? t_cut : time[new_root];
        t_c += rng_exponential(1.0);
        time[coal] = t_c;
        klass[coal] = CLASS_S;
        population[coal] = 0;
        left_child[coal] = NULL_NODE;
        right_sib[coal] = NULL_NODE;
        parent[coal] = NULL_NODE;
        sample_id[coal] = NULL_NODE;

        add_child(parent, left_child, right_sib, coal, new_root);
        add_child(parent, left_child, right_sib, coal, target);
        return coal;
    }
}

/* ================================================================
 * Structured reattach (exported)
 * ================================================================ */

int smc_structured_reattach(double *time, int *parent, int *left_child,
                             int *right_sib, signed char *klass,
                             signed char *population, int *sample_id,
                             int n_nodes, int root,
                             int target, signed char fclass, double t_cut,
                             double p_inv, double c, double rho, double phi_x,
                             double t_inv) {
    double p_std = 1.0 - p_inv;
    double t = t_cut;
    signed char fc = fclass;
    int branch_ids[8192];

    for (int iter = 0; iter < 50000; iter++) {
        /* Check t_inv */
        if (t_inv > 0 && t >= t_inv) {
            /* Panmictic: coalesce with root */
            double dt = rng_exponential(1.0);
            int coal = n_nodes;
            time[coal] = t + dt;
            klass[coal] = CLASS_S;
            population[coal] = 0;
            left_child[coal] = NULL_NODE;
            right_sib[coal] = NULL_NODE;
            parent[coal] = NULL_NODE;
            sample_id[coal] = NULL_NODE;
            add_child(parent, left_child, right_sib, coal, root);
            add_child(parent, left_child, right_sib, coal, target);
            return coal;
        }

        /* Find same-class branches at time t */
        int k_same = 0;
        for (int i = 0; i < n_nodes; i++) {
            int p = parent[i];
            if (p >= 0 && time[i] <= t && time[p] > t && klass[i] == fc) {
                branch_ids[k_same++] = i;
            }
        }

        double p_same = (fc == CLASS_S) ? p_std : p_inv;
        double p_other = (fc == CLASS_S) ? p_inv : p_std;
        double rate_coal = (k_same > 0 && p_same > 0) ? (double)k_same / p_same : 0.0;
        double rate_flux = c * (rho / 2.0) * p_other * phi_x;
        double total = rate_coal + rate_flux;

        if (total <= 0) {
            /* Try jumping to root time or t_inv */
            double root_time = time[root];
            if (t < root_time) { t = root_time; continue; }
            if (t_inv > 0) { t = t_inv; continue; }
            t += 50.0;
            continue;
        }

        double dt = rng_exponential(total);
        if (t_inv > 0 && t + dt >= t_inv) { t = t_inv; continue; }

        /* Check if above root */
        double root_time = time[root];
        if (t + dt > root_time && t < root_time) {
            t = root_time;
            /* Above root: two-lineage structured coalescent */
            signed char rclass = klass[root];
            for (int iter2 = 0; iter2 < 100000; iter2++) {
                if (t_inv > 0 && t >= t_inv) {
                    double dt2 = rng_exponential(1.0);
                    int coal = n_nodes;
                    time[coal] = t + dt2;
                    klass[coal] = CLASS_S;
                    population[coal] = 0;
                    left_child[coal] = NULL_NODE;
                    right_sib[coal] = NULL_NODE;
                    parent[coal] = NULL_NODE;
                    sample_id[coal] = NULL_NODE;
                    add_child(parent, left_child, right_sib, coal, root);
                    add_child(parent, left_child, right_sib, coal, target);
                    return coal;
                }
                double rc2 = 0;
                if (fc == rclass) {
                    double ps = (fc == CLASS_S) ? p_std : p_inv;
                    rc2 = (ps > 0) ? 1.0 / ps : 0.0;
                }
                double pof = (fc == CLASS_S) ? p_inv : p_std;
                double por = (rclass == CLASS_S) ? p_inv : p_std;
                double rff = c * (rho / 2.0) * pof * phi_x;
                double rfr = c * (rho / 2.0) * por * phi_x;
                double tot2 = rc2 + rff + rfr;
                if (tot2 <= 0) {
                    if (t_inv > 0) { t = t_inv; continue; }
                    t += 50.0; break;
                }
                double dt2 = rng_exponential(tot2);
                if (t_inv > 0 && t + dt2 >= t_inv) { t = t_inv; continue; }
                t += dt2;
                double u2 = rng_uniform() * tot2;
                if (u2 < rc2) {
                    int coal = n_nodes;
                    time[coal] = t; klass[coal] = fc;
                    population[coal] = 0;
                    left_child[coal] = NULL_NODE;
                    right_sib[coal] = NULL_NODE;
                    parent[coal] = NULL_NODE;
                    sample_id[coal] = NULL_NODE;
                    add_child(parent, left_child, right_sib, coal, root);
                    add_child(parent, left_child, right_sib, coal, target);
                    return coal;
                } else if (u2 < rc2 + rff) {
                    fc = (fc == CLASS_S) ? CLASS_I : CLASS_S;
                    int fn = n_nodes++;
                    time[fn] = t; klass[fn] = fc;
                    population[fn] = 0;
                    left_child[fn] = NULL_NODE;
                    right_sib[fn] = NULL_NODE;
                    parent[fn] = NULL_NODE;
                    sample_id[fn] = NULL_NODE;
                    add_child(parent, left_child, right_sib, fn, target);
                    target = fn;
                } else {
                    rclass = (rclass == CLASS_S) ? CLASS_I : CLASS_S;
                    int fn = n_nodes++;
                    time[fn] = t; klass[fn] = rclass;
                    population[fn] = 0;
                    left_child[fn] = NULL_NODE;
                    right_sib[fn] = NULL_NODE;
                    parent[fn] = NULL_NODE;
                    sample_id[fn] = NULL_NODE;
                    add_child(parent, left_child, right_sib, fn, root);
                    root = fn;
                }
            }
            break;
        }

        t += dt;
        double u = rng_uniform() * total;
        if (u < rate_coal && k_same > 0) {
            int idx = (int)(rng_uniform() * k_same);
            if (idx >= k_same) idx = k_same - 1;
            int attach = branch_ids[idx];
            int ap = parent[attach];
            int coal = n_nodes;
            time[coal] = t; klass[coal] = fc;
            population[coal] = population[attach];
            left_child[coal] = NULL_NODE;
            right_sib[coal] = NULL_NODE;
            parent[coal] = NULL_NODE;
            sample_id[coal] = NULL_NODE;
            if (ap >= 0) {
                remove_child(parent, left_child, right_sib, ap, attach);
                add_child(parent, left_child, right_sib, ap, coal);
            }
            add_child(parent, left_child, right_sib, coal, attach);
            add_child(parent, left_child, right_sib, coal, target);
            klass[target] = fc;
            return (ap < 0) ? coal : root;
        } else {
            /* Gene flux */
            fc = (fc == CLASS_S) ? CLASS_I : CLASS_S;
            int fn = n_nodes++;
            time[fn] = t; klass[fn] = fc;
            population[fn] = 0;
            left_child[fn] = NULL_NODE;
            right_sib[fn] = NULL_NODE;
            parent[fn] = NULL_NODE;
            sample_id[fn] = NULL_NODE;
            add_child(parent, left_child, right_sib, fn, target);
            target = fn;
        }
    }
    return root;
}
