Evidence behind the Task 9 report's soft_B LD z-scores (ld_S_1 +4.6 SD, ld_I_1 +2.3 SD): one SLiM rep and 10 simulator reps.
slim_bench_soft_B_task000000.tsv: run_one --params inline JSON with soft_B rounded to 3 s.f. (t_inv 480411, p0 0.0515,
  t_sel 432374, s 2.65e-4, p* 0.622, Q 200), set name "bench_soft_B", seed base 77 (before the per-set seed offset).
sim_soft_B.tsv: slim_vs_msinv simulate --set soft_B --reps 10 --seed 9002 (before the per-set seed offset), msprime engine.
z = (slim - mean_sim) / sd_sim per statistic. Kept out of the comparison globs (different set name, n_slim = 1).
