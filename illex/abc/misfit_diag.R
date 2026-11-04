# Misfit diagnostic for the trajectory ABC: what pushes soft p0 up (and t_inv down) in a reference table?
#
#   Rscript illex/abc/misfit_diag.R --table results/illex/abc_pilot_noflux/abc_table.parquet \
#       --observed results/illex/abc_observed.json --spec results/illex/abc_rf/spec.json \
#       --out-dir results/illex/misfit_diag [--ntree 1000] [--ncores 16] [--p0-hi 0.5] [--tinv-lo 3e5] [--tinv-hi 1.5e6]
#       [--p0-bins 0.001,0.01,0.1,0.5]   (p0 bin edges for step 2; default 1e-3, 0.01, 0.1, p0-hi)
#
# 1. Predictive check: for every statistic (fit keys from spec.json plus the PPC-only keys in the observed vector),
#    the quantile of the observed value among the table rows of each model x class (hard rows weighted by w_est).
#    The rows are already conditioned on p_now acceptance, so this is the prior predictive given p_now. A statistic
#    is flagged when the observed value is outside the 2.5-97.5% range for EVERY model x class.
# 2. Soft rows of N and D1: Spearman rho of each statistic with log10 p0 and with t_inv, binned means over p0 bins,
#    and in which p0 bins the observed value is inside the bin's 2.5-97.5% range; needs_top_bin = inside only the
#    highest p0 bin.
# 3. Ablation: regAbcrf posterior median and 90% interval of p0 (log10 fit) and t_inv on N and D1 soft rows with
#    cumulative statistic groups a = ratios (+ p_obs), b = a + per-class SFS, c = b + jSFS, d = c + LD, plus
#    LD alone and d minus LD-I / LD-S, for comparison.
# The prior bounds of the table being diagnosed are arguments (the table's priors may differ from the current
# constants); they are only used to express posterior positions as fractions of the prior range.

suppressPackageStartupMessages({ library(abcrf); library(jsonlite) })

args <- local({
  a <- list(ntree = 1000, ncores = 16, p0_lo = 1e-3, p0_hi = 0.5, tinv_lo = 3e5, tinv_hi = 1.5e6, seed = 1)
  v <- commandArgs(trailingOnly = TRUE); i <- 1
  while (i <= length(v)) { a[[gsub("-", "_", sub("^--", "", v[i]))]] <- v[i + 1]; i <- i + 2 }
  for (k in c("ntree", "ncores", "p0_lo", "p0_hi", "tinv_lo", "tinv_hi", "seed")) a[[k]] <- as.numeric(a[[k]])
  a$p0_bins <- if (is.null(a$p0_bins)) c(1e-3, 0.01, 0.1, a$p0_hi) else as.numeric(strsplit(a$p0_bins, ",")[[1]])
  a
})
dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)
spec <- fromJSON(args$spec, simplifyVector = FALSE)
fit_keys <- unlist(spec$fit_keys)
obsj <- fromJSON(args$observed)
obs <- c(obsj$stats, list(p_obs = obsj$p_obs))
ppc_keys <- setdiff(names(obsj$stats), fit_keys)
all_keys <- c(fit_keys, ppc_keys)
tab <- as.data.frame(arrow::read_parquet(args$table))
tab <- tab[tab$n_nan == 0, ]
tab$w <- ifelse(tab$cls == "hard", tab$w_est, 1)
tab$unit <- paste(tab$model, tab$flux, tab$cls, sep = "_")
lines <- c(sprintf("misfit_diag: table %s (%d rows), observed %s", args$table, nrow(tab), args$observed), "")

wq <- function(x, w, p) { o <- order(x); x <- x[o]; w <- w[o]; cw <- cumsum(w) / sum(w)
  sapply(p, function(pp) x[which(cw >= pp)[1]]) }
wrank <- function(x, w, v) (sum(w[x < v]) + 0.5 * sum(w[x == v])) / sum(w)

# ---- 1. predictive check
pc <- list()
for (u in sort(unique(tab$unit))) {
  d <- tab[tab$unit == u, ]
  for (k in all_keys) {
    x <- d[[k]]; ok <- is.finite(x); x <- x[ok]; w <- d$w[ok]
    q <- wq(x, w, c(0.025, 0.5, 0.975))
    pc[[length(pc) + 1]] <- data.frame(unit = u, stat = k, ppc_only = k %in% ppc_keys, obs = obs[[k]],
                                       q025 = q[1], median = q[2], q975 = q[3], obs_quantile = wrank(x, w, obs[[k]]),
                                       n = length(x))
  }
}
pc <- do.call(rbind, pc)
pc$outside <- pc$obs < pc$q025 | pc$obs > pc$q975
write.csv(pc, file.path(args$out_dir, "predictive_check.csv"), row.names = FALSE)
summ <- do.call(rbind, lapply(split(pc, pc$stat), function(s) data.frame(
  stat = s$stat[1], ppc_only = s$ppc_only[1], obs = s$obs[1], sim_median_range = sprintf("%.4g..%.4g",
  min(s$median), max(s$median)), obs_q_min = min(s$obs_quantile), obs_q_max = max(s$obs_quantile),
  n_units_inside = sum(!s$outside), flag_all_outside = all(s$outside))))
summ <- summ[match(all_keys, summ$stat), ]
write.csv(summ, file.path(args$out_dir, "predictive_check_summary.csv"), row.names = FALSE)
lines <- c(lines, "1. Predictive check (obs quantile among rows, per model x class; flag = outside 2.5-97.5% in all)",
           capture.output(print(summ, row.names = FALSE, digits = 4)), "")

# ---- 2. dependence on p0 and t_inv (soft N, D1)
bins <- args$p0_bins; bins[length(bins)] <- bins[length(bins)] + 1e-12
blab <- sprintf("p0_%g_%g", head(args$p0_bins, -1), args$p0_bins[-1])
dep <- list()
for (m in c("N", "D1")) {
  d <- tab[tab$model == m & tab$cls == "soft" & tab$flux == 0, ]
  b <- cut(d$p0, bins, right = FALSE, labels = blab)
  for (k in all_keys) {
    x <- d[[k]]
    r <- data.frame(model = m, stat = k, obs = obs[[k]],
                    rho_log_p0 = suppressWarnings(cor(x, log10(d$p0), method = "spearman", use = "complete.obs")),
                    rho_t_inv = suppressWarnings(cor(x, d$t_inv, method = "spearman", use = "complete.obs")))
    for (lv in levels(b)) {
      xb <- x[b == lv & is.finite(x)]
      r[[paste0("mean_", lv)]] <- mean(xb)
      qq <- quantile(xb, c(0.025, 0.975), names = FALSE)
      r[[paste0("in_", lv)]] <- obs[[k]] >= qq[1] & obs[[k]] <= qq[2]
      r[[paste0("n_", lv)]] <- length(xb)
    }
    ins <- unlist(r[paste0("in_", blab)])
    r$needs_top_bin <- all(!ins[-length(ins)]) & ins[length(ins)]
    dep[[length(dep) + 1]] <- r
  }
}
dep <- do.call(rbind, dep)
write.csv(dep, file.path(args$out_dir, "p0_tinv_dependence.csv"), row.names = FALSE)
lines <- c(lines, "2. Soft N/D1: Spearman rho with log10 p0 and t_inv; binned means; obs inside bin 2.5-97.5%?",
           capture.output(print(dep[, c("model", "stat", "obs", "rho_log_p0", "rho_t_inv", paste0("mean_", blab),
                                        paste0("in_", blab), "needs_top_bin")], row.names = FALSE, digits = 3)), "")

# ---- 3. ablation
grp <- list(ratios = c("pi_ratio", "dxy_pi", "f1_ratio", "p_obs"))
grp$sfs <- grep("^sfs_", fit_keys, value = TRUE)
grp$jsfs <- grep("^jsfs_", fit_keys, value = TRUE)
grp$ld <- grep("^ld_", fit_keys, value = TRUE)
sets <- list(a_ratios = grp$ratios, b_plus_sfs = c(grp$ratios, grp$sfs), c_plus_jsfs = c(grp$ratios, grp$sfs, grp$jsfs),
             d_plus_ld = c(grp$ratios, grp$sfs, grp$jsfs, grp$ld), ld_only = grp$ld,
             d_minus_ldI = setdiff(c(grp$ratios, grp$sfs, grp$jsfs, grp$ld), grep("^ld_I", fit_keys, value = TRUE)),
             d_minus_ldS = setdiff(c(grp$ratios, grp$sfs, grp$jsfs, grp$ld), grep("^ld_S", fit_keys, value = TRUE)))
stopifnot(setequal(sets$d_plus_ld, fit_keys))
ab <- list()
set.seed(args$seed)
for (m in c("N", "D1")) {
  d <- tab[tab$model == m & tab$cls == "soft" & tab$flux == 0, ]
  for (sn in names(sets)) for (p in c("p0", "t_inv")) {
    st <- sets[[sn]]
    y <- if (p == "p0") log10(d$p0) else d$t_inv
    df <- data.frame(y = y, d[, st, drop = FALSE])
    rf <- regAbcrf(y ~ ., data = df, ntree = args$ntree, paral = TRUE, ncores = args$ncores)
    pr <- predict(rf, as.data.frame(obs[st]), training = df, quantiles = c(0.05, 0.5, 0.95), paral = TRUE,
                  ncores = args$ncores)
    q <- as.numeric(pr$quantiles[1, ])
    lo <- if (p == "p0") log10(args$p0_lo) else args$tinv_lo
    hi <- if (p == "p0") log10(args$p0_hi) else args$tinv_hi
    pos <- (q - lo) / (hi - lo)
    bt <- if (p == "p0") function(x) 10^x else identity
    ab[[length(ab) + 1]] <- data.frame(model = m, set = sn, n_stats = length(st), parameter = p,
                                       median = bt(q[2]), q05 = bt(q[1]), q95 = bt(q[3]),
                                       pos_median = pos[2], pos_q05 = pos[1], pos_q95 = pos[3],
                                       oob_nmae = rf$model.rf$prediction.error)
  }
}
ab <- do.call(rbind, ab)
write.csv(ab, file.path(args$out_dir, "ablation.csv"), row.names = FALSE)
lines <- c(lines, "3. Ablation (regAbcrf on soft rows; pos = position in the table's prior range, fit scale)",
           capture.output(print(ab, row.names = FALSE, digits = 3)))
writeLines(lines, file.path(args$out_dir, "misfit_diag.txt"))
cat(lines, sep = "\n")
