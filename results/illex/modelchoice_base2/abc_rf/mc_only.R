source("illex/abc/abc_infer.R (branch illex/baseline-refit)")
mc_main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  t0 <- Sys.time()
  options(warn = 1)
  a <- parse_args(argv)
  MC_MIN_NODE_SIZE <<- a$mc_min_node_size
  dir.create(a$out_dir, recursive = TRUE, showWarnings = FALSE)
  spec <- fromJSON(a$spec, simplifyVector = FALSE)
  file.copy(a$spec, file.path(a$out_dir, "spec_used.json"), overwrite = TRUE)
  keys <- unlist(spec$fit_keys)
  if (is.null(spec$p_hard)) stop("spec lacks p_hard")
  p_hard <- as.numeric(spec$p_hard)
  tab <- as.data.frame(arrow::read_parquet(a$table))
  accf <- read.csv(a$acceptance)
  obsj <- fromJSON(a$observed)
  obs_all <- c(obsj$stats, list(p_obs = obsj$p_obs))
  missing_obs <- setdiff(keys, names(obs_all))
  if (length(missing_obs)) stop("observed vector lacks: ", paste(missing_obs, collapse = ","))
  obs <- as.data.frame(lapply(obs_all[keys], as.numeric))

  tab$unit <- paste(tab$model, tab$flux, sep = "_")
  n0 <- nrow(tab)
  drop_nan <- tab$n_nan > 0
  dropped <- aggregate(list(n_dropped_nan = drop_nan), by = list(unit = tab$unit, cls = tab$cls), FUN = sum)
  tab <- tab[!drop_nan, ]
  bad <- !stats::complete.cases(tab[, keys])
  n_dropped_na <- sum(bad)
  tab <- tab[!bad, ]
  tab$src <- seq_len(nrow(tab))
  units <- sort(unique(tab$unit))
  cat(sprintf("rows %d; dropped n_nan>0: %d; dropped NA in fit keys: %d; models: %s\n", n0, sum(drop_nan),
              n_dropped_na, paste(units, collapse = " ")))

  # per-model acceptance and mixture weights (each unit's cw sums to 1)
  info <- list()
  tab$cw <- NA_real_
  for (j in seq_along(units)) {
    u <- units[j]; i <- tab$unit == u; d <- tab[i, ]
    m <- d$model[1]; fl <- d$flux[1]
    acc <- lookup_acc(accf, m, fl)
    if (any(is.na(acc[unique(d$cls)]))) stop("no acceptance for ", u)
    for (c in c("hard", "soft")) if (!is.na(acc[[c]]) && acc[[c]] > 0 && !any(d$cls == c))
      stop(sprintf("%s: class %s has acceptance %g but no usable table rows", u, c, acc[[c]]))
    lo <- lookup_acc(accf, m, fl, "ci_lo"); hi <- lookup_acc(accf, m, fl, "ci_hi")
    w <- mixture_weights(d, acc, p_hard)
    tab$cw[i] <- w
    acc0 <- ifelse(is.na(acc), 0, acc)   # a class absent from both the CSV and the table: acceptance 0
    lo0 <- ifelse(is.na(lo), acc0, lo); hi0 <- ifelse(is.na(hi), acc0, hi)
    info[[u]] <- data.frame(model = m, flux = fl, n_rows = nrow(d), n_hard = sum(d$cls == "hard"),
                            n_soft = sum(d$cls == "soft"),
                            acc_hard = acc0[["hard"]], acc_soft = acc0[["soft"]],
                            acc = p_hard * acc0[["hard"]] + (1 - p_hard) * acc0[["soft"]],
                            acc_lo = p_hard * lo0[["hard"]] + (1 - p_hard) * lo0[["soft"]],
                            acc_hi = p_hard * hi0[["hard"]] + (1 - p_hard) * hi0[["soft"]],
                            mix_hard = sum(w[d$cls == "hard"]), ess_mix = ess(w),
                            ess_hard = if (any(d$cls == "hard")) ess(d$w_est[d$cls == "hard"]) else 0,
                            ess_soft = sum(d$cls == "soft"))
  }
  info <- do.call(rbind, info)
  stats <- drop_constant(tab, keys)
  if (length(setdiff(keys, stats))) cat("constant statistics dropped:", setdiff(keys, stats), "\n")
  obs_s <- obs[, stats, drop = FALSE]
  write.csv(dropped, file.path(a$out_dir, "dropped_raw.csv"), row.names = FALSE)

  # ---- model choice
  mtab <- data.frame(model = factor(tab$unit), tab[, stats])
  prior <- setNames(rep(1 / length(units), length(units)), units)
  acc_cv <- acc_oob_unw <- NA_real_
  vote <- setNames(rep(NA_real_, length(units)), units)
  if (length(units) > 1) {
    cat("model choice ...\n")
    mp <- model_probs(mtab, obs_s, tab$cw, ncores = a$ncores, ntree = a$ntree)
    ps <- mp$prob_stats[1, ]
    fit <- mp$fit
    pv <- predict(fit, obs_s, training = mtab, ntree = a$ntree, paral = a$ncores > 1, ncores = a$ncores)$vote
    vote <- setNames(as.numeric(pv[1, ]) / sum(pv[1, ]), colnames(pv))[units]
    oob <- confusion_df(mtab$model, as.character(fit$model.rf$predictions), tab$cw)
    cat("grouped CV confusion ...\n")
    cvc <- cv_confusion(mtab, tab$src, tab$cw, folds = a$folds, ntree = a$ntree, ncores = a$ncores, seed = a$seed)
    write.csv(oob, file.path(a$out_dir, "confusion_oob.csv"), row.names = FALSE)
    write.csv(cvc, file.path(a$out_dir, "confusion_cv.csv"), row.names = FALSE)
    acc_oob_unw <- accuracy(oob, "w"); acc_cv <- accuracy(cvc, "w")
  } else {
    ps <- setNames(1, units); mp <- list(map = units)
  }
  accM <- setNames(info$acc, units)
  pt <- combine_total(prior, accM, ps)
  out <- data.frame(unit = units, prob_stats = ps[units], vote = vote[units], acc = accM, prob_total = pt[units])
  print(out); write.csv(out, file.path(a$out_dir, "model_probs.csv"), row.names = FALSE)
  write.csv(info, file.path(a$out_dir, "unit_info.csv"), row.names = FALSE)
  cat(sprintf("accuracy CV %.3f OOB %.3f; %.0f s\n", acc_cv, acc_oob_unw, as.numeric(Sys.time() - t0, units = "secs")))
}
mc_main(c('--table','/sietch_colab/ssmall/projects/msinv_dir/msinv_pboot_wt/results/illex/modelchoice_base2/abc_table.parquet','--acceptance','/sietch_colab/ssmall/projects/msinv_dir/msinv_pboot_wt/results/illex/modelchoice_base2/abc_acceptance.csv','--observed','/sietch_colab/ssmall/projects/msinv_dir/msinv_pboot_wt/results/illex/abc_observed_ctl2.json','--spec','/sietch_colab/ssmall/projects/msinv_dir/msinv_pboot_wt/results/illex/modelchoice_base2/spec.json','--out-dir','/sietch_colab/ssmall/projects/msinv_dir/msinv_pboot_wt/results/illex/modelchoice_base2/abc_rf','--ncores','16'))
