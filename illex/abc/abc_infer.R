# ABC random-forest inference for the Illex chr2 trajectory ABC (Task 8). Inference only: definitions (fit keys,
# parameters, prior bounds) come from spec.json, dumped from illex/abc/constants.py by
# `python -m illex.abc.abc_report spec`; formatting is done by `abc_report.py report`.
#
#   Rscript illex/abc/abc_infer.R --table T.parquet --acceptance A.csv --observed abc_observed.json \
#       --spec spec.json --out-dir results/illex/abc_rf [--ntree 1000] [--seed 1] [--ncores 16]
#       [--n-pods 200] [--folds 5] [--n-draws 1000] [--abc-accept 100] [--sample-fraction 0.2]
#
# Model unit = (model, flux); hard/soft are classes within a model, each simulated separately. A row of class c gets
# mixture weight  0.5 * acc_c * w_est / sum_{c} w_est  (hard rows carry the establishment importance weight w_est,
# soft rows w_est = 1). Every forest trains on the UNIQUE rows with these weights passed to ranger as case.weights
# (abcrf 2.0 forwards `...` to ranger in abcrf(), regAbcrf() and the post.prob error forest of predict.abcrf()), so
# no row is duplicated and OOB is honest; each model's weights sum to 1, so models are balanced. prob_total(M) ∝
# prior(M) * acc(M) * prob_stats(M), acc(M) = 0.5 acc_hard + 0.5 acc_soft, prior uniform over the models present.
# Weighted forests subsample each tree to RF_SAMPLE_FRACTION of the rows (abcrf `sampsize`): with the full bootstrap
# (fraction 1) a row's in-bag probability 1 - exp(-n w_i) saturates, so fully grown trees under-weight heavy rows
# (noise test: weighted class mass 0.25 is recovered as 0.35-0.38 at fraction 1, 0.26-0.28 at 0.1-0.3).
# The LDA axes abcrf adds (lda = TRUE) are fitted unweighted. Functions are sourceable (tests); main() runs under
# Rscript.

suppressPackageStartupMessages({
  library(abcrf)
  library(jsonlite)
})

# ---------------------------------------------------------------------------------------------------- weighting --
mixture_weights <- function(df, acc) {
  # df: cls ("hard"/"soft"), w_est; acc: named c(hard=, soft=). Returns weights summing to 1.
  w <- numeric(nrow(df))
  for (c in unique(df$cls)) {
    i <- df$cls == c
    we <- ifelse(is.na(df$w_est[i]), 1, df$w_est[i])
    w[i] <- 0.5 * acc[[c]] * we / sum(we)
  }
  w / sum(w)
}

resample_rows <- function(w, n, seed) {
  set.seed(seed)
  sample.int(length(w), n, replace = TRUE, prob = w)
}

ess <- function(w) sum(w)^2 / sum(w^2)

hard_share_acc <- function(p, acc_h, acc_s) acc_h * p / (acc_h * p + acc_s * (1 - p))

combine_total <- function(prior, acc, prob_stats) {
  x <- prior * acc[names(prior)] * prob_stats[names(prior)]
  x / sum(x)
}

drop_constant <- function(d, stats) stats[vapply(stats, function(k) {
  v <- d[[k]]; length(unique(v[is.finite(v)])) > 1
}, logical(1))]

# ------------------------------------------------------------------------------------------------- model choice --
RF_SAMPLE_FRACTION <- 0.2   # per-tree subsample fraction for case-weighted forests (main() sets it from --sample-fraction)
ss <- function(n) max(10L, as.integer(ceiling(RF_SAMPLE_FRACTION * n)))

fit_model_choice <- function(train, ntree = 1000, ncores = 1, resp = "model", case_w = NULL) {
  train[[resp]] <- droplevels(factor(train[[resp]]))
  f <- as.formula(paste(resp, "~ ."))
  if (is.null(case_w))
    abcrf(f, data = train, lda = nlevels(train[[resp]]) > 1, ntree = ntree, paral = ncores > 1, ncores = ncores)
  else
    abcrf(f, data = train, lda = nlevels(train[[resp]]) > 1, ntree = ntree, paral = ncores > 1, ncores = ncores,
          sampsize = ss(nrow(train)), case.weights = case_w)
}

reg_rf <- function(d, ntree, ncores, case_w = NULL) {
  if (is.null(case_w)) regAbcrf(y ~ ., data = d, ntree = ntree, paral = ncores > 1, ncores = ncores)
  else regAbcrf(y ~ ., data = d, ntree = ntree, paral = ncores > 1, ncores = ncores, sampsize = ss(nrow(d)),
                case.weights = case_w)
}

predict_mc <- function(fit, obs, train, ntree, ncores, case_w = NULL) {
  if (is.null(case_w)) predict(fit, obs, training = train, ntree = ntree, paral = ncores > 1, ncores = ncores)
  else predict(fit, obs, training = train, ntree = ntree, sampsize = ss(nrow(train)), paral = ncores > 1,
               ncores = ncores, case.weights = case_w)
}

model_probs <- function(fit, train, obs, ncores = 1, resp = "model", ntree = 1000, case_w = NULL) {
  # prob_stats: abcrf's posterior probability for the MAP model; the remaining mass split across the other models in
  # proportion to regAbcrf estimates of P(model = M | stats) (indicator regression), which are also returned.
  train[[resp]] <- droplevels(factor(train[[resp]]))
  pr <- predict_mc(fit, obs, train, ntree, ncores, case_w)
  levs <- levels(train[[resp]])
  map <- as.character(pr$allocation[1])
  vote <- setNames(as.numeric(pr$vote[1, levs]), levs) / sum(pr$vote[1, ])
  q <- setNames(numeric(length(levs)), levs)
  stats <- setdiff(names(train), resp)
  for (m in levs) {
    d <- data.frame(y = as.numeric(train[[resp]] == m), train[, stats, drop = FALSE])
    rf <- reg_rf(d, ntree, ncores, case_w)
    q[m] <- predict(rf, obs, training = d, paral = ncores > 1, ncores = ncores)$expectation[1]
  }
  q <- pmax(q, 0); q <- q / sum(q)
  ps <- setNames(numeric(length(levs)), levs)
  ps[map] <- pr$post.prob[1]
  others <- setdiff(levs, map)
  if (length(others)) {
    qo <- q[others]
    ps[others] <- (1 - pr$post.prob[1]) * (if (sum(qo) > 0) qo / sum(qo) else rep(1 / length(qo), length(qo)))
  }
  list(map = map, post_prob_map = pr$post.prob[1], prob_stats = ps, vote = vote, prob_indicator = q)
}

confusion_df <- function(true, pred, w = NULL) {
  # n = unique rows; w = summed mixture weight (each true model's rows sum to 1 under the per-model weights)
  if (is.null(w)) w <- rep(1, length(true))
  ok <- !is.na(pred)
  lev <- levels(factor(true))
  tf <- factor(true[ok], levels = lev); pf <- factor(pred[ok], levels = lev)
  n <- as.data.frame(table(true = tf, predicted = pf), responseName = "n")
  ws <- tapply(w[ok], list(tf, pf), sum); ws[is.na(ws)] <- 0
  n$w <- ws[cbind(as.character(n$true), as.character(n$predicted))]
  n
}

accuracy <- function(conf, col = "n") sum(conf[[col]][conf$true == conf$predicted]) / sum(conf[[col]])

grouped_folds <- function(src_id, folds, seed) {
  u <- unique(src_id)
  set.seed(seed)
  f <- setNames(sample(rep_len(seq_len(folds), length(u))), u)
  unname(f[as.character(src_id)])
}

cv_confusion <- function(train, src_id, folds = 5, ntree = 1000, ncores = 1, seed = 1, resp = "model",
                         case_w = NULL) {
  # Grouped K-fold: all rows sharing a source id are in one fold (rows are unique here, so it is plain K-fold).
  fold <- grouped_folds(src_id, folds, seed)
  pred <- character(nrow(train))
  for (k in seq_len(folds)) {
    te <- fold == k
    cwk <- if (is.null(case_w)) NULL else case_w[!te]
    fit <- fit_model_choice(train[!te, ], ntree = ntree, ncores = ncores, resp = resp, case_w = cwk)
    p <- predict_mc(fit, train[te, setdiff(names(train), resp), drop = FALSE], droplevels(train[!te, ]), ntree,
                    ncores, cwk)
    pred[te] <- as.character(p$allocation)
  }
  confusion_df(train[[resp]], pred, case_w)
}

# --------------------------------------------------------------------------------------------------- parameters --
param_frame <- function(d, param, stats, log) {
  y <- d[[param]]
  if (log) y <- log10(y)
  data.frame(y = y, d[, stats, drop = FALSE])
}

QS <- c(0.025, 0.25, 0.5, 0.75, 0.975)

fit_param <- function(d, param, stats, log, obs, ntree = 1000, ncores = 1, case_w = NULL) {
  pf <- param_frame(d, param, stats, log)
  rf <- reg_rf(pf, ntree, ncores, case_w)
  pr <- predict(rf, obs, training = pf, quantiles = QS, paral = ncores > 1, ncores = ncores, rf.weights = TRUE)
  bt <- if (log) function(x) 10^x else identity
  q <- bt(as.numeric(pr$quantiles[1, ]))
  list(q025 = q[1], q25 = q[2], median = q[3], q75 = q[4], q975 = q[5], mean_fit = pr$expectation[1],
       sd_fit = sqrt(pr$variance[1]), weights = as.numeric(pr$weights[, 1]))
}

coverage_cv <- function(d, src_id, param, stats, log, n_pods = 200, folds = 5, ntree = 1000, ncores = 1, seed = 1,
                        case_w = NULL) {
  # Pods: n_pods rows with distinct source ids, drawn without replacement in proportion to case_w (uniform if NULL),
  # i.e. from the weighted reference distribution. Grouped K-fold: each pod is predicted by a forest trained without
  # its source row.
  set.seed(seed)
  first <- which(!duplicated(src_id))
  pw <- if (is.null(case_w)) NULL else case_w[first]
  pods <- first[sample.int(length(first), min(n_pods, length(first)), prob = pw)]
  fold <- grouped_folds(src_id, folds, seed)
  pf <- param_frame(d, param, stats, log)
  c50 <- c95 <- logical(0)
  for (k in seq_len(folds)) {
    pk <- pods[fold[pods] == k]
    if (!length(pk)) next
    tr <- pf[fold != k, ]
    rf <- reg_rf(tr, ntree, ncores, if (is.null(case_w)) NULL else case_w[fold != k])
    pr <- predict(rf, pf[pk, stats, drop = FALSE], training = tr, quantiles = QS, paral = ncores > 1,
                  ncores = ncores)
    q <- pr$quantiles; y <- pf$y[pk]
    c95 <- c(c95, y >= q[, 1] & y <= q[, 5])
    c50 <- c(c50, y >= q[, 2] & y <= q[, 4])
  }
  list(n_pods = length(c95), cover50 = mean(c50), cover95 = mean(c95))
}

wmedian <- function(x, w = NULL) {
  if (is.null(w)) w <- rep(1, length(x))
  o <- order(x); x <- x[o]; w <- w[o]
  x[which(cumsum(w) / sum(w) >= 0.5)[1]]
}

abc_crosscheck <- function(d, param, stats, log, obs, n_accept = 100, case_w = NULL) {
  # tol = n_accept / n, so a fixed number of rows is accepted; the accepted rows' medians are weighted by case_w.
  pf <- param_frame(d, param, stats, log)
  if (is.null(case_w)) case_w <- rep(1, nrow(pf))
  tol <- min(1, n_accept / nrow(pf))
  bt <- if (log) function(x) 10^x else identity
  out <- list(tol = tol, rejection = NA_real_, loclinear = NA_real_, n_accept = NA_integer_, note = "")
  run <- function(method) suppressWarnings(abc::abc(target = unlist(obs[1, stats]), param = data.frame(y = pf$y),
                                                     sumstat = as.matrix(pf[, stats]), tol = tol, method = method))
  r <- tryCatch(run("rejection"), error = function(e) e)
  if (inherits(r, "error")) { out$note <- conditionMessage(r); return(out) }
  out$rejection <- bt(wmedian(r$unadj.values[, 1], case_w[r$region])); out$n_accept <- nrow(r$unadj.values)
  l <- tryCatch(run("loclinear"), error = function(e) e)
  if (inherits(l, "error")) out$note <- conditionMessage(l)
  else out$loclinear <- bt(wmedian(l$adj.values[, 1], l$weights * case_w[l$region]))
  out
}

# --------------------------------------------------------------------------------------------------------- main --
parse_args <- function(argv) {
  a <- list(ntree = 1000, seed = 1, ncores = 16, n_pods = 200, folds = 5, n_draws = 1000, abc_accept = 100,
            sample_fraction = 0.2)
  i <- 1
  while (i <= length(argv)) {
    k <- gsub("-", "_", sub("^--", "", argv[i])); a[[k]] <- argv[i + 1]; i <- i + 2
  }
  for (k in c("ntree", "seed", "ncores", "n_pods", "folds", "n_draws", "abc_accept", "sample_fraction"))
    a[[k]] <- as.numeric(a[[k]])
  a
}

DRAWS_NOTE <- paste0("# NOTE: joint draws = rows resampled by RF weights averaged over this model's all-row parameter ",
                     "forests; their soft-only marginals (p0, t_sel) need not match the per-parameter regAbcrf ",
                     "posteriors in abc_posteriors.csv. Task 8b SMC draws replace these downstream. Read with ",
                     "comment='#'.")

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  t0 <- Sys.time()
  options(warn = 1)
  a <- parse_args(argv)
  RF_SAMPLE_FRACTION <<- a$sample_fraction
  dir.create(a$out_dir, recursive = TRUE, showWarnings = FALSE)
  spec <- fromJSON(a$spec, simplifyVector = FALSE)
  file.copy(a$spec, file.path(a$out_dir, "spec_used.json"), overwrite = TRUE)
  keys <- unlist(spec$fit_keys)
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

  # per-model acceptance and mixture weights (case.weights; each model's sum to 1)
  info <- list()
  tab$cw <- NA_real_
  for (j in seq_along(units)) {
    u <- units[j]; i <- tab$unit == u; d <- tab[i, ]
    m <- d$model[1]; fl <- d$flux[1]
    acc <- c(hard = NA_real_, soft = NA_real_)
    for (c in c("hard", "soft")) {
      r <- accf[accf$model == m & accf$flux == fl & accf$cls == c, ]
      if (nrow(r)) acc[[c]] <- r$acceptance[1]
    }
    if (any(is.na(acc[unique(d$cls)]))) stop("no acceptance for ", u)
    w <- mixture_weights(d, acc)
    tab$cw[i] <- w
    acc0 <- ifelse(is.na(acc), 0, acc)   # a class absent from the table counts as acceptance 0
    info[[u]] <- data.frame(model = m, flux = fl, n_rows = nrow(d), n_hard = sum(d$cls == "hard"),
                            n_soft = sum(d$cls == "soft"),
                            acc_hard = acc0[["hard"]], acc_soft = acc0[["soft"]],
                            acc = 0.5 * acc0[["hard"]] + 0.5 * acc0[["soft"]],
                            mix_hard = sum(w[d$cls == "hard"]), ess_mix = ess(w),
                            ess_hard = if (any(d$cls == "hard")) ess(d$w_est[d$cls == "hard"]) else 0,
                            ess_soft = sum(d$cls == "soft"))
  }
  info <- do.call(rbind, info)
  stats <- drop_constant(tab, keys)
  if (length(setdiff(keys, stats))) cat("constant statistics dropped:", setdiff(keys, stats), "\n")
  obs_s <- obs[, stats, drop = FALSE]
  write.csv(dropped, file.path(a$out_dir, "dropped_raw.csv"), row.names = FALSE)

  # ---- model choice (unique rows, case.weights = mixture weights, models balanced)
  mtab <- data.frame(model = factor(tab$unit), tab[, stats])
  prior <- setNames(rep(1 / length(units), length(units)), units)
  acc_oob <- acc_cv <- NA_real_
  if (length(units) > 1) {
    cat("model choice ...\n")
    fit <- fit_model_choice(mtab, ntree = a$ntree, ncores = a$ncores, case_w = tab$cw)
    mp <- model_probs(fit, mtab, obs_s, ncores = a$ncores, ntree = a$ntree, case_w = tab$cw)
    oob <- confusion_df(mtab$model, as.character(fit$model.rf$predictions), tab$cw)
    cat("grouped CV confusion ...\n")
    cvc <- cv_confusion(mtab, tab$src, folds = a$folds, ntree = a$ntree, ncores = a$ncores, seed = a$seed,
                        case_w = tab$cw)
    write.csv(oob, file.path(a$out_dir, "confusion_oob.csv"), row.names = FALSE)
    write.csv(cvc, file.path(a$out_dir, "confusion_cv.csv"), row.names = FALSE)
    acc_oob <- accuracy(oob, "w"); acc_cv <- accuracy(cvc, "w")
    ps <- mp$prob_stats
  } else {
    mp <- list(map = units, post_prob_map = 1, vote = setNames(1, units), prob_indicator = setNames(1, units))
    ps <- setNames(1, units)
  }
  accM <- setNames(info$acc, units)
  pt <- combine_total(prior, accM, ps)
  mc <- data.frame(model = info$model, flux = info$flux, prior = prior[units], acc_hard = info$acc_hard,
                   acc_soft = info$acc_soft, acc = info$acc, vote = mp$vote[units],
                   prob_indicator = mp$prob_indicator[units], prob_stats = ps[units], prob_total = pt[units],
                   map_stats = units == mp$map, n_rows = info$n_rows, n_hard = info$n_hard, n_soft = info$n_soft,
                   ess_mix = info$ess_mix, ess_hard = info$ess_hard, ess_soft = info$ess_soft,
                   mix_hard = info$mix_hard)
  write.csv(mc, file.path(a$out_dir, "modelchoice_raw.csv"), row.names = FALSE)

  # ---- hard share, parameters, coverage, cross-check, draws
  hs <- list(); post <- list(); cov <- list(); xc <- list()
  for (j in seq_along(units)) {
    u <- units[j]; d <- tab[tab$unit == u, ]; ir <- info[j, ]
    cat("model", u, "...\n")
    if (all(c("hard", "soft") %in% d$cls)) {
      # class-balanced weights: each class sums to 0.5 (hard rows by w_est), i.e. P(hard) = 0.5 without acceptance
      bw <- mixture_weights(d, c(hard = 1, soft = 1))
      bt <- data.frame(cls = factor(d$cls), d[, stats])
      cf <- fit_model_choice(bt, ntree = a$ntree, ncores = a$ncores, resp = "cls", case_w = bw)
      cp <- predict_mc(cf, obs_s, bt, a$ntree, a$ncores, bw)
      p_hard <- if (as.character(cp$allocation[1]) == "hard") cp$post.prob[1] else 1 - cp$post.prob[1]
      ind <- data.frame(y = as.numeric(d$cls == "hard"), d[, stats])
      rf <- reg_rf(ind, a$ntree, a$ncores, d$cw)
      p_mix <- predict(rf, obs_s, training = ind, paral = a$ncores > 1, ncores = a$ncores)$expectation[1]
      co <- confusion_df(as.character(bt$cls), as.character(cf$model.rf$predictions), bw)
      hs[[u]] <- data.frame(model = d$model[1], flux = d$flux[1], n_hard = ir$n_hard, n_soft = ir$n_soft,
                            ess_hard = ir$ess_hard, hard_share_stats = p_hard,
                            hard_share_total = hard_share_acc(p_hard, ir$acc_hard, ir$acc_soft),
                            hard_share_mix = p_mix, prior_mix_hard = ir$mix_hard,
                            class_oob_error = 1 - accuracy(co, "w"))
    }
    wsum <- numeric(nrow(d)); nw <- 0
    for (p in names(spec$params)) {
      sp <- spec$params[[p]]
      if (!(d$model[1] %in% unlist(sp$models))) next
      if (!is.null(sp$flux) && d$flux[1] != sp$flux) next
      sel <- if (sp$rows == "soft") d$cls == "soft" else rep(TRUE, nrow(d))
      sel <- sel & is.finite(d[[p]]) & (!sp$log | d[[p]] > 0)
      pd <- d[sel, ]
      if (nrow(pd) < 20 || length(unique(pd[[p]])) < 2) next
      cw <- pd$cw / sum(pd$cw)
      fp <- fit_param(pd, p, stats, sp$log, obs_s, ntree = a$ntree, ncores = a$ncores, case_w = cw)
      post[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p,
                                        scale = if (sp$log) "log10" else "linear", rows = sp$rows,
                                        n_train = nrow(pd), ess_train = ess(cw), median = fp$median,
                                        q025 = fp$q025, q25 = fp$q25, q75 = fp$q75, q975 = fp$q975,
                                        mean_fitscale = fp$mean_fit, sd_fitscale = fp$sd_fit,
                                        prior_lo = sp$lo, prior_hi = sp$hi)
      if (sp$rows == "all") { wsum[sel] <- wsum[sel] + fp$weights; nw <- nw + 1 }
      cv <- coverage_cv(pd, pd$src, p, stats, sp$log, n_pods = a$n_pods, folds = a$folds, ntree = a$ntree,
                        ncores = a$ncores, seed = a$seed + j, case_w = cw)
      cov[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p, n_pods = cv$n_pods,
                                       cover50 = cv$cover50, cover95 = cv$cover95)
      x <- abc_crosscheck(pd, p, stats, sp$log, obs_s, a$abc_accept, cw)
      xc[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p, tol = x$tol,
                                      n_accept = x$n_accept, rf_median = fp$median, rejection_median = x$rejection,
                                      loclinear_median = x$loclinear, note = x$note)
    }
    # joint draws: rows resampled by the RF weights averaged over this model's all-row parameter forests
    set.seed(a$seed + 200 + j)
    di <- sample.int(nrow(d), a$n_draws, replace = TRUE, prob = wsum / nw)
    pcols <- c("model", "flux", "cls", "t_inv", "hard", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff", "src")
    fn <- file.path(a$out_dir, sprintf("abc_post_%s.csv", u))
    writeLines(DRAWS_NOTE, fn)
    suppressWarnings(write.table(d[di, intersect(pcols, names(d))], fn, sep = ",", row.names = FALSE,
                                 append = TRUE, qmethod = "double"))
  }
  write.csv(do.call(rbind, post), file.path(a$out_dir, "posteriors_raw.csv"), row.names = FALSE)
  write.csv(do.call(rbind, cov), file.path(a$out_dir, "coverage_raw.csv"), row.names = FALSE)
  write.csv(do.call(rbind, xc), file.path(a$out_dir, "crosscheck_raw.csv"), row.names = FALSE)
  hsd <- if (length(hs)) do.call(rbind, hs) else data.frame(model = character(), flux = numeric(),
    hard_share_stats = numeric(), hard_share_total = numeric(), hard_share_mix = numeric())
  write.csv(hsd, file.path(a$out_dir, "hardshare_raw.csv"), row.names = FALSE)
  meta <- list(table = a$table, spec = a$spec, n_rows_in = n0, n_dropped_nan = sum(drop_nan),
               n_dropped_na_key = n_dropped_na, models = paste(units, collapse = " "),
               training = "unique rows, ranger case.weights = mixture weights",
               accuracy_oob_weighted = acc_oob, accuracy_grouped_cv_weighted = acc_cv,
               sample_fraction = a$sample_fraction, ntree = a$ntree, seed = a$seed, folds = a$folds, n_pods = a$n_pods, abc_accept = a$abc_accept,
               n_stats = length(stats), stats_dropped_constant = paste(setdiff(keys, stats), collapse = " "),
               R = R.version.string, abcrf = as.character(packageVersion("abcrf")),
               ranger = as.character(packageVersion("ranger")), abc = as.character(packageVersion("abc")),
               arrow = as.character(packageVersion("arrow")),
               runtime_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  write_json(meta, file.path(a$out_dir, "run_meta.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("done in %.1f s\n", meta$runtime_s))
}

if (sys.nframe() == 0L) main()
