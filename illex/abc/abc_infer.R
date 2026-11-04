# ABC random-forest inference for the Illex chr2 trajectory ABC (Task 8). Inference only: definitions (fit keys,
# parameters, prior bounds, p_hard) come from spec.json, dumped from illex/abc/constants.py by
# `python -m illex.abc.abc_report spec`; formatting is done by `abc_report.py report`.
#
#   Rscript illex/abc/abc_infer.R --table T.parquet --acceptance A.csv --observed abc_observed.json \
#       --spec spec.json --out-dir results/illex/abc_rf [--ntree 1000] [--seed 1] [--ncores 16]
#       [--n-pods 200] [--folds 5] [--n-draws 1000] [--abc-accept 100] [--param-sample-fraction 1]
#       [--mc-min-node-size 50]
#
# Model unit = (model, flux); hard/soft are classes within a model, each simulated separately. A row of class c gets
# mixture weight  P(c) * acc_c * w_est / sum_{c} w_est  (P(hard) = spec p_hard; hard rows carry the establishment
# importance weight w_est, soft rows w_est = 1), normalised to 1 per unit.
#
# Importance weights are applied POST HOC (fix round 2): every forest is grown UNWEIGHTED on the unique rows; at the
# point of interest the forest's weights over the training rows (abcrf findweights, the quantile-forest kernel) are
# multiplied by the mixture weights cw and renormalised. That applies w_est and the class acceptances exactly at any
# class mass (ranger case.weights could not: with ~1 hard row per tree a 1% class was over-weighted ~300x at the
# observed point). Uses:
#   prob_stats(M)   P(model = M | stats) from the abcrf model-choice classifier (ntree trees, its LDA axes appended as
#                   abcrf builds them): its own forest kernel at the observed point (per tree, in-bag counts of the
#                   training rows sharing the observation's leaf, normalised; averaged over trees) x cw, each unit's
#                   cw summing to 1 (balanced models); one kernel, so the model probabilities sum to 1 by
#                   construction (fix round 3, ruling N1). prob_total(M) ∝ prior(M) * acc(M) * prob_stats(M),
#                   acc(M) = p_hard acc_hard + (1 - p_hard) acc_soft.
#   hard share      kernel of an unweighted forest on the hard indicator within the model, x class-balanced weights
#                   (noacc, prior p_hard) or x cw (acc).
#   parameters      regAbcrf per parameter per model (per-tree subsample --param-sample-fraction); posterior =
#                   kernel x cw; weighted quantiles.
#   joint draws     rows resampled by the posterior weights averaged over the model's all-row parameter forests.
#   confusion       grouped K-fold CV of the argmax of prob_stats (headline) and, for reference, the plain OOB
#                   allocation of the abcrf classifier (unweighted; LDA fitted on all rows: slightly optimistic).
# abcrf's own post.prob is an aggregate of an error forest and cannot be reweighted post hoc, so it is not used.
# A kernel whose whole mass falls on zero-weight rows is an error (posthoc / model_probs stop), never a silent NA.
# Functions are sourceable (tests); main() runs under Rscript.

suppressPackageStartupMessages({
  library(abcrf)
  library(jsonlite)
})

# ---------------------------------------------------------------------------------------------------- weighting --
mixture_weights <- function(df, acc, p_hard = 0.5) {
  # df: cls ("hard"/"soft"), w_est; acc: named c(hard=, soft=). Returns weights summing to 1.
  prior <- c(hard = p_hard, soft = 1 - p_hard)
  w <- numeric(nrow(df))
  for (c in unique(df$cls)) {
    i <- df$cls == c
    we <- ifelse(is.na(df$w_est[i]), 1, df$w_est[i])
    w[i] <- prior[[c]] * acc[[c]] * we / sum(we)
  }
  w / sum(w)
}

ess <- function(w) sum(w)^2 / sum(w^2)

# p = P(hard | stats) without acceptance (any class prior); returns P(hard | stats, accepted). The class prior cancels.
hard_share_acc <- function(p, acc_h, acc_s) acc_h * p / (acc_h * p + acc_s * (1 - p))

combine_total <- function(prior, acc, prob_stats) {
  x <- prior * acc[names(prior)] * prob_stats[names(prior)]
  x / sum(x)
}

drop_constant <- function(d, stats) stats[vapply(stats, function(k) {
  v <- d[[k]]; length(unique(v[is.finite(v)])) > 1
}, logical(1))]

posthoc <- function(alpha, cw) {
  # alpha: n_train x n_obs forest weights; cw: n_train importance weights. Columns renormalised to 1.
  a <- alpha * cw
  s <- colSums(a)
  if (any(!(s > 0)))
    stop(sprintf("posthoc: the forest kernel of %d point(s) falls entirely on zero-weight rows", sum(!(s > 0))))
  sweep(a, 2, s, "/")
}

wquantile <- function(y, w, qs) {
  o <- order(y); y <- y[o]; cw <- cumsum(w[o]) / sum(w)
  vapply(qs, function(q) y[min(which(cw >= q - 1e-12))], numeric(1))
}

# ------------------------------------------------------------------------------------------------ forest kernel --
reg_rf <- function(d, ntree, ncores, sample_fraction = 1)
  regAbcrf(y ~ ., data = d, ntree = ntree, paral = ncores > 1, ncores = ncores,
           sampsize = max(10L, as.integer(ceiling(sample_fraction * nrow(d)))))

rf_kernel <- function(rf, newdata, training, ncores = 1) {
  # n_train x n_new forest weights (abcrf findweights over in-bag counts; each column sums to 1)
  w <- predict(rf, newdata, training = training, paral = ncores > 1, ncores = ncores, rf.weights = TRUE)$weights
  as.matrix(w)
}

indicator_share <- function(stats_df, is_target, cw, newdata, ntree = 1000, ncores = 1, rf = NULL) {
  # P(target | stats) at each row of newdata: unweighted forest on 1[target], kernel x cw. Returns list(p, rf).
  d <- data.frame(y = as.numeric(is_target), stats_df)
  if (is.null(rf)) rf <- reg_rf(d, ntree, ncores)
  a <- posthoc(rf_kernel(rf, newdata, d, ncores), cw)
  list(p = as.numeric(colSums(a[is_target, , drop = FALSE])), rf = rf)
}

# ------------------------------------------------------------------------------------------------- model choice --
MC_MIN_NODE_SIZE <- 50  # ranger min.node.size of the abcrf model-choice classifier (--mc-min-node-size; abcrf default 1).
# 50 (fix round 3): with pure leaves (1) the classifier kernel is a vote share, high-variance and over-confident
# (1-D Gaussian test, 6 datasets: RMSE 0.196 at 1, 0.097 at 50); v3 grouped-CV accuracy 0.534 at 1, 0.542 at 50.

fit_model_choice <- function(train, ntree = 1000, ncores = 1, resp = "model", min_node_size = MC_MIN_NODE_SIZE) {
  train[[resp]] <- droplevels(factor(train[[resp]]))
  abcrf(as.formula(paste(resp, "~ .")), data = train, lda = nlevels(train[[resp]]) > 1, ntree = ntree,
        paral = ncores > 1, ncores = ncores, min.node.size = min_node_size)
}

class_kernel <- function(fit, train, newdata, ncores = 1, resp = "model") {
  # Kernel of the abcrf classifier at each row of newdata (n_train x n_new; columns sum to 1): the LDA axes are
  # appended as abcrf builds them; per tree, the in-bag counts (keep.inbag) of the training rows that share the new
  # point's leaf, normalised; averaged over trees.
  st <- train[, setdiff(names(train), resp), drop = FALSE]
  xtr <- if (fit$lda) cbind(st, predict(fit$model.lda, st)$x) else st
  xnew <- if (fit$lda) cbind(newdata, predict(fit$model.lda, newdata)$x) else newdata
  tn_tr <- predict(fit$model.rf, xtr, type = "terminalNodes", num.threads = ncores)$predictions
  tn_new <- matrix(predict(fit$model.rf, xnew, type = "terminalNodes", num.threads = ncores)$predictions,
                   nrow = nrow(newdata))
  inb <- do.call(cbind, fit$model.rf$inbag.counts)
  if (is.null(inb) || !all(dim(inb) == dim(tn_tr))) stop("class_kernel: abcrf forest lacks in-bag counts")
  K <- matrix(0, nrow(st), nrow(newdata))
  for (j in seq_len(nrow(newdata))) {
    w <- inb * (tn_tr == rep(tn_new[j, ], each = nrow(tn_tr)))
    cs <- colSums(w)
    K[, j] <- rowSums(w[, cs > 0, drop = FALSE] / rep(cs[cs > 0], each = nrow(w))) / ncol(w)
  }
  sweep(K, 2, colSums(K), "/")
}

model_probs <- function(train, obs, cw, ncores = 1, resp = "model", ntree = 1000, fit = NULL) {
  # prob_stats(M) = sum over M's rows of (abcrf classifier kernel x cw), renormalised; cw sums to 1 per unit
  # (balanced prior over models). obs may have several rows. Stops if a kernel has no mass on positive weights.
  train[[resp]] <- droplevels(factor(train[[resp]]))
  if (is.null(fit)) fit <- fit_model_choice(train, ntree, ncores, resp)
  a <- class_kernel(fit, train, obs, ncores, resp) * cw
  tot <- colSums(a)
  if (any(!(tot > 0)))
    stop(sprintf("model_probs: the classifier kernel of %d point(s) falls entirely on zero-weight rows",
                 sum(!(tot > 0))))
  p <- t(rowsum(a, train[[resp]], reorder = TRUE)) / tot
  levs <- levels(train[[resp]])
  p <- matrix(p[, levs, drop = FALSE], nrow = nrow(obs), dimnames = list(NULL, levs))
  list(prob_stats = p, map = levs[max.col(p, ties.method = "first")], fit = fit)
}

confusion_df <- function(true, pred, w = NULL) {
  # n = unique rows; w = summed weight (each true model's rows sum to 1 under the per-unit weights)
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

cv_confusion <- function(train, src_id, cw, folds = 5, ntree = 1000, ncores = 1, seed = 1, resp = "model") {
  # Grouped K-fold of the post-hoc-weighted argmax classifier. Training-fold weights are renormalised per unit so
  # the models stay balanced.
  fold <- grouped_folds(src_id, folds, seed)
  pred <- character(nrow(train))
  for (k in seq_len(folds)) {
    te <- fold == k
    tr <- train[!te, ]
    cwk <- cw[!te]; cwk <- cwk / ave(cwk, as.character(tr[[resp]]), FUN = sum)
    mp <- model_probs(tr, train[te, setdiff(names(train), resp), drop = FALSE], cwk, ncores, resp, ntree)
    pred[te] <- mp$map
  }
  confusion_df(train[[resp]], pred, cw)
}

# --------------------------------------------------------------------------------------------------- parameters --
param_frame <- function(d, param, stats, log) {
  y <- d[[param]]
  if (log) y <- log10(y)
  data.frame(y = y, d[, stats, drop = FALSE])
}

QS <- c(0.025, 0.25, 0.5, 0.75, 0.975)

fit_param <- function(d, param, stats, log, obs, ntree = 1000, ncores = 1, case_w = NULL, sample_fraction = 1) {
  # Unweighted forest; posterior weights = kernel at obs x case_w (renormalised). Quantiles on the fitting scale.
  pf <- param_frame(d, param, stats, log)
  if (is.null(case_w)) case_w <- rep(1, nrow(pf))
  rf <- reg_rf(pf, ntree, ncores, sample_fraction)
  a <- posthoc(rf_kernel(rf, obs, pf, ncores), case_w)[, 1]
  bt <- if (log) function(x) 10^x else identity
  q <- bt(wquantile(pf$y, a, QS))
  m <- sum(a * pf$y)
  list(q025 = q[1], q25 = q[2], median = q[3], q75 = q[4], q975 = q[5], mean_fit = m,
       sd_fit = sqrt(sum(a * (pf$y - m)^2)), weights = a)
}

coverage_cv <- function(d, src_id, param, stats, log, n_pods = 200, folds = 5, ntree = 1000, ncores = 1, seed = 1,
                        case_w = NULL, sample_fraction = 1) {
  # Pods: n_pods rows with distinct source ids, drawn without replacement in proportion to case_w (uniform if NULL),
  # i.e. from the weighted reference distribution. Grouped K-fold: each pod is predicted by an unweighted forest
  # trained without its fold; its posterior = kernel x case_w (training rows). Coverage is global, not local.
  set.seed(seed)
  if (is.null(case_w)) case_w <- rep(1, nrow(d))
  first <- which(!duplicated(src_id))
  pods <- first[sample.int(length(first), min(n_pods, length(first)), prob = case_w[first])]
  fold <- grouped_folds(src_id, folds, seed)
  pf <- param_frame(d, param, stats, log)
  c50 <- c95 <- logical(0)
  for (k in seq_len(folds)) {
    pk <- pods[fold[pods] == k]
    if (!length(pk)) next
    tri <- fold != k
    tr <- pf[tri, ]
    rf <- reg_rf(tr, ntree, ncores, sample_fraction)
    a <- posthoc(rf_kernel(rf, pf[pk, stats, drop = FALSE], tr, ncores), case_w[tri])
    for (j in seq_along(pk)) {
      q <- wquantile(tr$y, a[, j], QS); y <- pf$y[pk[j]]
      c95 <- c(c95, y >= q[1] & y <= q[5]); c50 <- c(c50, y >= q[2] & y <= q[4])
    }
  }
  list(n_pods = length(c95), cover50 = mean(c50), cover95 = mean(c95))
}

wmedian <- function(x, w = NULL) {
  if (is.null(w)) w <- rep(1, length(x))
  o <- order(x); x <- x[o]; w <- w[o]
  x[which(cumsum(w) / sum(w) >= 0.5)[1]]
}

abc_crosscheck <- function(d, param, stats, log, obs, n_accept = 100, case_w = NULL, lo = -Inf, hi = Inf) {
  # tol = n_accept / n, so a fixed number of rows is accepted; the accepted rows' medians are weighted by case_w.
  # A loclinear median outside the prior support [lo, hi] (natural scale) is set to NA and flagged in `note`.
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
  else {
    v <- bt(wmedian(l$adj.values[, 1], l$weights * case_w[l$region]))
    if (!is.finite(v) || v < lo || v > hi) {
      out$note <- sprintf("loclinear median %.4g outside prior support [%g, %g]: set NA", v, lo, hi)
    } else out$loclinear <- v
  }
  out
}

# --------------------------------------------------------------------------------------------------------- main --
PARAM_SAMPLE_FRACTION <- 1   # per-tree subsample of the parameter forests (--param-sample-fraction)

parse_args <- function(argv) {
  a <- list(ntree = 1000, seed = 1, ncores = 16, n_pods = 200, folds = 5, n_draws = 1000, abc_accept = 100,
            param_sample_fraction = PARAM_SAMPLE_FRACTION, mc_min_node_size = MC_MIN_NODE_SIZE)
  i <- 1
  while (i <= length(argv)) {
    k <- gsub("-", "_", sub("^--", "", argv[i])); a[[k]] <- argv[i + 1]; i <- i + 2
  }
  for (k in c("ntree", "seed", "ncores", "n_pods", "folds", "n_draws", "abc_accept", "param_sample_fraction",
             "mc_min_node_size"))
    a[[k]] <- as.numeric(a[[k]])
  a
}

DRAWS_NOTE <- paste0("# NOTE: joint draws = rows resampled by the post-hoc posterior weights averaged over this model's ",
                     "all-row parameter forests; their soft-only marginals (p0, t_sel) need not match the ",
                     "per-parameter regAbcrf posteriors in abc_posteriors.csv. Task 8b SMC draws replace these ",
                     "downstream. Read with comment='#'.")

lookup_acc <- function(accf, m, fl, col = "acceptance") {
  acc <- c(hard = NA_real_, soft = NA_real_)
  for (c in c("hard", "soft")) {
    r <- accf[accf$model == m & accf$flux == fl & accf$cls == c, ]
    if (nrow(r) > 1) stop("duplicate acceptance rows for ", m, "/", fl, "/", c)
    if (nrow(r) && col %in% names(r)) acc[[c]] <- r[[col]][1]
  }
  acc
}

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
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
  # acceptance-CI range of prob_total: this model at its CI end, every other model at the opposite end
  pt_lo <- pt_hi <- setNames(numeric(length(units)), units)
  for (u in units) {
    lo <- setNames(info$acc_hi, units); lo[u] <- info$acc_lo[units == u]
    hi <- setNames(info$acc_lo, units); hi[u] <- info$acc_hi[units == u]
    pt_lo[u] <- combine_total(prior, lo, ps)[u]; pt_hi[u] <- combine_total(prior, hi, ps)[u]
  }
  mc <- data.frame(model = info$model, flux = info$flux, prior = prior[units], acc_hard = info$acc_hard,
                   acc_soft = info$acc_soft, acc = info$acc, vote_unweighted = vote[units],
                   prob_stats = ps[units], prob_total = pt[units], prob_total_acc_ci_lo = pt_lo[units],
                   prob_total_acc_ci_hi = pt_hi[units], map_stats = units == mp$map[1],
                   n_rows = info$n_rows, n_hard = info$n_hard, n_soft = info$n_soft,
                   ess_mix = info$ess_mix, ess_hard = info$ess_hard, ess_soft = info$ess_soft,
                   mix_hard = info$mix_hard)
  write.csv(mc, file.path(a$out_dir, "modelchoice_raw.csv"), row.names = FALSE)

  # ---- hard share, parameters, coverage, cross-check, draws
  hs <- list(); post <- list(); cov <- list(); xc <- list(); dmeta <- list()
  for (j in seq_along(units)) {
    u <- units[j]; d <- tab[tab$unit == u, ]; ir <- info[j, ]
    cat("model", u, "...\n")
    if (all(c("hard", "soft") %in% d$cls)) {
      # one unweighted forest on the hard indicator; kernel x class-balanced weights (prior p_hard, no acceptance)
      # and kernel x cw (with acceptance)
      bw <- mixture_weights(d, c(hard = 1, soft = 1), p_hard)
      isd <- d$cls == "hard"
      s1 <- indicator_share(d[, stats], isd, bw, obs_s, a$ntree, a$ncores)
      s2 <- indicator_share(d[, stats], isd, d$cw, obs_s, rf = s1$rf, ncores = a$ncores)
      hs[[u]] <- data.frame(model = d$model[1], flux = d$flux[1], n_hard = ir$n_hard, n_soft = ir$n_soft,
                            ess_hard = ir$ess_hard, hard_share_stats = s1$p, hard_share_total = s2$p,
                            prior_mix_hard = ir$mix_hard)
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
      fp <- fit_param(pd, p, stats, sp$log, obs_s, ntree = a$ntree, ncores = a$ncores, case_w = cw,
                      sample_fraction = a$param_sample_fraction)
      post[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p,
                                        scale = if (sp$log) "log10" else "linear", rows = sp$rows,
                                        n_train = nrow(pd), ess_train = ess(cw), ess_post = ess(fp$weights),
                                        median = fp$median, q025 = fp$q025, q25 = fp$q25, q75 = fp$q75,
                                        q975 = fp$q975, mean_fitscale = fp$mean_fit, sd_fitscale = fp$sd_fit,
                                        prior_lo = sp$lo, prior_hi = sp$hi,
                                        post_hard_mass = sum(fp$weights[pd$cls == "hard"]))
      if (sp$rows == "all") { wsum[sel] <- wsum[sel] + fp$weights; nw <- nw + 1 }
      cv <- coverage_cv(pd, pd$src, p, stats, sp$log, n_pods = a$n_pods, folds = a$folds, ntree = a$ntree,
                        ncores = a$ncores, seed = a$seed + j, case_w = cw,
                        sample_fraction = a$param_sample_fraction)
      cov[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p, n_pods = cv$n_pods,
                                       cover50 = cv$cover50, cover95 = cv$cover95)
      x <- abc_crosscheck(pd, p, stats, sp$log, obs_s, a$abc_accept, cw, lo = sp$lo, hi = sp$hi)
      xc[[paste(u, p)]] <- data.frame(model = d$model[1], flux = d$flux[1], parameter = p, tol = x$tol,
                                      n_accept = x$n_accept, rf_median = fp$median, rejection_median = x$rejection,
                                      loclinear_median = x$loclinear, note = x$note)
    }
    # joint draws: rows resampled by the post-hoc posterior weights averaged over the all-row parameter forests
    if (nw > 0) {
      pw <- wsum / nw
      set.seed(a$seed + 200 + j)
      di <- sample.int(nrow(d), a$n_draws, replace = TRUE, prob = pw)
      pcols <- c("model", "flux", "cls", "t_inv", "hard", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff", "src")
      fn <- file.path(a$out_dir, sprintf("abc_post_%s.csv", u))
      writeLines(DRAWS_NOTE, fn)
      suppressWarnings(write.table(d[di, intersect(pcols, names(d))], fn, sep = ",", row.names = FALSE,
                                   append = TRUE, qmethod = "double"))
      dmeta[[u]] <- data.frame(model = d$model[1], flux = d$flux[1], n_draws = a$n_draws,
                               expected_hard = a$n_draws * sum(pw[d$cls == "hard"]),
                               drawn_hard = sum(d$cls[di] == "hard"), unique_rows = length(unique(di)))
    }
  }
  write.csv(do.call(rbind, post), file.path(a$out_dir, "posteriors_raw.csv"), row.names = FALSE)
  write.csv(do.call(rbind, cov), file.path(a$out_dir, "coverage_raw.csv"), row.names = FALSE)
  write.csv(do.call(rbind, xc), file.path(a$out_dir, "crosscheck_raw.csv"), row.names = FALSE)
  write.csv(do.call(rbind, dmeta), file.path(a$out_dir, "draws_raw.csv"), row.names = FALSE)
  hsd <- if (length(hs)) do.call(rbind, hs) else data.frame(model = character(), flux = numeric(),
    hard_share_stats = numeric(), hard_share_total = numeric())
  write.csv(hsd, file.path(a$out_dir, "hardshare_raw.csv"), row.names = FALSE)
  meta <- list(table = a$table, spec = a$spec, n_rows_in = n0, n_dropped_nan = sum(drop_nan),
               n_dropped_na_key = n_dropped_na, models = paste(units, collapse = " "), p_hard = p_hard,
               training = "unweighted forests on unique rows; importance weights applied post hoc to the forest kernel",
               accuracy_grouped_cv_weighted = acc_cv, accuracy_oob_unweighted_abcrf = acc_oob_unw,
               coverage_note = "coverage is global (prior predictive), not local to the observed point",
               param_sample_fraction = a$param_sample_fraction, mc_min_node_size = a$mc_min_node_size, ntree = a$ntree, seed = a$seed, folds = a$folds, n_pods = a$n_pods, abc_accept = a$abc_accept,
               n_stats = length(stats), stats_dropped_constant = paste(setdiff(keys, stats), collapse = " "),
               R = R.version.string, abcrf = as.character(packageVersion("abcrf")),
               ranger = as.character(packageVersion("ranger")), abc = as.character(packageVersion("abc")),
               arrow = as.character(packageVersion("arrow")),
               runtime_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  write_json(meta, file.path(a$out_dir, "run_meta.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("done in %.1f s\n", meta$runtime_s))
}

if (sys.nframe() == 0L) main()
