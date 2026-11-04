# Tests for illex/abc/abc_infer.R.  Run from the repo root:
#   Rscript -e 'testthat::test_file("illex/abc/tests/test_abc_infer.R")'
library(testthat)
src <- Find(file.exists, c("../abc_infer.R", "illex/abc/abc_infer.R"))  # test_file() runs in tests/
source(src)

NTREE <- 500
NCORES <- 8

toy_table <- function(n, seed) {
  set.seed(seed)
  mk <- function(mu, lab) {
    theta <- rnorm(n, mu, 1)
    x <- matrix(rnorm(n * 10, theta, 1), n, 10)
    data.frame(model = lab, theta = theta, xbar = rowMeans(x), xmed = apply(x, 1, median),
               xsd = apply(x, 1, sd), noise = runif(n))
  }
  d <- rbind(mk(0, "M0"), mk(3, "M3"))
  d$model <- factor(d$model)
  d
}
STATS <- c("xbar", "xmed", "xsd", "noise")
unit_w <- function(lab) 1 / ave(rep(1, length(lab)), as.character(lab), FUN = sum)   # each unit sums to 1

test_that("toy: RF model choice puts > 0.95 on the true model at x = 3 (analytic 0.983)", {
  # mean over 3 reference tables: a single table's estimate varies by a few percent (table 1 alone gives ~0.95)
  obs <- data.frame(xbar = 3, xmed = 3, xsd = 1, noise = 0.5)
  p <- sapply(1:3, function(sd) {
    d <- toy_table(2000, sd)
    mp <- model_probs(d[, c("model", STATS)], obs, unit_w(d$model), ncores = NCORES, ntree = NTREE)
    expect_equal(mp$map, "M3")
    mp$prob_stats[1, "M3"]
  })
  expect_gt(mean(p), 0.95)
})

test_that("toy: 95% interval coverage is 0.95 +- 0.03 over 600 pods (binomial SD 0.009)", {
  d <- toy_table(3000, 2)
  d3 <- d[d$model == "M3", ]
  cv <- coverage_cv(d3, src_id = seq_len(nrow(d3)), param = "theta", stats = STATS, log = FALSE,
                    n_pods = 600, folds = 5, ntree = NTREE, ncores = NCORES, seed = 3)
  expect_equal(cv$n_pods, 600)
  expect_lt(abs(cv$cover95 - 0.95), 0.03 + 1e-12)
})

test_that("weighting algebra: class acceptances, w_est and p_hard give the right mixture", {
  n <- 1000
  df <- data.frame(cls = rep(c("hard", "soft"), each = n),
                   w_est = c(rep(c(1, 3), each = n / 2), rep(1, n)))
  w <- mixture_weights(df, c(hard = 0.2, soft = 0.6))
  expect_equal(sum(w), 1, tolerance = 1e-12)
  expect_equal(sum(w[df$cls == "hard"]), 0.25, tolerance = 1e-12)       # 0.5*0.2 / (0.5*0.2 + 0.5*0.6)
  hw <- w[df$cls == "hard"]
  expect_equal(sum(hw[df$w_est[df$cls == "hard"] == 3]) / sum(hw), 0.75, tolerance = 1e-12)
  w2 <- mixture_weights(df, c(hard = 0.2, soft = 0.6), p_hard = 0.25)   # 0.25*0.2 / (0.25*0.2 + 0.75*0.6)
  expect_equal(sum(w2[df$cls == "hard"]), 0.05 / 0.5, tolerance = 1e-12)
  expect_equal(ess(rep(1, 10)), 10)
  expect_equal(ess(c(1, 0, 0)), 1)
  expect_equal(hard_share_acc(0.5, 0.2, 0.6), 0.25)
  pt <- combine_total(prior = c(A = 0.5, B = 0.5), acc = c(A = 0.1, B = 0.3), prob_stats = c(A = 0.5, B = 0.5))
  expect_equal(unname(pt), c(0.25, 0.75))
  expect_equal(wquantile(1:4, c(0, 0, 1, 1), c(0.25, 0.75)), c(3, 4))
})

# The test that would have caught review I1: small class mass, informative statistic.
small_mass_table <- function(n = 3000, seed = 5) {
  set.seed(seed)
  cls <- rep(c("hard", "soft"), each = n)
  x <- rnorm(2 * n, ifelse(cls == "hard", 1, 0), 1)
  data.frame(cls = cls, w_est = 1, x = x, z = runif(2 * n), theta = ifelse(cls == "hard", 10, 0) + rnorm(2 * n))
}

test_that("post-hoc weights recover a 1% class mass at an informative observed point", {
  d <- small_mass_table()
  cw <- mixture_weights(d, c(hard = 0.01, soft = 0.99))             # hard mass 0.01
  expect_equal(sum(cw[d$cls == "hard"]), 0.01, tolerance = 1e-12)
  obs <- data.frame(x = 1, z = 0.5)
  truth <- 0.01 * dnorm(0) / (0.01 * dnorm(0) + 0.99 * dnorm(1))    # 0.0164
  bw <- mixture_weights(d, c(hard = 1, soft = 1))
  s1 <- indicator_share(d[, c("x", "z")], d$cls == "hard", bw, obs, NTREE, NCORES)
  s2 <- indicator_share(d[, c("x", "z")], d$cls == "hard", cw, obs, rf = s1$rf, ncores = NCORES)
  expect_equal(s1$p, dnorm(0) / (dnorm(0) + dnorm(1)), tolerance = 0.12)   # 0.62, no acceptance
  # analytic truth 0.0164. The round-1 case.weights mechanism gave 0.0067 here (0.4x truth); on the v3 B table it
  # over-weighted the 1% hard class ~300x at the observed point (review I1)
  expect_gt(s2$p, truth / 2); expect_lt(s2$p, truth * 2)
  # parameter posterior: hard rows (theta ~ 10) carry ~1.6% of the posterior mass, zero-weight rows none
  fp <- fit_param(d, "theta", c("x", "z"), FALSE, obs, NTREE, NCORES, case_w = cw)
  expect_equal(sum(fp$weights), 1, tolerance = 1e-9)
  expect_gt(sum(fp$weights[d$cls == "hard"]), truth / 2); expect_lt(sum(fp$weights[d$cls == "hard"]), truth * 2)
  cw0 <- cw; cw0[d$cls == "hard"] <- 0
  fp0 <- fit_param(d, "theta", c("x", "z"), FALSE, obs, NTREE, NCORES, case_w = cw0)
  expect_equal(sum(fp0$weights[d$cls == "hard"]), 0)
  expect_lt(fp0$q975, 5)
})

test_that("model_probs (abcrf classifier kernel x cw) matches the analytic posterior and follows the weights", {
  d <- small_mass_table(1500, 6)
  d$model <- factor(ifelse(d$cls == "hard", "A", "B"))
  obs <- data.frame(x = c(-1, 1, 3), z = 0.5)
  p <- model_probs(d[, c("model", "x", "z")], obs, unit_w(d$model), NCORES, ntree = NTREE)
  # balanced models, x ~ N(1, 1) under A and N(0, 1) under B: P(A | x) = 1 / (1 + exp(-(x - 0.5))).
  # Pinned on the mean over 4 reference tables (per-table SD ~0.06 at min.node.size 50, ~0.22 at abcrf's default 1)
  pa <- sapply(c(6, 11, 12, 13), function(sd) {
    dd <- small_mass_table(1500, sd); dd$model <- factor(ifelse(dd$cls == "hard", "A", "B"))
    model_probs(dd[, c("model", "x", "z")], obs, unit_w(dd$model), NCORES, ntree = NTREE)$prob_stats[, "A"]
  })
  expect_equal(mean(pa[2, ]), 1 / (1 + exp(-0.5)), tolerance = 0.08)   # 0.6225
  expect_equal(mean(pa[1, ]), 1 / (1 + exp(1.5)), tolerance = 0.10)    # 0.182
  expect_equal(mean(pa[3, ]), 1 / (1 + exp(-2.5)), tolerance = 0.08)   # 0.924
  expect_equal(p$map[c(1, 3)], c("B", "A"))
  # down-weighting the A rows with x > 0 within unit A (still summing to 1) lowers P(A | x = 1)
  w <- unit_w(d$model); ia <- d$model == "A"
  w2 <- w; w2[ia & d$x > 0] <- w2[ia & d$x > 0] * 0.1; w2[ia] <- w2[ia] / sum(w2[ia])
  p2 <- model_probs(d[, c("model", "x", "z")], obs[2, , drop = FALSE], w2, NCORES, ntree = NTREE, fit = p$fit)
  expect_lt(p2$prob_stats[1, "A"], p$prob_stats[2, "A"])
  # the kernel is a proper weight vector over the training rows
  K <- class_kernel(p$fit, d[, c("model", "x", "z")], obs, NCORES)
  expect_equal(unname(colSums(K)), rep(1, 3), tolerance = 1e-9); expect_true(all(K >= 0))
})

test_that("zero-mass kernels stop with a clear error (N5)", {
  expect_error(posthoc(matrix(c(1, 0, 0.5, 0.5), 2, 2), c(0, 1)), "zero-weight rows")
  d <- small_mass_table(300, 8); d$model <- factor(ifelse(d$cls == "hard", "A", "B"))
  expect_error(model_probs(d[, c("model", "x", "z")], data.frame(x = 1, z = 0.5), rep(0, nrow(d)), NCORES,
                           ntree = 50), "zero-weight rows")
})

test_that("cv_confusion: counts, weights per true model and accuracy", {
  d <- toy_table(400, 7)
  cw <- unit_w(d$model)
  cc <- cv_confusion(d[, c("model", STATS)], seq_len(nrow(d)), cw, folds = 4, ntree = 200, ncores = NCORES)
  expect_equal(sum(cc$n), nrow(d))
  expect_equal(as.numeric(tapply(cc$w, cc$true, sum)), c(1, 1), tolerance = 1e-9)
  expect_gt(accuracy(cc, "w"), 0.85)
  expect_equal(grouped_folds(c(1, 1, 2, 3), 2, 1)[1], grouped_folds(c(1, 1, 2, 3), 2, 1)[2])
})

test_that("abc cross-check: fixed number accepted; loclinear outside support set to NA", {
  d <- toy_table(3000, 4); d3 <- d[d$model == "M3", ]
  obs <- data.frame(xbar = 3, xmed = 3, xsd = 1, noise = 0.5)
  x <- abc_crosscheck(d3, "theta", STATS, FALSE, obs, n_accept = 100)
  expect_equal(x$n_accept, 100)
  expect_equal(x$loclinear, 3, tolerance = 0.3)
  y <- abc_crosscheck(d3, "theta", STATS, FALSE, obs, n_accept = 100, lo = 10, hi = 20)
  expect_true(is.na(y$loclinear)); expect_match(y$note, "outside prior support")
})

# ------------------------------------------------------------------------------------------- end-to-end main() --
mini_inputs <- function(dir, seed = 9, drop_d1_hard = FALSE) {
  set.seed(seed)
  rows <- list()
  for (m in c("N", "D1")) for (c in c("hard", "soft")) {
    n <- if (c == "hard") 80 else 200
    shift <- (m == "D1") * 1.5 + (c == "hard") * 2
    t_inv <- if (c == "hard") rep(1.9e6, n) else runif(n, 1e5, 1.5e6)   # hard rows identifiable by t_inv
    rows[[paste(m, c)]] <- data.frame(
      model = m, flux = 0L, cls = c, hard = c == "hard", t_inv = t_inv,
      p0 = if (c == "soft") 10^runif(n, -3, -1) else NA_real_,
      t_sel = if (m == "D1") (if (c == "soft") runif(n) * t_inv else t_inv) else NA_real_,
      s = if (m == "D1") 10^runif(n, -6, -2) else NA_real_, t_stop = NA_real_, p_star = NA_real_,
      gamma_eff = NA_real_, w_est = if (c == "hard") runif(n, 0.5, 2) else 1,
      x1 = rnorm(n, shift + t_inv / 1e6, 0.5), x2 = runif(n), p_obs = 0.62 + rnorm(n, 0, 0.005), n_nan = 0L)
  }
  tab <- do.call(rbind, rows)
  if (drop_d1_hard) tab$n_nan[tab$model == "D1" & tab$cls == "hard"] <- 1L
  arrow::write_parquet(tab, file.path(dir, "t.parquet"))
  acc <- data.frame(model = c("N", "N", "D1", "D1"), flux = 0, cls = c("hard", "soft", "hard", "soft"),
                    acceptance = c(1e-4, 1e-2, 2e-3, 4e-3), ci_lo = c(0.5e-4, 0.9e-2, 1e-3, 3e-3),
                    ci_hi = c(2e-4, 1.1e-2, 3e-3, 5e-3))
  write.csv(acc, file.path(dir, "a.csv"), row.names = FALSE)
  write_json(list(stats = list(x1 = 3.0, x2 = 0.5), p_obs = 0.62), file.path(dir, "o.json"), auto_unbox = TRUE)
  P <- function(rows, models, lo, hi, log = FALSE, flux = NULL) {
    x <- list(rows = rows, models = models, lo = lo, hi = hi, log = log); if (!is.null(flux)) x$flux <- flux; x }
  spec <- list(fit_keys = c("x1", "x2", "p_obs"), log_params = c("s", "p0", "gamma_eff"), p_hard = 0.5,
               params = list(t_inv = P("all", c("N", "D1"), 1e5, 2e6), s = P("all", "D1", 1e-6, 1e-2, TRUE),
                             p0 = P("soft", c("N", "D1"), 1e-3, 0.1, TRUE), t_sel = P("soft", "D1", 0, 2e6),
                             gamma_eff = P("all", c("N", "D1"), 1e-9, 1e-5, TRUE, flux = 1)))
  write_json(spec, file.path(dir, "s.json"), auto_unbox = TRUE, digits = NA)
  tab
}
run_main <- function(dir) main(c("--table", file.path(dir, "t.parquet"), "--acceptance", file.path(dir, "a.csv"),
                                 "--observed", file.path(dir, "o.json"), "--spec", file.path(dir, "s.json"),
                                 "--out-dir", file.path(dir, "out"), "--ntree", "200", "--ncores", "4",
                                 "--n-pods", "20", "--folds", "2", "--n-draws", "300", "--abc-accept", "30"))

test_that("main(): per-unit weighting, acceptance lookup, row selection, prob_total, draws alignment", {
  dir <- tempfile("abcmain"); dir.create(dir)
  tab <- mini_inputs(dir)
  capture.output(run_main(dir))
  o <- file.path(dir, "out")
  mc <- read.csv(file.path(o, "modelchoice_raw.csv"))
  mN <- mc[mc$model == "N", ]; mD <- mc[mc$model == "D1", ]
  expect_equal(mN$acc_hard, 1e-4); expect_equal(mN$acc_soft, 1e-2)               # lookup by (model, flux, cls)
  expect_equal(mD$acc_hard, 2e-3); expect_equal(mD$acc_soft, 4e-3)
  expect_equal(mN$acc, 0.5 * 1e-4 + 0.5 * 1e-2)
  expect_equal(mN$mix_hard, 1e-4 / (1e-4 + 1e-2), tolerance = 1e-12)
  expect_equal(mD$mix_hard, 2e-3 / 6e-3, tolerance = 1e-12)
  pt <- combine_total(c(D1_0 = 0.5, N_0 = 0.5), c(D1_0 = mD$acc, N_0 = mN$acc),
                      c(D1_0 = mD$prob_stats, N_0 = mN$prob_stats))
  expect_equal(c(mD$prob_total, mN$prob_total), unname(pt), tolerance = 1e-9)
  expect_equal(sum(mc$prob_stats), 1, tolerance = 1e-9)
  expect_true(all(mc$prob_total_acc_ci_lo <= mc$prob_total & mc$prob_total <= mc$prob_total_acc_ci_hi))
  po <- read.csv(file.path(o, "posteriors_raw.csv"))
  expect_setequal(po$parameter[po$model == "N"], c("t_inv", "p0"))
  expect_setequal(po$parameter[po$model == "D1"], c("t_inv", "s", "p0", "t_sel"))
  expect_true(all(po$n_train[po$parameter %in% c("p0", "t_sel")] == 200))       # soft rows only
  expect_true(all(po$n_train[po$parameter %in% c("t_inv", "s")] == 280))
  expect_true(all(po$post_hard_mass[po$rows == "soft"] == 0))
  # hard share against an independent estimator: weighted 60-nearest-neighbour share on x1 within each unit
  hs <- read.csv(file.path(o, "hardshare_raw.csv"))
  for (m in c("N", "D1")) {
    du <- tab[tab$model == m, ]
    acc <- if (m == "N") c(hard = 1e-4, soft = 1e-2) else c(hard = 2e-3, soft = 4e-3)
    nn <- order(abs(du$x1 - 3.0))[1:60]
    knn <- function(w) sum(w[nn] * (du$cls[nn] == "hard")) / sum(w[nn])
    h <- hs[hs$model == m, ]
    expect_equal(h$hard_share_stats, knn(mixture_weights(du, c(hard = 1, soft = 1))), tolerance = 0.2)
    kt <- knn(mixture_weights(du, acc))
    if (kt > 0) { expect_gt(h$hard_share_total, kt / 4); expect_lt(h$hard_share_total, kt * 4) }
    else expect_lt(h$hard_share_total, 0.02)            # no hard row among the 60 neighbours (D1 at x1 = 3)
  }
  # draws: comment header + n_draws rows; expected hard count = mean over all-row forests of post_hard_mass
  dr <- read.csv(file.path(o, "draws_raw.csv"))
  for (m in c("N", "D1")) {
    f <- read.csv(file.path(o, sprintf("abc_post_%s_0.csv", m)), comment.char = "#")
    expect_equal(nrow(f), 300)
    expect_true(all((f$cls == "hard") == (f$t_inv == 1.9e6)))
    ph <- mean(po$post_hard_mass[po$model == m & po$rows == "all"])
    expect_equal(dr$expected_hard[dr$model == m], 300 * ph, tolerance = 1e-6)
    expect_equal(dr$drawn_hard[dr$model == m], sum(f$cls == "hard"))
  }
  expect_match(readLines(file.path(o, "abc_post_N_0.csv"), n = 1), "^# NOTE")
  cc <- read.csv(file.path(o, "confusion_cv.csv")); expect_equal(sum(cc$n), nrow(tab))
  expect_true(file.exists(file.path(o, "spec_used.json")))
})

test_that("main(): a class with acceptance but no usable rows is an error", {
  dir <- tempfile("abcmain2"); dir.create(dir)
  mini_inputs(dir, drop_d1_hard = TRUE)
  expect_error(capture.output(run_main(dir)), "no usable table rows")
  expect_equal(lookup_acc(read.csv(file.path(dir, "a.csv")), "D1", 0), c(hard = 2e-3, soft = 4e-3))
})
