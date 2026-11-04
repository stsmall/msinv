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

test_that("toy: RF model choice puts > 0.95 on the true model at x = 3", {
  d <- toy_table(2000, 1)
  stats <- c("xbar", "xmed", "xsd", "noise")
  fit <- fit_model_choice(d[, c("model", stats)], ntree = NTREE, ncores = NCORES)
  obs <- data.frame(xbar = 3, xmed = 3, xsd = 1, noise = 0.5)
  p <- model_probs(fit, d[, c("model", stats)], obs, ncores = NCORES)
  expect_equal(p$map, "M3")
  expect_gt(p$prob_stats[["M3"]], 0.95)
  expect_equal(sum(p$prob_stats), 1, tolerance = 1e-9)
})

test_that("toy: 95% interval coverage is 0.95 +- 0.03 over 600 pods (binomial SD 0.009)", {
  d <- toy_table(3000, 2)
  d3 <- d[d$model == "M3", ]
  stats <- c("xbar", "xmed", "xsd", "noise")
  cv <- coverage_cv(d3, src_id = seq_len(nrow(d3)), param = "theta", stats = stats, log = FALSE,
                    n_pods = 600, folds = 5, ntree = NTREE, ncores = NCORES, seed = 3)
  expect_equal(cv$n_pods, 600)
  expect_lt(abs(cv$cover95 - 0.95), 0.03 + 1e-12)
})

test_that("weighting: two classes with known acceptances recover the right mixture", {
  n <- 1000
  df <- data.frame(cls = rep(c("hard", "soft"), each = n),
                   w_est = c(rep(c(1, 3), each = n / 2), rep(1, n)))
  acc <- c(hard = 0.2, soft = 0.6)
  w <- mixture_weights(df, acc)
  expect_equal(sum(w), 1, tolerance = 1e-12)
  expect_equal(sum(w[df$cls == "hard"]), 0.25, tolerance = 1e-12)       # 0.5*0.2 / (0.5*0.2 + 0.5*0.6)
  hw <- w[df$cls == "hard"]
  expect_equal(sum(hw[df$w_est[df$cls == "hard"] == 3]) / sum(hw), 0.75, tolerance = 1e-12)
  idx <- resample_rows(w, 40000, seed = 7)
  expect_equal(mean(df$cls[idx] == "hard"), 0.25, tolerance = 0.01)
  h <- idx[df$cls[idx] == "hard"]
  expect_equal(mean(df$w_est[h] == 3), 0.75, tolerance = 0.02)
  expect_identical(idx, resample_rows(w, 40000, seed = 7))             # fixed seed
  expect_equal(ess(rep(1, 10)), 10)
  expect_equal(ess(c(1, 0, 0)), 1)
  # acceptance combination: within-model hard share and model total
  expect_equal(hard_share_acc(0.5, 0.2, 0.6), 0.25)
  pt <- combine_total(prior = c(A = 0.5, B = 0.5), acc = c(A = 0.1, B = 0.3), prob_stats = c(A = 0.5, B = 0.5))
  expect_equal(unname(pt), c(0.25, 0.75))
})

test_that("case.weights reach ranger: weighted forests recover the weighted class mix", {
  set.seed(11)
  n <- 2000
  df <- data.frame(cls = rep(c("hard", "soft"), each = n / 2), w_est = 1, x1 = runif(n), x2 = runif(n))
  w <- mixture_weights(df, c(hard = 0.2, soft = 0.6))                 # hard mass 0.25; stats uninformative
  d <- data.frame(y = as.numeric(df$cls == "hard"), df[, c("x1", "x2")])
  grid <- data.frame(x1 = runif(100), x2 = runif(100))   # single points are noisy (sd ~0.09); average 100
  pw <- mean(predict(reg_rf(d, NTREE, NCORES, w), grid, training = d, paral = TRUE, ncores = NCORES)$expectation)
  pu <- mean(predict(reg_rf(d, NTREE, NCORES), grid, training = d, paral = TRUE, ncores = NCORES)$expectation)
  expect_equal(pw, 0.25, tolerance = 0.04)   # needs the per-tree subsample (RF_SAMPLE_FRACTION); 0.34 at fraction 1
  expect_equal(pu, 0.50, tolerance = 0.05)
  obs <- data.frame(x1 = 0.5, x2 = 0.5)
  # weighted classification forest: P(hard) from abcrf post.prob with the same weights is near 0.25 too
  bt <- data.frame(cls = factor(df$cls), df[, c("x1", "x2")])
  cf <- fit_model_choice(bt, ntree = NTREE, ncores = NCORES, resp = "cls", case_w = w)
  cp <- predict_mc(cf, obs, bt, NTREE, NCORES, w)
  ph <- if (as.character(cp$allocation) == "hard") cp$post.prob else 1 - cp$post.prob
  expect_lt(ph, 0.40)
})

test_that("abc cross-check accepts a fixed number of rows", {
  d <- toy_table(3000, 4); d3 <- d[d$model == "M3", ]
  x <- abc_crosscheck(d3, "theta", c("xbar", "xmed", "xsd", "noise"), FALSE,
                      data.frame(xbar = 3, xmed = 3, xsd = 1, noise = 0.5), n_accept = 100)
  expect_equal(x$n_accept, 100)
  expect_equal(x$loclinear, 3, tolerance = 0.3)
})
