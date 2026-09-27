# R twin of reanalysis/python/run_analysis.py: same split files, same starting factors, same SGD order.
#   Rscript reanalysis/R/run_analysis.R      # after prepare.py; ~25 min; needs Rcpp (+ a C++ compiler)
suppressPackageStartupMessages(library(Rcpp))

root <- normalizePath(".")
for (k in 1:4) if (!file.exists(file.path(root, "data", "work", "prepare_info.txt"))) root <- dirname(root)
WORK <- file.path(root, "data", "work"); OUT <- file.path(root, "reanalysis", "results")
K <- 50L; LR <- 0.005; REG <- 0.02; REG_B <- 0.02; EPOCHS <- 20L; LAMBDA_I <- 25; LAMBDA_U <- 10
n_users <- as.integer(sub(".*'n_users': (\\d+).*", "\\1", readLines(file.path(WORK, "prepare_info.txt"))))

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double sgd_epoch(IntegerVector u, IntegerVector i, NumericVector r, double mu, NumericVector bu, NumericVector bi,
                 NumericMatrix P, NumericMatrix Q, double lr, double reg, double regb) {
  int K = P.nrow(); double sse = 0;                       // P, Q are K x n: one column per user / movie
  for (R_xlen_t t = 0; t < r.size(); t++) {
    int a = u[t], b = i[t];
    double pred = mu + bu[a] + bi[b];
    for (int f = 0; f < K; f++) pred += P(f, a) * Q(f, b);
    double e = r[t] - pred; sse += e * e;
    bu[a] += lr * (e - regb * bu[a]);
    bi[b] += lr * (e - regb * bi[b]);
    for (int f = 0; f < K; f++) { double pf = P(f, a), qf = Q(f, b);
      P(f, a) += lr * (e * qf - reg * pf); Q(f, b) += lr * (e * pf - reg * qf); }
  }
  return std::sqrt(sse / r.size());
}
// [[Rcpp::export]]
NumericVector predict_mf(IntegerVector u, IntegerVector i, double mu, NumericVector bu, NumericVector bi,
                         NumericMatrix P, NumericMatrix Q) {
  NumericVector out(u.size());
  for (R_xlen_t t = 0; t < u.size(); t++) {
    double pred = mu + bu[u[t]] + bi[i[t]];
    for (int f = 0; f < P.nrow(); f++) pred += P(f, u[t]) * Q(f, i[t]);
    out[t] = std::min(5.0, std::max(1.0, pred));
  }
  return out;
}
// [[Rcpp::export]]
NumericVector bincount(IntegerVector x, NumericVector w, int n) {
  NumericVector out(n);
  for (R_xlen_t t = 0; t < x.size(); t++) out[x[t]] += w[t];
  return out;
}')

read_bin <- function(name) {
  f <- function(tag, size) { p <- file.path(WORK, sprintf("%s_%s.bin", name, tag)); readBin(p, "integer", file.size(p) / size, size, endian = "little") }
  list(u = f("u", 4), i = f("i", 2), r = as.numeric(f("r", 1)))
}
rmse <- function(y, p) sqrt(mean((y - p)^2))

rows <- list()
for (split in c("probe", "random")) {
  tr <- read_bin(paste0(split, "_train")); te <- read_bin(paste0(split, "_test"))
  mu <- sum(tr$r) / length(tr$r)
  ni <- bincount(tr$i, rep(1, length(tr$i)), 17771L)
  movie_mean <- ifelse(ni > 0, bincount(tr$i, tr$r, 17771L) / pmax(ni, 1), mu)
  bi0 <- bincount(tr$i, tr$r - mu, 17771L) / (LAMBDA_I + ni)
  nu <- bincount(tr$u, rep(1, length(tr$u)), n_users)
  bu0 <- bincount(tr$u, tr$r - mu - bi0[tr$i + 1], n_users) / (LAMBDA_U + nu)
  res <- c(global_mean = rmse(te$r, mu), movie_mean = rmse(te$r, movie_mean[te$i + 1]),
           biases = rmse(te$r, pmin(5, pmax(1, mu + bu0[te$u + 1] + bi0[te$i + 1]))))
  P <- matrix(readBin(file.path(WORK, "P0.bin"), "double", n_users * K, endian = "little"), nrow = K)
  Q <- matrix(readBin(file.path(WORK, "Q0.bin"), "double", 17771 * K, endian = "little"), nrow = K)
  bu <- numeric(n_users); bi <- numeric(17771)
  for (ep in seq_len(EPOCHS)) {
    t0 <- Sys.time()
    trn <- sgd_epoch(tr$u, tr$i, tr$r, mu, bu, bi, P, Q, LR, REG, REG_B)       # updates bu, bi, P, Q in place
    tst <- rmse(te$r, predict_mf(te$u, te$i, mu, bu, bi, P, Q))
    message(sprintf("%s epoch %2d: train %.4f test %.4f (%.0fs)", split, ep, trn, tst, as.numeric(Sys.time() - t0, units = "secs")))
  }
  res["mf"] <- tst
  rows[[split]] <- data.frame(split = split, t(round(res, 5)))
  rm(tr, te, P, Q); gc(verbose = FALSE)
}
out <- do.call(rbind, rows)
write.csv(out, file.path(OUT, "rmse_r.csv"), row.names = FALSE)
print(out, row.names = FALSE)

py <- file.path(OUT, "rmse.csv")
if (file.exists(py)) {
  p <- read.csv(py)[, c("split", "global_mean", "movie_mean", "biases", "mf")]
  cat("\nParity with Python (max abs difference):\n")
  print(signif(sapply(c("global_mean", "movie_mean", "biases", "mf"), function(k) max(abs(p[[k]] - out[[k]]))), 3))
}
