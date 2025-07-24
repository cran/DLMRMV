#' CSLMI: Consensus-based Stochastic Linear Multiple Imputation (Simplified Version)
#'
#' Performs multiple imputation and parameter estimation using a consensus-based approach.
#' This version simplifies the interface by removing yidx, xidx, midx, n parameters.
#' It assumes:
#'   - The response variable is in the first column
#'   - All other columns are predictors
#'   - Missing values are automatically detected
#'   - The whole dataset is treated as one block
#'
#' @param data A list containing:
#'   \item{all}{A matrix or data frame with missing values.}
#'   \item{p}{Number of variables (optional, defaults to ncol(all)).}
#' @param M Number of multiple imputations.
#'
#' @return A list containing:
#'   \item{Yhat}{Imputed response values.}
#'   \item{betahat}{Average regression coefficients across imputations.}
#'   \item{comm}{Communication cost (number of messages passed).}
#' @export
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   y = c(rnorm(50), rep(NA, 10)),
#'   x1 = rnorm(60),
#'   x2 = rnorm(60)
#' )
#' my_data <- list(all = as.matrix(data))
#' result <- CSLMI(data = my_data, M = 10)
#' head(result$Yhat)
#' print(result$betahat)
#' print(result$comm)

CSLMI <- function(data, M) {
  # Automatically recognize data structure
  X <- data
  if (!is.matrix(X)) X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(data$p)) data$p <- p

  # Response variable is the first column, predictors are the rest
  yidx <- 1
  xidx <- 2:p

  # Automatically detect columns with missing values (only for response variable)
  midx <- which(colSums(is.na(X)) > 0)
  q <- length(midx)
  if (q == 0) stop("No missing values found in the dataset.")
  if (!(yidx %in% midx)) stop("Response variable has no missing values; nothing to impute.")

  # Initialize outputs
  betahats <- matrix(0, length(xidx), M)
  Yhat <- matrix(NA, nrow = n, ncol = M)
  comm <- 0
  fit.imp <- vector("list", q)  # Explicitly initialize as a list vector

  # Backup original data for multiple imputations
  original_all <- X

  # --- Start of Inlined CSLLS Function Logic ---
  for (j in 1:q) {
    if (midx[j] != yidx) {
      next  # Only process missing in response variable
    }

    yidx_local <- midx[j]
    xidx_local <- xidx
    lam <- 0.005

    p_local <- length(xidx_local)
    cc <- apply(is.na(X[, c(yidx_local, xidx_local)]), 1, sum) == 0
    idx <- which(cc)
    ncc <- length(idx)

    if (ncc == 0) stop("No complete cases available for CSLLS fitting.")

    Xc <- X[idx, xidx_local]
    yc <- X[idx, yidx_local]

    XX <- t(Xc) %*% Xc + diag(lam, p_local)
    Xy <- t(Xc) %*% yc

    if (any(is.nan(XX)) || any(is.infinite(XX))) {
      stop("Invalid Gram matrix in CSLLS")
    }

    cXX <- tryCatch(chol(XX), error = function(e) FALSE)
    if (is.logical(cXX)) stop("Cholesky decomposition failed in CSLLS")

    iXX <- chol2inv(cXX)
    beta <- iXX %*% Xy

    e <- yc - Xc %*% beta
    Xe <- t(Xc) %*% e
    beta_adj <- beta + iXX %*% Xe

    SSE <- sum((yc - Xc %*% beta_adj)^2)
    gram <- XX * ncc / ncc
    cgram <- tryCatch(chol(gram), error = function(e) FALSE)
    if (is.logical(cgram)) stop("Cholesky decomposition failed in gram matrix")

    # Ensure it's a list
    fit.imp[[j]] <- list(
      beta = beta_adj,
      df = ncc,
      SSE = SSE,
      gram = gram,
      cgram = cgram,
      comm = 0
    )
    comm <- comm + fit.imp[[j]]$comm + 1
  }
  # --- End of Inlined CSLLS Function ---


  # --- Main Imputation Loop ---
  for (m in 1:M) {
    X <- original_all  # Start from original data each time

    for (j in 1:q) {
      if (midx[j] != yidx) next  # Only process response variable missingness

      # Accessing a list object, can use $
      sig <- sqrt(1 / rgamma(1, (fit.imp[[j]]$df + 1)/2, (fit.imp[[j]]$SSE + 1)/2))
      alpha <- fit.imp[[j]]$beta + sig * backsolve(fit.imp[[j]]$cgram, rnorm(p - 1))

      na_rows <- which(is.na(X[, yidx]))
      if (length(na_rows) > 0) {
        pred <- as.vector(X[na_rows, xidx] %*% alpha)
        X[na_rows, yidx] <- pred + rnorm(length(na_rows), 0, sig)
      }
    }

    # --- Start of Inlined PPLS Function Logic ---
    lam_ppls <- 0.005
    cc <- apply(is.na(X), 1, sum) == 0
    idx <- which(cc)
    Xk <- X[idx, xidx]
    yk <- X[idx, yidx]

    XX <- t(Xk) %*% Xk + diag(lam_ppls, ncol(Xk))
    Xy <- t(Xk) %*% yk

    if (any(is.nan(XX)) || any(is.infinite(XX))) {
      stop("Invalid Gram matrix in PPLS")
    }

    cA <- tryCatch(chol(XX), error = function(e) FALSE)
    if (is.logical(cA)) stop("Cholesky decomposition failed in PPLS")

    beta <- backsolve(cA, forwardsolve(t(cA), Xy))
    comm <- comm + 1
    # --- End of Inlined PPLS Function ---

    betahats[, m] <- beta
    Yhat[, m] <- X[, yidx]
  }

  betahat <- apply(betahats, 1, mean)
  return(list(Yhat = Yhat, betahat = betahat, comm = comm))
}
