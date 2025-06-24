#' Impute Missing Values in Response Variable Y Using Distributed AVGMMI Method (With Grouping)
#'
#' This function implements the Distributed Averaged Generalized Method of Moments Imputation (DAVGMMI)
#' to fill in missing values in the response variable Y based on observed covariates X.
#' Assumes a single group structure and does not require group size input (`n`).
#'
#' @param data A data frame or matrix where the first column is the response variable Y (may contain NA),
#'             and remaining columns are covariates X.
#' @param R Number of simulations for stable Beta estimation.
#' @param M Number of multiple imputations.
#'
#' @return A list containing:
#'   \item{Yhat}{The vector of Y with missing values imputed.}
#'   \item{betahat}{Final averaged regression coefficient estimates used for imputation.}
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   y = c(rnorm(50), rep(NA, 10)),
#'   x1 = rnorm(60),
#'   x2 = rnorm(60)
#' )
#' result <- DAVGMMI(data, R = 50, M = 10)
#' head(result$Yhat)
#'
#' @export
DAVGMMI <- function(data, R, M) {

  # Extract observed data
  obs_idx <- !is.na(data[, 1])
  yidx <- data[obs_idx, 1]
  xidx <- data[obs_idx, -1]
  p <- ncol(xidx)

  # Initialize storage for Beta estimates across simulations
  Beta <- matrix(0, p, R)
  comm <- rep(0, R)

  for (r in 1:R) {

    # AVGM logic starts here
    betahats <- matrix(0, p, M)  # Store Beta estimates per imputation
    d_all <- cbind(yidx, xidx)
    N <- nrow(d_all)

    # Create missing pattern for simulation: each imputation has its own missingness
    miss_list <- lapply(1:M, function(m) {
      sample(N, floor(0.1 * N), replace = FALSE)
    })

    d_imp <- d_all  # Copy original data for imputation

    # LS: Ordinary least squares fitting on observed data
    X_sub <- as.matrix(d_all[, 2:(p+1)])
    y_sub <- d_all[, 1]
    fit <- lm.fit(X_sub, y_sub)
    resid <- residuals(fit)
    df <- N - length(coef(fit))
    SSE <- sum(resid^2)
    beta_ls <- coef(fit)
    cgram <- chol(crossprod(X_sub))

    # AVGMLS: compute average covariance and gram matrix
    Cov <- chol2inv(cgram)  # Approximate covariance
    gram <- chol2inv(chol(Cov))
    cgram <- chol(gram)

    # Simulated fit object
    fit.imp <- list(
      beta = beta_ls,
      df = df,
      SSE = SSE,
      gram = gram,
      cgram = cgram
    )

    # Multiple imputation loop
    for (m in 1:M) {

      # Add random noise to beta for variation
      sig <- sqrt(1 / rgamma(1, (fit.imp$df + 1)/2, (fit.imp$SSE + 1)/2))
      alpha <- fit.imp$beta + sig * backsolve(fit.imp$cgram, rnorm(p))

      # Get current missing indices
      current_miss <- miss_list[[m]]

      # Impute missing Y values
      if (length(current_miss) > 0) {
        d_imp[current_miss, 1] <-
          as.matrix(d_imp[current_miss, -1]) %*% alpha + rnorm(length(current_miss), 0, sig)
      }

      # Estimate new beta using linear regression on imputed data
      X <- as.matrix(d_imp[, -1])
      y <- d_imp[, 1]
      fit_ppls <- lm.fit(X, y)
      betahats[, m] <- coef(fit_ppls)
    }

    # Average over M imputations
    betahat <- apply(betahats, 1, mean)
    Beta[, r] <- betahat
    comm[r] <- 1
  }

  # Final average Beta
  m <- as.matrix(apply(Beta, 1, mean))

  # Predict missing Y values
  missing_idx <- is.na(data[, 1])
  Xmis <- as.matrix(data[missing_idx, -1])
  Ymishat <- Xmis %*% m

  # Build final output
  Yhat <- data[, 1]
  Yhat[missing_idx] <- Ymishat

  return(list(Yhat = Yhat, betahat = betahat))
}