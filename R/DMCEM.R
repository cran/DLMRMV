#' Distributed Monte Carlo Expectation-Maximization (DMCEM) Algorithm
#'
#' This function implements a distributed version of the Monte Carlo EM algorithm
#' for handling missing response variables in linear regression models.
#' By running multiple simulations (R) and averaging the results,
#' it provides more stable parameter estimates compared to standard EM.
#'
#' @param data A data frame where the first column is the response variable (with missing values)
#'   and subsequent columns are predictors.
#' @param R Integer specifying the number of Monte Carlo simulations.
#'   Larger values improve stability but increase computation time (default = 50).
#' @param tol Numeric value indicating the convergence tolerance.
#'   The algorithm stops when the change in coefficients between iterations
#'   is below this threshold (default = 0.01).
#' @param nb Integer specifying the maximum number of iterations per simulation.
#'   Prevents infinite loops if convergence is not achieved (default = 50).
#'
#' @return A list containing:
#'   \item{Yhat}{A vector of imputed response values with missing data filled in.}
#'   \item{betahat}{A vector of final regression coefficients, averaged across simulations.}
#'
#' @details
#' The DMCEM algorithm works by:
#' 1. Splitting data into observed and missing response subsets
#' 2. Running multiple MCEM simulations with random imputations
#' 3. Averaging results across simulations to reduce variance
#' 4. Using robust matrix inversion to handle near-singular designs
#'
#' This approach is particularly useful for datasets with a large proportion
#' of missing responses or high variability in the data.
#'
#' @export
#'
#' @examples
#' # Generate data with 20% missing responses
#' set.seed(123)
#' data <- data.frame(
#'   Y = c(rnorm(80), rep(NA, 20)),
#'   X1 = rnorm(100),
#'   X2 = runif(100)
#' )
#'
#' # Run DMCEM with 50 simulations
#' result <- DMCEM(data, R = 50, tol = 0.001, nb = 100)
#'
#' # View imputed values and coefficients
#' head(result$Yhat)
#' result$betahat
#'
#' # Check convergence and variance
#' result$converged_ratio
#' result$sigma2
DMCEM <- function(data, R = 50, tol = 0.01, nb = 50) {
  # Load required package for matrix operations
  if (!requireNamespace("MASS", quietly = TRUE)) {
    install.packages("MASS", dependencies = TRUE)
  }

  # Data preparation
  Y <- as.matrix(data[, 1])
  p <- ncol(data) - 1
  n <- nrow(data)

  # Check for missing values
  n_missing <- sum(is.na(Y))
  if (n_missing == 0) {
    warning("No missing values in response variable. Performing standard regression.")
    X <- as.matrix(data[, -1])
    betahat <- solve(t(X) %*% X) %*% t(X) %*% Y
    return(list(Yhat = Y, betahat = betahat))
  }

  # Split data into observed and missing cases
  complete <- data[!is.na(Y), ]
  missing <- data[is.na(Y), ]

  # Validate sufficient observed data for parameter estimation
  if (nrow(complete) <= p) {
    stop("Insufficient observed cases to estimate model parameters.")
  }

  Xobs <- as.matrix(complete[, -1])
  Yobs <- as.matrix(complete[, 1])
  Xmis <- as.matrix(missing[, -1])

  # Initialize storage for regression coefficients across simulations
  Beta <- matrix(0, p, R)
  converged <- logical(R)

  # Compute initial residual variance for stochastic imputation
  beta_init <- solve(t(Xobs) %*% Xobs) %*% t(Xobs) %*% Yobs
  sigma2_init <- sum((Yobs - Xobs %*% beta_init)^2) / (nrow(complete) - p)

  # Perform R Monte Carlo EM simulations
  for (r in 1:R) {
    # Initial parameter estimates
    betaold <- beta_init

    # EM iterations
    d <- Inf
    niter <- 1

    while ((d >= tol) && (niter <= nb)) {
      # E-step: Stochastic imputation of missing responses
      Ymis <- Xmis %*% betaold + rnorm(nrow(Xmis), 0, sqrt(sigma2_init))

      # M-step: Update regression coefficients
      X_full <- rbind(Xobs, Xmis)
      Y_full <- rbind(Yobs, Ymis)

      # Check for near-singular design matrix
      if (det(t(X_full) %*% X_full) < 1e-10) {
        betahat <- ginv(t(X_full) %*% X_full) %*% t(X_full) %*% Y_full
      } else {
        betahat <- solve(t(X_full) %*% X_full) %*% t(X_full) %*% Y_full
      }

      # Convergence check
      d <- sqrt(mean((betahat - betaold)^2))
      betaold <- betahat  # Update previous coefficients
      niter <- niter + 1  # Increment iteration counter
    }

    # Store final coefficients and convergence status
    Beta[, r] <- betahat
    converged[r] <- (d < tol)
  }

  # Compute average coefficients across simulations
  betahat <- rowMeans(Beta)

  # Final imputation of missing Y values
  Yhat <- Y
  Yhat[is.na(Y)] <- Xmis %*% betahat

  # Compute final residual variance
  Y_full_imputed <- Y
  Y_full_imputed[is.na(Y)] <- Xmis %*% betahat
  X_full <- as.matrix(data[, -1])
  residuals <- Y_full_imputed - X_full %*% betahat
  sigma2 <- sum(residuals^2) / (n - p)

  # Return results
  return(list(
    Yhat = Yhat,                 # Imputed response vector
    betahat = betahat           # Averaged regression coefficients
  ))
}
