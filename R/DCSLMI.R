#' Distributed and Consensus-Based Stochastic Linear Multiple Imputation (DCSLMI)
#'
#' Performs multiple imputation for missing response variables in linear regression models.
#' This method iteratively updates parameter estimates using ordinary least squares (OLS)
#' and generates M complete datasets by imputing missing values with different parameter draws.
#'
#' @param data A data frame or matrix. The first column contains the response variable `y`
#'             (which may include NA values), and the remaining columns are predictors `X`.
#' @param R Number of internal iterations for parameter estimation per imputation.
#' @param M Number of multiple imputations to generate.
#'
#' @return A list containing:
#' \describe{
#'   \item{Yhat}{A matrix of size n x M, where each column is a completed response vector.}
#'   \item{betahat}{A matrix of size (p+1) x M, where each column contains the estimated regression coefficients.}
#'   \item{missing_count}{The number of missing values in the original response variable.}
#' }
#' @export
#'
#' @examples
#' # Simulate data with missing responses
#' set.seed(123)
#' data <- data.frame(
#'   y = c(rnorm(50), rep(NA, 10)),
#'   x1 = rnorm(60),
#'   x2 = rnorm(60)
#' )
#'
#' # Perform multiple imputation
#' result <- DCSLMI(data, R = 500, M = 10)
#'
#' # View imputed response values
#' head(result$Yhat)
#'
#' # View coefficient estimates
#' apply(result$betahat, 1, mean)  # average estimates
#' apply(result$betahat, 1, sd)    # uncertainty across imputations
#'
DCSLMI <- function(data, R = 1000, M = 20) {
  
  # Input validation
  if (!is.matrix(data) && !is.data.frame(data)) 
    stop("Input data must be a matrix or data frame")
  if (ncol(data) < 2) stop("Data must contain at least one predictor")

  p <- ncol(data) - 1
  y <- data[, 1]
  X <- as.matrix(data[, -1])
  
  Beta <- matrix(0, p + 1, M)
  Y_samples <- matrix(y, nrow = length(y), ncol = M)

  X_design <- cbind(1, X)  # Add intercept
  miss <- is.na(y)         # Missing indices

  # Multiple imputation loop
  for (m in 1:M) {
    theta <- rnorm(p + 1)  # Initial parameter guess
    
    for (r in 1:R) {
      # Update parameters using observed data
      X_obs <- X_design[!miss, , drop = FALSE]
      y_obs <- y[!miss]
      
      if(nrow(X_obs) > 0) {
        theta <- solve(t(X_obs) %*% X_obs, t(X_obs) %*% y_obs)
      }
    }

    Beta[, m] <- theta

    # Impute missing values using current theta
    if (sum(miss) > 0) {
      X_miss <- X_design[miss, , drop = FALSE]
      Y_samples[miss, m] <- X_miss %*% theta
    } else {
      Y_samples[, m] <- y
    }
  }

  return(list(
    Yhat = Y_samples,
    betahat = Beta,
    missing_count = sum(miss)
  ))
}