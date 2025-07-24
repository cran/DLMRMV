#' MCEM Algorithm for Missing Response Variables
#'
#' Implements Monte Carlo EM algorithm for handling missing response data in linear regression models
#'
#' @param data Data frame with response in 1st column and predictors in other columns
#' @param d Initial convergence threshold (default=5)
#' @param tol Termination tolerance (default=0.01)
#' @param nb Maximum iterations (default=50)
#' @return List containing:
#' \item{Yhat}{Imputed response vector}
#' \item{betahat}{Final regression coefficients}
#' \item{iterations}{Number of iterations performed}
#' @examples
#' # Create dataset with 20% missing responses
#' set.seed(123)
#' data <- data.frame(
#'   Y = c(rnorm(80), rep(NA, 20)),
#'   X1 = rnorm(100),
#'   X2 = runif(100)
#' )
#' result <- MCEM(data, d=5, tol=0.001, nb=100)
MCEM <- function(data, d = 5, tol = 0.01, nb = 50) {
  # Data preparation
  Y <- as.matrix(data[, 1])
  p <- ncol(data) - 1
  nobs <- sum(!is.na(Y))
  
  # Split complete and missing cases
  complete <- data[!is.na(Y), ]
  missing <- data[is.na(Y), ]
  
  Xobs <- as.matrix(complete[, -1])
  Yobs <- as.matrix(complete[, 1])
  Xmis <- as.matrix(missing[, -1])
  
  # Initial parameter estimates
  betahat <- solve(t(Xobs) %*% Xobs) %*% t(Xobs) %*% Yobs
  
  # Initialize iteration counter
  niter <- 1
  
  # EM iterations
  while ((d >= tol) && (niter <= nb)) {
    beta_old <- betahat
    
    # E-step: Impute missing responses
    Ymis <- Xmis %*% beta_old
    
    # M-step: Update parameters
    X_full <- rbind(Xobs, Xmis)
    Y_full <- rbind(Yobs, Ymis)
    betahat <- solve(t(X_full) %*% X_full) %*% t(X_full) %*% Y_full
    
    # Convergence check
    d <- sqrt(mean((betahat - beta_old)^2))
    niter <- niter + 1
  }
  
  # Final imputation
  Y[is.na(Y)] <- Ymis
  
  return(list(
    Yhat = Y,
    betahat = betahat,
    iterations = niter - 1
  ))
}