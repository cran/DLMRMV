#' Distributed Full-information Multiple Imputation (DfiMI)
#'
#' Perform multiple imputation of the response variable Y via R independent
#' runs and M stochastic imputations per run.  
#' No grouping information is required.  
#' Missing values in Y are imputed by means of (intercept-adjusted) OLS
#' regression on the complete predictors.
#'
#' @param data A data frame  whose  first column  contains the response
#'   variable Y (possibly with NAs) and whose remaining columns contain
#'   numeric predictors.
#' @param R Positive integer-number of simulation runs used to stabilise the
#'   coefficient estimates.
#' @param M Positive integer-number of multiple imputations drawn within each
#'   run.
#'
#' @return A list containing:
#'   \item{Yhat}{The vector of Y with missing values imputed.}
#'   \item{betahat}{Final averaged regression coefficient estimates used for imputation.}
#'
#'
#' @examples
#' set.seed(123)
#' n  <- 60
#' data <- data.frame(
#'   Y  = c(rnorm(n - 10), rep(NA, 10)),  # 50 observed,10 missing
#'   X1 = rnorm(n),
#'   X2 = rnorm(n)
#' )
#'
#' res <- DfiMI(data, R = 3, M = 5)
#' head(res$Yhat)   # inspect imputed Y
#' res$betahat      # inspect coefficients
#'
#' @export

DfiMI <- function(data, R, M) {
  
  #Input validation 
if (!is.data.frame(data)) stop("Input 'data' must be a data frame")
  if (ncol(data) < 2) stop("Data requires at least one predictor")
  if (any(!sapply(data, is.numeric))) stop("All variables must be numeric")
  
  p <- ncol(data) - 1  # Number of predictors
  N <- nrow(data)      # Total sample size
  
  # Prepare to collect beta estimates
  Beta <- matrix(0, nrow = p, ncol = R)
  
  # Loop over R runs
  cat("Running DfiMI with", R, "runs...\n")
  for (r in 1:R) {
    cat("Run", r, "of", R, "\n")
    
    # Step 1: Initialize imputed data
    d_imp <- as.matrix(data)
    miss <- is.na(d_imp[, 1])  # Missing indices in Y
    
    # Step 2: Perform M imputations
    betahats <- matrix(0, p, M)
    
    for (m in 1:M) {
      # Copy original data for this imputation
      d_temp <- d_imp
      
      # Impute missing Y values using mean + noise
      if (sum(miss) > 0) {
        obs_idx <- which(!miss)
        y_mean <- mean(d_temp[obs_idx, 1], na.rm = TRUE)
        y_sd <- sd(d_temp[obs_idx, 1], na.rm = TRUE)
        d_temp[miss, 1] <- rnorm(sum(miss), mean = y_mean, sd = y_sd)
      }
      
      # Calculate regression coefficients using complete cases
      complete_cases <- complete.cases(d_temp)
      X <- as.matrix(d_temp[complete_cases, -1])
      Y <- d_temp[complete_cases, 1]
      
      # Compute beta using standard OLS formula
      if (qr(X)$rank < ncol(X)) {
        warning("X matrix not full rank in run ", r, ", imputation ", m)
        betahats[, m] <- rep(NaN, p)
      } else {
        beta <- solve(t(X) %*% X) %*% t(X) %*% Y
        betahats[, m] <- beta
      }
    }
    
    # Average over M imputations
    Beta[, r] <- apply(betahats, 1, mean, na.rm = TRUE)
  }
  
  # Final average beta across R runs
  betahat <- apply(Beta, 1, mean, na.rm = TRUE)
  
  # Extract observed Y and X
  Y <- as.numeric(data[[1]]) # Extract the first column as the response variable Y
  X <- as.matrix(data[,-1])   
  
  # Find missing indices
  miss_idx <- which(is.na(Y))
  
  # If no missing values, return original Y
  if (length(miss_idx) == 0) {
    return(list(
      Yhat = Y,
      betahat = as.numeric(betahat)
    ))
  }
  
  # Predict missing Y values
  cat("Predicting missing Y values...\n")
  X_miss <- X[miss_idx, , drop = FALSE]
  Y_imputed <- Y
  Y_imputed[miss_idx] <- cbind(1, X_miss) %*% c(1, betahat)  # Add the intercept term
  
  return(list(
    Yhat = as.numeric(Y_imputed),
    betahat = as.numeric(betahat)
  ))
}