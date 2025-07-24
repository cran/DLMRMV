#' Distributed Full-information Multiple Imputation (DfiMI) using LASSO
#'
#' Performs multiple imputation of response variable Y via R independent 
#' runs and M stochastic imputations per run. Missing Y values are imputed 
#' using LASSO regression on predictors.
#'
#' @param data Data frame where:
#'   - First column contains response Y (may contain NA)
#'   - Remaining columns contain numeric predictors
#' @param R Positive integer - number of simulation runs for stable coefficient estimation
#' @param M Positive integer - number of multiple imputations per run
#'
#' @return A list containing:
#'   \item{Yhat}{The vector of Y with missing values imputed.}
#'   \item{betahat}{Final averaged regression coefficient estimates used for imputation.}
#'
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   Y = c(rnorm(50), rep(NA, 10)),  # 50 observed + 10 missing
#'   X1 = rnorm(60),
#'   X2 = rnorm(60)
#' )
#' res <- DfiMI_lasso(data, R = 3, M = 5)
#' head(res$Yhat)

DfiMI_lasso <- function(data, R, M) {
  # Package dependency check
  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("Package 'glmnet' required. Please install with: install.packages('glmnet')")
  
  # Input validation
  if (!is.data.frame(data)) stop("Input must be a data frame")
  if (ncol(data) < 2) stop("Data must contain at least one predictor")
  if (!all(vapply(data, is.numeric, logical(1L))))
    stop("All columns must be numeric")

  p <- ncol(data) - 1L  # Number of predictors
  
  # Initialize coefficient storage (includes intercept)
  Beta <- matrix(0, nrow = p, ncol = R) 
  
  # Main loop: R simulation runs
  cat(sprintf("Executing DfiMI_lasso with %d runs...\n", R))
  for (r in seq_len(R)) {
    cat(sprintf("  Run %2d of %d\n", r, R))
    
    # Step 1: Initialize imputation data
    d_imp <- as.matrix(data)
    miss  <- is.na(d_imp[, 1L])  # Missing value indices in Y
    
    # Step 2: Perform M imputations
    betahats <- matrix(0, p, M)  # Store coefficients (excluding intercept)
    for (m in seq_len(M)) {
      # Create copy for current imputation
      d_temp <- d_imp
      
      #1. Initialize missing Y with mean + noise
      if (any(miss)) {
        obs_idx <- which(!miss)
        y_mean  <- mean(d_temp[obs_idx, 1L], na.rm = TRUE)
        y_sd    <- sd(d_temp[obs_idx, 1L], na.rm = TRUE)
        d_temp[miss, 1L] <- rnorm(sum(miss), mean = y_mean, sd = y_sd)
      }
      
      #2. Fit model using complete cases
      cc <- complete.cases(d_temp)
      X  <- as.matrix(d_temp[cc, -1L, drop = FALSE])
      Y  <- d_temp[cc, 1L]
      
      #3. LASSO regression with cross-validation
      fit <- glmnet::glmnet(X, Y, alpha = 1)  # alpha=1 for LASSO
      cv  <- glmnet::cv.glmnet(X, Y, alpha = 1)
      # Store coefficients (excluding intercept)
      betahats[, m] <- as.numeric(coef(cv, s = "lambda.min"))[-1]
    }
    
    #4. Average coefficients across M imputations
    Beta[, r] <- rowMeans(betahats, na.rm = TRUE)
  }
  
  # Final average coefficients across all R runs
  betahat <- rowMeans(Beta, na.rm = TRUE)
  
  # Prepare data for final imputation
  Y  <- as.numeric(data[[1L]])  # Response variable
  X  <- as.matrix(data[, -1L, drop = FALSE])  # Predictors
  na <- is.na(Y)  # Missing value indices
  
  # Return original data if no missing values
  if (!any(na)) {
    cat("No missing values found; returning original data.\n")
    return(list(Yhat = Y, betahat = betahat))
  }
  
  # Final imputation of missing Y values
  cat("Predicting missing Y values...\n")
  cc  <- complete.cases(data)  # Complete cases
  # Fit LASSO model to complete cases
  cv  <- glmnet::cv.glmnet(X[cc, ], Y[cc], alpha = 1)
  # Predict missing values
  Y[na] <- predict(cv, newx = X[na, , drop = FALSE], s = "lambda.min")
  
  # Return results
  list(Yhat = as.numeric(Y), betahat = as.numeric(betahat))
}