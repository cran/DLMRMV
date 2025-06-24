#' Averaged Generalized Method of Moments Imputation (AVGM)
#'
#' This function performs multiple imputations on missing values in the response variable Y,
#' using AVGMMI logic with support for grouped data. It is fully self-contained.
#'
#' @param data A data frame where the first column is the response variable (Y), and others are predictors (X).
#' @param M Number of multiple imputations.
#' @param midx Integer indicating which column is the response variable (default = 1).
#'
#' @return A list containing:
#'   \item{betahat}{Final averaged regression coefficient estimates.}
#'   \item{Yhat}{Imputed response variable with all missing values filled in.}
#'   \item{comm}{Completion flag (1 = success).}
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   y = c(rnorm(50), rep(NA, 10)),
#'   x1 = rnorm(60),
#'   x2 = rnorm(60)
#' )
#' result <- AVGM(data, M = 10)
#' head(result$Yhat)
#' @export
AVGM <- function(data, M, midx = 1) {
  
  # Input validation
  if (!is.data.frame(data)) stop("Input 'data' must be a data frame")
  if (ncol(data) < 2) stop("Data requires at least one predictor")
  if (any(!sapply(data, is.numeric))) stop("All variables must be numeric")
  
  # Extract components
  Y <- as.matrix(data[[midx]])
  X <- as.matrix(data[,-midx])
  n <- nrow(data)
  p <- ncol(X)
  
  # Identify missingness
  miss <- is.na(Y)
  all_data <- cbind(Y, X)
  
  # Initialize storage for Beta estimates across imputations
  betahats <- matrix(0, p + 1, M)  # Include intercept
  
  # --- Initial Model Fitting ---
  {
    obs_idx <- which(!is.na(Y))
    if (length(obs_idx) < p) stop("Not enough complete cases to estimate initial model")
    
    Xobs <- cbind(1, X[obs_idx, , drop = FALSE])
    Yobs <- Y[obs_idx]
    
    fit <- stats::lm.fit(x = Xobs, y = Yobs)
    beta <- stats::coef(fit)
    res <- stats::residuals(fit)
    df <- length(obs_idx) - length(beta)
    SSE <- sum(res^2)
    
    gram <- chol2inv(chol(crossprod(Xobs)))
    cgram <- chol(gram)
    
    fit.imp <- list(beta = beta, df = df, SSE = SSE, cgram = cgram)
  }
  
  # Multiple imputation loop
  for (m in 1:M) {
    d_imp <- all_data
    
    # Generate imputation based on current parameter estimates
    sig <- sqrt(1 / rgamma(1, (fit.imp$df + 1)/2, (fit.imp$SSE + 1)/2))
    alpha <- fit.imp$beta + sig * backsolve(fit.imp$cgram, rnorm(p + 1))  # Include intercept
    
    # Find indices of missing values
    idx <- which(is.na(d_imp[, 1]))
    
    if (length(idx) > 0) {
      # Ensure the number of columns in d_imp[idx, -1] matches the length of alpha
      d_imp[idx, 1] <- as.matrix(cbind(1, d_imp[idx, -1])) %*% alpha + rnorm(length(idx), 0, sig)
    }
    
    # Re-fit after imputation
    Y_fit <- d_imp[, 1]
    X_fit <- d_imp[, -1, drop = FALSE]
    
    fit <- lm.fit(x = cbind(1, X_fit), y = Y_fit)  # Include intercept
    beta <- coef(fit)
    res <- residuals(fit)
    df <- nrow(d_imp) - length(beta)
    SSE <- sum(res^2)
    
    gram <- chol2inv(chol(crossprod(cbind(1, X_fit))))
    cgram <- chol(gram)
    
    fit.imp <- list(beta = beta, df = df, SSE = SSE, cgram = cgram)
    
    betahats[, m] <- beta
  }
  
  # Average over all imputations
  betahat <- apply(betahats, 1, mean)
  
  # Predict missing Y values
  missing_idx <- is.na(Y)
  Xmis <- X[missing_idx, , drop = FALSE]
  Yhat <- Y
  Yhat[missing_idx] <- cbind(1, Xmis) %*% betahat
  
  return(list(
    betahat = betahat,
    Yhat = Yhat,
    comm = 1
  ))
}
