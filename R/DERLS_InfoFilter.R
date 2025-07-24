#' @title Distributed Exponentially Weighted Recursive Least Squares (DERLS) using Information Filter
#' @description Impute missing values in the response variable Y using distributed ERLS method.
#'              Multiple independent runs are performed to stabilize coefficient estimates.
#'              Missing values are imputed recursively and refined over multiple iterations.
#' @param data A data frame whose first column is the response Y (with possible NAs),
#'             and remaining columns are predictors X.
#' @param rho Regularization parameter
#' @param lambda Forgetting factor
#' @param R Number of independent runs to stabilize estimates
#' @param nb Number of iterations per run
#' @return A list containing:
#'   \item{Yhat}{The vector of Y with missing values imputed.}
#'   \item{betahat}{Final averaged regression coefficient estimates used for imputation.}
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' n <- 60
#' data <- data.frame(
#'   Y = c(rnorm(n - 10), rep(NA, 10)),
#'   X1 = rnorm(n),
#'   X2 = rnorm(n)
#' )
#' result <- DERLS_InfoFilter(data, rho = 0.01, lambda = 0.95, R = 3, nb = 50)
#' head(result$Yhat) # inspect imputed Y
#' result$betahat      # inspect estimated coefficients

DERLS_InfoFilter <- function(data, rho, lambda, R, nb) {
  # Input validation
  if (!is.data.frame(data)) stop("Input must be a dataframe")
  if (ncol(data) < 2) stop("Data requires at least one predictor")
  if (any(!sapply(data, is.numeric))) stop("All variables must be numeric")

  n <- nrow(data)
  p <- ncol(data) - 1L  # number of predictors
  Y <- as.matrix(data[, 1])  # Extract response variable Y
  X <- as.matrix(data[, -1])  # Extract predictors X
  miss_idx <- which(is.na(Y))  # Indices of missing values in Y
  n_miss <- length(miss_idx)  # Number of missing values

  # Store beta estimates from each run
  Beta <- matrix(0, nrow = p, ncol = R)

  # Main loop over R runs
  cat("Running DERLS with", R, "runs...\n")
  for (r in 1:R) {
    cat("Run", r, "of", R, "\n")
    
    # Step 1: Initial imputation using mean + noise
    Y_imp <- Y
    if (n_miss > 0) {
      obs <- Y_imp[!is.na(Y_imp)]
      mu <- mean(obs)
      sigma <- sd(obs)
      Y_imp[miss_idx] <- rnorm(n_miss, mean = mu, sd = sigma)
    }

    # Step 2: Initialize ERLS parameters
    delta <- ifelse(is.na(Y), 0, 1)
    Pinv <- rho * diag(p)  # Information matrix (inverse of covariance matrix)
    eta <- matrix(rnorm(p), p, 1)  # Information vector

    # Step 3: Online recursive update
    for (iter in seq_len(nb)) {
      for (i in seq_len(n)) {
        xi <- matrix(X[i, ], ncol = 1)
        y_pred <- crossprod(xi, solve(Pinv, eta))[1]
        e <- if (delta[i] == 0) 0 else (Y_imp[i] - y_pred)

        # Information filter update
        Pinv <- Pinv + (1 / lambda) * tcrossprod(xi)
        eta <- eta + (1 / lambda) * e * xi
      }
    }

    # Store results (including intercept)
    Beta[, r] <- solve(Pinv, eta)
  }

  # Average over all R runs
  betahat <- apply(Beta, 1, mean, na.rm = TRUE)

  # Final prediction on missing values
  Y_imputed <- Y
  if (n_miss > 0) {
    X_miss <- X[miss_idx, , drop = FALSE]
    Y_imputed[miss_idx] <- as.vector(cbind(1, X_miss) %*% c(1, betahat))  # Add intercept
  }

  list(Yhat = Y_imputed, betahat = as.numeric(betahat))
}