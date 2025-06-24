
#' 
#'Exponentially Weighted Recursive Least Squares with Missing Value Imputation
#'
#' @param data Linear regression dataset (1st column as Y, others as X)
#' @param rho Regularization parameter 
#' @param lambda Forgetting factor
#' @param nb Maximum iterations 
#' @param niter Initial iteration count (typically 1)
#' @return List containing:
#' \item{Yhat}{Imputed response vector}
#' \item{betahat}{Estimated coefficients}
#' @export
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   y = c(rnorm(50), rep(NA, 10)),
#'   x1 = rnorm(60),
#'   x2 = rnorm(60)
#' )
#' result <- ERLS(data, rho = 0.01, lambda = 0.95, nb = 100, niter = 1)
#' head(result$Yhat)
ERLS <- function(data, rho=0.01, lambda=0.95, nb=100, niter=1) {
  # Input validation
  if(!is.data.frame(data)) stop("Input must be a dataframe")
  if(ncol(data)<2) stop("Data requires at least 1 predictor")
  if(any(!sapply(data, is.numeric))) stop("Non-numeric variables detected")
  
  # Data preparation
  n <- nrow(data)
  X0 <- as.matrix(data[,-1,drop=FALSE])
  Y <- as.matrix(data[,1])
  p <- ncol(X0)
  delta <- ifelse(is.na(Y), 0, 1)  # Missing value indicator
  
  # Initialization
  Pstar <- (rho)^-1 * diag(p)  # Initial precision matrix
  betastar <- matrix(rnorm(p), p, 1)  # Initial coefficients
  Y2 <- matrix(0, n, 1)  # Placeholder for imputed values
  
  # Main iteration loop
  for(iter in seq_len(nb)){
    for(i in seq_len(n)){
      xi <- matrix(X0[i,], ncol=1)
      y_pred <- crossprod(xi, betastar)[1,1]
      Y2[i] <- ifelse(delta[i]==0, y_pred, Y[i])
      
      # Kalman update
      K <- Pstar %*% xi / (lambda + crossprod(xi, Pstar %*% xi)[1,1])
      e <- if(delta[i]==0) 0 else (Y[i] - y_pred)
      betastar <- betastar + K * e
      Pstar <- (Pstar - tcrossprod(K, xi) %*% Pstar)/lambda
    }
  }
  
  # Result compilation
  Y[is.na(Y)] <- Y2[is.na(Y)]
  list(Yhat=Y, betahat=betastar)
}