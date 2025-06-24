
#' EM Algorithm for Linear Regression with Missing Data
#'
#' @param data Dataframe with first column as response (Y) and others as predictors (X)
#' @param d Initial convergence threshold (default=1)
#' @param tol Termination tolerance (default=1e-6)
#' @param nb Maximum iterations (default=100)
#' @param niter Starting iteration counter (default=1)
#' @return List containing:
#' \item{Yhat}{Imputed response vector}
#' \item{betahat}{Estimated coefficients}
#' @export
#'
#' @examples
#' # Generate data with 20% missing Y values
#' set.seed(123)
#' data <- data.frame(Y=c(rnorm(80),rep(NA,20)), X1=rnorm(100), X2=rnorm(100))
#' 
#' # Run EM algorithm
#' result <- EMRE(data, d=1, tol=1e-5, nb=50)
#' print(result$betahat) # View coefficients
EMRE <- function(data, d=1, tol=1e-6, nb=100, niter=1){
  # Input validation
  if(!is.data.frame(data)) stop("Data must be dataframe")
  if(ncol(data)<2) stop("Requires at least 1 predictor")
  
  # Data preparation
  Y <- as.matrix(data[,1])
  X0 <- as.matrix(data[,-1])
  p <- ncol(X0)
  nobs <- sum(!is.na(Y))
  
  # Initialization
  obs_idx <- which(!is.na(Y))
  Xobs <- X0[obs_idx,]
  Yobs <- matrix(Y[obs_idx], ncol=1)
  betahat <- solve(t(Xobs)%*%Xobs) %*% t(Xobs)%*%Yobs
  
  # EM iterations
  while((d >= tol) && (niter <= nb)){
    beta_old <- betahat
    
    # E-step: Impute missing Y
    mis_idx <- which(is.na(Y))
    if(length(mis_idx)>0){
      Xmis <- X0[mis_idx,]
      Ymis <- Xmis %*% beta_old
      Y[mis_idx] <- Ymis
    }
    
    # M-step: Update coefficients
    betahat <- solve(t(X0)%*%X0) %*% t(X0)%*%Y
    d <- sqrt(mean((betahat-beta_old)^2))
    niter <- niter+1
  }
  
  list(Yhat=Y, betahat=betahat)
}
