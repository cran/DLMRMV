#' Least Squares Estimation for Grouped Data with Ridge Regularization
#'
#' This function implements the least squares estimation for grouped data, 
#' supporting ridge regression regularization. It can handle missing data 
#' and returns regression coefficients and the sum of squared residuals for each group.
#'
#' @param d A data frame containing dependent and independent variables.
#' @param yidx The column index of the dependent variable.
#' @param Xidx The column indices of the independent variables.
#' @param n A vector of starting indices for the groups.
#' @param lam Regularization parameter for ridge regression, default is 0.005.
#'
#' @return A list containing the following elements:
#' \item{beta}{A matrix of regression coefficients for each group.}
#' \item{SSE}{The sum of squared residuals for each group.}
#' \item{df}{The sample size for each group.}
#' \item{gram}{The Gram matrix for each group.}
#' \item{cgram}{The Cholesky decomposition result for each group.}
#' \item{comm}{An unused variable (reserved for future expansion).}
#'
#' @examples
#' # Example data
#' set.seed(123)
#' n <- 1000
#' p <- 5
#' d <- list(all = cbind(rnorm(n), matrix(rnorm(n*p), ncol=p)))
#'
#' # Call the LS function
#' result <- LS(d, yidx = 1, Xidx = 2:(p + 1), n = c(1, 300, 600, 1000))
#'
#' # View the results
#' print(result$beta)  # Regression coefficients
#' print(result$SSE)   # Sum of squared residuals
#'
#' @export
LS <- function(d, yidx, Xidx, n, lam = 0.005) {
    # Initialize parameters
    p <- length(Xidx)  # Number of independent variables
    K <- length(n)     # Number of groups
    ni <- diffinv(n)   # Cumulative indices
    XX <- array(0, c(p, p, K))  # Store Gram matrices
    df <- rep(0, K)    # Sample size for each group
    cA <- array(0, c(p, p, K))  # Store Cholesky decomposition results
    beta <- matrix(0, p, K)     # Store regression coefficients
    SSE <- rep(0, K)            # Store sum of squared residuals
    
    # Filter complete cases
    complete_cases <- complete.cases(d$all[, c(yidx, Xidx)])
    d$complete_cases <- complete_cases
    
    # Process each group
    for (k in 1:K) {
        # Extract complete observations for the current group
        idx <- ni[k] + which(complete_cases[(ni[k] + 1):ni[k + 1]])
        df[k] <- length(idx)
        
        # Extract dependent and independent variables for the current group
        Xk <- matrix(d$all[idx, Xidx], df[k], p)
        yk <- matrix(d$all[idx, yidx], df[k], 1)
        
        # Compute Gram matrix and add regularization term
        XX[,,k] <- crossprod(Xk) + diag(lam, p)
        Xy <- crossprod(Xk, yk)
        yy <- sum(yk^2)
        
        # Solve the linear equations using Cholesky decomposition
        cA[,,k] <- chol(XX[,,k])
        beta[,k] <- backsolve(cA[,,k], forwardsolve(t(cA[,,k]), Xy))
        
        # Compute the sum of squared residuals
        SSE[k] <- yy - sum(Xy * beta[,k])
    }
    
    # Return the results
    list(
        beta = beta,  # Regression coefficients
        SSE = SSE,    # Sum of squared residuals
        df = df,      # Sample size for each group
        gram = XX,    # Gram matrices
        cgram = cA,   # Cholesky decomposition results
        comm = 0      # Unused variable
    )
}