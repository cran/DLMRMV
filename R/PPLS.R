#' Penalized Partial Least Squares (PPLS) Estimation
#'
#' This function performs Penalized Partial Least Squares (PPLS) estimation for grouped data.
#' It supports ridge regression regularization and handles missing data by excluding incomplete cases.
#' The function returns regression coefficients, residual sum of squares, and other diagnostic information.
#'
#' @param d Containing the dependent and independent variables.
#' @param yidx Column index of the dependent variable in \code{d}.
#' @param Xidx Column indices of the independent variables in \code{d}.
#' @param n Vector of sample sizes for each group.
#' @param lam Regularization parameter for ridge regression (default is 0.005).
#'
#' @return A list containing the following elements:
#' \item{beta}{Regression coefficients.}
#' \item{SSE}{Residual sum of squares.}
#' \item{df}{Number of complete cases used in the estimation.}
#' \item{gram}{Gram matrix (\eqn{X^TX + \lambda I}).}
#' \item{cgram}{Cholesky decomposition of the Gram matrix.}
#' \item{comm}{Indicator variable (0 for single group, 1 for multiple groups).}
#'
#' @details
#' This function assumes that the data is grouped and that the sample sizes for each group are provided.
#' It excludes cases with missing values in the dependent or independent variables.
#' The function uses Cholesky decomposition to solve the regularized least squares problem.
#'
#' @examples
#' # Example data
#' set.seed(123)
#' n_total <- 1000
#' p <- 5
#' n_groups <- c(300, 300, 400)
#' d <- list(all = cbind(rnorm(n_total), matrix(rnorm(n_total*p), ncol=p)),p = p)
#'
#' # Call PPLS function
#' result <- PPLS(d, yidx=1, Xidx=2:(p+1), n=n_groups)
#'
#' # View results
#' print(result$beta)  # Regression coefficients
#' print(result$SSE)   # Residual sum of squares
#'
#' @export
PPLS <- function(d, yidx, Xidx, n, lam = 0.005) {
    p <- length(Xidx)  # Number of independent variables
    K <- length(n)     # Number of groups
    N <- sum(n)        # Total sample size
    complete_cases <- complete.cases(d$all[1:N, c(yidx, Xidx)])
    d$complete_cases <- complete_cases
    df <- sum(complete_cases)  # Number of complete cases
    idx <- which(complete_cases)  # Indices of complete cases

    # Extract complete data
    Xk <- d$all[idx, Xidx, drop=FALSE]
    yk <- d$all[idx, yidx]

    # Compute Gram matrix with regularization
    XX <- t(Xk) %*% Xk + diag(lam, p)
    Xy <- t(Xk) %*% yk
    yy <- sum(yk^2)

    # Solve using Cholesky decomposition
    cA <- tryCatch(chol(XX), error = function(e) {
        warning("Cholesky decomposition failed. Using identity matrix.")
        diag(p)
    })
    beta <- backsolve(cA, forwardsolve(t(cA), Xy))

    # Compute residual sum of squares
    SSE <- yy - sum(Xy * beta)

    # Indicator for single vs. multiple groups
    comm <- ifelse(K == 1, 0, 1)

    # Return results
    list(
        beta = beta,    # Regression coefficients
        SSE = SSE,     # Residual sum of squares
        df = df,       # Number of complete cases
        gram = XX,     # Gram matrix
        cgram = cA,    # Cholesky decomposition
        comm = comm    # Group indicator
    )
}