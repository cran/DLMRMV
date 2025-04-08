#' Improved Multiple Imputation (IMI) Estimation
#'
#' This function performs Improved Multiple Imputation (IMI) estimation for grouped data with missing values.
#' It iteratively imputes missing values using the \code{LS} function and estimates regression coefficients using the \code{PPLS} function.
#' The final regression coefficients are averaged across multiple imputations.
#'
#' @param d \code{data.frame} containing the dependent variable (\code{Y}) and independent variables (\code{X}).
#' @param M Number of multiple imputations to perform.
#' @param midx Column indices of the missing variables in \code{d}.
#' @param n Vector of sample sizes for each group.
#'
#' @return A list containing the following elements:
#' \item{betahat}{Average regression coefficients across all imputations.}
#' \item{comm}{Indicator variable (0 for single group, 1 for multiple groups).}
#'
#' @details
#' The function assumes the data is grouped and contains missing values in specified columns (\code{midx}).
#' It uses the \code{LS} function to impute missing values and the \code{PPLS} function to estimate regression coefficients.
#' The process is repeated \code{M} times, and the final regression coefficients are averaged.
#' @importFrom stats rgamma rnorm diffinv complete.cases
#' @examples
#' # Example data
#' 
#' set.seed(123)
#' n <- c(300, 300, 400)  # Sample sizes for each group
#' p <- 5  # Number of independent variables
#' Y <- rnorm(sum(n))  # Dependent variable
#' X0 <- matrix(rnorm(sum(n) * p), ncol = p)  # Independent variables matrix
#' d <- list(p = p, Y = Y, X0 = X0)  # Data list
#' d$all <- cbind(Y, X0)
#'# Indices of missing variables (assuming some variables are missing)
#'midx <- c(2, 3)  # For example, the second and third variables are missing
#'# Call IMI function
#'result <- IMI(d, M = 5, midx = midx, n = n)
#'# View results
#'print(result$betahat)  # Average regression coefficients
#'
#' @export
IMI <- function(d, M, midx, n) {
    p <- d$p  # Number of independent variables
    q <- length(midx)  # Number of missing variables
    N <- sum(n)  # Total sample size
    K <- length(n)  # Number of groups
    ni <- diffinv(n)  # Cumulative indices for each group
    betahats <- matrix(0, d$p, M)  # Matrix to store regression coefficients for each imputation
    d$all <- cbind(d$Y, d$X0)  # Combine dependent and independent variables
    miss <- is.na(d$all)  # Logical matrix for missing values
    comm <- 0  # Indicator variable
    fit.imp <- NULL  # List to store imputation results

    # Impute missing values for each missing variable
    for (j in 1:q) {
        fit.imp[[j]] <- LS(d, midx[j], setdiff(1:(p + 1), midx), n)
    }

    # Perform multiple imputations
    for (m in 1:M) {
        for (k in 1:K) {
            for (j in 1:q) {
                sig <- sqrt(1 / rgamma(1, (fit.imp[[j]]$df[k] + 1) / 2, (fit.imp[[j]]$SSE[k] + 1) / 2))
                alpha <- fit.imp[[j]]$beta[, k] + sig * backsolve(fit.imp[[j]]$cgram[,, k], rnorm(p))
                idx <- ni[k] + which(miss[(ni[k] + 1):ni[k + 1], midx[j]])
                d$all[idx, midx[j]] <- d$all[idx, setdiff(1:(p + 1), midx)] %*% alpha + rnorm(length(idx), 0, sig)
            }
        }

        # Estimate regression coefficients using PPLS
        fit <- PPLS(d, 1, 2:(p + 1), sum(n))
        betahats[, m] <- fit$beta
    }

    # Calculate average regression coefficients
    betahat <- apply(betahats, 1, mean)

    # Return results
    list(betahat = betahat, comm = comm)
}