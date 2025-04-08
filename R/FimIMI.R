#' FimIMI: Multiple Runs of Improved Multiple Imputation (IMI)
#'
#' This function performs multiple runs of the Improved Multiple Imputation (IMI) estimation
#' and collects the results. It is designed to facilitate batch processing and repeated runs of IMI.
#' @param d The data structure.
#' @param R Number of runs to perform.
#' @param n Vector of sample sizes for each group.
#' @param M Number of multiple imputations per run.
#' @param batch Batch number (default is 0). This can be used to distinguish different batches of runs.
#'
#' @return A list containing:
#' \item{R}{Vector of run numbers.}
#' \item{Beta}{Matrix of regression coefficients for each run.}
#' \item{comm}{Vector of indicator variables for each run.}
#'
#' @details
#' This function assumes that the data structure \code{d} is properly defined and contains the necessary information.
#' The function repeatedly calls the \code{IMI} function and collects the regression coefficients and indicator variables.
#'
#' @examples
#' # Example data
#' set.seed(123)
#' n <- c(300, 300, 400)  # Sample sizes for each group
#' p <- 5  # Number of independent variables
#' d <- list(p = p, Y = rnorm(sum(n)), X0 = matrix(rnorm(sum(n) * p), ncol = p))
#'
#' # Call FimIMI function
#' result <- FimIMI(d = d, R = 10, n = n, M = 20, batch = 1)
#'
#' # View results
#' print(result$Beta)  # Regression coefficients for each run
#'
#' @export
FimIMI <- function(d,R, n, M, batch = 0) {
    K <- length(n)  # Number of groups
    p <- d$p  # Number of independent variables (assuming d is a list containing p)
    Beta <- matrix(0, p, R)  # Matrix to store regression coefficients for each run
    comm <- rep(0, R)  # Vector to store indicator variables for each run

    for (r in 1:R) {
        fit <- IMI(d = d, M = M, midx = c(1), n = n)  # Call IMI function
        Beta[, r] <- fit$betahat  # Extract regression coefficients
        comm[r] <- fit$comm  # Extract indicator variable
    }

    # Return results
    list(
        R = batch + 1:R,  # Vector of run numbers
        Beta = Beta,  # Matrix of regression coefficients
        comm = comm  # Vector of indicator variables
    )
}