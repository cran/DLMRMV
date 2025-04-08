#' fiMI: Predict Missing Response Variables using Multiple Imputation
#'
#' This function predicts missing response variables in a linear regression dataset
#' using multiple imputation. It leverages the \code{FimIMI} function to perform multiple runs
#' of improved multiple imputation and averages the regression coefficients to predict
#' the missing response values.
#'
#' @param data \code{data.frame} containing the linear regression model dataset with missing response variables.
#' @param R Number of runs for multiple imputation.
#' @param n Number of rows in the dataset.
#' @param M Number of multiple imputations per run.
#'
#' @return A list containing:
#' \item{Yhat}{Predicted response values with missing values imputed.}
#'
#' @details
#' This function assumes that the first column of \code{data} is the response variable
#' and the remaining columns are the independent variables. The function uses the \code{FimIMI}
#' function to perform multiple runs of improved multiple imputation and averages the
#' regression coefficients to predict the missing response values.
#'
#' @examples
#' # Example data
#' set.seed(123)
#' n <- 1000  # Number of rows
#' p <- 5  # Number of independent variables
#' data <- data.frame(Y = rnorm(n), X1 = rnorm(n), X2 = rnorm(n))
#' data[sample(n, 100), 1] <- NA  # Introduce missing response values
#'
#' # Call fiMI function
#' result <- fiMI(data, R = 10, n = n, M = 20)
#'
#' # View results
#' print(result$Yhat)  # Predicted response values
#'
#' @export
fiMI <- function(data, R, n, M) {
    
    if(ncol(data) < 2) stop("Data must contain at least 1 predictor")
    
    d <- list(
        p = ncol(data) - 1,
        Y = data[,1],
        X0 = as.matrix(data[,-1]),
        all = as.matrix(data)
    )
    
    fit <- FimIMI(d = d, R = R, n = n, M = M, batch = 0)
    
    Yhat <- data[,1]
    na_idx <- is.na(Yhat)
    Yhat[na_idx] <- as.matrix(data[na_idx, -1]) %*% rowMeans(fit$Beta)
    
    list(Yhat = Yhat)
}