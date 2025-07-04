% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fiMI.R
\name{fiMI}
\alias{fiMI}
\title{fiMI: Predict Missing Response Variables using Multiple Imputation}
\usage{
fiMI(data, R, n, M)
}
\arguments{
\item{data}{\code{data.frame} containing the linear regression model dataset with missing response variables.}

\item{R}{Number of runs for multiple imputation.}

\item{n}{Number of rows in the dataset.}

\item{M}{Number of multiple imputations per run.}
}
\value{
A list containing:
\item{Yhat}{Predicted response values with missing values imputed.}
}
\description{
This function predicts missing response variables in a linear regression dataset
using multiple imputation. It leverages the \code{FimIMI} function to perform multiple runs
of improved multiple imputation and averages the regression coefficients to predict
the missing response values.
}
\details{
This function assumes that the first column of \code{data} is the response variable
and the remaining columns are the independent variables. The function uses the \code{FimIMI}
function to perform multiple runs of improved multiple imputation and averages the
regression coefficients to predict the missing response values.
}
\examples{
# Example data
set.seed(123)
n <- 1000  # Number of rows
p <- 5  # Number of independent variables
data <- data.frame(Y = rnorm(n), X1 = rnorm(n), X2 = rnorm(n))
data[sample(n, 100), 1] <- NA  # Introduce missing response values

# Call fiMI function
result <- fiMI(data, R = 10, n = n, M = 20)

# View results
print(result$Yhat)  # Predicted response values

}
