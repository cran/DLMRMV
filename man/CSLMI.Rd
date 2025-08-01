% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CSLMI.R
\name{CSLMI}
\alias{CSLMI}
\title{CSLMI: Consensus-based Stochastic Linear Multiple Imputation (Simplified Version)}
\usage{
CSLMI(data, M)
}
\arguments{
\item{data}{Dataframe with response variable in 1st column and predictors in others}
\item{M}{Number of imputations}}
\value{
  A list containing:
  \item{Yhat}{Imputed response values.}
  \item{betahat}{Average regression coefficients across imputations.}
  \item{comm}{Communication cost (number of messages passed).}
  A list containing the following components:
  \item{Yhat}{Imputed response vector with missing values filled in.}
  \item{betahat}{Final regression coefficients.}
}
\description{
Performs multiple imputation and parameter estimation using a consensus-based approach.
  The response variable is in the first column,
   all other columns are predictors,
   missing values are automatically detected,
   the whole dataset is treated as one block.
}
\examples{
set.seed(123)
data <- data.frame(
  y = c(rnorm(50), rep(NA, 10)),
  x1 = rnorm(60),
  x2 = rnorm(60)
)
result <- CSLMI(data = data, M = 10)
head(result$Yhat)
print(result$betahat)
print(result$comm)
}
