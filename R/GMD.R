#' @name GMD
#' @title Generate Missing Data function
#' @description This function generates missing data in a specified column of a 
#' data frame according to a given missing ratio.
#' @param data A data frame containing the linear regression model dataset
#' @param ratio The missing ratio (e.g., 0.5 means 1/2 of data will be made missing)
#' @usage GMD(data, ratio)
#' @return 
#' \item{data0}{A modified version of `data` with missing values inserted.}
#' @examples
#' set.seed(123) # for reproducibility
#' data <- data.frame(x = 1:10, y = rnorm(10))
#' modified_data <- GMD(data, ratio = 0.5)
#' summary(modified_data)
#' @export
GMD <- function(data, ratio) {
  n <- nrow(data)
  nob <- round(n * (1 - ratio))  # Calculate the number of non-missing observations
  data0 <- data
  data0[sample(n, n - nob), 1] <- NA  # Randomly select rows and make the first column NA
  return(data0)
}