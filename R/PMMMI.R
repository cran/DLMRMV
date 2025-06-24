#' Predictive Mean Matching with Multiple Imputation
#' 
#' Implements PMM algorithm for handling missing data in linear regression models.
#' Uses chained equations approach to generate multiple imputed datasets and 
#' pools results using Rubin's rules.
#'
#' @param data Dataframe with response variable in 1st column and predictors in others
#' @param k Number of nearest neighbors for matching (default=5)
#' @param m Number of imputations (default=5)
#' @return List containing:
#' \item{Y}{Original response vector with NAs}
#' \item{Yhat}{Final imputed response vector (averaged across imputations)}
#' \item{betahat}{Pooled regression coefficients}
#' \item{imputations}{List of m completed datasets}
#' \item{m}{Number of imputations performed}
#' \item{k}{Number of neighbors used}
#' @export
#' @examples
#' # Create dataset with 30% missing values

#' data <- data.frame(Y=c(rnorm(70),rep(NA,30)), X1=rnorm(100))
#' results <- PMMI(data, k=5, m=5)
PMMI <- function(data, k=5, m=5) {
  # Ensure the first column is the target variable
  target_var <- names(data)[1]
  Y <- data[,1]
  
  # Initialize output structure
  result <- list(
    Yhat = vector("list", m),
    betahat = vector("list", m),
    imputed_data = vector("list", m)
  )
  
  for (imp in 1:m) {
    temp_data <- data
    na_idx <- which(is.na(Y))
    
    if (length(na_idx) > 0) {
      # Build model using complete cases
      complete_data <- temp_data[!is.na(Y), ]
      model <- stats::lm(stats::as.formula(paste(target_var, "~ .")), 
                 data=complete_data)
      
      # Predictive mean matching
      pred <- stats::predict(model, newdata=temp_data[na_idx, -1, drop=FALSE])
      y_obs <- complete_data[[target_var]]
      
      # Perform KNN matching
      for (i in seq_along(na_idx)) {
        distances <- abs(pred[i] - y_obs)
        candidates <- order(distances)[1:min(k, length(y_obs))]
        temp_data[na_idx[i], 1] <- sample(y_obs[candidates], 1)
      }
      
      # Store results
      result$betahat[[imp]] <- coef(model)
      result$Yhat[[imp]] <- predict(model, newdata=temp_data)
    } else {
      model <- lm(as.formula(paste(target_var, "~ .")), data=temp_data)
      result$betahat[[imp]] <- coef(model)
      result$Yhat[[imp]] <- predict(model)
    }
    result$imputed_data[[imp]] <- temp_data
  }
  
  # Combine results using Rubin's rules
  final_betahat <- colMeans(do.call(rbind, result$betahat))
  final_Yhat <- rowMeans(do.call(cbind, result$Yhat))
  
  return(list(
    Y = Y,
    Yhat = final_Yhat,
    betahat = final_betahat,
    imputations = result$imputed_data,
    m = m,
    k = k
  ))
}

