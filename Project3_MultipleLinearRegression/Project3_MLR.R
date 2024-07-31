# Resilience Prediction Using Brain Network Connectivity
# This script performs a series of linear regression analyses to explore
# the relationship between brain network connectivity and psychological resilience.
  
# steps include:
# weight calculation for each site
# iterating over brain networks (SN, DMN, CEN, SNDMN, SNCEN, CENDMN)
# and performing separate weighted linear regressions for each modality.
# independent variables: pre-stress, stress reactivity, and recovery
# covariates: age, site, gender, and handedness
# dependent variable: psychological resilience
# model evaluation
# plotting

library(readr)
library(dplyr)
library(stats)
library(car)
library(tidyr)
library(coefplot)

# Load data
path <- "/path_to_data/ProcessedDataAndFeatures.csv"
df <- read_csv(path)

# Ensure covariate is factors
df$handedness <- as.factor(df$handedness_write)

# Compute weights for each site
std_devs <- df %>% group_by(site) %>% summarise(std_deviation = sd(resilience))
weights <- 1 / std_devs$std_deviation
df$weights <- weights[df$site]

# Linear regressions: brain network connectivity -> resilience
modalities <- c('SN', 'DMN', 'CEN', 'SNDMN', 'SNCEN', 'CENDMN')
covars <- c('age', 'site', 'gender', 'handedness')
p_values <- c()

for (modality in modalities) {
  cat(paste("\n------\n", toupper(modality), "Linear Regression Resilience\n------\n"))
  
  # Columns of interest
  cols_of_interest <- c(paste0(modality, '_pre'), paste0(modality, '_reactivity'), paste0(modality, '_recovery'), covars)
  df_cleaned <- df %>% select(all_of(c(cols_of_interest, 'resilience', 'weights'))) %>% filter(complete.cases(.))
  # Fit the model
  model_formula <- as.formula(paste('resilience ~ ', paste0(cols_of_interest, collapse = ' + ')))
  model <- lm(model_formula, data = df_cleaned, weights = df_cleaned$weights)
  # Extract and store p-values
  model_pvalues <- summary(model)$coefficients[-1,4]  # -1 to skip intercept
  p_values <- c(p_values, model_pvalues)
  
  print(summary(model))
  print(p_values)
}
#---------------
# Visualization
# Residuals vs. Fitted values plot
plot(model$fitted.values, model$residuals, 
     main = "Residuals vs. Fitted values", 
     xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")

# Normal Q-Q plot
qqnorm(model$residuals, main = "Normal Q-Q plot")
qqline(model$residuals)

# Scale-Location plot
sqrt_abs_res <- sqrt(abs(model$residuals))
plot(model$fitted.values, sqrt_abs_res, 
     main = "Scale-Location Plot", 
     xlab = "Fitted values", ylab = "Square root of |Residuals|")

# Cook's distance plot
cooksd <- cooks.distance(model)
plot(cooksd, main = "Cook's distance plot", ylab = "Cook's distance")
abline(h = 4*mean(cooksd, na.rm = TRUE), col = "red")

# Coefficient plots
coefplot(model, main = "Coefficient plot")

