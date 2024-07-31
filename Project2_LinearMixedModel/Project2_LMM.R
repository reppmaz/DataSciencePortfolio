# This script analyzes psychological and physiological stress markers using
# linear mixed-effects models. 

# steps include:
# - normalization of continuous Variables
# - model fitting and summary
# separate models are fit for each response variable (affect, heart rate,
# cortisol, and alpha-amylase),
# including fixed effects for phase, age, gender, site,
# and oral_contraceptive (only for salivary markers)
# and a random effect for participants

# Load necessary libraries
library(lme4) 
library(ggplot2)
library(lmerTest)
library(forcats)
library(tidyr)
library(dplyr)

# Import and clean data
data <- read.csv('path_to_data/PreprocessedDataAndFeatures.csv') %>%
  na.omit()

# Convert relevant columns to factors
factor_vars <- c('phase', 'gender', 'site', 'oral_contraceptive')
data[factor_vars] <- lapply(data[factor_vars], as.factor)

# Relevel 'phase' factor and check levels
data$phase <- fct_relevel(data$phase, "stress")
print(levels(data$phase))

# Normalize continuous variables
continuous_vars <- c('affect', 'heartrate', 'cortisol', 'alphaamylase')

if (!all(continuous_vars %in% colnames(data))) {
  stop("Some continuous variables are missing in the 'data' DataFrame.")
}

data[continuous_vars] <- scale(data[continuous_vars])

# Define a function to fit and summarize models
fit_and_summarize <- function(formula, data) {
  model <- lmer(formula, data = data)
  summary(model)
}

# Fit and summarize models for each response variable
fit_and_summarize(affect ~ phase + age + gender + site + (1 | ID), data)
fit_and_summarize(heartrate ~ phase + age + gender + site + (1 | ID), data)
fit_and_summarize(cortisol ~ phase + age + gender + site + oral_contraceptive + (1 | ID), data)
fit_and_summarize(alphaamylase ~ phase + age + gender + site + oral_contraceptive + (1 | ID), data)
