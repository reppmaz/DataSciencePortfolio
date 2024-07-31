#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:09:06 2023
@author: reppmazc

Random Forest Regression Script

This script implements a Random Forest regression model for predicting
psychological resilience based on various demographic, behavioral, features and various markers
of the stress response (negative affect, salivary cortisol, salivary alpha-amylase, heart rate,
within-network connectivity before and after stress of three large-scale brain networks).
The data were aquired at different study site and one study site is used as the validation dataset
since it is demographically distinct from the other study sites. 

The implemented steps include:
    - Basic sample statistics calculation
    - Dummy coding of categorical variables
    - Identification and removal of near-zero variance features
    - Normalization of continuous variables
    - Feature selection using Recursive Feature Elimination (RFE)
    - Random Forest model training with hyperparameter tuning
    - Model evaluation and validation
    - Feature importance analysis using SHAP values

The input is the preprocessed validation and training datasets (preprocessed though the following script:
Project1_ML_Preprocessing.py). Both datasets contain the exact same columns. The script outputs
the results of the random forest regression. 

"""
# -------
# Imports
# -------
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

# Set seed for reproducibility
np.random.seed(42)

# -------------------------
# LOADING AND DEFINING DATA
# -------------------------
val_data = pd.read_csv('input_path/data_ml_val.csv')
train_data = pd.read_csv('input_path/data_ml_train.csv')

# Setting index and defining target and features
val_data.set_index("ID", inplace=True)
train_data.set_index("ID", inplace=True)

# Categorical variables
categorical_vars = ['gender', 'education', 'exercise_frequency', 'alcohol_use',
                    'cannabis_use', 'cigarette_use', 'meditation']

# Splitting features and target variable
X_train = train_data.drop("resilience", axis=1)
y_train = train_data["resilience"]
X_validation = val_data.drop("resilience", axis=1)
y_validation = val_data["resilience"]

# ------------------------
# PRINT BASIC SAMPLE INFO
# ------------------------
def get_stats(df, column_name):
    if column_name == "gender":
        return (df[column_name] == 1).mean() * 100  # percentage of females given female=1
    else:
        return df[column_name].mean(), df[column_name].std()  # mean and standard deviation

# For validation data
val_data_age_mean, val_data_age_sd = get_stats(val_data, "age")
val_data_resilience_mean, val_data_resilience_sd = get_stats(val_data, "resilience")
val_data_gender_percentage = get_stats(val_data, "gender")

print("Validation dataset statistics:")
print(f"Age: Mean = {val_data_age_mean}, SD = {val_data_age_sd}")
print(f"Resilience: Mean = {val_data_resilience_mean}, SD = {val_data_resilience_sd}")
print(f"Gender: % Female = {val_data_gender_percentage}%")

# For training data
train_data_age_mean, train_data_age_sd = get_stats(train_data, "age")
train_data_resilience_mean, train_data_resilience_sd = get_stats(train_data, "resilience")
train_data_gender_percentage = get_stats(train_data, "gender")

print("\nTraining dataset statistics:")
print(f"Age: Mean = {train_data_age_mean}, SD = {train_data_age_sd}")
print(f"Resilience: Mean = {train_data_resilience_mean}, SD = {train_data_resilience_sd}")
print(f"Gender: % Female = {train_data_gender_percentage}%")

# --------------------------------
# DUMMY CODING OF CATEGORICAL VARS
# --------------------------------
# Combine the training and validation data before dummy coding
combined_data = pd.concat([X_train, X_validation], axis=0)

# Dummy coding on the combined data
combined_data = pd.get_dummies(combined_data, columns=categorical_vars, drop_first=True)

# Split them back into training and validation sets
X_train = combined_data.iloc[:len(X_train), :]
X_validation = combined_data.iloc[len(X_train):, :]

# -------------------------------------------------
# IDENTIFY AND REMOVE VARS WITH NEAR ZERO VARIANCE
# -------------------------------------------------
# Removing variables with zero or near-zero variance
threshold = 0.02
variances = X_train.var(axis=0)
columns_to_keep = variances[variances > threshold].index

# Print dropped and final columns
dropped_columns = set(X_train.columns) - set(columns_to_keep)

print("Columns removed due to near-zero variance:")
for col in dropped_columns:
    print(col)

X_train = X_train[columns_to_keep]
X_validation = X_validation[columns_to_keep]

print('Final set of features:')
print(columns_to_keep)

# --------------------------
# NORMALIZATION OF VARIABLES
# --------------------------
numerical_vars = X_train.select_dtypes(include=['float64', 'int']).columns.tolist()
scaler = MinMaxScaler()
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])
X_validation[numerical_vars] = scaler.transform(X_validation[numerical_vars])

# ------------------------------------------------------
# FEATURE SELECTION: RECURSIVE FEATURE ELIMINATION (RFE)
# ------------------------------------------------------
# Initialize Random Forest model for feature selection
model = RandomForestRegressor(random_state=42)
selector = RFE(estimator=model, n_features_to_select=20, step=1, verbose=1)

# Fit RFE for feature selection
selector.fit(X_train, y_train)

# Select the important features
X_train_selected = selector.transform(X_train)
X_validation_selected = selector.transform(X_validation)

print("Number of features selected by RFE:", X_train_selected.shape[1])

# --------------------------------------------
# RANDOM FOREST MODEL & HYPERPARAMETER TUNING
# --------------------------------------------
# Hyperparameters for Random Forest
param_distributions = {
    'n_estimators': np.arange(10, 200, 10),  # Number of trees
    'max_features': ['auto', 'sqrt'],        # Number of features for best split
    'max_depth': np.arange(5, 50, 5)         # Maximum depth of the tree
}

# Repeated K-Fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Randomized Search Cross Validation
search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    cv=rkf,
    n_iter=100,                                  # Number of combinations to try
    scoring='neg_mean_squared_error',            # Evaluation metric
    random_state=42,
    verbose=1,
    error_score='raise'
)

# Fit the search object
search.fit(X_train_selected, y_train)

# Extract best model and parameters
best_rf_model = search.best_estimator_
best_parameters = search.best_params_

print("Best hyperparameters for Random Forest:", best_parameters)

# -----------------
# MODEL VALIDATION
# -----------------
# Predict using the best model
y_pred = best_rf_model.predict(X_validation_selected)

# Calculate R^2, MAE, and MSE
r2 = r2_score(y_validation, y_pred)
mae = mean_absolute_error(y_validation, y_pred)
mse = mean_squared_error(y_validation, y_pred)

print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# -------------------------
# FEATURE IMPORTANCE & SHAP
# -------------------------
# Convert the numpy array back to DataFrame for SHAP analysis
X_train_selected_df = pd.DataFrame(X_train_selected, columns=X_train.columns[selector.support_])

# Initialize JavaScript visualizations for SHAP
shap.initjs()

# Explainer object for SHAP values
explainer = shap.TreeExplainer(best_rf_model)
explanation = explainer(X_train_selected_df)

# Visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, explanation.values[0, :], X_train_selected_df.iloc[0, :])

# Summarize the effects of all features
shap.summary_plot(explanation.values, X_train_selected_df.iloc[:1000, :])
