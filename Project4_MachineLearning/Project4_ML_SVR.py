#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:38:52 2023
@author: reppmazc

Support Vector Regression Script

This script implements a Support Vector Regression (SVR) model for predicting
psychological resilience based on various demographic, behavioral, and physiological features,
including markers of stress response. The data were acquired at different study sites, with one site
used as the validation dataset due to its demographic distinctiveness.

Steps include:
    - Data loading and preprocessing
    - Dummy coding of categorical variables
    - Feature selection using Recursive Feature Elimination (RFE)
    - Hyperparameter tuning using Randomized Search CV
    - Model validation and performance evaluation
    - Feature importance analysis using SHAP values

The input is the preprocessed validation and training datasets (preprocessed though the following script:
Project1_ML_Preprocessing.py). Both datasets contain the exact same columns.

"""
#--------
# IMPORTS
#--------
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

#------------
# IMPORT DATA
#------------
val_data = pd.read_csv('input_path/data_ml_val.csv')
train_data = pd.read_csv('input_path/data_ml_train.csv')

# Setting index and defining target and features
val_data.set_index("ID", inplace=True)
train_data.set_index("ID", inplace=True)

# Categorical variables
categorical_vars = ['gender', 'education', 'exercise_frequency', 'alcohol_use',
                    'cannabis_use', 'cigarette_use', 'meditation']

# ------------------------
# PRINT BASIC SAMPLE INFO
# ------------------------
def get_stats(df, column_name):
    if column_name == "gender":
        return (df[column_name] == 1).mean() * 100  # Percentage of females given female=1
    else:
        return df[column_name].mean(), df[column_name].std()  # Mean and standard deviation

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

#-----------------
# SPLIT OFF TARGET
#-----------------
# Splitting features and target variable
X_train = train_data.drop("resilience", axis=1)
y_train = train_data["resilience"]
X_validation = val_data.drop("resilience", axis=1)
y_validation = val_data["resilience"]

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

#-------------------------------------------------
# IDENTIFY AND REMOVE VARS WITH NEAR ZERO VARIANCE
#-------------------------------------------------
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

#---------------------------------
# NORMALIZATION OF CONTINUOUS VARS
#---------------------------------
numerical_vars = X_train.select_dtypes(include=['float64', 'int']).columns.tolist()
scaler = MinMaxScaler()
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])
X_validation[numerical_vars] = scaler.transform(X_validation[numerical_vars])

#----------------------------------
# FEATURE SELECTION USING LinearSVR
#----------------------------------
print("*** Starting Feature Selection...")
# Initialize LinearSVR model for feature selection
selector_model = LinearSVR(max_iter=100)

# Initialize recursive feature elimination (RFE)
selector = RFE(estimator=selector_model, n_features_to_select=20, step=1, verbose=1)

# Fit RFE for feature selection
selector.fit(X_train, y_train)

# Extract important features
X_train_selected = selector.transform(X_train)
X_validation_selected = selector.transform(X_validation)
selected_feature_names = X_train.columns[selector.support_]

print("Number of features selected by RFE:", X_train_selected.shape[1])
print("Features selected by RFE:", selected_feature_names)
print("*** Feature Selection Done!")

#--------------------------
# PRINT GENERAL INFORMATION
#--------------------------
print("Total sample size:", len(combined_data))
print("Number of participants in the training set:", len(X_train))
print("Number of participants in the validation set:", len(X_validation))

#--------------------------------------------
# SUPPORT VECTOR REGRESSION & HYPERPARAMETERS
#--------------------------------------------
print("*** Starting Hyperparameter Tuning...")
param_distributions = {
    'C': [0.01, 0.1, 1, 10, 100],         # Regularization parameter
    'kernel': ['poly', 'rbf', 'sigmoid'], # Kernel type
    'degree': [2, 3, 4, 5],               # Degree for polynomial kernel
    'gamma': ['scale', 'auto']}           # Kernel coefficient

# Initialize SVR model
model = SVR()

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Randomized Search Cross Validation
search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    cv=rkf,
    n_iter=100,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=1,
    error_score='raise')

# Fit the randomized search object to the data
search.fit(X_train_selected, y_train)

# Extract best model and parameters
best_svr_model = search.best_estimator_
best_parameters = search.best_params_

print("Best hyperparameters for Support Vector Regression:", best_parameters)
print("*** Hyperparameter Tuning Done!")

#-----------------
# MODEL VALIDATION
#-----------------
print("*** Starting Model Validation...")
# Predict using the best model
y_pred = best_svr_model.predict(X_validation_selected)

# Calculate R^2, MAE, and MSE
r2 = r2_score(y_validation, y_pred)
mae = mean_absolute_error(y_validation, y_pred)
mse = mean_squared_error(y_validation, y_pred)

print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

#--------------------------
# FEATURE IMPORTANCE & SHAP
#--------------------------
# Create SHAP explainer
explainer = shap.KernelExplainer(best_svr_model.predict, X_train_selected)

# Convert transformed validation data back to DataFrame
X_validation_df = pd.DataFrame(X_validation_selected, columns=selected_feature_names)

# Calculate SHAP values for validation data
shap_values = explainer.shap_values(X_validation_df)

# Plot SHAP values
shap.initjs()
instance_index = 0  
shap.force_plot(explainer.expected_value, shap_values[instance_index], X_validation_selected[instance_index])

# Plot the summary plot for all instances
shap.summary_plot(shap_values, X_validation_selected)
