#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:34:40 2023
@author: reppmazc

Data Preprocessing for Random Forest Regression

This script implements various preprocessing steps to prepare multi-site data
for a random forest regression. The processing steps include:
    - Handling missing values
    - Removing implausible values based on predefined metadata
    - One-hot encoding categorical variables
    - Multiple imputation for missing values
    - Splitting the data into training and validation sets

The input:
- datasheet: Contains participant IDs, study site identifiers, demographic variables,
  markers of the stress response, and the outcome variable.
- metadata: Defines plausible value ranges for each variable (e.g., realistic heart rate values).

The output:
- Processed datasets saved as CSV files for validation and training.

"""
# -------
# IMPORTS
# -------
import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

# Mute warnings and set display options
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read the metadata and data (replace with generic paths)
meta_data = pd.read_excel("path_to_meta_data.xlsx")
data = pd.read_csv("path_to_data.csv")

# List of relevant columns
relevant_cols = [
    'ID', 'site',  # Sample details
    'age', 'gender', 'income', 'education',  # Demographics
    'exercise_frequency', 'alcohol_use', 'cannabis_use', 'cigarette_use', 'meditation',  # Health behaviors
    'negativeaffect_reactivity', 'negativeaffect_recovery',  # Negative affect stress response
    'cortisol_reactivity', 'cortisol_recovery',  # Cortisol stress response
    'alphaamylase_reactivity', 'alphaamylase_recovery',  # Alpha amylase stress response
    'heartrate_reactivity', 'heartrate_recovery',  # Heart rate stress response
    'DMN_mean_conn_prestress', 'DMN_mean_conn_poststress',  # Default mode network connectivity pre and post stress
    'SN_mean_conn_prestress', 'SN_mean_conn_poststress',  # Salience network connectivity pre and post stress
    'resilience'  # Outcome variable: psychological resilience
]

data = data[relevant_cols]

# Define categorical variables for one-hot encoding (aka dummy coding)
categorical_vars = ['gender', 'education', 'exercise_frequency', 'alcohol_use',
                    'cannabis_use', 'cigarette_use', 'meditation']

# Remove participants without data in the target variable (psychological resilience)
removed_ids = data[data['resilience'].isna()]['ID'].tolist()
data = data.dropna(subset=['resilience'])
print(f"Removed {len(removed_ids)} participants due to missing 'resilience' values.")

# Detect and handle implausible values based on metadata -> codes implausible values as missing data
for _, row in meta_data.iterrows():
    var_name, min_val, max_val = row['var_name'], row['min'], row['max']
    if var_name in data.columns:
        implausible_mask = ~data[var_name].between(min_val, max_val) & ~data[var_name].isna()
        data.loc[implausible_mask, var_name] = np.nan

# Remove columns with > 30% missing values since variables with more than 30% missing should not be imputed
missing_percentage = data.isnull().mean(axis=0)
cols_to_remove = missing_percentage[missing_percentage > 0.30].index
data = data.drop(columns=cols_to_remove)
print(f"Columns with > 30% missing values removed: {cols_to_remove.tolist()}")

# Function for multiple imputation and final dataset processing
def impute_dataset(dataset):
    if "ID" in dataset.columns:
        dataset.set_index("ID", inplace=True)
    data_index = dataset.index

    # One-Hot Encoding
    dummies = pd.get_dummies(dataset[categorical_vars], drop_first=False)
    dataset = pd.concat([dataset.drop(categorical_vars, axis=1), dummies], axis=1)

    # Apply IterativeImputer
    imputer = IterativeImputer(random_state=100, max_iter=10)
    data_imputed = imputer.fit_transform(dataset)

    # Round the imputed values for one-hot encoded categorical columns to integers
    for col in dummies.columns:
        data_imputed[:, dataset.columns.get_loc(col)] = np.round(data_imputed[:, dataset.columns.get_loc(col)])

    data_imputed_df = pd.DataFrame(data_imputed, columns=dataset.columns)

    # Decode the imputed one-hot encoded columns back to original representation
    for var in categorical_vars:
        dummy_cols = [col for col in dummies.columns if col.startswith(var)]
        data_imputed_df[var] = data_imputed_df[dummy_cols].idxmax(axis=1).str.replace(f"{var}_", "").astype(float).astype(int)
        data_imputed_df.drop(dummy_cols, axis=1, inplace=True)

    data_imputed_df.index = data_index
    data_imputed_df.reset_index(inplace=True)
    return data_imputed_df

# Split data into validation and training sets
val_data = data[data["site"] == 4]
train_data = data[data["site"] != 4]

# Apply the imputation to both datasets separately to avoid data leakage
val_processed = impute_dataset(val_data)
train_processed = impute_dataset(train_data)

# Save the processed data
val_processed.to_csv("output_path/data_ml_val.csv", index=False)
train_processed.to_csv("output_path/data_ml_train.csv", index=False)
