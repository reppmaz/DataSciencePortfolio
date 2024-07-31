#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:23:43 2023

@author: reppmazc

Preprocessing data for feature calculation and subsequent implementation of
linear mixed models

Description:
This script preprocesses a dataset containing time series of different 
stress markers (negative affect, heart rate, salivary cortisol
and alpha-amylase) to prepare it for feature calculation and the
susequent statistical analysis of the impact of a stress induction procedure
(ScanSTRESS task) on the stress markers over time.

Input data:
- Metadata: includes names of all variables (var_names) and acceptable
value ranges (min and max) used to identify implausible values.
- Data: Includes individual records for each participant (ID) in wide format,
encompassing various demographic variables (study site, gender, age, and
oral contraceptive use) and physiological measures (heart rate, salivary cortisol
and alpha-amylase). Stress markers (negative affect, heart rate, salivary cortisol
and alpha-amylase) are collected at different timepoints. Specifically, before
stress induction, during stress (only in the case of heart rate), and after stress).

This code:
- Removes participants with more than 30% missing values in relevant columns.
- Excludes columns where more than 30% of the data is missing.
- Identifies and sets implausible values to NaN based on predefined
  minimum and maximum values from a metadata codebook.
- Uses Iterative Imputer to fill missing values, including handling of
  categorical variables via one-hot encoding.
- Detects outliers separately for each modality (affect, cortisol, alpha-amylase,
  heart rate) and overall data using Mahalanobis distance.
- Sets values to NaN for identified outliers across modalities and overall.
- Applies logarithmic transformation to cortisol, alpha-amylase,
  and heart rate data to normalize distributions.
- Outputs the cleaned dataset to a CSV file, excluding all rows with NaN values
  in all columns (except the ID).

"""

# -------
# IMPORTS
# -------
import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Mute warnings and set display options
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read the metadata and data
meta_data = pd.read_excel("path_to_meta_data/meta_data.xlsx")
data = pd.read_csv("path_to_data/data.csv",
                   usecols=['ID', 'site', 'gender', 'age', 'oral_contraceptive',
                            'affect_pre', 'affect_post1', 'affect_post2', 'affect_post3', 'affect_post4',
                            'cortisol_pre', 'cortisol_post1', 'cortisol_post2', 'cortisol_post3',
                            'alphaamylase_pre', 'alphaamylase_post1', 'alphaamylase_post2', 'alphaamylase_post3',
                            'heartrate_pre', 'heartrate_stress', 'heartrate_post'])

# ---------------------------------------------
# Remove participants with > 30% missing values
# ---------------------------------------------
missing_percentage = data.isnull().mean(axis=1)
affected_participants = data[missing_percentage > 0.3]['ID']

print(f"Participants with >30% missing values: {len(affected_participants)}")
for participant in affected_participants:
    missing_columns = data.columns[data.loc[data['ID'] == participant].isnull().any()].tolist()
    print(f"Participant {participant} has missing values in: {', '.join(missing_columns)}")

data = data.drop(data[missing_percentage > 0.3].index)

# ----------------------------------------
# Remove columns with > 30% missing values
# ----------------------------------------
missing_percentage = data.isnull().mean(axis=0)
cols_to_remove = missing_percentage[missing_percentage > 0.30].index
print(f"Columns with > 30% missing values: {cols_to_remove.tolist()}")
data = data.drop(columns=cols_to_remove)

# ----------------------------------------------------
# Check for implausible values based on metadata
# ----------------------------------------------------
implausible_records = []

for _, row in meta_data.iterrows():
    var_name, min_val, max_val = row['var_name'], row['min'], row['max']
    if var_name in data.columns and not pd.isna(min_val) and not pd.isna(max_val):
        implausible_mask = ~data[var_name].between(min_val, max_val) & ~data[var_name].isna()
        implausible_ids = data.loc[implausible_mask, 'ID']
        implausible_values = data.loc[implausible_mask, var_name]
        for idx, imp_value in zip(implausible_ids, implausible_values):
            implausible_records.append({
                'ID': idx,
                'Variable': var_name,
                'Value': imp_value})

        data.loc[implausible_mask, var_name] = np.nan

implausible_df = pd.DataFrame(implausible_records)
print(implausible_df)

# -------------------------------------------------
# Implement multiple imputation to replace missings
# -------------------------------------------------
categorical_vars = ['site', 'gender', 'oral_contraceptive']

data.set_index("ID", inplace=True)
data_index = data.index

# One-Hot Encoding
original_cols = data.columns.tolist()
dummies = pd.get_dummies(data[categorical_vars], columns=categorical_vars, drop_first=False, prefix=categorical_vars)
data = pd.concat([data.drop(categorical_vars, axis=1), dummies], axis=1)

# Apply IterativeImputer
imputer = IterativeImputer(random_state=100, max_iter=10)
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Decode the imputed one-hot encoded columns back to original representation
for var in categorical_vars:
    dummy_cols = [col for col in dummies.columns if col.startswith(var)]
    data[var] = data[dummy_cols].idxmax(axis=1).str.replace(f"{var}_", "").astype(float).astype(int)
    data.drop(dummy_cols, axis=1, inplace=True)

data.index = data_index
data.reset_index(inplace=True)

# -------------------------------------------------------------------------
# Check for outliers based on Mahalanobis distance per modality and overall
# -------------------------------------------------------------------------
affect = ['affect_pre', 'affect_post1', 'affect_post2', 'affect_post3', 'affect_post4']
cortisol = ['cortisol_pre', 'cortisol_post1', 'cortisol_post2', 'cortisol_post3']
alphaamylase = ['alphaamylase_pre', 'alphaamylase_post1', 'alphaamylase_post2', 'alphaamylase_post3']
heartrate = ['heartrate_pre', 'heartrate_post', 'heartrate_stress']

def detect_outliers(data, features):
    subset = data[features].dropna()
    subset = subset.loc[:, subset.var() > 0]
    inv_cov = np.linalg.inv(subset.cov())
    mean = subset.mean()
    mahalanobis_dist = subset.apply(lambda row: distance.mahalanobis(row, mean, inv_cov), axis=1)
    threshold = mahalanobis_dist.quantile(0.99)
    outlier_ids = subset[mahalanobis_dist > threshold].index.tolist()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mahalanobis_dist, bins=50, color='#23312d', edgecolor='grey')
    ax.set_xlabel('Mahalanobis Dist.', fontsize=15)
    ax.set_ylabel('Frequency', fontsize=15)
    ax.set_title(f'Distribution of Mahalanobis Distances for {modality} modality', fontsize=18)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

    return mahalanobis_dist > threshold

modalities = {
    'affect': affect,
    'cortisol': cortisol,
    'alphaamylase': alphaamylase,
    'heartrate': heartrate
}

outliers_df = pd.DataFrame(index=data.index)
for modality, features in modalities.items():
    outliers_df[modality] = detect_outliers(data, features)

data_for_outliers = data.copy()
data_for_outliers = data_for_outliers.loc[:, data_for_outliers.var() > 0]
inv_cov_overall = np.linalg.inv(data_for_outliers.cov())
mean_overall = data_for_outliers.mean()
mahalanobis_dist_overall = data_for_outliers.apply(lambda row: distance.mahalanobis(row, mean_overall, inv_cov_overall), axis=1)

threshold_overall = mahalanobis_dist_overall.quantile(0.99)
outliers_df['overall'] = mahalanobis_dist_overall > threshold_overall

all_outliers = outliers_df[outliers_df.any(axis=1)]
print(f"Total unique outliers detected: {len(all_outliers)}")
print(f"Outlier IDs: {', '.join(map(str, all_outliers.index.tolist()))}")

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(mahalanobis_dist_overall, bins=50, color='#23312d', edgecolor='grey')
ax.set_xlabel('Mahalanobis Dist.', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.set_title('Distribution of Overall Mahalanobis Distances', fontsize=18)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Removing outliers per modality and overall outliers
# ---------------------------------------------------
column_map = {
    'affect': affect,
    'cortisol': cortisol,
    'alphaamylase': alphaamylase,
    'heartrate': heartrate,
    'overall': data.columns.tolist()
}

affected_counts = {modality: 0 for modality in column_map.keys()}

for idx, row in outliers_df.iterrows():
    for outlier_column, actual_columns in column_map.items():
        if row[outlier_column]:
            affected = False
            for actual_column in actual_columns:
                if actual_column in data.columns and not pd.isna(data.at[idx, actual_column]):
                    data.at[idx, actual_column] = np.nan
                    affected = True
            if affected:
                affected_counts[outlier_column] += 1

            print(f"Participant ID {idx} had outlier data in the '{outlier_column}' modality")

affected_participants = outliers_df.index[outliers_df['overall']].tolist()
if affected_participants:
    print(f"Participants with IDs {', '.join(map(str, affected_participants))} were set to NaN across all modalities due to being overall outliers.")

for modality, count in affected_counts.items():
    print(f"{count} participants were affected in the '{modality}' modality.")
    
all_nan_rows = data[data.isnull().all(axis=1)]
print(f"Number of IDs removed due to overall outlier: {len(all_nan_rows)}")
print(f"Removed IDs: {all_nan_rows.index.tolist()}")
data_cleaned = data.dropna(how='all')

# ---------------------------------------------------------
# Log transform cortisol, alpha-amylase, and heart rate data
# ---------------------------------------------------------
for col in cortisol + alphaamylase + heartrate:
    if col in data_cleaned.columns:
        data_cleaned[col] = np.log(data_cleaned[col])

# -------------------------
# Remove NaNs and save data
# -------------------------
data_cleaned.reset_index(inplace=True)
data_cleaned.to_csv("path_to_save/PreprocessedData.csv", index=False)
