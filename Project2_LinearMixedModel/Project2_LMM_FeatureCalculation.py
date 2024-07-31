#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:23:07 2023

@author: reppmazc

Feature calculation for linear mixed models

This script calculates features for various stress markers (negative affect, 
heart rate, salivary cortisol, alpha-amylase) to subsequently analyze changes
across different phases (pre-stress, stress, post-stress) in the different markers.
The input data are preprocessed timeseries of different stress markers per participant.

Implemented steps include:
Cortisol Feature Calculation:
   - Determine the peak cortisol response
   - Calculate mean cortisol levels across selected post-stress timepoints
     to define 'cortisol_stress' and 'cortisol_post'.
Alpha-Amylase Feature Calculation:
   - Identify the peak alpha-amylase response
   - Define 'alphaamylase_stress' and 'alphaamylase_post' based on predefined timepoints
Affect Ratings Feature Calculation:
   - Compute mean affect ratings across selected timepoints to define
     'affect_stress' and 'affect_post'
Heart Rate Feature Calculation:
   - Rename columns for heart rate to follow a consistent naming scheme.

- Combine cortisol, alpha-amylase, affect, and heart rate features into a single
  DataFrame
- Reshape the data into a long format with a 'phase' column indicating
  the measurement phase (pre-stress, stress, post-stress).
- Combine features (long format) with the rest of the processed data

Output: a final data file that contains all relevant preprocessed data and features

"""

#--------
# IMPORTS
#--------
import pandas as pd
import numpy as np

# Load the preprocessed data
path = "path_to_data/PreprocessedData.csv"

#------------------------------
# CALCULATING CORTISOL FEATURES
#------------------------------
# Cortisol measurements in nmol/l at different timepoints
cortisol_data = pd.read_csv(path, usecols=['ID',
                                           'cortisol_pre',
                                           'cortisol_post1',
                                           'cortisol_post2',
                                           'cortisol_post3'])
cortisol_data.set_index("ID", inplace=True)

# Calculate the peak response as per the preregistered plan
cols_of_interest = ['cortisol_post1', 'cortisol_post2', 'cortisol_post3']
peak_col = cortisol_data[cols_of_interest].mean().idxmax()
peak_mean_value = cortisol_data[peak_col].mean()
print(f"The peak cortisol response is in {peak_col} with a mean value of {peak_mean_value:.2f}")

# Define cortisol features
cortisol_data['cortisol_post'] = cortisol_data[['cortisol_post2', 'cortisol_post3']].mean(axis=1)
cortisol_data = cortisol_data.rename(columns={'cortisol_pre': 'cortisol_pre',
                                              'cortisol_post1': 'cortisol_stress'})

#-----------------------------------
# CALCULATING ALPHA-AMYLASE FEATURES
#-----------------------------------
# Alpha-amylase measurements in U/ml at different timepoints
alphaamylase_data = pd.read_csv(path, usecols=["ID",
                                               'alphaamylase_pre',
                                               'alphaamylase_post1',
                                               'alphaamylase_post2',
                                               'alphaamylase_post3'])
alphaamylase_data.set_index("ID", inplace=True)

# Calculate alpha-amylase peak response
cols_of_interest = ['alphaamylase_post1', 'alphaamylase_post2']
peak_col = alphaamylase_data[cols_of_interest].mean().idxmax()
peak_mean_value = alphaamylase_data[peak_col].mean()
print(f"The peak alpha-amylase response is in {peak_col} with a mean value of {peak_mean_value:.2f}")

# Define alpha-amylase features
alphaamylase_data['alphaamylase_post'] = alphaamylase_data['alphaamylase_post3']
alphaamylase_data = alphaamylase_data.rename(columns={'alphaamylase_pre': 'alphaamylase_pre',
                                                      'alphaamylase_post2': 'alphaamylase_stress'})

#------------------------------------
# CALCULATING AFFECT RATINGS FEATURES
#------------------------------------
affect_data = pd.read_csv(path, usecols=["ID",
                                         'affect_pre',
                                         'affect_post1', 
                                         'affect_post2',
                                         'affect_post3',
                                         'affect_post4'])
affect_data.set_index("ID", inplace=True)

# Define affect features
affect_data['affect_post'] = affect_data[['affect_post2', 'affect_post3', 'affect_post4']].mean(axis=1)
affect_data = affect_data.rename(columns={'affect_pre': 'affect_pre',
                                          'affect_post1': 'affect_stress'})

#--------------------------------
# CALCULATING HEART RATE FEATURES
#--------------------------------
heartrate_data = pd.read_csv(path, usecols=["ID",
                                            'heartrate_pre',
                                            'heartrate_post',
                                            'heartrate_stress'])
heartrate_data.set_index("ID", inplace=True)

# Rename columns to follow a consistent naming pattern
heartrate_data = heartrate_data.rename(columns={'heartrate_pre': 'heartrate_pre',
                                                'heartrate_stress': 'heartrate_stress',
                                                'heartrate_post': 'heartrate_post'})

#----------------------------------------
# MERGING ALL FEATURES INTO ONE DATAFRAME
#----------------------------------------
print("Unique IDs in each dataset:")
for df_name, df in zip(['Cortisol', 'Alpha-Amylase', 'Affect', 'Heart Rate'],
                       [cortisol_data, alphaamylase_data, affect_data, heartrate_data]):
    print(f"{df_name}: {df.index.nunique()} unique IDs out of {len(df)} rows.")

merged_data = cortisol_data.merge(alphaamylase_data, left_index=True, right_index=True)
merged_data = merged_data.merge(affect_data, left_index=True, right_index=True)
merged_data = merged_data.merge(heartrate_data, left_index=True, right_index=True)
merged_data.reset_index(inplace=True)

#----------------------------------------------------------------------------
# CREATING A LONG FORMAT DATAFRAME WITH PHASE (PRE, STRESS, POST) INFORMATION
#----------------------------------------------------------------------------
def melt_and_rename(df, name):
    """
    Melt the dataframe and rename the columns.

    Args:
    - df (pd.DataFrame): DataFrame to melt
    - name (str): base name for the value columns
    """
    cols_to_melt = [f'{name}_pre', f'{name}_stress', f'{name}_post']
    melted = pd.melt(df, id_vars=['ID'], value_vars=cols_to_melt)
    melted['variable'] = melted['variable'].str.split('_').str[1]
    melted.rename(columns={'variable': 'phase', 'value': name}, inplace=True)
    return melted

cortisol_long = melt_and_rename(cortisol_data, 'cortisol')
alphaamylase_long = melt_and_rename(alphaamylase_data, 'alphaamylase')
affect_long = melt_and_rename(affect_data, 'affect')
heartrate_long = melt_and_rename(heartrate_data, 'heartrate')

# Merge the long format dataframes
long_data = cortisol_long
for df in [alphaamylase_long, affect_long, heartrate_long]:
    long_data = pd.merge(long_data, df, on=['ID', 'phase'])

# Save the processed data to CSV
long_data.to_csv("path_to_save/PreprocessedFeatures.csv", index=False)

#----------------------------------------------------------------------------
# CREATING FINAL DATA FRAME FOR LINEAR MIXED MODELING CONTAINING FEATURES AND
# THE REST OF THE DATA
#----------------------------------------------------------------------------
data_path = "path_to_data/PreprocessedData.csv"
features_path = "path_to_features/PreprocessedFeatures.csv"

variables = pd.read_csv(data_path, usecols=['ID', 
                                            'site',
                                            'age',
                                            'gender',
                                            'oral_contraceptive'])
features = pd.read_csv(features_path)
data = pd.merge(variables, features, on='ID')
data.to_csv("path_to_save/PreprocessedDataAndFeatures.csv", index = False)