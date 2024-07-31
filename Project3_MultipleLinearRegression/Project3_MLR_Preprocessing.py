#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Sep 25 10:26:09 2023

@author: reppmazc

This script merges brain connectivity feature pre-, during, and post stress
(ScanSTRESS) features with demograohic data and the outcome variable (psychological
Resilience). Subsequently, all continuous variables are normalized and the final
data frame for multiple linear regressions is generated.

Input:
- brain network connectivity features
- datafile with additional variables (demographics, outcome, etc.)

"""
#--------
# IMPORTS
#--------
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = "/path_to_data/data.csv"
feature_path = "/path_to_features/Features.csv"

data = pd.read_csv(data_path, usecols=['ID',
                                       'site',
                                       'age',
                                       'gender',
                                       'handedness',
                                       'resilience'])

features = pd.read_csv(feature_path)
data = pd.merge(data, features, on='ID')

#--------------------------------------
# normalization of continuous variables
#--------------------------------------
continuous_vars = ['resilience',
                   'SN_pre', 'SN_reactivity', 'SN_recovery',
                   'DMN_pre', 'DMN_reactivity', 'DMN_recovery',
                   'CEN_pre', 'CEN_reactivity', 'CEN_recovery',
                   'SNCEN_pre', 'SNCEN_reactivity', 'SNCEN_recovery',
                   'SNDMN_pre', 'SNDMN_reactivity', 'SNDMN_recovery', 
                   'CENDMN_pre', 'CENDMN_reactivity', 'CENDMN_recovery']

# Ensure that features DataFrame contains the continuous variables
assert all(var in data.columns for var in continuous_vars), "Some continuous variables are missing in 'data'"

# Normalize using StandardScaler
scaler = StandardScaler(copy=False)
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

#------------------------------------------------
# saving data final data for statistical analysis
#------------------------------------------------
data.to_csv("/path_to_save/ProcessedDataAndFeatures.csv", index = False)
