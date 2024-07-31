Project 2: Linear Mixed Models for Investigation of the Stress Response

This project explores the effects of a stress induction procedure (ScanSTRESS)
on various psychological and physiological stress markers over time using
linear mixed models (LMMs). The analysis includes preprocessing of raw data,
feature calculation, and modeling of stress responses across different phases.

Project Components
1. Data Preprocessing (Project2_LMM_Preprocessing.py)
This script cleans and prepares the dataset for analysis:
Participant and Column Filtering, Handling Implausible Values, Imputation,
Outlier Detection and Handling, Data Normalization

2. Feature Calculation (Project2_LMM_FeatureCalculation.py)
This script calculates key features from the data:
Feature calculation for each marker (affect, heart rate, cortisol, alpha amylase)
and the combines features into a unified DataFrame and reshapes data into a long
format with phase indicators.

3. Linear Mixed Models (Project2_LMM.R)
This script models the effects of the stress induction procedure.
Predictor: Phase (baseline, stress, post-stress).
Covariates: Includes study site, age, gender, oral contraceptive intake.
random intercepts for participants
dependent variable (one per model): affect, heart rate, salivary cortisol, and
salivary alpha-amylase