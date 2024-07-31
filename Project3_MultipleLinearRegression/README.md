Project 3: Brain Network Connectivity and Psychological Resilience Prediction

This project focuses on using brain connectivity features derived from a multi-site brain imaging study to predict psychological resilience. The study includes resting-state functional connectivity data collected before and after a stress task (ScanSTRESS), with the analysis targeting key brain networks and their interactions. The project aims to showcase the application of data preprocessing, feature calculation, and statistical modeling techniques, specifically using Multiple Linear Regression (MLR).

Project Components
1. Data Preprocessing (Project3_MLR_FeatureCalculation.py)
This script processes functional connectivity data from a brain imaging study, focusing on two resting-state scans (pre-stress and post-stress) around a stress task.
- Extracts correlation matrices of specific brain networks: Salience Network (SN), Default Mode Network (DMN), and Central Executive Network (CEN), including cross-network interactions (SN x DMN, SN x CEN).
- Calculates mean connectivity for each network and task
- Applies Fisher's Z-transformation to stabilize variance in correlation data.

2. Feature Calculation (Project3_MLR_Preprocessing.py)
This script merges brain connectivity features with demographic data and the outcome variable (psychological resilience), preparing the final dataset for MLR analysis.

3. Multiple Linear Regression Analysis (Project3_MLR.R)
This script implements multiple linear regression models to analyze the relationship between brain network connectivity and psychological resilience.
- calculates weights for each study site based on the standard deviation of resilience scores
- Performs linear regressions on various brain networks and their interactions, adjusting for covariates   like age, site, gender, and handedness.
- extracts model statistics and visualizes results