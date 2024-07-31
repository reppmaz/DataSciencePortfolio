Project 4: Machine Learning for Psychological Resilience Prediction

This project demonstrates the application of machine learning techniques to predict psychological resilience using a range of demographic, behavioral, and physiological features, as well as markers of stress response. The data were collected from multiple study sites, with one site specifically designated as the validation dataset. For data protection reasons, the original scripts have been adapted to exclude any participant-specific information. The primary goal is to showcase proficiency in data preprocessing, feature selection, and model implementation using Random Forest Regression and Support Vector Regression (SVR).

Elements of the Project

1. Data Preprocessing (Project4_ML_Preprocessing.py)
This script implements various data cleaning steps, including handling missing values, detecting outliers, and feature scaling. These processes ensure that the data is in optimal condition for model training

2. Machine Learning Algorithms
Two different supervised machine learning algorithms are implemented:
Random Forest Regression (Project4_ML_RandomForest.py)
This script covers:
Feature Selection using Recursive Feature Elimination (RFE)
Hyperparameter tuning using Randomized Search Cross Validation
Model evaluation using metrics such as R², MAE, and MSE
Feature importance analysis using SHAP (SHapley Additive exPlanations)

Support Vector Regression (Project4_ML_SVR.py)
This script includes:
Feature Selection utilizing LinearSVR for RFE
Hyperparameter tuning to optimize model performance
Evaluation of model metrics (R², MAE, MSE)
Feature importance interpretation through SHAP analysis