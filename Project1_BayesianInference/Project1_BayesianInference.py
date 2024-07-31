#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 08:16:28 2024
@author: reppmazc

Bayesian Analysis of fMRI BOLD Response to ScanSTRESS Task

This script performs a Bayesian ANOVA to investigate the effect of study site
on the BOLD (Blood-Oxygen-Level-Dependent) response to an fMRI task (ScanSTRESS task)
by calculating a Region of Interest (ROI) wise Bayes factors (BF). The BF allows
quantification of evidence for the H0 as well as the alternative hypothesis.

Implemented steps include:
- loading data
- Prepare a masker using the NiftiLabelsMasker with the loaded atlas to extract time series data for each ROI
- For each participant, load the individual fMRI data, transform it using the masker, and store the ROI data
- Prepare the data and perform Bayes Factor calculations for each ROI using Bayesian ANOVA in R
- Map Bayes Factors back to the atlas space and save as NIFTI image
- generate a csv file that contains results per ROI

Input:
- First-level contrast image (stress vs. control runs) of the ScanSTRESS task for each participant
- Brainnetome atlas for ROI definition
- look up table for the brainnetome atlas defining ROIs (including names and MNI coordinates, etc.)

Output:
- NIFTI image mapping Bayes Factor values for each ROI
- csv file with results per ROI

"""

#------------
# Import Libraries
#------------
import os
import pandas as pd
import numpy as np
from nilearn import image, masking, datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
import logging
from rpy2.robjects import r as R
from rpy2.rinterface_lib.embedded import RRuntimeError
from nilearn.image import new_img_like

#--------------
# Setup Logging
#--------------
logging.getLogger('').handlers = []  # Clear existing log handlers
logging.basicConfig(filename='log_file_path.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#---------------------------
# Define Paths and Load Data
#---------------------------
# Base path for data
basepath = 'base_path_to_data'

# Brainnetome atlas path
atlas_path = 'path_to_brainnetome_atlas.nii.gz'

# Participant directories (directories follow a 'sub-XXXX' naming pattern)
all_participant_dirs = [d for d in os.listdir(basepath) if d.startswith('sub-')]
logging.info(f"Found {len(all_participant_dirs)} participant directories.")

LUT_path = "path_to_LUT_file.csv" # look up table of brain atlas

#------------------
# Atlas Preparation
#------------------
# Load the atlas
atlas_img = image.load_img(atlas_path)

# Define the path to a specific fMRI data file as a reference image
reference_image_path = 'path_to_reference_image.nii.gz'
reference_image = image.load_img(reference_image_path)

# Resample the atlas to match the reference image space
resampled_atlas = image.resample_to_img(atlas_img, reference_image, interpolation='nearest')
atlas_img = resampled_atlas

#-----------------
# Extract ROI Data
#-----------------
# Load the atlas and prepare the masker
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True, memory='nilearn_cache', verbose=0)

# Dictionary to hold ROI data per participant
participant_roi_data = {}

# Extract data per participant
for dir_name in all_participant_dirs:
    pid = dir_name.split('-')[-1]
    file_path = os.path.join(basepath, dir_name, 'func', f'{dir_name}_task-stress_feature-taskStress1_taskcontrast-stress_stat-effect_statmap.nii.gz')
    if os.path.exists(file_path):
        logging.info(f"Processing image for participant {dir_name}")
        participant_img = image.load_img(file_path)
        roi_time_series = masker.fit_transform(participant_img)
        participant_roi_data[pid] = roi_time_series
        logging.info(f"Data extraction complete for participant {dir_name}")
    else:
        logging.warning(f"Image file not found for participant {dir_name}")
        continue

logging.info("ROI data has been extracted for all participants.")

#-----------------------------------
# Prepare for Bayes Factor Calculation
#-----------------------------------
# Load BayesFactor package in R
R('library(BayesFactor)')

# Define study sites and labels
site_dict = {
    '1': 'Site1',
    '2': 'Site2',
    '3': 'Site3',
    '4': 'Site4',
    '5': 'Site5'
}
# Assume site information can be derived from participant IDs or folder structure, update this mapping accordingly
site_labels = [site_dict[pid[0]] for pid in participant_roi_data.keys()]
R.globalenv['site_labels'] = R.StrVector(site_labels)

# Prepare data for R
R.globalenv['participant_ids'] = R.StrVector(list(participant_roi_data.keys()))
R.globalenv['participant_sites'] = R.StrVector(site_labels)

#-------------------------------
# ROI-wise Bayes Factor Calculation
#-------------------------------
roi_labels = masker.labels_
roi_bayes_factors = {}

for i, roi_name in enumerate(roi_labels):
    logging.info(f"Processing ROI {roi_name}")

    # Gather data for this ROI across all participants
    roi_data = np.array([participant_roi_data[pid][:, i] for pid in participant_roi_data.keys()])
    
    # Check for NaN values and handle them appropriately
    if np.isnan(roi_data).any():
        roi_bayes_factors[roi_name] = np.nan
        logging.info(f"ROI {roi_name} contains NaN values and will be skipped for Bayes Factor calculation.")
        continue

    # Prepare the data frame for R analysis
    R.globalenv['roi_data'] = R.FloatVector(roi_data.flatten())
    R('df = data.frame(y = roi_data, id = rep(participant_ids, each=nrow(roi_data)/length(participant_ids)), group = factor(rep(participant_sites, each=nrow(roi_data)/length(participant_ids))))')
    
    # Perform Bayesian ANOVA
    try:
        bf_result = R('res <- tryCatch({anovaBF(y ~ group, data = df)@bayesFactor$bf}, error = function(e) NA)')
        if bf_result[0] is R.NA_Real:
            roi_bayes_factors[roi_name] = np.nan
            logging.warning(f"Integration error in ROI {roi_name}. Assigned NaN.")
        else:
            roi_bayes_factors[roi_name] = bf_result[0]
            logging.info(f"Bayes Factor for ROI {roi_name}: {bf_result[0]}")
    except RRuntimeError as e:
        roi_bayes_factors[roi_name] = np.nan
        logging.error(f"R Runtime Error in ROI {roi_name}: {e}")

# Log the completion of Bayes Factor calculations
logging.info("Bayes Factor calculation completed for all ROIs.")

# -------------------------------
# Conduct Bayesian Post Hoc Tests
# -------------------------------
groups = list(set(site_labels))
post_hoc_results = {roi: [] for roi in roi_labels if roi_bayes_factors.get(roi, 0) > 0}

def bayesian_t_test(group1_data, group2_data):
    R('library(BayesFactor)')
    R.globalenv['group1_data'] = R.FloatVector(group1_data)
    R.globalenv['group2_data'] = R.FloatVector(group2_data)
    
    try:
        bf_result = R('ttestBF(x=group1_data, y=group2_data)')
        bayes_factor = bf_result.slots['bayesFactor'][0][0]
        return bayes_factor
    except RRuntimeError as e:
        logging.error(f"R Runtime Error during Bayesian t-test: {e}")
        return np.nan
    except Exception as e:
        logging.error(f"General Error extracting Bayes Factor: {e}")
        return np.nan

for roi_name in post_hoc_results.keys():
    roi_index = roi_labels.index(roi_name)
    roi_data = np.array([participant_roi_data[pid][:, roi_index] for pid in participant_roi_data.keys()])
    df = pd.DataFrame(roi_data, columns=['data'])
    df['group'] = site_labels

    for (group1, group2) in combinations(groups, 2):
        group1_data = df[df['group'] == group1]['data']
        group2_data = df[df['group'] == group2]['data']
        if not group1_data.empty and not group2_data.empty:
            bf = bayesian_t_test(group1_data, group2_data)
            post_hoc_results[roi_name].append((group1, group2, bf))
            logging.info(f"Bayesian t-test for ROI {roi_name} between {group1} and {group2}: BF={bf}")

#------------------
# Save and Clean Up
#------------------
bf_values = [roi_bayes_factors.get(roi, np.nan) for roi in roi_labels]
bf_image = masker.inverse_transform(bf_values)
bf_image_path = 'path_to_save_bf_image.nii'
bf_image.to_filename(bf_image_path)
logging.info(f"Bayes Factor image saved as {bf_image_path}")

atlas_data = atlas_img.get_fdata()
non_roi_mask = (atlas_data == 0)
bf_image_data = bf_image.get_fdata()
bf_image_data[non_roi_mask] = np.nan
bf_image_nan = new_img_like(bf_image, bf_image_data)
bf_image_nan.to_filename(bf_image_path)
logging.info(f"Bayes Factor image with NaN in non-ROI areas saved as {bf_image_path}")

#--------------------------
# Generate results CSV File
#--------------------------
roi_bayes_df = pd.DataFrame(list(roi_bayes_factors.items()), columns=['ROI', 'Log(BF10)'])

def format_post_hoc_results(results):
    return '; '.join([f"{g1} vs {g2}: BF={bf:.2f}" for g1, g2, bf in results if bf > 0])

roi_bayes_df['Post Hoc'] = roi_bayes_df['ROI'].apply(lambda x: format_post_hoc_results(post_hoc_results.get(x, [])))

def evidence_category(log_bf):
    if log_bf > 4.61:
        return "Extreme evidence for H1"
    elif 3.40 <= log_bf <= 4.61:
        return "Very strong evidence for H1"
    elif 2.30 <= log_bf < 3.40:
        return "Strong evidence for H1"
    elif 1.10 <= log_bf < 2.30:
        return "Moderate evidence for H1"
    elif 0 <= log_bf < 1.10:
        return "Anecdotal evidence for H1"
    elif log_bf == 0:
        return "No evidence"
    elif -1.10 <= log_bf < 0:
        return "Anecdotal evidence for H0"
    elif -2.30 <= log_bf < -1.10:
        return "Moderate evidence for H0"
    elif -3.40 <= log_bf < -2.30:
        return "Strong evidence for H0"
    elif -4.61 <= log_bf < -3.40:
        return "Very strong evidence for H0"
    else:
        return "Extreme evidence for H0"

roi_bayes_df['Evidence Category'] = roi_bayes_df['Log(BF10)'].apply(lambda x: evidence_category(x) if x is not None and not np.isnan(x) else "Data not available")
roi_bayes_df['Log(BF10)'] = roi_bayes_df['Log(BF10)'].apply(lambda x: x if x is not None and not np.isnan(x) else "NaN")

lut_df = pd.read_csv(LUT_path)
combined_df = pd.merge(roi_bayes_df, lut_df, left_on='ROI', right_on='Label_ID')
csv_path = 'path_to_save_bf_csv.csv'
combined_df.to_csv(csv_path, index=False)
logging.info(f"Evidence CSV with LUT data and post hoc results saved as {csv_path}")