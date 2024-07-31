Project 1: Bayesian Inference for Investigating Differences in Brain Response Across Study Sites

This project applies Bayesian inference, specifically Bayesian ANOVAs, to investigate potential differences in blood oxygen level-dependent (BOLD) signals across five different study sites in response to an fMRI task (ScanSTRESS task). The primary goal is to identify and quantify differences in brain activity patterns between the sites. For data protection reasons, the original script has been adapted to exclude any participant or site-specific information. 

Project Overview
- Brain regions of interest (ROIs) are defined based on the Brainnetome atlas.
- For each participant, BOLD signals are extracted for every voxel (3D pixel in the brain). These signals are averaged across all voxels within the same ROI to produce a single value per ROI per participant.
- Bayesian ANOVAs are conducted for each ROI to assess the differences in BOLD signals between the study sites.
- Bayes Factors are calculated for each ROI to quantify the evidence for or against differences between study sites.
- Bayes Factors are reconstructed to a brain image (nifti file)
- Post hoc analyses are performed for each ROI to pinpoint which (if any) study sites exhibit significant differences.