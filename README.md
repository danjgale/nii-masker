# roi-extractor
Command-line tool for extracting out ROI data

This package is serves as a command-line wrapper for `nilearn`'s `NiftiMasker` utilities, which extracts voxelwise or region-averaged timecourses based on binary masks. The command-line interface extends `nilearn`'s powerful capabilities to non-Python users. Some of these features include additional preprocessing (temporal filtering and detrending, spatial smoothing, etc), standardization, and confound removal (e.g., regress out motion parameters). 

`roi-extractor` also generates an optional quality report for each extracted ROI so you can flag down outlier voxels, examine signal to noise ratios, and generally preview your data to ensure that everything is in order. If you're looking for whole brain quality control, refer to MRIQC and fmriprep.

All outputs are saved to an "ROI Directory", which includes tab-delimited timeseries files for each ROI, and if the report is specified, a report file (with accompanying figures). 
