# Glacier surface velocity comparison project

This notebook is intended to be a roadmap/intro to the project contained in `new_retreat` which is a comparison of glacier surface velocity datasets derived from satellite imagery (ITSLIVE and RETREAT). 

## RETREAT data
RETREAT data is downloaded from [insert webpage] and stored locally. It is formatted as a STAC catalog and read in using `stackstac` tools. The STAC catalog is created in `notebooks/310_working_stac_catalog.ipynb` and lives in `/324stac_catalog`. 

There have been a few different processing iterations:

`328_retreat.ipynb` (double check this) contains processing to start with a list of glaciers and a RETREAT data cube and write to file csvs with seasonal mean velocities and standard error of measurements for each glacier (full area and each elevation quartile). These are stored in `328_results` (double check)

`retreat_43.ipynb` contains a workflow to start again with a RETREAT data cube and list of glaciers and write RETREAT objects that have been clipped to the extent of an individual glacier to file. Various other processing steps are included like removing time steps with minimal coverage, adding NASADEM, subsetting by elevation quartiles and calculating standard error of measurement. These objects have the full time series preserved, not grouped by season. 

## ITSLIVE data
ITS_LIVE data is accessed from AWS s3 using fsspec (in itslive_tools). 
`328_itslive.ipynb` contains processing to go from an ITSLIVE granule and a list of glaciers to csvs of mean seasonal velocities and standard error of measurement for each glacier for the full glacier area as well as for each elevation quartile of each glacier.

`itslive_43.ipynb` contains processing to go from ITSLIVE granule and list of glaciers to written netcdfs of velocity and elevation data for each glacier. 