# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:26:29 2025

Download the 'Global Ocean 3D Chlorophyll-a concentration, Particulate 
Backscattering coefficient and Particulate Organic Carbon' product, monthly
climatologies.

https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_BGC_3D_REP_015_010/description
https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_BGC_3D_REP_015_010/files?subdataset=cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg-climatology_P1M-m_202411
https://biogeochemical-argo.org/bgc-data-products-soca-multiobs.php


@author: Markel
"""

#%% IMPORTS

import copernicusmarine
import xarray as xr
import os


#%% LOGIN

# Note: if you do not have a Copernicus account yet, you can easly register
# here -> https://data.marine.copernicus.eu/register

my_username = ''
my_password = ''
copernicusmarine.login(username=my_username, password=my_password,
                       force_overwrite=True)


#%% DOWNLOAD DATA

# Set dataset ID
dataset_id = 'cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg-climatology_P1M-m'

# Directory to save the data
output_dpath = 'rawdata/poc3d/'
if not os.path.exists(output_dpath):
    os.makedirs(output_dpath)

# Directory where the script is executed
# script_directory = 'C:/users/username/Your_Folder/**/' 

# We can check the names of the output files before download
# query_metadata = copernicusmarine.get(dataset_id=dataset_id, dry_run=True)
# 
# for file in query_metadata.files:
#     print(file.file_path)


# Download data
# See documentation of .get(): https://toolbox-docs.marine.copernicus.eu/en/stable/python-interface.html#copernicusmarine.get
copernicusmarine.get(dataset_id=dataset_id,
                     output_directory=output_dpath,
                     no_directories=True,
                     overwrite=True)


#%% CLIMATOLOGY

# Jointly open the downloaded files
downloaded_files = [(output_dpath + f) for f in os.listdir(output_dpath)]
ds = xr.open_mfdataset(downloaded_files)

# Average data over months to get a mean climatological value
ds['poc'] = ds.poc.mean(dim='time')

# Export mean climatology
dpath = 'deriveddata/poc3d/'
if not os.path.exists(dpath): os.makedirs(dpath)
fpath = 'deriveddata/poc3d/cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg-climatology_P1M-m_MEAN_P202411_____.nc'
ds.poc.to_netcdf(fpath)

# Close the dataset before deleting the files
ds.close()

# Delete the downloaded files
for f in downloaded_files: os.remove(f)




