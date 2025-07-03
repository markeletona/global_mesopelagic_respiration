# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:23:08 2025

Download GLODAP dataset from:
https://glodap.info/index.php/merged-and-adjusted-data-product-v2-2023/

Download GLODAP dataset from:
https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0227166/

@author: Markel
"""

#%% IMPORTS

import urllib.request
import os


#%% SETUP

# When downloading, check whether file already exists to avoid unnecessarily 
# downloading files repeatedly.         
# 
# But, also give option to force download and overwrite in case this 
# is desired
#
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

force_download_and_overwrite = False

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


#%% DOWNLOAD GLODAP

# Set paths in server and local
url = 'https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_Merged_Master_File.csv.zip'
fpath_local = 'rawdata/glodap/GLODAPv2.2023_Merged_Master_File.csv.zip'

# Download if does not exist, or if exists and forced to overwrite
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo:
    # Download file from server
    urllib.request.urlretrieve(url, fpath_local)


## Ancillary files

url = 'https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_EXPOCODES.txt'
fpath_local = 'rawdata/glodap/GLODAPv2.2023_EXPOCODES.txt'
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo: urllib.request.urlretrieve(url, fpath_local)

url = 'https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_DOIs.csv'
fpath_local = 'rawdata/glodap/GLODAPv2.2023_DOIs.csv' 
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo: urllib.request.urlretrieve(url, fpath_local)



#%% DOWNLOAD HANSELL DOM

# Set paths in server and local
url = 'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0227166/All_Basins_Data_Merged_Hansell_2022.xlsx'
fpath_local = 'rawdata/dom_hansell/All_Basins_Data_Merged_Hansell_2022.xlsx'

# Download if does not exist, or if exists and forced to overwrite
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo: urllib.request.urlretrieve(url, fpath_local)