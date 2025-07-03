# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:32:16 2025

Download World Ocean Atlas temperature climatological fields.

https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23.pl

It will be used to estimate b values following Marsay et al. (2015), Fig. 2A

@author: Markel
"""

#%% IMPORTS

import urllib.request
import os


#%% SETUP

# Create necessary directories
dpath = 'rawdata/woa/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    
    
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

# Grid ---> 1º
# Field --> statistical mean
# Period -> averaged decades

# Set paths in server and local
url = 'https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/csv/decav/1.00/woa23_decav_t00mn01.csv.gz'
fpath_local = 'rawdata/woa/woa23_decav_t00mn01.csv.gz'

# Download if does not exist, or if exists and forced to overwrite
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo:
    # Download file from server
    urllib.request.urlretrieve(url, fpath_local)