# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:35:23 2025ç

Download atmospheric histories of tracer gases.

https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0164584


@author: Markel
"""

#%% IMPORTS

import urllib.request
import os
import tarfile


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


#%% DOWNLOAD

# Set paths in server and local
url = 'https://www.ncei.noaa.gov/archive/archive-management-system/OAS/bin/prd/jquery/download/164584.2.2.tar.gz'
fpath_local = 'rawdata/tracers_atmosphere/164584.2.2.tar.gz'

# Download if does not exist, or if exists and forced to overwrite
boo = ((os.path.exists(fpath_local) & force_download_and_overwrite) |
       (not os.path.exists(fpath_local)))
if boo:
    # Download file from server
    urllib.request.urlretrieve(url, fpath_local)

# Uncompress contents
file = tarfile.open(fpath_local) 
file.extractall('rawdata/tracers_atmosphere/') 
file.close()