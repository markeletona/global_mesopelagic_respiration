# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:08:42 2025

Download NPP and associated SST data from:
http://orca.science.oregonstate.edu/npp_products.php

Eppley-VGPM and CbPM models, monthly data, 2160 x 4320, MODIS R2022 data.

Data from 2003-2021 year, i.e., the years that are complete, with no month
missing, so as to not bias the climatology. 2022 and 2023 have data but are
missing a month.


@author: Markel
"""

#%% IMPORTS

import urllib.request
import urllib.error
import os

#%% DOWNLOAD DATA

# Set models to download, and required codes to fill paths etc.:
mods = {'cbpm': {'url_subdir': 'cbpm2.modis.r2022',
                 'url_fname_code': 'cbpm.m.',
                 'local_dir': 'cbpm/'},
        'eppley': {'url_subdir': 'eppley.r2022.m.chl.m.sst',
                   'url_fname_code': 'eppley.m.',
                   'local_dir': 'eppley-vgpm/'}}


# Set years to download data from
years = range(2003, 2022) # this creates 2003...2021

# Set parent directory in server and local
server_parent = 'http://orca.science.oregonstate.edu/data/2x4/monthly/'
local_parent = 'rawdata/npp/'


# When downloading, check whether file already exists to avoid unnecessarily 
# downloading  files repeatedly.         
# 
# But, also give option to force download and overwrite in case this 
# is desired
#
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

force_download_and_overwrite = False

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

# It seems that Oregon State University's webpage currently has outdated
# certificates. This might be fixed in the future. As a workaround, temporarily 
# disable SSL verification.
try:
    # Test if webpage has issues with certificates
    main_site = 'https://orca.science.oregonstate.edu/npp_products.php'
    test_response = urllib.request.urlopen(main_site, timeout=20)
except urllib.error.URLError:
    # If issues, temporarily disable SSL verification.
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
        


for im, m in enumerate(mods):
    for y in years:
        
        # Check existence of file
        fname = mods[m]['url_fname_code'] + str(y) + '.tar'
        fpath_local = local_parent + mods[m]['local_dir'] + fname
        if os.path.exists(fpath_local):
            if not force_download_and_overwrite:
                continue
                # File is only skipped if exists and download is NOT forced
                
        # Download file from server
        url = (server_parent + mods[m]['url_subdir'] + '/hdf/' + fname)
        urllib.request.urlretrieve(url, fpath_local)

