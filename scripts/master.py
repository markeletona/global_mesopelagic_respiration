# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:24:36 2025

A master script that executes all individual scripts to reproduce the results
of Gómez-Letona & Álvarez-Salgado (XXXX)

@author: Markel
"""

#%% IMPORTS

import subprocess
from datetime import datetime as dt


#%% EXECUTE SCRIPTS


# Define list of scripts to execute (in order).

list_of_scripts = [
    
    # Download data -----------------------------------------------------------
    'scripts/download_data/00_download_glodap_and_hansell.py',
    'scripts/download_data/01_download_tracers_atmosphere.py',
    'scripts/download_data/02_download_npp.py',
    'scripts/download_data/03_download_woa.py',
    # 
    # !! IMPORTANT !! 
    # 04_download_poc3d.py needs to be edited to introduce your Copernicus
    # credentials in order to download the POC data. 
    # (it's easy to get them!, see script)
    # 
    'scripts/download_data/04_download_poc3d.py',
    
    # Preprocessing -----------------------------------------------------------
    'scripts/npp/01_npp_create_climatology.py',
    'scripts/ocean_wm_defs/01_ocean_water_mass_polys.py',
    'scripts/glodap/01_filter_glodap_dataset.py',
    'scripts/dom_hansell/01_filter_hansell_dataset.py',
    'scripts/hansell_glodap/01_merge_glodap_hansell.py',
    'scripts/hansell_glodap/02_estimate_tracer_ages.py',
    'scripts/hansell_glodap/02b_tracer_data_availability.py',
    'scripts/hansell_glodap/helper_sects_maps.py',
    'scripts/hansell_glodap/helper_sigma_sects.py',
    
    # Estimate OURs -----------------------------------------------------------
    'scripts/hansell_glodap/03_estimate_our_global.py',
    'scripts/hansell_glodap/03b_merge_regression_pdfs.py',
    'scripts/hansell_glodap/04_our_auxillary_plots.py',

    # Literature review -------------------------------------------------------
    'scripts/litrev/literature_review_vis.py',
    
    # Ancillary scripts (optional) --------------------------------------------
    'scripts/misc/test_RegressConsensus.py',
    'scripts/woa/01_woa_b_visualisation.py',
    'scripts/mixcor/test_water_mass_mix_corrections.py'

    ]


## Execute scripts one by one:

for i in list_of_scripts:
    
    subprocess.call(['python', i])
    now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Finished: " + i.split('/')[-1] + " (" + now + ")")
