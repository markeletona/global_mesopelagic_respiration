# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:08:50 2024

Read NPP data and create mean climatologies.

Do it for ancillary data too (SST).

Data source: http://orca.science.oregonstate.edu/npp_products.php

@author: Markel
"""

#%% IMPORTS

import os
import shutil
import gzip
import tarfile
import numpy as np
import datetime
import rioxarray as rxr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo


#%% READ FILES ~ NPP

# I don't know how to directly read an hdf4 file that is gz-ed within a tar 
# file (it might not even be possible?), so I'll unpack everything temporarily

# Get list of *directories* (one per model of NPP)
dpath = 'rawdata/npp/'
ds = [fn for fn in os.listdir(dpath) if os.path.isdir(os.path.join(dpath,fn))]

## Go through the models
npp_all = {}
npp_mean = {}
npp_dates = {}
for d in ds:

    # List tar files that pack monthly data per year
    dpath_model = 'rawdata/npp/' + d
    flist = [x for x in os.listdir(dpath_model) if '.tar' in x]

    # Go through all files
    hdf_list = []
    hdf_dates = []
    for f in flist:
        
        # open tar file and extract contents 
        fpath = dpath + d + '/' + f
        dpath_temp = dpath + 'tempfolder/'
        with tarfile.open(fpath) as tfile:
            tfile.extractall(dpath_temp) # with closes file when exiting
        
        # List extracted files
        flist_temp = os.listdir(dpath_temp)
        
        # Extracted files are .gz files that I need to ungzip to read the hdf
        for f2 in flist_temp:
            
            # .gz files end in ".hdf.gz"
            # set paths for .gz file and the .hdf file that contains
            fpath_gz = dpath_temp + f2
            fpath_hdf = fpath_gz.replace(".gz", "")
            
            # Uncompress .gz file
            with gzip.open(fpath_gz, 'rb') as f2_in:
                with open(fpath_hdf, 'wb') as f2_out:
                    shutil.copyfileobj(f2_in, f2_out)
            
            # Read hdf file and append to list
            with rxr.open_rasterio(fpath_hdf, masked=True) as f_hdf:
                
                # Get data
                hdf_data = f_hdf.data
                # Replace bad values
                hdf_data[hdf_data==-9999] = np.nan
                # Append to list of arrays
                hdf_list.append(hdf_data)
                
                # Save date
                dstr = f_hdf.attrs['Start Time String'].split(" ")[0]
                dt = datetime.datetime.strptime(dstr, "%m/%d/%Y")
                hdf_dates.append(dt)

        # Remove temporary folder and all its contents
        shutil.rmtree(dpath_temp)


    # Concatenate the arrays from all months
    npp_all[d] = np.concatenate(hdf_list)

    # Estimate average value across months to obtain global climatology
    npp_mean[d] = np.nanmean(npp_all[d], axis=0)

    # Note that nanmean() returns a warning when all values are NaN
    # RuntimeWarning: Mean of empty slice
    # but behaves correctly and returns a NaN.
    # Ignore it.
        
    # Store dates
    npp_dates[d] = hdf_dates
    

#%% CREATE COORDINATES

#### From the OSU webpage FAQs:
# 
#--- What projection are these data in?
# 
# These hdf files are in an Equidistant Cylindrical projection.
# 
#--- How do we convert rows and columns to lats and lons?
# 
# The rows represent lines of latitude, and the columns coincide with longitude.
# 
# For 1080 by 2160 data, the grid spacing is 1/6 of a degree in both latitude and longitude.
# 1080 rows * 1/6 degree per row = 180 degrees of latitude (+90 to -90).
# 2160 columns * 1/6 degree per column = 360 degrees of longitude (-180 to +180).
# 
# The north west corner of the start of the hdf file is at +90 lat, -180 lon.
# 
# To obtain the location of the center of any pixel:
# - take the number of rows and columns you are away from the NW corner,
# - multiply by the grid spacing to get the change in latitude and longitude,
# - subtract the change in latitude from +90 lat,
# - add the change in longitude to -180 lon;
# - shift the latitude down (subtract) by 1/2 of a grid spacing
# - and shift the longitude over (add) by 1/2 of a grid spacing
# 
# This gives the center of any pixel. To get the NW corner of any pixel, do not shift over and down by 1/2 of a grid spacing.

# Get the length of the latitude (rows) and longitude (cols) coords
len_lat = npp_mean['cbpm'].shape[0] # cbpm just as reference, all are the same
len_lon = npp_mean['cbpm'].shape[1]

# Compute grid spacing
grid_spacing = (90 - (-90))/len_lat

# Compute coordinates (center of pixel) following instructions
lat = [90 - grid_spacing * px - grid_spacing/2 for px in range(len_lat)]
lon = [-180 + grid_spacing * px + grid_spacing/2 for px in range(len_lon)]
lat = np.array(lat)
lon = np.array(lon)


#%% EXPORT

dpath_out = 'deriveddata/npp/climatology/'
if not os.path.exists(dpath_out):
    os.makedirs(dpath_out)
for d in npp_mean:
    
    # Export climatology of each model as numpy compressed .npz, along:
    # - latitude array
    # - longitude array
    # - array of dates the climatology is based on
    fout = dpath_out + d
    dt_a = np.array(npp_dates[d], dtype='datetime64[D]')
    np.savez_compressed(fout, lat=lat, lon=lon, npp=npp_mean[d], dates=dt_a)
    

#%% MAPS

cm = 1/2.54
mproj = ccrs.Mollweide(central_longitude=-160)

model_labels = {'cbpm': "CbPM", 'eppley-vgpm': "Eppley-VGPM"}
fd = {'fontsize': 11, 
      'fontweight': 'bold'}


#%%% NPP

nr = len(npp_mean)
fig, ax = plt.subplots(nrows=nr, ncols=1, figsize=(15*cm, 7.5*cm*nr),
                       subplot_kw={'projection': mproj})
for i, d in enumerate(npp_mean):
    pcm = ax[i].pcolormesh(lon, lat, npp_mean[d], vmin=200, vmax=2000,
                           cmap=cmo.cm.algae,
                           transform=ccrs.PlateCarree(),
                           rasterized=True,
                           zorder=0)
    cb = fig.colorbar(pcm, ax=ax[i],
                      ticks=range(200, 2300, 300),
                      extend='both',
                      pad=.04,
                      shrink=.75)
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_ylabel("NPP [mgC · m$^{-2}$ · day$^{-1}$]", fontsize=7)
    ax[i].set_title(model_labels[d], loc='left', fontdict=fd)
    ax[i].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                  name='land',
                                                  scale='50m'),
                      facecolor='#ccc',
                      edgecolor='black',
                      linewidth=.1,
                      zorder=1)
    ax[i].set_global()
    
fpath = "figures/npp/climatology/npp_climatologies.pdf"
fig.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False,
            dpi=500)

fpath = fpath.replace("pdf", "svg")
fig.savefig(fpath, format='svg', bbox_inches='tight', transparent=False,
            dpi=500)


