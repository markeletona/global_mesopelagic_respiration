# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:50:40 2024

Plot SIGMA0 sections across oceans

@author: Markel
"""

#%% IMPORTS

import numpy as np
import pandas as pd
import pathlib
import os
from shapely import geometry, from_geojson
from scipy.interpolate import RBFInterpolator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string


#%% LOAD DATA

# Unfiltered, merged Hansell+GLODAP dataset:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_unfiltered.csv'
tbl = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                   'CRUISE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})


## Ocean polygons
fpath = pathlib.Path('deriveddata/ocean_wm_defs/').glob("*")
dlist = [x for x in fpath if x.is_dir()]

# List polygon files (they have .geojson extension), read and store them.
# There is one polygon file for each ocean, found in the 'ocean' subfolder of
# each ocean
ocean_polys = {}
for d in dlist:
    
    # Get path to polygon file
    fpath = [*pathlib.Path(str(d) + "\\ocean").glob("*.geojson")][0]
    
    # Read and store it
    o = str(d).split("\\")[2]
    ocean_polys[o] = from_geojson(fpath.read_text())


#%% DATA HANDLING

# Assign missing/invalid values as NAN:
tbl.replace(-9999, np.nan, inplace=True)

# When representing the water mass assignations, enforce the depth range set
# for water masses
# (we have loaded the unfiltered version to represent SIGMA0 up to the surface
#  but do not do the same with water masses)
drange = [150, 1500]
idx = ((tbl.CTD_PRESSURE >= drange[0]) &
       (tbl.CTD_PRESSURE <= drange[1]) &
       (~(tbl.WATER_MASS=="NO_WATER_MASS")))
tbl_f = tbl.loc[idx, :]



#%% SECTIONS

def sigma_col(vmin=21, vmax=29, specific=None, reverse_levels=False):
    
    col_dict = {21: ['#f9f8d7', # yellow
                     '#f4f1b0',
                     '#eeeb88',
                     '#e9e461',
                     '#e3dd39'],
                22: ['#d7eff8', # light blue
                     '#afdef1',
                     '#87cee9',
                     '#5fbde2',
                     '#37addb'],
                23: ['#fff9f0', # orange
                     '#ffe5c0',
                     '#ffcc85',
                     '#ffb650',
                     '#ff990a'],
                24: ['#cceacd', # green
                     '#99d59a',
                     '#67c068',
                     '#34ab35',
                     '#019603'],
                25: ['#f1cece', # red
                     '#e39e9e',
                     '#d66d6d',
                     '#c83d3d',
                     '#ba0c0c'],
                26: ['#e0d6e2', # purple
                     '#c1adc5',
                     '#a383a9',
                     '#845a8c',
                     '#65316f'],
                27: ['#ced4e0', # blue
                     '#9ea9c2',
                     '#6d7da3',
                     '#3d5285',
                     '#0c2766'],
                28: ['#cfebe7', # greenblue
                     '#a0d6cf',
                     '#70c2b8',
                     '#41ada0',
                     '#119988'],
                29: ['#ddd4d1', # brown
                     '#bbaaa3',
                     '#987f74',
                     '#765546',
                     '#542a18']}
    
    # If specific levels are not requested, return all levels between vmin and
    # vmax. If they are requested, return only those levels.
    if specific==None:
        col_list = []
        for i in range(vmin, vmax + 1):
            if reverse_levels:
                # reverse order of colours *within* each level
                col_list.extend(list(reversed(col_dict[i])))
            else:
                col_list.extend(col_dict[i])
    else:
        col_list = []
        for i in specific:
            if reverse_levels:
                col_list.extend(list(reversed(col_dict[i])))
            else:
                col_list.extend(col_dict[i])
        
    return col_list
    
wm_markers = {'Atlantic': {'WNACW' : ['$W$', '#e9e461'],
                           'ENACW' : ['$E$', '#87cee9'],
                           'WSACW' : ['$X$', '#ffcc85'],
                           'ESACW' : ['$Y$', '#67c068'],
                           'SPMW'  : ['$S$', '#d66d6d'],
                           'SAIW'  : ['$I$', '#a383a9'],
                           'MW'    : ['$M$', '#6d7da3'],
                           'AAIW_A': ['$A$', '#70c2b8'],
                           'LSW'   : ['$L$', '#777777'],
                           'UNADW' : ['$U$', '#e9e461'],
                           'CDW'   : ['$C$', '#87cee9']
                           },
              'Indian'  : {'ASW'   : ['$S$', '#e9e461'],
                           'IUW'   : ['$U$', '#87cee9'],
                           'SICW'  : ['$W$', '#ffcc85'],
                           'IEW'   : ['$E$', '#67c068'],
                           'IIW'   : ['$I$', '#d66d6d'],
                           'AAIW_I': ['$A$', '#70c2b8'],
                           'RSPGIW': ['$R$', '#a383a9']
                           }, 
              'Pacific' : {'WNPCW' : ['$W$', '#a383a9'],
                           'ENPCW' : ['$E$', '#777777'],
                           'CCST'  : ['$C$', '#ffcc85'],
                           'NPEW'  : ['$P$', '#67c068'],
                           'SPEW'  : ['$X$', '#d66d6d'],
                           'WSPCW' : ['$Y$', '#e9e461'],
                           'ESPCW' : ['$Z$', '#6d7da3'],
                           'PCCST' : ['$T$', '#ffcc85'],
                           'PSUW'  : ['$U$', '#87cee9'],
                           'NPIW'  : ['$N$', '#eeeb88'],
                           'PEqIW' : ['$Q$', '#87cee9'],
                           'AAIW_P': ['$A$', '#70c2b8'],
                           },
              'Mediterranean' : {'WMCW'  : ['$W$', '#a383a9'],
                                 'WMIW'  : ['$I$', '#777777'],
                                 'WMDW'  : ['$D$', '#ffcc85'],
                                 'EMCW'  : ['$E$', '#67c068'],
                                 'EMIW'  : ['$Y$', '#d66d6d'],
                                 'EMDW'  : ['$Z$', '#e9e461']}
              }


    
cm = 1/2.54
sections = {'s1': ['Atlantic', -22],
            's2': ['Atlantic', -60],
            's3': ['Indian', 55],
            's4': ['Indian', 95],
            's5': ['Pacific', -105],
            's6': ['Pacific', -150],
            's7': ['Pacific', 177]
            }
d = 5 # section width at each side, in degrees (e.g., if 5, total width = 10º)
nrow = len(sections)
ncol = 3
# Set the desired Z variables:
zs = ['SIGMA0']
# Create dictionaries to store the results:
rvalues = {}
nn = {}
for iz, zname in enumerate(zs):
    
    # Set up plot
    # (with to separate gridspecs to control spacing between sections and map)
    fig = plt.figure(figsize=(8*cm*ncol, 5*cm*nrow))
    gs = GridSpec(nrow, 4, width_ratios=[1, .85, .05, .4], figure=fig)
    # gs_map = GridSpec(nrow, 1, figure=fig)
    cntr = 0

    for s in sections:
        
        #### Interpolation
        
        o = sections[s][0] # ocean of the section
        l = sections[s][1] # longitude of the section
        l_e = l + d # eastern limit based on the provided d
        l_w = l - d # western limit based on the provided d
        
        # Subset data (account for antimeridian crossing)
        longitudes = tbl.LONGITUDE.copy()
        if (l_e > 180):
            longitudes[longitudes < 0] = longitudes[longitudes < 0] + 360
        elif (l_w < -180):
            longitudes[longitudes > 0] = longitudes[longitudes > 0] - 360
            
        idx = (tbl.OCEAN==o) & (longitudes > l_w) & ((longitudes < l_e))
        ss = tbl.loc[idx, :].copy()
        
        # Average duplicated coordinates (otherwise interpolation breaks)
        ss2 = ss.groupby(['LATITUDE', 'CTD_PRESSURE']).agg({zname: 'mean'}).reset_index()
    
        # Set the desired X and Y variables:
        ix = ss2.LATITUDE
        iy = ss2.CTD_PRESSURE

        # Get the min/max values to set up the interpolation mesh, and the standard
        # deviation to scale the axes (otherwise they differ in orders of magnitude,
        # and the interpolation is greatly influenced horizontally due to this, 
        # resulting in a banded pattern).
        xmin = min(ix)
        xmax = max(ix)
        ymin = min(iy)
        ymax = max(iy)
        xsd = np.std(ix)
        ysd = np.std(iy)

        # Create the interpolation grid fitting to the requirements of RBFInterpolator:
        ndim = 250
        rx, ry = np.meshgrid(np.linspace(xmin, xmax, ndim)/xsd,
                             np.linspace(ymin, ymax, ndim)/ysd)
        meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)

        # Assemble the data point coordinates and values as required:
        dpc = np.array([ix/xsd,iy/ysd]).T
        dpv = ss2.loc[:, zname]
        
        # Exclude missing values:
        notnan = ~np.isnan(dpv)
        dpc = dpc[notnan]
        dpv = dpv[notnan]
        
        # Check that zname has valid values (if all nan -> the interpolator breaks):
        if any(notnan):
        
            # Perform interpolation with RBF:
            rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                                   neighbors=100, smoothing=0)
            rv = rbfi(meshflat)
            rv = rv.reshape(ndim, ndim)
        
        # Exclude continents from interpolated data (using ocean polygons)
        # Consider the longitudinal extent of the section and account for 
        # antimedirian crossing.
        pe = l_e
        pw = l_w
        if (l_e > 180):
            pe = pe - 360
        elif (l_w < -180):
            pw = pw + 360
            
        # Get which grid points fall outside of ocean
        latitudes = (rx*xsd)[1, :]
        pts_east = [geometry.Point(pe, i) for i in latitudes]
        pts_west = [geometry.Point(pw, i) for i in latitudes]
        boo_east = ocean_polys[o.lower()].contains(pts_east)
        boo_west = ocean_polys[o.lower()].contains(pts_west)
        boo = (boo_east | boo_west)
        
        # Adjust to last data off the coast before/after each landmass
        ocean = latitudes[boo] # get latitudes corresponding to ocean
        # For this, find when the ocean latitude values in grid jump (means 
        # change from one land gap to another)
        diff = abs(ocean[:-1] - ocean[1:])
        idx2 = (diff > np.median(diff) + .001).nonzero()[0] # add .001 to avoid floating point issues
        boo2 = np.empty((len(idx2), len(latitudes)))
        for iv, v in enumerate(idx2):
            # For each gap, find the closest measurement point to gap (land)
            gap_lower = max(ix[ix < ocean[v]])
            gap_upper = min(ix[ix > ocean[v]])
            # Record positions to mask with nans
            boo2[iv, :] = (latitudes > gap_lower) & (latitudes < gap_upper)
        # Merge booleans for each gap into a single array
        boo2 = np.any(boo2, axis=0)
        rv[:, boo2] = np.nan

        

        #### Plotting
        
        ## CONTOURF SECTION
        
        ax_sec = fig.add_subplot(gs[cntr, 0])
        ax_sec.scatter(ix.unique(), [-30]*len(ix.unique()),
                       marker='v',
                       s=3, c='k',
                       edgecolor='none',
                       clip_on=False)
        lvls_cf = np.arange(21, 29.2, .2)
        cfsig = ax_sec.contourf(rx*xsd, ry*ysd, rv,
                                levels=lvls_cf,
                                extend='both',
                                colors=sigma_col(21, 28),
                                zorder=0)
        ax_sec.set(xlim=[-80, 80],
                   xticks=range(-80, 85, 20),
                   ylim=[0, 1500],
                   yticks=range(0, 1800, 300))
        ax_sec.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax_sec.set_ylabel("Depth [dbar]", fontsize=7)
        ax_sec.tick_params(axis='both', which='major', length=2, labelsize=6)
        ax_sec.invert_yaxis()
        if cntr==(nrow - 1):
            ax_sec.set_xlabel("Latitude [$\degree$N]",
                              fontsize=7, labelpad=2)
        cb_sec = fig.colorbar(cfsig, ax=ax_sec,
                              ticks=range(21, 30))
        cb_sec.ax.tick_params(labelsize=6)
        cb_sec.ax.set_ylabel("$\sigma _\\theta$ [g·kg$^{-1}$]", fontsize=7)
        cb_sec.ax.yaxis.set_minor_locator(ticker.MultipleLocator(.2))
        ax_sec.text(-.25, .97, string.ascii_lowercase[cntr],
                    weight='bold',
                    ha='left',
                    fontsize=9,
                    transform=ax_sec.transAxes)
        
        ## WATER MASSES
        
        # Scatter plot of samples with water mass assignations
        ax_wm = fig.add_subplot(gs[cntr, 1])
        lvls_c1 = np.arange(21, 29)
        csig1 = ax_wm.contour(rx*xsd, ry*ysd, rv,
                              levels=lvls_c1,
                              colors='#ddd',
                              linewidths=.7,
                              zorder=1)
        lvls_c2 = np.arange(21.5, 29.5)
        csig2 = ax_wm.contour(rx*xsd, ry*ysd, rv,
                              levels=lvls_c2,
                              colors='#ddd',
                              linewidths=.2,
                              zorder=1)
        handles = []
        
        # (Subset data again but only from the filtered table to have the
        #  proper depth range)
        longitudes_f = tbl_f.LONGITUDE.copy()
        idx_f = ((tbl_f.OCEAN==o) &
                 (longitudes_f > l_w) & ((longitudes_f < l_e)))
        ss_f = tbl_f.loc[idx_f, :].copy()
        for w in wm_markers[o]:
            ss3 = ss_f.loc[ss_f.WATER_MASS==w, :]
            ax_wm.scatter(ss3.LATITUDE, ss3.CTD_PRESSURE,
                          marker=wm_markers[o][w][0],
                          c=wm_markers[o][w][1],
                          s=1, linewidth=.1,
                          zorder=0)
            handles.append(Line2D([0], [0],
                                  color='none',
                                  marker=wm_markers[o][w][0],
                                  markeredgecolor='none',
                                  markerfacecolor=wm_markers[o][w][1],
                                  label=w,
                                  markersize=4,
                                  markeredgewidth=.1))
            
            
        ax_wm.set(xlim=[-80, 80],
                  xticks=range(-80, 85, 20),
                  ylim=[0, 1500],
                  yticks=range(0, 1800, 300),
                  yticklabels=[])
        ax_wm.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax_wm.set_ylabel(None)
        ax_wm.tick_params(axis='both', which='major', length=2, labelsize=6)
        ax_wm.invert_yaxis()
        if cntr==(nrow - 1):
            ax_wm.set_xlabel("Latitude [$\degree$N]",
                             fontsize=7, labelpad=2)
            

        leg1 = ax_wm.legend(handles=handles,
                            title="Water mass",
                            title_fontsize=5,
                            handletextpad=.1,
                            prop={'size': 5},
                            bbox_to_anchor=(1.02, .5), 
                            loc='center left')
        leg1.get_frame().set_edgecolor('none')
            

        ## MAP
            
        ss2 = tbl.loc[idx, :].copy()
        ss2 = ss2.loc[~ss2.duplicated(['EXPOCODE', 'STATION']),:]
        
        proj = ccrs.Mollweide(central_longitude=-160)
        ax_map = fig.add_subplot(gs[cntr, 3], projection=proj)
        ax_map.add_feature(cfeature.LAND, facecolor='#ccc', edgecolor='k')
        ax_map.scatter(x=ss2.LONGITUDE,
                       y=ss2.LATITUDE,
                       c='#a81c1c',
                       edgecolor='none',
                       s=.5,
                       transform=ccrs.PlateCarree(),
                       zorder=2)
        ax_map.set_global()
        
        cntr += 1
        
        # "Fix" aliasing problem (white lines between contours) before exporting to pdf:
        # (although the white lines are truly more a problem of the pdf viewer)
        # for c in cfsig.collections:
        #     c.set_edgecolor("face")
        
dpath = 'figures/ocean_wm_defs/helpers/'
if not os.path.exists(dpath): os.makedirs(dpath)
fpath = 'figures/ocean_wm_defs/helpers/sections_sigma0_watermass.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)
        


#%%% MEDITERRANEAN

# Custom section for the Mediterranean

# Mediterranean cruises overall follow an east-west axis so just do section
# against longitude
sections = {'s1': ['Mediterranean']}


nrow = len(sections)
ncol = 3
# Set the desired Z variables:
zs = ['SIGMA0']
# Create dictionaries to store the results:
for iz, zname in enumerate(zs):
    
    # Set up plot
    # (with to separate gridspecs to control spacing between sections and map)
    fig = plt.figure(figsize=(8*cm*ncol, 5*cm*nrow))
    gs = GridSpec(nrow, 4, width_ratios=[1, .85, .05, .4], figure=fig)
    # gs_map = GridSpec(nrow, 1, figure=fig)
    cntr = 0

    for s in sections:
        
        #### Interpolation
        
        o = sections[s][0] # ocean of the section
        
        # Subset data 
        idx = (tbl.OCEAN==o)
        ss = tbl.loc[idx, :].copy()
        
        # Average duplicated coordinates (otherwise interpolation breaks)
        ss2 = ss.groupby(['LONGITUDE', 'CTD_PRESSURE']).agg({zname: 'mean'}).reset_index()
    
        # Set the desired X and Y variables:
        ix = ss2.LONGITUDE
        iy = ss2.CTD_PRESSURE

        # Get the min/max values to set up the interpolation mesh, and the standard
        # deviation to scale the axes (otherwise they differ in orders of magnitude,
        # and the interpolation is greatly influenced horizontally due to this, 
        # resulting in a banded pattern).
        xmin = min(ix)
        xmax = max(ix)
        ymin = min(iy)
        ymax = max(iy)
        xsd = np.std(ix)
        ysd = np.std(iy)

        # Create the interpolation grid fitting to the requirements of RBFInterpolator:
        ndim = 250
        rx, ry = np.meshgrid(np.linspace(xmin, xmax, ndim)/xsd,
                             np.linspace(ymin, ymax, ndim)/ysd)
        meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)

        # Assemble the data point coordinates and values as required:
        dpc = np.array([ix/xsd,iy/ysd]).T
        dpv = ss2.loc[:, zname]
        
        # Exclude missing values:
        notnan = ~np.isnan(dpv)
        dpc = dpc[notnan]
        dpv = dpv[notnan]
        
        # Check that zname has valid values (if all nan -> the interpolator breaks):
        if any(notnan):
        
            # Perform interpolation with RBF:
            rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                                   neighbors=100, smoothing=0)
            rv = rbfi(meshflat)
            rv = rv.reshape(ndim, ndim)

        

        #### Plotting
        
        ## CONTOURF SECTION
        
        ax_sec = fig.add_subplot(gs[cntr, 0])
        ax_sec.scatter(ix.unique(), [-30]*len(ix.unique()),
                       marker='v',
                       s=3, c='k',
                       edgecolor='none',
                       clip_on=False)
        lvls_cf = np.arange(21, 30.2, .2)
        cfsig = ax_sec.contourf(rx*xsd, ry*ysd, rv,
                                levels=lvls_cf,
                                extend='both',
                                colors=sigma_col(21, 29),
                                zorder=0)
        ax_sec.set(xlim=[-7, 37],
                   xticks=range(-5, 40, 5),
                   ylim=[0, 1500],
                   yticks=range(0, 1800, 300))
        ax_sec.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
        ax_sec.set_ylabel("Depth [dbar]", fontsize=7)
        ax_sec.tick_params(axis='both', which='major', length=2, labelsize=6)
        ax_sec.invert_yaxis()
        if cntr==(nrow - 1):
            ax_sec.set_xlabel("Longitude [$\degree$N]",
                              fontsize=7, labelpad=2)
        cb_sec = fig.colorbar(cfsig, ax=ax_sec,
                              ticks=range(21, 31))
        cb_sec.ax.tick_params(labelsize=6)
        cb_sec.ax.set_ylabel("$\sigma _\\theta$ [g·kg$^{-1}$]", fontsize=7)
        cb_sec.ax.yaxis.set_minor_locator(ticker.MultipleLocator(.2))
        ax_sec.text(-.25, .97, string.ascii_lowercase[cntr],
                    weight='bold',
                    ha='left',
                    fontsize=9,
                    transform=ax_sec.transAxes)
        
        ## WATER MASSES
        
        # Scatter plot of samples with water mass assignations
        ax_wm = fig.add_subplot(gs[cntr, 1])
        lvls_c1 = np.arange(21, 30)
        csig1 = ax_wm.contour(rx*xsd, ry*ysd, rv,
                              levels=lvls_c1,
                              colors='#ddd',
                              linewidths=.7,
                              zorder=1)
        lvls_c2 = np.arange(21.5, 30.5)
        csig2 = ax_wm.contour(rx*xsd, ry*ysd, rv,
                              levels=lvls_c2,
                              colors='#ddd',
                              linewidths=.2,
                              zorder=1)
        handles = []
        
        # (Subset data again but only from the filtered table to have the
        #  proper depth range)
        idx_f = (tbl_f.OCEAN==o)
        ss_f = tbl_f.loc[idx_f, :].copy()
        for w in wm_markers[o]:
            ss3 = ss_f.loc[ss_f.WATER_MASS==w, :]
            ax_wm.scatter(ss3.LONGITUDE, ss3.CTD_PRESSURE,
                          marker=wm_markers[o][w][0],
                          c=wm_markers[o][w][1],
                          s=1, linewidth=.1,
                          zorder=0)
            handles.append(Line2D([0], [0],
                                  color='none',
                                  marker=wm_markers[o][w][0],
                                  markeredgecolor='none',
                                  markerfacecolor=wm_markers[o][w][1],
                                  label=w,
                                  markersize=4,
                                  markeredgewidth=.1))
            
            
        ax_wm.set(xlim=[-7, 37],
                  xticks=range(-5, 40, 5),
                  ylim=[0, 1500],
                  yticks=range(0, 1800, 300),
                  yticklabels=[])
        ax_wm.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
        ax_wm.set_ylabel(None)
        ax_wm.tick_params(axis='both', which='major', length=2, labelsize=6)
        ax_wm.invert_yaxis()
        if cntr==(nrow - 1):
            ax_wm.set_xlabel("Latitude [$\degree$N]",
                             fontsize=7, labelpad=2)
            

        leg1 = ax_wm.legend(handles=handles,
                            title="Water mass",
                            title_fontsize=5,
                            handletextpad=.1,
                            prop={'size': 5},
                            bbox_to_anchor=(1.02, .5), 
                            loc='center left')
        leg1.get_frame().set_edgecolor('none')
            

        ## MAP
            
        ss2 = tbl.loc[idx, :].copy()
        ss2 = ss2.loc[~ss2.duplicated(['EXPOCODE', 'STATION']),:]
        
        proj = ccrs.Mercator(central_longitude=20)
        ax_map = fig.add_subplot(gs[cntr, 3], projection=proj)
        ax_map.add_feature(cfeature.LAND, facecolor='#ccc', edgecolor='k',
                           lw=.1)
        ax_map.scatter(x=ss2.LONGITUDE,
                       y=ss2.LATITUDE,
                       c='#a81c1c',
                       edgecolor='none',
                       s=.5,
                       transform=ccrs.PlateCarree(),
                       zorder=2)
        # ax_map.set_global()
        
        cntr += 1
        
fpath = 'figures/ocean_wm_defs/helpers/sections_sigma0_watermass_MED.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%% MAPS
    
    
# Map SIGMA0 contours at specific depths to get isopycnal extensions
sigma0_depths = [[0, 10],
                 [190, 210],
                 [590, 610],
                 [990, 1010]] # small range

# Set interpolation grid resolution
deg_res_lon = 1
deg_res_lat = 1


nrow = len(sigma0_depths)
kws = {'projection': ccrs.Mollweide(central_longitude=-160)}
fig_m, ax_m = plt.subplots(nrows=nrow, ncols=1,
                           squeeze=0,
                           figsize=(10*cm * 1, 5*cm * nrow),
                           subplot_kw=kws)
fig_i, ax_i = plt.subplots(nrows=nrow, ncols=2,
                           squeeze=0,
                           figsize=(8*cm * 2, 6*cm * nrow))

offset = 20
for iz, z in enumerate(sigma0_depths):
    
    #### Interpolation
    
    # Subset data and remove NaNs
    idx = (tbl.CTD_PRESSURE > z[0]) & (tbl.CTD_PRESSURE < z[1])
    tbl_z = tbl.loc[idx, :].copy()
    tbl_z = tbl_z.loc[~np.isnan(tbl_z.loc[:, 'SIGMA0']), :]
    
    # Wrap around antimeridian by 'offset' degrees, duplicating data, to aid
    # interpolation in such edge case (will be clipped after interpolation).
    # (basically copies patches of data left and right of data edges)
    # Copy data and transform longitude to put it to the right of the antimed.
    idx1 = (tbl_z.LONGITUDE < (-180 + offset)) & (tbl_z.LONGITUDE > (-180))
    tbl_z_copy1 = tbl_z.loc[idx1, :].copy()
    tbl_z_copy1.LONGITUDE = tbl_z_copy1.LONGITUDE + 360
    
    # Same to put it left
    idx2 = (tbl_z.LONGITUDE > (180 - offset)) & (tbl_z.LONGITUDE < (180))
    tbl_z_copy2 = tbl_z.loc[idx2, :].copy()
    tbl_z_copy2.LONGITUDE = tbl_z_copy2.LONGITUDE - 360
    
    # Gather "duplicated" data with original
    tbl_z_wrapped = pd.concat([tbl_z, tbl_z_copy1, tbl_z_copy2])

    # Average duplicated coordinates (otherwise interpolation breaks)
    # (round a bit because when interpolating superclose points with different
    # values return weird interpolation patterns locally)
    tbl_z_wrapped.LONGITUDE = round(tbl_z_wrapped.LONGITUDE, 0)
    tbl_z_wrapped.LATITUDE = round(tbl_z_wrapped.LATITUDE, 0)
    tbl_z_wrapped = tbl_z_wrapped.groupby(['LATITUDE', 'LONGITUDE']).agg({'SIGMA0': 'mean'}).reset_index()
    
    # Smooth data averaging values around point 
    # (because when interpolating superclose points with different values 
    # interpolation returns weird patterns locally)
    radius = 2
    tbl_z_wrapped['SIGMA0_S'] = np.nan
    for i, r in tbl_z_wrapped.iterrows():
        
        # Estimate distance from sample r to the other samples
        dist = ((tbl_z_wrapped.LONGITUDE - r.LONGITUDE)**2 + 
                (tbl_z_wrapped.LATITUDE - r.LATITUDE)**2) ** .5
        
        # Retain only samples within set radius
        within_radius = dist < radius
        dist = dist[within_radius]
        
        # Estimate weighted average depending on distance from sample
        dist_frac = ((radius - dist)/radius)**2 # **1 = linear
        weights = dist_frac/sum(dist_frac)
        values = tbl_z_wrapped.loc[within_radius, 'SIGMA0']
        tbl_z_wrapped.loc[i, 'SIGMA0_S'] = np.nansum(values * weights)
        
    
    # Set the desired X and Y variables:
    ix = tbl_z_wrapped.LONGITUDE
    iy = tbl_z_wrapped.LATITUDE
    xmin = min(ix)
    xmax = max(ix)
    ymin = min(iy)
    ymax = max(iy)
    xsd = np.std(ix)
    ysd = np.std(iy)
    
    # Create the interpolation grid fitting to the requirements of RBFInterpolator:
    ndim_lon = int(round((360 + offset*2) / deg_res_lon))
    ndim_lat = int(round(180 / deg_res_lat))
    rx, ry = np.meshgrid(np.linspace(xmin, xmax, ndim_lon)/xsd,
                         np.linspace(ymin, ymax, ndim_lat)/ysd)
    meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)

    # Assemble the data point coordinates and values as required:
    dpc = np.array([ix/xsd,iy/ysd]).T
    dpv = tbl_z_wrapped.loc[:, 'SIGMA0_S']
            
    # Exclude missing values:
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
            
    # Perform interpolation with RBF
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv = rbfi(meshflat)
    rv = rv.reshape(ndim_lat, ndim_lon)
    
    # Clip the duplicated data for edge cases
    idx = (ix > -180) & (ix < 180)
    ix2 = ix[idx]
    iy2 = iy[idx]
    ndim_lon2 = int(round(360 / deg_res_lon))
    idx = (rx*xsd > -180) & (rx*xsd < 180)
    rx2 = rx[idx].reshape(ndim_lat, ndim_lon2)
    ry2 = ry[idx].reshape(ndim_lat, ndim_lon2)
    rv2 = rv[idx].reshape(ndim_lat, ndim_lon2)
    
    
    #### Plotting
    
    ## Map results
    
    #--- Interpolation control
    
    ax_i[iz, 0].scatter(dpc[:,0]*xsd,
                        dpc[:,1]*ysd,
                        c=dpv,
                        marker='o',
                        s=3,
                        vmin=21, vmax=29,
                        edgecolor='none')
    cfsig = ax_i[iz, 1].contourf(rx*xsd,
                                 ry*ysd,
                                 rv,
                                 levels=np.arange(21, 29, .2),
                                 extend='both',
                                 vmin=21, vmax=29)

    for c in range(2):
        for lon in [-180, 180]:
            ax_i[iz, c].axvline(x=lon, c='#555',
                                linestyle='dashed', linewidth=.8)
        ax_i[iz, c].set(xlim=[(-180 - offset), (180 + offset)],
                        ylim=[-90, 90],
                        yticks=range(-90, 100, 45))
        ax_i[iz, c].tick_params(axis='both', which='major',
                                length=2, labelsize=6)

    ax_i[iz, 0].text(.005, .95, str(z[0]) + " – " + str(z[1]) + " dbar",
                     size=6, fontweight='bold', ha='left',
                     transform=ax_i[iz, 0].transAxes)
    cb_m1 = fig.colorbar(cfsig, ax=ax_i[iz, 1],
                         ticks=np.arange(21, 29.5, 1),
                         shrink=1)
    cb_m1.ax.tick_params(labelsize=6, length=2, width=.6)
    cb_m1.ax.set_ylabel("$\sigma _\\theta$ [g·kg$^{-1}$]", fontsize=7)
    
    fpath = "figures/ocean_wm_defs/helpers/help_dens_map_interpolation_control.pdf"
    fig_i.savefig(fpath, format='pdf', bbox_inches='tight')
    
    
    #--- Actual results
    
    csig = ax_m[iz, 0].contour(rx2*xsd,
                               ry2*ysd,
                               rv2,
                               levels=np.arange(21, 29.2, .2),
                               vmin=21, vmax=29,
                               colors=sigma_col(21, 28, reverse_levels=True),
                               linewidths=.5,
                               zorder=0,
                               transform=ccrs.PlateCarree())
    ax_m[iz, 0].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                         name='land',
                                                         scale='50m'),
                            facecolor='#ccc',
                            edgecolor='black',
                            linewidth=.25,
                            zorder=1)
    ax_m[iz, 0].set_global()
    ax_m[iz, 0].text(.005, .975, str(z[0]) + " – " + str(z[1]) + " dbar", 
                     size=6, fontweight='bold', ha='left',
                     transform=ax_m[iz, 0].transAxes)
    cb_m2 = fig.colorbar(csig, ax=ax_m[iz, 0],
                         ticks=np.arange(21, 29.5, 1),
                         shrink=.75)
    cb_m2.outline.set_linewidth(.6)
    cb_m2.ax.tick_params(labelsize=6, length=2, width=.6)
    cb_m2.ax.set_ylabel("$\sigma _\\theta$ [g·kg$^{-1}$]", fontsize=7)
    
    fpath = "figures/ocean_wm_defs/helpers/help_dens_map.pdf"
    fig_m.savefig(fpath, format='pdf', bbox_inches='tight')



