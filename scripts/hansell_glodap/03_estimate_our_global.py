# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:47:41 2024

@author: Markel Gómez Letona

Estimate Oxygen Utilisation Rates (OUR) in water masses.

"""

#%% IMPORTS

# general
import numpy as np
import pandas as pd
import pathlib
import os
from shapely import geometry, from_geojson, to_geojson
import warnings
import datetime as dt
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import sys
import gc

# plotting & mapping
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeoaxes
from scipy.stats import gaussian_kde, kruskal, spearmanr, norm, t

# regressions
import statsmodels.formula.api as smf
import scripts.modules.RegressConsensus as rc


#%% NOTE ON PARALLEL PROCESSING

# This script (specifically, the Monte Carlo simulations) will be run in 
# parallel. As it was written in Windows, the main code needs to be 'protected'
# within if __name__ == '__main__':, otherwise each child process will execute
# everyting. The code inside the if __name__ == '__main__': block will only be
# executed once by the parent process. This avoid wasting resources repeating
# the pre-processing, plots etc. every time a child process is run.

# https://stackoverflow.com/questions/20360686/compulsory-usage-of-if-name-main-in-windows-while-using-multiprocessi

if __name__ == '__main__':
    
    #%% DIRECTORIES FOR FIGURES
    
    # Create folders to save figures
    ds = ['figures/hansell_glodap/global/',
          'figures/hansell_glodap/global/helper/',
          'figures/hansell_glodap/global/helper/latitudinal/',
          'figures/hansell_glodap/global/regressions/',
          'figures/hansell_glodap/global/rates/',
          'figures/hansell_glodap/global/rates/profiles/',
          'figures/hansell_glodap/global/rates/maps/',
          'deriveddata/hansell_glodap/global/',
          'deriveddata/hansell_glodap/global/regression_pixel_polys/']
    for d in ds:
        if not os.path.exists(d): os.makedirs(d)
        
        
    #%% LOAD DATA
    
    # Filtered, merged Hansell+GLODAP dataset:
    fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
    tbl = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                       'CRUISE': str,
                                                       'BOTTLE': str,
                                                       'DATE': int})
        
    
    #%%% POLYGONS
    
    ## Oceans
    
    fpath = pathlib.Path('deriveddata/ocean_wm_defs/').glob("*")
    dlist = [x for x in fpath if x.is_dir()]
    
    # List polygon files (they have .geojson extension), read and store them.
    # There is one polygon file for each ocean, found in the 'ocean' subfolder of
    # each ocean
    ocean_polys = {}
    for d in dlist:
        
        # Get path to polygon file
        fpath = list((d / "ocean").glob("*.geojson"))[0]
        
        # Read and store it
        o = d.name
        ocean_polys[o] = from_geojson(fpath.read_text())
        
    
    ## Water masses
    
    wm_polys = {}
    wm_polys_plot = {} # load uncut version of polygons too (for mapping)
    wm_depths = ['central', 'intermediate', 'deep']
    for d in dlist:
        
        o = d.name
        wm_polys[o] = {}
        wm_polys_plot[o] = {}
        
        for z in wm_depths:
            
            # Get wm paths at depth z and ocean d
            flist = list((d / "wms" / z).glob("*.geojson"))
            
            # Skip iteration if certain depth is absent (i.e. flist is empty)
            # (to skip 'deep' in Indian and Pacific)
            if not flist: continue
            
            wm_polys[o][z] = {}
            wm_polys_plot[o][z] = {}
            for f in flist:
                
                # Get wm name (accounts for when the name itself has underscore, 
                # e.g. AAIW_P)            
                w = "_".join(f.stem.split("_")[0:-1])
    
                # Load polygon
                wm_polys[o][z][w] = from_geojson(f.read_text())
                wm_polys_plot[o][z][w] = from_geojson(f.read_text())
                
                # For Pacific, replace with uncut version in wm_polys_plot
                if o=='pacific':
                    fp = list(f.parts)
                    fp[-1] = fp[-1].replace("polygon", "polygon_uncut") 
                    f2 = (pathlib.Path(*fp[:fp.index(z)]).
                          joinpath('uncut').
                          joinpath(*fp[fp.index(z):]))
                    wm_polys_plot[o][z][w] = from_geojson(f2.read_text())
                    
    
    # Create a second version of the water mass dict, which instead of being nested
    # has composite keys following the pattern -> ocean-depth_layer-watermass
    # 
    # Create function to flatten nested dictionaries, resulting in a single dict
    # with structure {'key1_key2_key': value} (where key1, key2,... are the keys 
    # of the nested dicts)
    # See: https://stackoverflow.com/a/6027615
    def flatten(dictionary, parent_key='', sep='_'):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                items.extend(flatten(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    # Create the flattened dictionary
    wm_polys_flat = flatten(wm_polys, sep=';')
    wm_polys_plot_flat = flatten(wm_polys_plot, sep=';')
    
                
                
    #%% DATA HANDLING
    
    # Assign missing/invalid values as NAN:
    tbl = tbl.replace(-9999, np.nan)
    
    
    # Summarise number of values for each tracer age
    def summary_values(x):
        
        d = {}
        d['nAGE_CFC11'] = sum(~np.isnan(x.AGE_CFC11))
        d['nAGE_CFC12'] = sum(~np.isnan(x.AGE_CFC12))
        d['nAGE_SF6'] = sum(~np.isnan(x.AGE_SF6))
        d['pAGE_CFC11'] = round(100 * d['nAGE_CFC11'] / x.shape[0], 1)
        d['pAGE_CFC12'] = round(100 * d['nAGE_CFC12'] / x.shape[0], 1)
        d['pAGE_SF6'] = round(100 * d['nAGE_SF6'] / x.shape[0], 1)
    
        
        vrs = ['AGE_CFC11', 'AGE_CFC12', 'AGE_SF6']
        summary_string = [(v + " -> n = " + str(d['n' + v]) + 
                           " (" + str(d['p' + v]) + " %)\n") for v in vrs]
        summary_string = "".join(summary_string)
        print(summary_string)
        
        return d
    
    summary_values(tbl)
    
    
    #%% PLOT STATION MAP
    
    # Filter down to unique cruise~station values to avoid unnecessarily 
    # repeated data (it increases file size)
    tblu = tbl.drop_duplicates(subset=['EXPOCODE', 'STATION']).copy()
    
    cm = 1/2.54
    fig_s = plt.figure(figsize=(15*cm, 8*cm))
    ax_s = plt.subplot(1, 1, 1, projection=ccrs.Mollweide())
    ax_s.add_feature(cfeature.LAND, facecolor='#ccc', edgecolor='black')
    ax_s.add_feature(cfeature.OCEAN, facecolor='white')
    s_h = ax_s.scatter(x=tblu['LONGITUDE'],
                       y=tblu['LATITUDE'],
                       c='#a81c1c',
                       s=.5,
                       linewidth=.5,
                       transform=ccrs.PlateCarree(),
                       zorder=2)
    ax_s.set_global()
    
    fpath = 'figures/hansell_glodap/global/hansell_glodap_station_map.pdf'
    fig_s.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
    
    
    
    #%%% PER WATER MASS AND TRACER AGE
    
    # Map stations with valid observations of O2 and tracer ages within each water 
    # mass
    ages = ['AGE_CFC11', 'AGE_CFC12', 'AGE_SF6']
    age_labs = ["Age$_\mathregular{CFC\u201011}$",
                "Age$_\mathregular{CFC\u201012}$",
                "Age$_\mathregular{SF_6}$"]
    ages = dict(zip(ages, ages))
    age_labs = dict(zip(ages, age_labs))
    
    # Set number of rows (water masses) and columns (tracer ages)
    nr = len(wm_polys_flat)
    nc = len(ages)
    mproj = ccrs.Mollweide(central_longitude=-160)
    fig_w, ax_w = plt.subplots(nrows = nr, ncols = nc,
                               figsize=(5*cm * nc, 3*cm * nr),
                               subplot_kw={'projection': mproj})
    for ik, k in enumerate(wm_polys_flat):
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        for ig, g in enumerate(ages):
            
            # Subset samples for each w, g combination
            idx = ((~np.isnan(tbl['OXYGEN'])) &
                   (~np.isnan(tbl[g])) &
                   (tbl['WATER_MASS']==w))
            ss = tbl.loc[idx, :].copy()
            ss = ss.drop_duplicates(subset=['CRUISE', 'STATION'])
            
            ax_w[ik, ig].add_feature(cfeature.LAND,
                                     facecolor='#ccc', 
                                     edgecolor='#444',
                                     linewidth=.2,
                                     zorder=0)
            ax_w[ik, ig].add_geometries(wm_polys_flat[k],
                                        facecolor='none',
                                        edgecolor='k',
                                        linewidth=.7,
                                        crs=ccrs.PlateCarree(),
                                        zorder=1)
            ax_w[ik, ig].scatter(ss.LONGITUDE, ss.LATITUDE,
                                 s=.5,
                                 linewidth=.05,
                                 transform=ccrs.PlateCarree(),
                                 zorder=2)
            ax_w[ik, ig].text(.01, 1.04, 
                              "$\mathbf{" + w + "}$, " + age_labs[g],
                              size=4,
                              transform=ax_w[ik, ig].transAxes)
            ax_w[ik, ig].set_global()
    
    
    fpath = 'figures/hansell_glodap/global/helper/hansell_glodap_water_mass_age_sample_map.pdf'
    fig_w.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
    
    
    
    #%% PIXELATE WATER MASSES
    
    # Analysing water masses that occupy vast areas does not make much
    # sense oceanographycally speaking: biogeochemistry might not be homogeneous
    #
    # One approach is to segment the water masses in pixels, and analyse trends 
    # across them. A reasonsable pixel could be of 20º (or 10º), balancing detail 
    # and data availability.
    
    #%%% SET PIXEL BOUNDS
    
    # Set lines dividing pixels
    px_w = 10 # pixel width
    px_h = 10 # pixel height
    lonlines = {}
    latlines = {}
    for o in ocean_polys:
        if o=='pacific':
            lonlines[o] = []
            latlines[o] = []
            for i, p in enumerate(ocean_polys[o].geoms):
                
                # For Pacific, do it accounting for antimeridian
                
                # Rounds down|up to closest pixel width
                lonmin = np.floor(p.bounds[0] / px_w) * px_w
                lonmax = np.ceil(p.bounds[2] / px_w) * px_w
                lons = [*np.arange(lonmin, lonmax + px_w, px_w)]
                lons = [(l1, l2) for l1, l2 in zip(lons[:-1], lons[1:])]
                lonlines[o].extend(lons)
                
                # Rounds down|up to closest pixel height
                latmin = np.floor(p.bounds[1] / px_h) * px_h
                latmax = np.ceil(p.bounds[3] / px_h) * px_h
                lats = [*np.arange(latmin, latmax + px_h, px_h)]
                lats = [(l1, l2) for l1, l2 in zip(lats[:-1], lats[1:])]
                latlines[o].extend(lats)
            
            # Make sure to retain unique instances of the lat ranges
            unique_tupples = list(set(latlines[o]))
            # Order them (set does not preserve order)
            latlines[o] = sorted(unique_tupples, key=lambda tup: tup[1])
            
            
        else: # regular case for atlantic, indian (and med)
            p = ocean_polys[o]
            lonmin = np.floor(p.bounds[0] / px_w) * px_w
            lonmax = np.ceil(p.bounds[2] / px_w) * px_w
            lons = [*np.arange(lonmin, lonmax + px_w, px_w)]
            lonlines[o] = [(l1, l2) for l1, l2 in zip(lons[:-1], lons[1:])]
            latmin = np.floor(p.bounds[1] / px_h) * px_h
            latmax = np.ceil(p.bounds[3] / px_h) * px_h
            lats = [*np.arange(latmin, latmax + px_h, px_h)]
            latlines[o] = [(l1, l2) for l1, l2 in zip(lats[:-1], lats[1:])]
    
    
    #%%% CREATE PIXELS
    
    pixel_polys = {}
    for k in wm_polys_flat:
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        # Water masses of the Mediterranean (and its outflow, MW), will be left
        # in their entirety and not pixelated, because they are particularly small
        # have have not many data points
        do_not_split = [';MW', ';WMIW', ';LIW', ';WMDW', ';EMDW']
        pixelate = all([x not in k for x in do_not_split]) # if k is not any of of do_not_split
        
        pixel_polys[k] = {}
        if pixelate:
            for lo in lonlines[o]:
                for la in latlines[o]:
                    
                    # Create pixel polygon
                    lo1 = lo[0]
                    lo2 = lo[1]
                    la1 = la[0]
                    la2 = la[1]
                    pxp = geometry.Polygon([(lo1, la1), (lo1, la2), (lo2, la2), (lo2, la1)])
                
                    # Crop pixel with water mass polygon to shed excess where needed
                    wmp = wm_polys_flat[k]
                    pxp2 = pxp.difference(wmp.symmetric_difference(pxp))
                
                    # If pixel was out of water mass, returned bounds are all nan,
                    # skip it
                    if np.isnan(pxp2.bounds).all():
                        continue
                    else:
                        px_code = ";".join([str(x) for x in [lo1, la1, lo2, la2]])
                        pixel_polys[k][px_code] = pxp2
                        
        else:
            wmp = wm_polys_flat[k]
            px_code = ";".join([str(x) for x in wmp.bounds])
            pixel_polys[k][px_code] = wmp
                
                        
    
    
    #%%% MAP PIXELS    
    
    # Map pixels to ensure they are correctly created
    
    nc = 3
    nwm = len(pixel_polys)
    nr = int(nwm / nc) if (nwm % nc)==0 else int(nwm / nc + 1)
    
    px_pal = mpl.colormaps.get_cmap('turbo')
    
    fig_px = plt.figure(figsize=(5*cm * nc, 3*cm * nr))
    for ik, k in enumerate(pixel_polys):
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        # Initialise subplot
        mproj = ccrs.Mollweide(central_longitude=-160)
        ax_px = fig_px.add_subplot(nr, nc, ik + 1, projection=mproj)
        
        # Plot land, water mass polygon and sample points
        ax_px.add_feature(cfeature.LAND,
                          facecolor='#ccc',
                          edgecolor='#444',
                          linewidth=.2,
                          zorder=0)
        
        # Normalise pixel indices between [0,1] to get distinct colours from
        # px_pal
        col_norm = mpl.colors.Normalize(vmin=1, vmax=len(pixel_polys[k]))
        
        # Map water mass pixels with distinct colour just to aid visualisation
        for ip, p in enumerate(pixel_polys[k]):
            ax_px.add_geometries(pixel_polys[k][p],
                                 facecolor=px_pal(col_norm(ip + 1)),
                                 edgecolor='k',
                                 linewidth=.2,
                                 crs=ccrs.PlateCarree(),
                                 zorder=1)
            
        # Overlay water mass border as reference
        ax_px.add_geometries(wm_polys_flat[k],
                             facecolor='none',
                             edgecolor='k',
                             linewidth=.4,
                             crs=ccrs.PlateCarree(),
                             zorder=2)
        
        # Add water mass label
        ax_px.text(.01, 1.03,
                   "$\mathbf{" + w + "}$",
                   size=5,
                   transform=ax_px.transAxes)
        ax_px.set_global()
        
        
    fpath = 'figures/hansell_glodap/global/helper/hansell_glodap_water_mass_pixel_grid.pdf'
    fig_px.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
    fpath = fpath.replace("pdf", "png")
    fig_px.savefig(fpath, format='png', bbox_inches='tight', transparent=True,
                   dpi=600)
    
    
    
    #%%% ASSIGN PIXELS
    
    # Create column to assign pixel polygon to each regression, 
    
    tbl['PIXEL'] = 'NO_PIXEL'
    rand_offset = np.random.default_rng(1934) # set seed for reproducibility
    for ik, k in enumerate(pixel_polys):
    
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        # Subset data for water mass w (to avoid going over all data points for
        # each pixel)
        tbl_ss = tbl.loc[tbl.WATER_MASS==w, ].copy()
        
        # Create coordinate points 
        # 
        # Some cruise happen to follow section at specific lat/lons.
        # If lat/lon are multiples of 10 (i.e., pixel border), randomly add an 
        # infinitesimal value when assigning to a pixel (otherwise they will be
        # assigned always just by the order of the last pixel evaluated among those
        # sharing the border).
        pts = list()
        for i, r in tbl_ss.iterrows():
            
            # Account for floating-point issues
            offset = rand_offset.choice([-1, 1], size=1)[0] * .001
            if_lon_in_border = np.isclose(r['LONGITUDE'] % 10, 0, atol=1e-5) or np.isclose(r['LONGITUDE'] % 10, 10, atol=1e-5)
            rlon = r['LONGITUDE'] + offset if if_lon_in_border else r['LONGITUDE']
            if_lat_in_border = np.isclose(r['LATITUDE'] % 10, 0, atol=1e-5) or np.isclose(r['LATITUDE'] % 10, 10, atol=1e-5)
            rlat = r['LATITUDE'] + offset if if_lat_in_border else r['LATITUDE']
            
            pts.append(geometry.Point([rlon, rlat]))
        
        # Iterate through pixels for water mass
        for ip, p in enumerate(pixel_polys[k]):
            
            # Get pixel polygon and find samples within it
            pp = pixel_polys[k][p]
            boo = ((pp.contains(pts)) | (pp.touches(pts)))
            
            # Assign code to those samples
            idx = tbl_ss.index[boo]
            tbl.loc[idx, 'PIXEL'] =  k + ";" + p
            
    
    #%%% CHECK ASSIGNMENT
    
    # Create water mass labels
    wm_labels = [w for o in wm_polys_plot for d in wm_polys_plot[o] for w in wm_polys_plot[o][d]]
    wm_labels = {k:v for k, v in zip(wm_labels, wm_labels)}
    # Most are already valid labels, except for the AAIWs
    # Do it bold manually as setting fontweight='bold' leaves subscripts of AAIWs
    # not being bold...
    wm_labels_b = {k:("$\mathbf{" + v + "}$") for k, v in zip(wm_labels, wm_labels)}
    
    
    # Map pixels with data points 
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    for ia, a in enumerate(ages):
        
        fig_px = plt.figure(figsize=(5*cm * nc, 3*cm * nr))
    
        for ik, k in enumerate(pixel_polys):
            
            # Split water mass key
            o = k.split(";")[0] # ocean
            d = k.split(";")[1] # depth layer
            w = k.split(";")[2] # water mass code
        
            
            # Initialise subplot
            # mproj = ccrs.Mollweide(central_longitude=-160)
            wmb = wm_polys_plot[o][d][w].bounds
            wmc = wm_polys_plot[o][d][w].centroid
            mproj = ccrs.Mercator(central_longitude=wmc.x, 
                                  min_latitude=wmb[1] - 2,
                                  max_latitude=wmb[3] + 2)
            ax_px = fig_px.add_subplot(nr, nc, ik + 1, projection=mproj)
            
            # Plot land, water mass polygon and sample points
            ax_px.add_feature(land,
                              facecolor='#ccc',
                              edgecolor='#444',
                              linewidth=.2,
                              zorder=0)
            
            # Normalise pixel indices between [0,1] to get distinct colours from
            # px_pal
            col_norm = mpl.colors.Normalize(vmin=1, vmax=len(pixel_polys[k]))
            
            # Map water mass pixels with distinct colour just to aid visualisation
            for ip, p in enumerate(pixel_polys[k]):
                
                # Subset samples for each pixel
                ss = tbl.loc[tbl['PIXEL']==(k + ";" + p), :].copy()
                
                # Subset samples with values for age a:
                ss = ss.loc[~np.isnan(ss[a]), :]
                
                # Just plot one point per station to avoid overplotting
                ss = ss.drop_duplicates(subset=['CRUISE', 'STATION'])
                
                # Map points
                ax_px.scatter(ss.LONGITUDE, ss.LATITUDE,
                              s=.5,
                              linewidth=.05,
                              facecolor=px_pal(col_norm(ip + 1)),
                              transform=ccrs.PlateCarree(),
                              zorder=2)
                # Map pixel
                ax_px.add_geometries(pixel_polys[k][p],
                                     facecolor='none',
                                     edgecolor=px_pal(col_norm(ip + 1)),
                                     linewidth=.2,
                                     crs=ccrs.PlateCarree(),
                                     zorder=1)
                
            # Water mass tag
            ax_px.text(-.05, 1, wm_labels_b[w],
                       ha='right', va='top',
                       fontsize=6,
                       transform=ax_px.transAxes)
            
            # Adjust extent
            ax_px.set_extent((wmb[0] - 2, wmb[2] + 2, wmb[1] - 2, wmb[3] + 2))
        
        fig_px.subplots_adjust(wspace=.4)
        
        fpath = ('figures/hansell_glodap/global/helper/' +
                 'hansell_glodap_water_mass_pixel_grid_data_' + a +
                 '.pdf')
        fig_px.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
        fpath = fpath.replace("pdf", "png")
        fig_px.savefig(fpath, format='png', bbox_inches='tight', transparent=True,
                       dpi=600)
    
    
    #%% MEASUREMENT ERRORS
    
    # This is relevant because we will perform Monte Carlo simulations to propagate
    # uncertainty through the (multiple) linear regressions.
    
    # As outlined in 01_merge_glodap_hansell, GLODAP paper reports uncertainty of
    # 0.005 for salinity but none for temperature. Following Alvarez et al. 2014 it
    # was set to 0.04.
    # Ages and AOU already have their error assigned in previous scripts of
    # the workflow.
    

    #%% WATER MASS MIXING CORRECTION (ESTIMATE RESIDUALS)
    
    # Perform multiple linear regressions with temperature and salinity as 
    # independent variables to account for the effect of water mass mixing on other
    # variables. Following De la Fuente et al. 2014 include squared terms of PT and
    # S. This makes possible to account for the mixing of more than two end 
    # members.
    # 
    # From regressions, extract residuals to use as "mixing-corrected" values.
    # 
    # Given than the error/uncertainty of SALINITY and PT are much smaller 
    # than those of biogeochemical variables it's fine to do model I regressions.
    #
    # Corrections, i.e., regressions, are done separately for each water mass and
    # cruise.
    #
    # Error propagation will be done using Monte Carlo simulations. Instead of
    # directly computing a regression for each case directly with the variable
    # values in the sample, we will take random values based on their uncertainties.
    

    # Compute squared PT and S columns (will be used for analytical solution)
    tbl['PT2'] = tbl['PT'] ** 2
    tbl['S2'] = tbl['CTD_SALINITY'] ** 2
    
    
    # Set variables for which residuals need to be estimated
    vrs = ['OXYGEN', 'AOU', 'AGE_CFC11', 'AGE_CFC12', 'AGE_SF6']
    vrs_flags = {'OXYGEN': 'OXYGEN_FLAG_W',
                 'AOU': 'OXYGEN_FLAG_W',
                 'AGE_CFC11': 'CFC_11_FLAG_W',
                 'AGE_CFC12': 'CFC_12_FLAG_W',
                 'AGE_SF6': 'SF6_FLAG_W'}


#------------------------------------------------------------------------------

# Get out of the if block so that all childs have access to the definition of
# run_single_MC_sim()

#------------------------------------------------------------------------------


# Function to run Monte Carlo simulations in parallel with multiprocessing

def run_single_MC_sim(task_args):
    
    # Unpack tuple of arguments
    sim_id, child_seed, tbl, vrs, vrs_flags, wm_polys_flat = task_args
    
    # Initialize the RNG with the specific child seed spawned by the parent.
    # independent & reproducible
    rng = np.random.default_rng(child_seed)
    
    # Create dataframe to store residuals for each simulation
    tbl_mc = pd.DataFrame(np.nan, index=tbl.index, columns=[v + '_RES' for v in vrs])
    
    # Set minimum number of observations to perform correction (regressions)
    min_obs = 5
    
    # Iterate across variables and water masses
    for v in vrs:
        for k in wm_polys_flat:
            
            # Split water mass key
            w = k.split(";")[2] # water mass code
                
            # Subset samples of water mass w, with valid flags
            boo = ((tbl['WATER_MASS']==w) &
                   (tbl[vrs_flags[v]].isin([0, 2, 6])) &
                   (~np.isnan(tbl[v])))
            ss = tbl.loc[boo, :].copy()
            if ss.empty: continue
        
            # Perform corrections separately for each cruise
            for c in ss.EXPOCODE.unique():
                
                # Subset samples for cruise c
                ssc = ss.loc[ss.EXPOCODE==c, :].copy()
        
                # If there is enough data for v in w in c, proceed
                # (at least 'min_obs' non-nan values)
                if len(ssc) >= min_obs:
                    
                    # Use rng to create random values within the uncertainty
                    # ranges of variables (assuming gaussian distributions)
                    # Assuming given uncertainties correspond to 95% confidence
                    # interval, we need to compute the standard deviation to
                    # provide it to the random number generator.
                    z_score = norm.ppf(.975)
                    ssc['PT'] += rng.normal(0, ssc['PT_U'] / z_score)
                    ssc['CTD_SALINITY'] += rng.normal(0, ssc['CTD_SALINITY_U'] / z_score)
                    ssc['PT2'] = ssc['PT']**2 # Compute squared terms after introducing randomness
                    ssc['S2'] = ssc['CTD_SALINITY']**2 
                    ssc[v] += rng.normal(0, ssc[v + '_U'] / z_score)
                    
                    # Perform regression
                    frml = v + ' ~ CTD_SALINITY + PT + S2 + PT2'
                    md1 = smf.ols(frml, data=ssc).fit()
                    
                    # Introduce residual values in table using the index
                    tbl_mc.loc[md1.resid.index, v + '_RES'] = md1.resid
                    
    return sim_id, tbl_mc


#------------------------------------------------------------------------------

# Return to an if block to avoid execution by child processes.

#------------------------------------------------------------------------------


#### Parallel execution of Monte Carlo simulations
if __name__ == '__main__':
    
    # Set number of Monte Carlo simulations
    n_sim = 100 # !!!
    
    # Set number of cores to use. Leave a core free.
    cpus = mp.cpu_count() // 2
    
    print(f"Running Monte Carlo simulations for water mass mixing corrections: {n_sim} sims on {cpus} cores...")
    
    # Create a root seed sequence
    seedseq = np.random.SeedSequence(12345)
    
    # Spawn n_sim independent child seed sequences
    child_seeds = seedseq.spawn(n_sim)
    
    # Zip child seed sequences into tasks -> each sim gets its own unique seed
    # [each task needs all variables to complete run_single_MC_sim()]
    tasks = [
        (i, child_seeds[i], tbl, vrs, vrs_flags, wm_polys_flat) 
        for i in range(n_sim)
    ]
    
    # Run simulations in parallel
    MC_SIM_WMMC = {}
    with mp.Pool(processes=cpus) as pool:
        for sim_id, result_df in tqdm(pool.imap_unordered(run_single_MC_sim, tasks), total=n_sim):
            MC_SIM_WMMC[sim_id] = result_df
    
    
    print("Monte Carlo WMMC done!")


    #### Do the water mass mixing correction but without uncertainty
    # for checking purposes
    
    # Iterate across variables and water masses
    for v in vrs:
        
        # Create new column for residual variable
        vres = v + '_RES'
        tbl[vres] = np.nan
    
        for k in wm_polys_flat:
            
            # Split water mass key
            w = k.split(";")[2] # water mass code
                
            # Subset samples of water mass w, with valid flags
            boo = ((tbl['WATER_MASS']==w) &
                   (tbl[vrs_flags[v]].isin([0, 2, 6])) &
                   (~np.isnan(tbl[v])))
            ss = tbl.loc[boo, :].copy()
            if ss.empty: continue
        
            # Perform corrections separately for each cruise
            for c in ss.EXPOCODE.unique():
                
                # Subset samples for cruise c
                ssc = ss.loc[ss.EXPOCODE==c, :].copy()
        
                # If there is enough data for v in w in c, proceed
                # (at least 'min_obs' non-nan values)
                min_obs = 5
                if len(ssc) >= min_obs:
                    
                    # Perform regression
                    frml = v + ' ~ CTD_SALINITY + PT + S2 + PT2'
                    md1 = smf.ols(frml, data=ssc).fit()
                    
                    # Introduce residual values in table using the index
                    tbl.loc[md1.resid.index, v + '_RES'] = md1.resid
    
    gc.collect()
    
    
    #%% REGRESSIONS: OXYGEN UTILISATION RATES
    
    # Perform linear regressions with residuals to estimate OUR.
    # Do so individually for each water mass pixel.
    
    print("Computing OURs with MC simulations...")
    
    
    # Set the x variables (including tracer ages) to iterate
    xvrs = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
    
    # And the dependent variables
    vres = ['OXYGEN_RES', 'AOU_RES']
    
    
    # Set labels
    xvrs_labs = ["Age$_\mathregular{CFC\u201011}$$_\Delta$ [y]",
                 "Age$_\mathregular{CFC\u201012}$$_\Delta$ [y]",
                 "Age$_\mathregular{SF_6}$$_\Delta$ [y]"]
    xvrs_labs = dict(zip(xvrs, xvrs_labs))
    vres_labs = ["O$_{2\Delta}$ [$\mu$mol kg$^{-1}$]",
                 "AOU$_\Delta$ [$\mu$mol kg$^{-1}$]"]
    vres_labs = dict(zip(vres, vres_labs))
    
    def p_label(p):
        if p >= .05:
            pl = "$\it{p}$ = " + '{:.2f}'.format(p)
        elif p >= .01:
            pl = "$\it{p}$ < 0.05"
        elif p >= .001:
            pl = "$\it{p}$ < 0.01"
        else:
            pl = "$\it{p}$ < 0.001"
        return pl
    
    # Optional -----------
    # Function to compute the weight of X in the regressions, from Álvarez-Salgado et al. (2014)
    # doi:10.1016/j.pocean.2013.12.009
    def Wx_value(X, Y, erX, erY):
        return (erX**2 / np.var(X)) / ((erX**2 / np.var(X)) + (erY**2 / np.var(Y)))
    # --------------------
    
    # set minimum number of observations required to do regression
    min_obs = 5
    
    # Set variables to store results etc.        
    reg = []
    skipped_instances = []
    start_time = dt.datetime.now()
    
    MC_SIM_OUR = {}
    for v in vres:
        
        # Create sub-dicts to store results
        MC_SIM_OUR[v] = {}
        
        for x in xvrs:
            
            # Sub-dict
            MC_SIM_OUR[v][x] = {}
            
            # Iterate through each water mass
            for k1 in pixel_polys:
                
                # Split water mass key
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                
                txt1 = v.replace("_RES", "")
                txt2 = x.replace("_RES", "")
                print("OURs | " + txt1 + " | " + txt2 + " | " + w)
                
                # Iterate through pixels within each water mass
                for ik2, k2 in enumerate(pixel_polys[k1]):
                    
                    # Create pixel key
                    px_k = k1 + ";" + k2
                    
                    # Sub-dict
                    MC_SIM_OUR[v][x][px_k] = {}
                    
                    # Identify samples for v-x-px_k with valid values for both 
                    # target variables of the regression.
                    # Identify nans just with the first MC simulation, as this
                    # is the same for all sims.
                    idx = ((tbl['PIXEL']==px_k) &
                           (~np.isnan(MC_SIM_WMMC[0][v])) &
                           (~np.isnan(MC_SIM_WMMC[0][x])))
                    nobs = sum(idx)
                    
    
                    #### Do linear regression
    
                    # Skip variable pair if no data available OR less than min_obs
                    # keep track of skipped
                    if nobs < min_obs:
                        msg = (v + ", " + x + ", " + px_k +
                               ": not enough observations (" + str(nobs) +
                               ", at least " + str(min_obs) + " required)")
                        skipped_instances.append(msg)
    
                    else: # otherwise do regression
                        
                        # Follow Monte Carlo simulations done before to propagate 
                        # uncertainty. For each a slope will be computed, and then
                        # estimate mean and sd.
                        for i in range(n_sim):
                            
                            # Subset the samples (with idx), values corresponding 
                            # to the residuals computed in MC simulation i
                            ssp = MC_SIM_WMMC[i].loc[idx, :]
                            
                            # Select independent and dependent variables
                            X = ssp[x]
                            Y = ssp[v]
                            # plt.scatter(X, Y)#, c='#333', alpha=.01)
                            
                            
                            # Set Wx = .5 to compute model II regression
                            do_weighted_Wx = False
                            if do_weighted_Wx:
                                # Optional
                                # Get mean uncertainty associated with these samples
                                # (from the main table, as ssp has only the residuals)
                                u_mean_X = np.mean(tbl.loc[ssp.index, x.replace("_RES", "_U")])
                                u_mean_Y = np.mean(tbl.loc[ssp.index, v.replace("_RES", "_U")])
                                Wx_val = Wx_value(X, Y, u_mean_X, u_mean_Y)
                            else:
                                Wx_val = .5
    
                            # Perform linear regression 
                            MC_SIM_OUR[v][x][px_k][i] = rc.RegressConsensusW(X, Y, Wx=Wx_val)
                            MC_SIM_OUR[v][x][px_k][i]['Wx'] = Wx_val
                         
                        # Compute mean, sd, se and ci of slopes
                        # Also p-value of mean being different from 0
                        slope_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['slope'] for i in range(n_sim)])
                        # mean of the SE of individual slopes
                        slope_se_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['sse'] for i in range(n_sim)])
                        # propagate SE of individual slopes when estimating slope_mean, instead of directly computing a simple SE
                        slope_se_prop = (1 / n_sim) * np.sqrt(sum([MC_SIM_OUR[v][x][px_k][i]['sse']**2 for i in range(n_sim)]))
                        # SD and SE of all slope values
                        # slope_sd = np.std([MC_SIM_OUR[v][x][px_k][i]['slope'] for i in range(n_sim)])
                        # slope_se = slope_sd / np.sqrt(n_sim)
                        slope_ci99 = t.ppf(.995, n_sim - 1) * slope_se_mean
                        t_value = (0 - slope_mean) / slope_se_mean
                        spvalue = 2 * (1 - t.cdf(abs(t_value), n_sim - 1)) # two sided
                        intercept_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['intercept'] for i in range(n_sim)])
                        intercept_se_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['ise'] for i in range(n_sim)])
                        intercept_se_prop = (1 / n_sim) * np.sqrt(sum([MC_SIM_OUR[v][x][px_k][i]['ise']**2 for i in range(n_sim)]))
                        # intercept_sd = np.std([MC_SIM_OUR[v][x][px_k][i]['intercept'] for i in range(n_sim)])
                        # intercept_se = intercept_sd / np.sqrt(n_sim)
                        intercept_ci99 = t.ppf(.995, n_sim - 1) * intercept_se_mean
                        t_value = (0 - intercept_mean) / intercept_se_mean
                        ipvalue = 2 * (1 - t.cdf(abs(t_value), n_sim - 1)) # two sided
                        r2_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['r2'] for i in range(n_sim)])
                        Wx_mean = np.mean([MC_SIM_OUR[v][x][px_k][i]['Wx'] for i in range(n_sim)])
    
                        
                        # Extract and store results, include mean values
                        # NOTE THAT np.nanmean() behaves correctly when all 
                        # elements are nan (returns a nan), but still displays a 
                        # warning. It has no use here, so supress it to avoid 
                        # confussion.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            spp_envars = tbl.loc[idx]
                            tmp = [v,    # dependent variable
                                   x,    # independent variable (age or other)
                                   k1,   # water mass
                                   k2,   # pixel
                                   px_k, # pixel key
                                   slope_mean,        # slope (OUR), mean across MC simulations
                                   slope_se_mean,     # mean of SE of slopes across MC simulations
                                   slope_se_prop,     # propagated SE of individual slopes
                                   # slope_sd,          # standard deviation of slopes across MC simulations
                                   # slope_se,          # standard error of slopes based on slope_sd
                                   slope_ci99,        # 99% confidence interval of slopes across MC simulations
                                   spvalue,           # slope p-value (H0: = 0), estimated from slope_mean and slope_se
                                   intercept_mean,    # same for intercept
                                   intercept_se_mean, #
                                   intercept_se_prop, #
                                   # intercept_sd,      # 
                                   # intercept_se,      # 
                                   intercept_ci99,    # 
                                   ipvalue,           # 
                                   r2_mean,           # coefficient of determination, R^2, mean across MC simulations
                                   Wx_mean,           # Wx value, mean across MC simulations
                                   nobs,              # number of samples in the regression
                                   np.nanmean(spp_envars['LONGITUDE']),     # mean actual lon of samples
                                   np.nanstd(spp_envars['LONGITUDE']),      # sd of lon of samples
                                   np.nanmean(spp_envars['LATITUDE']),      # mean actual lat of samples
                                   np.nanstd(spp_envars['LATITUDE']),       # sd of lat of samples
                                   np.nanmean(spp_envars['CTD_PRESSURE']),  # mean depth of samples
                                   np.nanstd(spp_envars['CTD_PRESSURE']),   # sd of depth of samples
                                   np.nanmean(spp_envars['PT']),            # etc.
                                   np.nanstd(spp_envars['PT']),
                                   np.nanmean(spp_envars['CTD_SALINITY']),
                                   np.nanstd(spp_envars['CTD_SALINITY']),
                                   np.nanmean(spp_envars['SIGMA0']),
                                   np.nanstd(spp_envars['SIGMA0']),
                                   np.nanmean(spp_envars['OXYGEN']),
                                   np.nanstd(spp_envars['OXYGEN']),
                                   np.nanmean(spp_envars['AOU']),
                                   np.nanstd(spp_envars['AOU']),
                                   np.nanmean(spp_envars['DOC']),
                                   np.nanstd(spp_envars['DOC']),
                                   np.nanmean(spp_envars['NITRATE']),
                                   np.nanstd(spp_envars['NITRATE']),
                                   np.nanmean(spp_envars['PHOSPHATE']),
                                   np.nanstd(spp_envars['PHOSPHATE']),
                                   np.nanmean(spp_envars['SILICIC_ACID']),
                                   np.nanstd(spp_envars['SILICIC_ACID']),
                                   np.nanmean(spp_envars['NPP_EPPL']),
                                   np.nanstd(spp_envars['NPP_EPPL']),
                                   np.nanmean(spp_envars['NPP_CBPM']),
                                   np.nanstd(spp_envars['NPP_CBPM']),
                                   np.nanmean(spp_envars['B']),
                                   np.nanstd(spp_envars['B']),
                                   np.nanmean(spp_envars['POC3D']),
                                   np.nanstd(spp_envars['POC3D']),
                                   np.nanmean(spp_envars['AGE_CFC11']),
                                   np.nanstd(spp_envars['AGE_CFC11']),
                                   np.nanmean(spp_envars['AGE_CFC12']),
                                   np.nanstd(spp_envars['AGE_CFC12']),
                                   np.nanmean(spp_envars['AGE_SF6']),
                                   np.nanstd(spp_envars['AGE_SF6'])]
                        
                        # Append to previous
                        reg.append(tmp)
                        
    # Convert reg into dataframe
    
    nms = ['y_var', 'x_tracer', 'water_mass', 'pixel', 'pixel_key',
           'slope', 'slope_se_mean', 'slope_se_propagated', 
           # 'slope_sd', 'slope_se',
           'slope_ci99', 'spvalue',
           'intercept', 'intercept_se_mean', 'intercept_se_propagated', 
           # 'intercept_sd', 'intercept_se',
           'intercept_ci99', 'ipvalue',
           'r2', 'Wx', 'n',
           'LONGITUDE', 'LONGITUDE_SD', 'LATITUDE', 'LATITUDE_SD',
           'CTD_PRESSURE', 'CTD_PRESSURE_SD',
           'PT', 'PT_SD', 'SALINITY', 'SALINITY_SD', 'SIGMA0', 'SIGMA0_SD',
           'OXYGEN', 'OXYGEN_SD',
           'AOU', 'AOU_SD',
           'DOC', 'DOC_SD',
           'NITRATE', 'NITRATE_SD',
           'PHOSPHATE', 'PHOSPHATE_SD',
           'SILICIC_ACID', 'SILICIC_ACID_SD',
           'NPP_EPPL', 'NPP_EPPL_SD', 'NPP_CBPM', 'NPP_CBPM_SD',
           'B', 'B_SD',
           'POC3D', 'POC3D_SD',
           'AGE_CFC11', 'AGE_CFC11_SD',
           'AGE_CFC12', 'AGE_CFC12_SD',
           'AGE_SF6', 'AGE_SF6_SD']
    reg = pd.DataFrame(reg, columns=nms)
    
    end_time = dt.datetime.now()
    print("Duration: {}".format(end_time - start_time))
    
    
    # Do the same but the analytical solution with the original sample data
    # to compare
    
    print("Computing OURs with sample data...")
    start_time = dt.datetime.now()
    
    AN_OUR = {}
    for v in vres:
        
        # Create sub-dicts to store results
        AN_OUR[v] = {}
        
        for x in xvrs:
            
            # Sub-dict
            AN_OUR[v][x] = {}
            
            # Iterate through each water mass
            for k1 in pixel_polys:
                
                # Split water mass key
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                
                txt1 = v.replace("_RES", "")
                txt2 = x.replace("_RES", "")
                print("OURs | " + txt1 + " | " + txt2 + " | " + w)
                
                # Iterate through pixels within each water mass
                for ik2, k2 in enumerate(pixel_polys[k1]):
                    
                    # Create pixel key
                    px_k = k1 + ";" + k2
                    
                    # Identify samples for v-x-px_k with valid values for both 
                    # target variables of the regression.
                    # Identify nans just with the first MC simulation, as this
                    # is the same for all sims.
                    idx = ((tbl['PIXEL']==px_k) &
                           (~np.isnan(tbl[v])) &
                           (~np.isnan(tbl[x])))
                    ss_res_o = tbl.loc[idx, :]
                    nobs = sum(idx)
                    
                    ## Do linear regression
    
                    # Skip variable pair if no data available OR less than min_obs
                    # keep track of skipped
                    if nobs < min_obs: continue
                    X = ss_res_o[x]
                    Y = ss_res_o[v]
                    if do_weighted_Wx:
                        u_mean_X = np.mean(ss_res_o[x.replace("_RES", "_U")])
                        u_mean_Y = np.mean(ss_res_o[v.replace("_RES", "_U")])
                        Wx_val = Wx_value(X, Y, u_mean_X, u_mean_Y)
                    else:
                        Wx_val = .5
                    AN_OUR[v][x][px_k] = rc.RegressConsensusW(X, Y, Wx=Wx_val)
                    AN_OUR[v][x][px_k]['Wx'] = Wx_val
                    
    end_time = dt.datetime.now()
    print("Duration: {}".format(end_time - start_time))
    gc.collect()
    sys.exit()
                    
    #%%% Plot regressions
    
    # To condense all results of MC simulations, overplot all regressions for each pixel.
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    for v in vres:
        for x in xvrs:
            
            # Iterate through each water mass
            for k1 in pixel_polys:
                
                # Split water mass key
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                                    
                txt1 = v.replace("_RES", "")
                txt2 = x.replace("_RES", "")
                print("Plot OURs | " + txt1 + " | " + txt2 + " | " + w)
                
                # Initialise plot for water mass w
                n_px = len(pixel_polys[k1])
                nc = nr = int(np.ceil(n_px ** .5))
                fig_reg = plt.figure(figsize=(5*cm*nc, 3.5*cm*nr), dpi=150)
                
                # Iterate through pixels within each water mass
                for ik2, k2 in enumerate(pixel_polys[k1]):
                    
                    # Create pixel key
                    px_k = k1 + ";" + k2

                    # Initialise subplot for v and x regression in px_k
                    ax_reg = fig_reg.add_subplot(nr, nc, ik2 + 1)
                    
                    # Get index to subset observations underlying regressions
                    # (use first MC sim data to identify non-nan, as it will
                    #  apply equally to all simulation of the same pixel)
                    idx = ((tbl['PIXEL']==px_k) &
                           (~np.isnan(MC_SIM_WMMC[0][v])) &
                           (~np.isnan(MC_SIM_WMMC[0][x])))
                    
                    # Subset "original" observations, with residuals not
                    # having uncertainty
                    ss_res_o = tbl.loc[idx, :]
                    
                    # Include clarification if not enough data
                    nobs = sum(idx)
                    if nobs < min_obs:
                        ax_reg.text(.5, .55,
                                    "Not enough data (" + str(nobs) + ")",
                                    size=6, fontweight='bold',
                                    c='#999',
                                    ha='center', va='center',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        
                    # Plot data points of observations
                    ax_reg.scatter(ss_res_o[x], ss_res_o[v],
                                   marker='o',
                                   s=7,
                                   linewidth=.5,
                                   facecolor='none',
                                   edgecolor='goldenrod',
                                   alpha=.9,
                                   zorder=2)
                    
                    # Observation slopes
                    if nobs >= min_obs:
                        obs_reg = AN_OUR[v][x][px_k]
                        x0 = np.min(ss_res_o[x])
                        x1 = np.max(ss_res_o[x])
                        y0 = obs_reg['slope'] * x0 + obs_reg['intercept']
                        y1 = obs_reg['slope'] * x1 + obs_reg['intercept']
                        ax_reg.plot([x0, x1], [y0, y1],
                                    c='firebrick', lw=1.3,
                                    alpha=.9,
                                    zorder=2)
                        
                        txt_slope_o = f"{obs_reg['slope']:.2f} ± {obs_reg['sci95']:.2f}"
                        ax_reg.text(.97, .90,
                                    txt_slope_o,
                                    size=3, fontweight='bold',
                                    c='firebrick',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        # Also add mean slope of MC sims
                        # (do it here to add it just once, not for every MC)
                        mc_reg = reg.loc[((reg['pixel_key']==px_k) &
                                          (reg['x_tracer']==x) &
                                          (reg['y_var']==v)), :]
                        mc_slope = mc_reg['slope'].item()
                        mc_ci = mc_reg['slope_ci99'].item()
                        txt_slope_mc = f"{mc_slope:.2f} ± {mc_ci:.2f}"
                        ax_reg.text(.97, .85,
                                    txt_slope_mc,
                                    size=3, fontweight='bold',
                                    c='steelblue',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        
                        # Also Wx
                        txt_wx_o = f"W$_x$ = {obs_reg['Wx']:.3f}"
                        ax_reg.text(.97, .78,
                                    txt_wx_o,
                                    size=3, fontweight='bold',
                                    c='firebrick',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        txt_wx_mc = f"W$_x$ = {mc_reg['Wx'].item():.3f}"
                        ax_reg.text(.97, .73,
                                    txt_wx_mc,
                                    size=3, fontweight='bold',
                                    c='steelblue',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        
                        # And R2
                        txt_R2_o = f"R$^2$ = {obs_reg['r2']:.3f}"
                        ax_reg.text(.97, .67,
                                    txt_R2_o,
                                    size=3, fontweight='bold',
                                    c='firebrick',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        txt_R2_mc = f"R$^2$ = {mc_reg['r2'].item():.3f}"
                        ax_reg.text(.97, .62,
                                    txt_R2_mc,
                                    size=3, fontweight='bold',
                                    c='steelblue',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        # And p-value
                        txt_pval_o = p_label(obs_reg['spvalue'])
                        ax_reg.text(.97, .56,
                                    txt_pval_o,
                                    size=3, fontweight='bold',
                                    c='firebrick',
                                    ha='right', va='bottom',
                                    transform=ax_reg.transAxes,
                                    zorder=1)
                        
                        
                    ## Add data points and regression slopes of MC sims
                    for i in range(n_sim):
                        
                        # Get observations and regression results for MC
                        # simulation i.
                        ss_res_mc = MC_SIM_WMMC[i].loc[idx, :]
                        
                        # Plot data points
                        # (as raster, otherwise the pdf reader will struggle
                        #  with so many points when viewing the plots)
                        ax_reg.scatter(ss_res_mc[x], ss_res_mc[v],
                                       marker='o',
                                       s=7,
                                       linewidth=.2,
                                       facecolor='#333',
                                       edgecolor='#333',
                                       alpha=.05,
                                       rasterized=True,
                                       zorder=-1)
                        
                        ## Add slopes 
                        if nobs < min_obs: continue
                        ss_reg = MC_SIM_OUR[v][x][px_k][i]
                        x0 = np.min(ss_res_mc[x])
                        x1 = np.max(ss_res_mc[x])
                        y0 = ss_reg['slope'] * x0 + ss_reg['intercept']
                        y1 = ss_reg['slope'] * x1 + ss_reg['intercept']
                        ax_reg.plot([x0, x1], [y0, y1],
                                    c='steelblue', lw=1.0,
                                    alpha=.1,
                                    zorder=1)
                            
                        
                    ## Customise axes
                    
                    # Make space for inset map
                    if nobs==0:
                        xmin = -1
                        xmax = +1
                    else:
                        # Set range based on results from all sims
                        X = np.concat([MC_SIM_WMMC[i].loc[idx, x] for i in range(n_sim)])
                        xmin = np.nanmin(X)
                        xmax = np.nanmax(X)
                        delta_x = xmax - xmin
                        xmin = np.nanmin(X) - delta_x * .05
                        xmax = np.nanmax(X) + delta_x * .48
                    ax_reg.set(xlim=[xmin, xmax])
                    
                    ax_reg.tick_params(which='major', axis='both',
                                       labelsize=5, pad=2,
                                       direction='in', length=1.5,
                                       top=True, right=True)
                    ax_reg.tick_params(which='minor', axis='both',
                                       direction='in', length=1.2,
                                       top=True, right=True)
                    if (ik2 + 1) in [*range(1, len(pixel_polys[k1]), nc)]:
                        ax_reg.set_ylabel(vres_labs[v], size=5)
                    
                    # x axis label in last row of each column
                    last_rows = [*range(n_px - nc + 1, n_px + 1)]
                    if (ik2 + 1) in last_rows:
                        ax_reg.set_xlabel(xvrs_labs[x], size=5)
                        
                        
                    ## Add inset map
                    clon = pixel_polys[k1][k2].centroid.x
                    clat = pixel_polys[k1][k2].centroid.y
                    mproj = ccrs.NearsidePerspective(central_longitude=clon,
                                                     central_latitude=clat)
                    axins = inset_axes(ax_reg, width="40%", height="40%",
                                       loc="lower right",
                                       bbox_to_anchor=(.1, -.05, 1, 1),
                                       bbox_transform=ax_reg.transAxes,
                                       axes_class=cgeoaxes.GeoAxes,
                                       axes_kwargs={'projection': mproj})
                    # Add pixel in map
                    axins.add_geometries(pixel_polys[k1][k2],
                                         facecolor='firebrick',
                                         edgecolor='k',
                                         linewidth=.1,
                                         crs=ccrs.PlateCarree(),
                                         zorder=1)
                    # Overlay water mass border as reference
                    axins.add_geometries(wm_polys_flat[k1],
                                         facecolor='none',
                                         edgecolor='k',
                                         linewidth=.3,
                                         crs=ccrs.PlateCarree(),
                                         zorder=2)
                    # Add land
                    axins.add_feature(land,
                                      facecolor='#ccc',
                                      edgecolor='#444',
                                      linewidth=.2,
                                      zorder=0)
            
                    axins.spines['geo'].set_linewidth(.3) # thiner border
                    
                    
                # Add title
                ptitle = (w + ": " + vres_labs[v].split(" ")[0] + 
                          " vs. " + xvrs_labs[x].split(" ")[0])
                fig_reg.suptitle(ptitle, x=.5, y = .98, weight='bold', color='#222')
                
                # Try to adjust manually because bbox_inches='tight' in savefig
                # breaks the mixed-mode rendering (raster + vector)
                fig_reg.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)
                
                # Save figure
                fpath = ('figures/hansell_glodap/global/regressions/' +
                         'hansell_glodap_regressions_' +
                         v + '_' + x + '_' + w + '.pdf')
                fig_reg.savefig(fpath, format='pdf', transparent=True, dpi=150)
                plt.close(fig_reg)
                fig_reg.clf() 
                plt.close('all')
                gc.collect()
                    

    # Once OURs are done and plotted, delete MC simulations dict to free space
    del MC_SIM_WMMC
    gc.collect()
    # sys.exit()
    
    #%% SET QUALITY OF REGRESSION RESULTS
    
    # for AOU rates with values < 0 (for OXYGEN those > 0)
    # do not make sense biologically...* but there are only a few outliers.
    # Get the actual number of such outliers. 
    #
    # * Could be a sporious result due to scarce data or some left over influence
    # of water mass mixing (with some input of a water mass with higher O2)
    
    ## Check range of results for OURs
    ages_res = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
    idx = ((reg.y_var=='AOU_RES') & 
           (reg.x_tracer.isin(ages_res)))
    all_aour = reg.loc[idx, :]
    
    print("Negative OURs are " + str(sum(all_aour.slope < 0)) +
          " out of " + str(len(all_aour)))
    
    
    ### Plot R2 distribution across all regressions (from MC)
    
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10*cm, 6*cm))
    ax_hist.hist(all_aour.r2, bins=np.arange(0, 1.02, .02),
                 facecolor='tab:green', edgecolor='#222')
    
    # Add reference line
    x_line = .15
    ax_hist.axvline(x_line, color='firebrick')
    ax_hist.text(x_line + .02, .8, '$R^2$ = ' + str(x_line),
                 fontsize=8,
                 ha='left',
                 transform=ax_hist.get_xaxis_transform())
    
    # Customise axes
    ax_hist.set_xlabel("$R^2$ of regression", fontsize=8)
    ax_hist.set_ylabel("No. of regressions", fontsize=8)
    ax_hist.set(xlim=[0, 1],
                xticks=np.arange(0, 1.2, .2),
                ylim=[0, 125],
                yticks=np.arange(0, 150, 25))
    ax_hist.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    ax_hist.tick_params(axis='both', which='major', labelsize=8)
    
    
    fpath = ('figures/hansell_glodap/global/regressions/_hist_regressions_R2_MC.svg')
    fig_hist.savefig(fpath, format='svg', bbox_inches='tight')
    plt.close(fig_hist)
    # plt.scatter(all_aour.r2, all_aour.n)
    
    
    # Same, but analytical results for OURs (regressions with original data)
    # -----------------------
    # Put AN_OUR into a table
    reg_AN = []
    for v in vres:
        for x in AN_OUR[v]:
            for px_k in AN_OUR[v][x]:
                md = AN_OUR[v][x][px_k]
                tmp = [v, x, 
                       ';'.join(px_k.split(';')[:3]),
                       ';'.join(px_k.split(';')[3:]),
                       px_k, 
                       md['slope'], md['spvalue'], md['sse'], md['sci95'], 
                       md['intercept'], md['ipvalue'], md['ise'], md['ici95'], 
                       md['r2'], md['r2_adj'], md['Wx'], md['n']]
                reg_AN.append(tmp)
    nms = ['y_var', 'x_tracer', 'water_mass', 'pixel', 'pixel_key', 
           'slope', 'spvalue', 'sse', 'sci95',
           'intercept', 'ipvalue', 'ise', 'ici95',
           'r2', 'r2_adj', 'Wx', 'n']
    reg_AN = pd.DataFrame(reg_AN, columns=nms)
    # -----------------------
    
    idx = ((reg_AN.y_var=='AOU_RES') & 
           (reg_AN.x_tracer.isin(ages_res)))
    all_aour_AN = reg_AN.loc[idx, :]
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10*cm, 6*cm))
    ax_hist.hist(all_aour_AN.r2, bins=np.arange(0, 1.02, .02),
                 facecolor='tab:green', edgecolor='#222')
    
    # Add reference line
    ax_hist.axvline(x_line, color='firebrick')
    ax_hist.text(x_line + .02, .8, '$R^2$ = ' + str(x_line),
                 fontsize=8,
                 ha='left',
                 transform=ax_hist.get_xaxis_transform())
    
    # Customise axes
    ax_hist.set_xlabel("$R^2$ of regression", fontsize=8)
    ax_hist.set_ylabel("No. of regressions", fontsize=8)
    ax_hist.set(xlim=[0, 1],
                xticks=np.arange(0, 1.2, .2),
                ylim=[0, 125],
                yticks=np.arange(0, 150, 25))
    ax_hist.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    ax_hist.tick_params(axis='both', which='major', labelsize=8)
    
    
    fpath = ('figures/hansell_glodap/global/regressions/_hist_regressions_R2_AN.svg')
    fig_hist.savefig(fpath, format='svg', bbox_inches='tight')
    plt.close(fig_hist)
    
    
    
    #------------------------------------------------------------------------------
    # v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v
    
    #### Set  pval and R2 limits to accept regressions...
    
    pval_limit = .001 # acceptable if lower than
    r2_limit = .15    # acceptable if greater than
    
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
    #------------------------------------------------------------------------------
    
    # Given that we deliverately introduce noise to the regressions in the MC
    # simulations, R2 are not really representative, or at least not to the 
    # degree they are with the analytical results of the regressions with the
    # original data. Use the R2 of the analytical results when filtering by R2
    reg['r2_AN'] = reg_AN.loc[reg.index, 'r2']
    
    # On top of R2 and p, also for AOU discard rates with values < 0; for OXYGEN 
    # those > 0. Get the actual number of such outliers
    
    ## Check range of results when filtering based on the said criteria
    idx = ((reg.y_var=='AOU_RES') & 
           (reg.x_tracer.isin(ages_res)) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit))
    all_aour_f = reg.loc[idx, :]
    
    neg_aour = all_aour_f.loc[all_aour_f.slope < 0, :]
    len(neg_aour)
    
    
    print("\nFiltering rates based on p-value and\nminimum R2 "
          "reduces OURs from " + str(len(all_aour)) +
          " to " + str(len(all_aour_f)))
    
    print("\nAfter filtering rates based on p-value and\nminimum R2, "
          "negative OURs are " + str(len(neg_aour)) +
          " out of " + str(len(all_aour_f)) + 
          " (" + str(round(100*len(neg_aour)/len(all_aour_f), 1)) + " %)")
    
    
    #%% PLOT RATES
    
    #### Range of results among those positive
    
    idx = ((reg.y_var=='AOU_RES') & 
           (reg.x_tracer.isin(ages_res)) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit) &
           (reg.slope >= 0))
    pos_aour = reg.loc[idx, :]
    
    print("OUR range ->  " + 
          str(round(min(pos_aour.slope), 1)) + 
          " - " +
          str(round(max(pos_aour.slope), 1)))
    
    
    print("OUR mean ± sd ->  " + 
          str(round(np.mean(pos_aour.slope), 1)) + 
          " ± " +
          str(round(np.std(pos_aour.slope), 1)))
    
    
    #------------------------------------------------------------------------------
    
    #### Ranges for central, intermediate and deep
    
    for d in wm_depths:
        v = pos_aour.slope[pos_aour.water_mass.str.contains(d)]
        vmean = str(round(np.mean(v), 1))
        vsd = str(round(np.std(v), 1))
        print(d.capitalize() + ": " + vmean + " ± " + vsd)
    
    #------------------------------------------------------------------------------
    
    
    
    #%%% AOU - O2 intercomparison
    
    # Check differences and correlation between AOU- and O2-based rates.
    
    # Subset data for those rates
    ss_aou = reg.loc[(reg['y_var']=='AOU_RES'), :].copy().reset_index(drop=True)
    ss_oxy = reg.loc[(reg['y_var']=='OXYGEN_RES'), :].copy().reset_index(drop=True)
    
    # As AOU and OXYGEN have the same regressions, no need to do complex merging,
    # just bind cols. Can check creating ID
    ss_aou['reg_id'] = ss_aou.x_tracer.astype(str) + ';' + ss_aou.pixel_key.astype(str)
    ss_oxy['reg_id'] = ss_oxy.x_tracer.astype(str) + ';' + ss_oxy.pixel_key.astype(str)
    all(ss_aou.reg_id == ss_oxy.reg_id)
    
    ss_joined = ss_aou.join(ss_oxy, lsuffix='_aou', rsuffix='_oxy')
    all(ss_joined.reg_id_aou == ss_joined.reg_id_oxy)
    
    
    
    ### Plot results
    
    #------------------------------------------------------------------------------
    
    # Age res labels
    ages_res_labs = ["Age$_\mathregular{CFC\u201011\Delta}$",
                     "Age$_\mathregular{CFC\u201012\Delta}$",
                     "Age$_\mathregular{SF_6\Delta}$"]
    ages_res_labs = dict(zip(ages_res, ages_res_labs))
    
    ages_pal = {'AGE_CFC11_RES': '#DDAA33',
                'AGE_CFC12_RES': '#BB5566',
                'AGE_SF6_RES':   '#004488'}
    
    
    # Palette for AOU/OXY
    aou_oxy_pal = {'aou': '#008080', 'oxy': '#B22222'}
    aou_oxy_labs = {'aou': 'AOU$_\\Delta$', 'oxy': '[O$_2$]$_\\Delta$'}
    
    nc = int(np.floor(len(pixel_polys)**.5))
    nr = int(np.ceil(len(pixel_polys) / nc))
    n_w = len(pixel_polys)
    last_row = [*range(n_w - (nc - 1), n_w + 1)]
    first_col = [*range(0, n_w, nc)]
    
    #------------------------------------------------------------------------------
    
        
    #### Boxplots
    
    ylab = "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
    
    fig_bx = plt.figure(figsize=(5.2*cm*nc, 5*cm*nr))
    for i, k1 in enumerate(pixel_polys):
        
        o = k1.split(";")[0] # ocean
        d = k1.split(";")[1] # depth layer
        w = k1.split(";")[2] # water mass code
        
        # Create subplot
        ax_bx = fig_bx.add_subplot(nr, nc, i + 1)
    
        # prepare to keep track of values in each water mass to set y-limits
        vals_track = list()
        
        for ia, a in enumerate(ages_res):
    
            # Subset data for tracer a in water_mass
            idx = ((ss_joined.x_tracer_aou==a) & 
                   (ss_joined.water_mass_aou==k1) &
                   (ss_joined.spvalue_aou < pval_limit) &
                   (ss_joined.r2_AN_aou > r2_limit) &
                   (ss_joined.slope_aou >= 0))
            ss_k1 = ss_joined.loc[idx, :]
            if ss_k1.empty: continue
        
            # Perform Kruskal-Wallis to check whether there are significant 
            # differences between them
            kwt = kruskal(ss_k1['slope_aou'], abs(ss_k1['slope_oxy']))
            
            for v in ['aou', 'oxy']:
                
                # Get values of either aou/oxy-based rates
                vals = abs(ss_k1['slope_' + v])
                vals_track.extend(vals)
                
                # Prepare offset to position boxplots side-by-side
                offset = -.2 if v=='aou' else +.2
                
                # Colour to differentiate AOU/O2 boxplot
                bx_col = aou_oxy_pal[v]
                
                # Do boxplots
                bplot = ax_bx.boxplot(vals,
                                      positions=[ia + offset],
                                      widths=.3,
                                      boxprops={'color': bx_col,
                                                'facecolor': '#fff',
                                                'linewidth': .5},
                                      medianprops={'color': bx_col,
                                                   'linewidth': .5},
                                      flierprops={'marker': 'o',
                                                  'markersize': 2,
                                                  'markeredgewidth': .2, 
                                                  'markerfacecolor': bx_col, 
                                                  'markeredgecolor': 'none'},
                                      capprops={'color': bx_col,
                                                'linewidth': .5},
                                      whiskerprops={'color': bx_col,
                                                    'linewidth': .5},
                                      patch_artist=True,
                                      capwidths=.1)
            
            # Add label with KW test results
            axis_to_data = ax_bx.transAxes + ax_bx.transData.inverted()
            if not np.isnan(kwt[1]):
                vals_a = pd.concat((ss_k1['slope_aou'], abs(ss_k1['slope_oxy'])))
                ypos = np.max(vals_a) + axis_to_data.transform((ia, .8))[1]
                if kwt[1] < .01:
                    pval = "$\mathit{p}$ < 0.01"
                elif kwt[1] < .05:
                    pval = "$\mathit{p}$ < 0.05"
                else:
                    pval = "$\mathit{p}$ = " + str(round(kwt[1], 2))
                    
                ax_bx.text(ia, ypos, pval,
                           fontsize=4, color='#444',
                           ha='center', va='bottom')
        
        # Customise axes etc.
        ax_bx.set(xlim=[-.7, len(ages_res) - 1 + .7],
                  xticks=range(0, len(ages_res)),
                  xticklabels=ages_res_labs.values())
        ax_bx.tick_params(axis='both', labelsize=5, length=1.8)
        
        # Extend y-limits to make room for labels
        if vals_track:
            ax_bx.set(ylim=[0, (np.max(vals_track) + np.max(vals_track)*.4)])
            ax_bx.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax_bx.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        else:
            ax_bx.text(.5, .5, "No valid results",
                       color='#777',
                       fontsize=8, fontweight='bold',
                       ha='center', va='center',
                       transform=ax_bx.transAxes)
            ax_bx.set(yticks=[])
            
        # Water mass label
        ax_bx.text(.03, .91, wm_labels_b[w],
                   color='#222',
                   fontsize=6,
                   ha='left', va='baseline',
                   transform=ax_bx.transAxes)
        
        # Y-axis labels
        if i in first_col:
            ax_bx.set_ylabel(ylab, fontsize=5, labelpad=3)
            
        # In the first subplot, add legend indicating which boxplot is for 
        # AOU-based and which for O2-based
        if i==0:
            hdl = [Line2D([0], [0],
                          color=kv,
                          label=aou_oxy_labs[k]) for k, kv in aou_oxy_pal.items()]
            legi = ax_bx.legend(handles=hdl,
                                handlelength=1.5,
                                   handletextpad=.5,
                                   labelspacing=.6,
                                   prop={'size': 4.5},
                                   frameon=False,
                                   loc='lower left',
                                   borderaxespad=1)
            legi.get_frame().set_edgecolor('none')
        
    # Adjust spacing
    fig_bx.subplots_adjust(wspace=.25)
    
    fpath = ('figures/hansell_glodap/global/rates/_our_rates_aou_oxygen_boxplot_kruskal.svg')
    fig_bx.savefig(fpath, format='svg', bbox_inches='tight')
    plt.close(fig_bx)
    
    
    #### Correlations
    
    xlab = "AOU$_\\Delta$-based OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
    ylab = "[O$_2$]$_\\Delta$-based OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
    
    fig_cor = plt.figure(figsize=(5.2*cm*nc, 5*cm*nr))
    for i, k1 in enumerate(pixel_polys):
        
        o = k1.split(";")[0] # ocean
        d = k1.split(";")[1] # depth layer
        w = k1.split(";")[2] # water mass code
        
        # Create subplot
        ax_cor = fig_cor.add_subplot(nr, nc, i + 1)
    
        # prepare to keep track of values in each water mass to set y-limits
        vals_track = list()
        
        for ia, a in enumerate(ages_res):
    
            # Subset data for tracer a in water_mass
            idx = ((ss_joined.x_tracer_aou==a) & 
                   (ss_joined.water_mass_aou==k1) &
                   (ss_joined.spvalue_aou < pval_limit) &
                   (ss_joined.r2_AN_aou > r2_limit) &
                   (ss_joined.slope_aou >= 0))
            ss_k1 = ss_joined.loc[idx, :]
            
            # Get values and keep track to set limits
            x = ss_k1.slope_aou
            y = ss_k1.slope_oxy
            vals_track.extend(x)
            vals_track.extend(abs(y))
            
            # Compute Spearman correlation coefficient
            scor = spearmanr(x, y)
            
            # Plot data points
            ax_cor.scatter(x, y,
                           facecolor=ages_pal[a],
                           edgecolor='none',
                           s=10,
                           alpha=.7)
            
            # Add label with correlation value
            if not np.isnan(scor[0]):
                rval = (ages_res_labs[a] + 
                        " → $\mathit{r}$ = " + 
                        str(round(scor[0], 2)))
                ax_cor.text(.04, (.05 + ia * .06), rval,
                           fontsize=4, color=ages_pal[a],
                           ha='left', va='bottom',
                           transform=ax_cor.transAxes)
            
        # Customise axes etc.
        ax_cor.tick_params(axis='both', labelsize=5, 
                           direction='in', length=1.8,
                           top=True, right=True)
        
        if vals_track:
            min_max_dif = np.max(vals_track) - np.min(vals_track)
            margin = .15
            round_factor = 2
            x_min = np.min(vals_track) - min_max_dif * margin
            x_min = np.floor(x_min/round_factor) * round_factor
            x_max = np.max(vals_track) + min_max_dif * margin
            x_max = np.ceil(x_max/round_factor) * round_factor
            y_min = -np.max(vals_track) - min_max_dif * margin
            y_min = np.floor(y_min/round_factor) * round_factor
            y_max = -np.min(vals_track) + min_max_dif * margin
            y_max = np.ceil(y_max/round_factor) * round_factor
            ax_cor.set(xlim=[x_min, x_max], ylim=[y_min, y_max])
            ax_cor.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax_cor.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        else:
            ax_cor.text(.5, .5, "No valid results",
                        color='#777',
                        fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        transform=ax_cor.transAxes)
            ax_cor.set(xticks=[], yticks=[])
    
        # Water mass label
        ax_cor.text(.96, .91, wm_labels_b[w],
                    color='#222',
                    fontsize=6,
                    ha='right', va='baseline',
                    transform=ax_cor.transAxes)
        
        # Axis labels
        if (i + 1) in last_row:
            ax_cor.set_xlabel(xlab, fontsize=5, labelpad=3)
        if i in first_col:
            ax_cor.set_ylabel(ylab, fontsize=5, labelpad=3)
            
    fig_cor.subplots_adjust(wspace=.25, hspace=.15)
    
    fpath = ('figures/hansell_glodap/global/rates/_our_rates_aou_oxygen_correlations.svg')
    fig_cor.savefig(fpath, format='svg', bbox_inches='tight')
    plt.close(fig_cor)
    
    
    #%%% Depth profiles
    
    # Set rate variable names (and labels)
    vres = ['OXYGEN_RES', 'AOU_RES']
    ages_res = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
    
    rate_labs = ["OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]",
                 "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"]
    rate_labs = dict(zip(vres, rate_labs))
    
    # Set depth variable
    depth_as = ['CTD_PRESSURE', 'SIGMA0']
    
    # Set axis parameters (limits and ticks) for each variable
    xaxpar = {'OXYGEN_RES': [[-40, 40],           # x limits
                             range(-40, 60, 20),  # x ticks
                             10,                  # x minor tick multiple
                             "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]" # xlab
                             ],
              'AOU_RES': [[-40, 40],
                          range(-40, 60, 20),
                          10,
                          "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
                          ]
              }
    yaxpar = {'CTD_PRESSURE': [[0, 1500],           # y limits
                               range(0, 1800, 300), # y ticks
                               150,                 # y minor tick multiple
                               "Depth [dbar]"       # y axis label
                               ],
              'SIGMA0': [[25.5, 28.0],
                         np.arange(26.0, 28.5, .5),
                         .1,
                         "$\sigma_\\theta$ [g kg$^{-1}$]"
                         ]
              }
    
    for v in vres:
        
        nr = len(ages_res)
        nc = len(depth_as)
        fig_pr, ax_pr = plt.subplots(nrows=nr, ncols=nc,
                                     figsize=(5*cm*nc, 6*cm*nr))
        
        for iz, z in enumerate(depth_as):
            
            # unpack axis params for v and d
            xl, xtck, xtck_mi, xlab = xaxpar[v]
            yl, ytck, ytck_mi, ylab = yaxpar[z]
            
            for ig, g in enumerate(ages_res):
                
                # subset data for tracer age g
                ss = reg.loc[(reg['y_var']==v) &
                             (reg['x_tracer']==g), :].copy()
                
                # plot only significant rates that have a reasonably good fit
                ss = ss.loc[(ss.spvalue<pval_limit) &
                            (ss.r2>r2_limit), :]
                
                # plot scatter and error bars
                ax_pr[ig, iz].errorbar(x=ss['slope'],
                                       y=ss[z],
                                       xerr=ss['slope_ci99'],
                                       yerr=ss[z + '_SD'],
                                       c='w',
                                       markersize=3,
                                       markeredgecolor='#222',
                                       markeredgewidth=.5,
                                       linestyle='none',
                                       capsize=1,
                                       ecolor='#222',
                                       elinewidth=.5,
                                       zorder=0)
                        
                ax_pr[ig, iz].set(xlim=xl, xticks=xtck, ylim=yl, yticks=ytck)
                ax_pr[ig, iz].xaxis.set_minor_locator(mticker.MultipleLocator(xtck_mi))
                ax_pr[ig, iz].yaxis.set_minor_locator(mticker.MultipleLocator(ytck_mi))
                ax_pr[ig, iz].tick_params(which='major', axis='both',
                                          labelsize=6, pad=2,
                                          direction='in', length=2.5,
                                          top=True, right=True)
                ax_pr[ig, iz].tick_params(which='minor', axis='both',
                                          direction='in', length=1.5,
                                          top=True, right=True)
                ax_pr[ig, iz].set_xlabel(xlab, fontsize=6.5, labelpad=2)
                ax_pr[ig, iz].set_ylabel(ylab, fontsize=6.5, labelpad=1.5)
                
                # add reference line for 0
                ax_pr[ig, iz].axvline(0, c='#999', linestyle=':', linewidth=1, zorder=1)
                
                # flip y axis
                ax_pr[ig, iz].invert_yaxis()
                
                
            # adjust spacing between subplots
            fig_pr.subplots_adjust(wspace=.5, hspace=.3)
        
            # save figure
            fpath = ('figures/hansell_glodap/global/rates/profiles/' +
                     'hansell_glodap_' + v + '_rates_profile_.pdf')
            fig_pr.savefig(fpath, format='pdf', bbox_inches='tight')
            plt.close(fig_pr)
    
    
    #%%% Tracer-age intercomparison 
    
    # Compare the rate estimates based on CFC-11, CFC-12 and SF6 ages
    
    def sign_string(num): return "-" if num < 0 else "+"
    
    #### Regressions against each other
    
    comparisons = [['AGE_CFC11_RES', 'AGE_CFC12_RES'],
                   ['AGE_CFC11_RES', 'AGE_SF6_RES'],
                   ['AGE_CFC12_RES', 'AGE_SF6_RES']]
    vres_lims = [[-30, 5], [-5, 30]]
    
    fig_com, ax_com = plt.subplots(nrows=len(vres), ncols=len(comparisons),
                                   figsize=(5*cm * len(comparisons),
                                            5*cm * len(vres)))
    for iv, v in enumerate(vres):
        for ic, c in enumerate(comparisons):
            
            # Subset data for rate with v, and tracer ages in c
            idx = ((reg['y_var']==v) &
                   (reg.spvalue < pval_limit) &
                   (reg.r2_AN > r2_limit))
            if v=='AOU_RES':
                idx = idx & (reg.slope >= 0)
            elif v=='OXYGEN_RES':
                idx = idx & (reg.slope <= 0)
            r = reg.loc[idx, :].copy()
            
            # get the rates (slopes) separately
            r_0 = r.loc[r['x_tracer']==c[0], ['pixel_key', 'slope']]
            r_1 = r.loc[r['x_tracer']==c[1], ['pixel_key', 'slope']]
            
            # Merge them back but in wide format, matching shared pixels
            r_01 = r_0.merge(r_1, on='pixel_key', how='inner',
                             suffixes=["_" + c[0], "_" + c[1]])
            X = r_01["slope_" + c[0]]
            Y = r_01["slope_" + c[1]]
            
            #### Compare average slopes (rates) between tracer ages
            #    (On shared pixels)
            txt = (v + ": on average " + c[1] + " rates = " +
                   str(round(np.mean(Y), 2)) + " and " +
                   c[0] + " rates = " +
                   str(round(np.mean(X), 2)))
            print(txt)
            txt = ("Thus, on average " + c[1] + " rates are " +
                   str(round(100*np.mean(Y)/np.mean(X), 1)) +
                   "% of " + c[0] + " rates.")
            print(txt)
            print("\n")
                        
            # Plot data and regression
            ax_com[iv, ic].scatter(X, Y,
                                   marker='o',
                                   s=5,
                                   linewidth=.1,
                                   facecolor='steelblue',
                                   edgecolor='w')
            if len(X) > 2:
                # Estimate regression
                md_com = rc.RegressConsensusW(X, Y, Wx=.5)
                
                # If regression was significant, add line
                if md_com['spvalue']<.05:
                    x0 = np.nanmin(X)
                    x1 = np.nanmax(X)
                    y0 = md_com['intercept'] + x0 * md_com['slope']
                    y1 = md_com['intercept'] + x1 * md_com['slope']
                    ax_com[iv, ic].plot([x0, x1], [y0, y1],
                                        c='#444', lw=1.0,
                                        zorder=1)
                    
                # Add regression parameters
                txt_eq = f"y = ({md_com['slope']:.2f} ± {md_com['sci95']:.2f})x {sign_string(md_com['intercept'])} ({abs(md_com['intercept']):.2f} ± {md_com['ici95']:.2f})"
                txt_spval = p_label(md_com['spvalue'])
                txt_r2 = ("$R^2$ = " + '{:.3f}'.format(md_com['r2']))
                ax_com[iv, ic].text(.05, .93,
                                    txt_eq,
                                    size=3,
                                    c='#555',
                                    ha='left', va='baseline',
                                    transform=ax_com[iv, ic].transAxes,
                                    zorder=1)
                ax_com[iv, ic].text(.05, .88,
                                    txt_spval,
                                    size=3,
                                    c='#555',
                                    ha='left', va='baseline',
                                    transform=ax_com[iv, ic].transAxes,
                                    zorder=1)
                ax_com[iv, ic].text(.05, .83,
                                    txt_r2,
                                    size=3,
                                    c='#555',
                                    ha='left', va='baseline',
                                    transform=ax_com[iv, ic].transAxes,
                                    zorder=1)
                
                # Add mean values for each OURs
                ax_com[iv, ic].text(.97, .05,
                                    f'Mean OUR ({ages_res_labs[c[0]]}) = {np.mean(X):.2f}',
                                    size=3,
                                    c='#555',
                                    ha='right', va='baseline',
                                    transform=ax_com[iv, ic].transAxes,
                                    zorder=1)
                ax_com[iv, ic].text(.97, .10,
                                    f'Mean OUR ({ages_res_labs[c[1]]}) = {np.mean(Y):.2f}',
                                    size=3,
                                    c='#555',
                                    ha='right', va='baseline',
                                    transform=ax_com[iv, ic].transAxes,
                                    zorder=1)
                
            
            xl = yl = vres_lims[iv]
            ax_com[iv, ic].set(xlim=xl, ylim=yl)
            ax_com[iv, ic].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax_com[iv, ic].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax_com[iv, ic].tick_params(which='major', axis='both',
                                      labelsize=5, pad=2,
                                      direction='in', length=1.5,
                                      top=True, right=True)
            ax_com[iv, ic].tick_params(which='minor', axis='both',
                                      direction='in', length=1,
                                      top=True, right=True)
            xlab = rate_labs[v] + "\n(" + ages_res_labs[c[0]] + ")"
            ylab = rate_labs[v] + "\n(" + ages_res_labs[c[1]] + ")"
            ax_com[iv, ic].set_xlabel(xlab, fontsize=5,
                                      labelpad=3, linespacing=1.5)
            ax_com[iv, ic].set_ylabel(ylab, fontsize=5,
                                      labelpad=3, linespacing=1.5)
            
            
    fig_com.subplots_adjust(wspace=.45, hspace=.35)
    fpath = ('figures/hansell_glodap/global/rates/_rates_comparison_regression.svg')
    fig_com.savefig(fpath, format='svg', bbox_inches='tight')
    
    
    ### Repeat it only for AOU-based rates
    
    v = 'AOU_RES'
    iv = 1
    fig_com, ax_com = plt.subplots(nrows=1, ncols=len(comparisons),
                                   figsize=(5*cm * len(comparisons),
                                            4*cm))
    for ic, c in enumerate(comparisons):
        
        # Subset data for rate with v, and tracer ages in c
        idx = ((reg['y_var']==v) &
               (reg.spvalue < pval_limit) &
               (reg.r2_AN > r2_limit))
        if v=='AOU_RES':
            idx = idx & (reg.slope >= 0)
        elif v=='OXYGEN_RES':
            idx = idx & (reg.slope <= 0)
        r = reg.loc[idx, :].copy()
        
        # get the rates (slopes) separately
        r_0 = r.loc[r['x_tracer']==c[0], ['pixel_key', 'slope']]
        r_1 = r.loc[r['x_tracer']==c[1], ['pixel_key', 'slope']]
        
        # Merge them back but in wide format
        r_01 = r_0.merge(r_1, on='pixel_key', how='inner',
                         suffixes=["_" + c[0], "_" + c[1]])
        X = r_01["slope_" + c[0]]
        Y = r_01["slope_" + c[1]]
        
        # Estimate regression
        md_com = rc.RegressConsensusW(X, Y, Wx=.5)
                    
        # Plot data and regression
        ax_com[ic].scatter(X, Y,
                           marker='o',
                           s=5,
                           linewidth=.1,
                           facecolor='steelblue',
                           edgecolor='w')
        
        # If regression was significant, add line
        if md_com['spvalue']<.05:
            x0 = np.nanmin(X)
            x1 = np.nanmax(X)
            y0 = md_com['intercept'] + x0 * md_com['slope']
            y1 = md_com['intercept'] + x1 * md_com['slope']
            ax_com[ic].plot([x0, x1], [y0, y1],
                            c='#444', lw=1.0,
                            zorder=1)
            
        # Add regression parameters
        txt_eq = f"y = ({md_com['slope']:.2f} ± {md_com['sci95']:.2f})x {sign_string(md_com['intercept'])} ({abs(md_com['intercept']):.2f} ± {md_com['ici95']:.2f})"
        txt_spval = p_label(md_com['spvalue'])
        txt_r2 = ("$R^2$ = " + '{:.3f}'.format(md_com['r2']))
        ax_com[ic].text(.05, .93,
                        txt_eq,
                        size=3,
                        c='#555',
                        ha='left', va='baseline',
                        transform=ax_com[ic].transAxes,
                        zorder=1)
        ax_com[ic].text(.05, .88,
                        txt_spval,
                        size=3,
                        c='#555',
                        ha='left', va='baseline',
                        transform=ax_com[ic].transAxes,
                        zorder=1)
        ax_com[ic].text(.05, .83,
                        txt_r2,
                        size=3,
                        c='#555',
                        ha='left', va='baseline',
                        transform=ax_com[ic].transAxes,
                        zorder=1)
        
        # Add mean values for each OURs
        ax_com[ic].text(.97, .05,
                        f'Mean OUR ({ages_res_labs[c[0]]}) = {np.mean(X):.2f}',
                        size=3,
                        c='#555',
                        ha='right', va='baseline',
                        transform=ax_com[ic].transAxes,
                        zorder=1)
        ax_com[ic].text(.97, .10,
                        f'Mean OUR ({ages_res_labs[c[1]]}) = {np.mean(Y):.2f}',
                        size=3,
                        c='#555',
                        ha='right', va='baseline',
                        transform=ax_com[ic].transAxes,
                        zorder=1)
        
        xl = yl = vres_lims[iv]
        ax_com[ ic].set(xlim=xl, ylim=yl)
        ax_com[ic].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_com[ic].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_com[ic].tick_params(which='major', axis='both',
                               labelsize=5, pad=2,
                               direction='in', length=1.5,
                               top=True, right=True)
        ax_com[ic].tick_params(which='minor', axis='both',
                               direction='in', length=1,
                               top=True, right=True)
        x = "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
        xlab = x + "\n(" + ages_res_labs[c[0]] + ")"
        ylab = x + "\n(" + ages_res_labs[c[1]] + ")"
        ax_com[ic].set_xlabel(xlab, fontsize=5,
                              labelpad=3, linespacing=1.5)
        ax_com[ic].set_ylabel(ylab, fontsize=5,
                              labelpad=3, linespacing=1.5)
        
        
    fig_com.subplots_adjust(wspace=.45, hspace=.35)
    fpath = ('figures/hansell_glodap/global/rates/_rates_comparison_regression_only_AOU.svg')
    fig_com.savefig(fpath, format='svg', bbox_inches='tight')
    
    
    
    #### Boxplots grouping by tracer, all rates
    
    vres_lims = [[-35, 5], [-5, 35], [-4, 4]]
    
    fig_com2, ax_com2 = plt.subplots(nrows=1, ncols=len(vres),
                                     figsize=(5*cm * len(vres), 5*cm))
    for iv, v in enumerate(vres):
            
            # Subset data for rate with v, and tracer ages in c
            idx = ((reg['y_var']==v) &
                   (reg.spvalue < pval_limit) &
                   (reg.r2_AN > r2_limit))
            if v=='AOU_RES':
                idx = idx & (reg.slope >= 0)
            elif v=='OXYGEN_RES':
                idx = idx & (reg.slope <= 0)
            r = reg.loc[idx, :].copy()
            
            # Get the rates (slopes) separately
            cols = ['pixel_key', 'slope']
            r_0 = r.loc[r['x_tracer']==ages_res[0], cols].set_index('pixel_key')
            r_1 = r.loc[r['x_tracer']==ages_res[1], cols].set_index('pixel_key')
            r_2 = r.loc[r['x_tracer']==ages_res[2], cols].set_index('pixel_key')
            
            # Merge results (merge will retain only values that are present in 
            # all three tracers).
            # Do it this way otherwise some tracers will be biased towards higher/
            # lower values: e.g., if everything is plotted, the boxplot of SF6 will
            # seem to indicate that SF6 rates are higher, when the opposite is true
            # but that would only seem so because SF6 is biased towards shallower
            # data, and hence higher rates. Paired data avoids this, because data 
            # will be equally distributed in the water column.
            r_all = (r_0.merge(r_1, on='pixel_key', suffixes=['_0', '_1'])
                     # suffixes are only added if column names overlap. force it
                     .merge(r_2.add_suffix('_2'), on='pixel_key'))
            
            ax_com2[iv].boxplot(r_all,
                                boxprops={'color': '#444',
                                          'linewidth': .7},
                                medianprops={'color': '#444',
                                             'linewidth': 1.2},
                                flierprops={'marker': 'o',
                                            'markersize': 3,
                                            'markeredgewidth': .2, 
                                            'markerfacecolor': '#444', 
                                            'markeredgecolor': 'w'},
                                capprops={'color': '#444',
                                          'linewidth': .7},
                                whiskerprops={'color': '#444',
                                              'linewidth': .7},
                                capwidths=.1,
                                tick_labels=list(ages_res_labs.values())
                                )
            
            yl = vres_lims[iv]
            ax_com2[iv].set(ylim=yl)
            ax_com2[iv].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax_com2[iv].tick_params(which='major', axis='both',
                                    labelsize=5, pad=2,
                                    direction='in', length=1.5,
                                    top=True, right=True)
            ax_com2[iv].tick_params(which='minor', axis='both',
                                    direction='in', length=1,
                                    top=True, right=True)
            ylab = rate_labs[v]
            ax_com2[iv].set_ylabel(ylab, fontsize=5, labelpad=1.5)
            
    fig_com2.subplots_adjust(wspace=.4)
    fpath = ('figures/hansell_glodap/global/rates/_rates_comparison_boxplot.svg')
    fig_com2.savefig(fpath, format='svg', bbox_inches='tight')
    
    
    
    #### Boxplots only for AOU, per water mass
    
    # Subset data for rate with AOU
    idx = ((reg['y_var']=='AOU_RES') &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit) &
           (reg.slope > 0))
    r = reg.loc[idx, :].copy()
    
    
    nc = int(np.floor(len(pixel_polys)**.5))
    nr = int(np.ceil(len(pixel_polys) / nc))
    n_w = len(pixel_polys)
    last_row = [*range(n_w - (nc - 1), n_w + 1)]
    first_col = [*range(0, n_w, nc)]
    
    ylab = "OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
    
    fig_com3 = plt.figure(figsize=(4.5*cm * nc, 5*cm * nr))
    for i, wm in enumerate(wm_polys_flat):
        
        # if i > 6: break
        
        # Subset data for water mass wm
        idx = ((r.water_mass==wm) &
               (r.x_tracer.isin(ages_res)))
        rss = r.loc[idx, :].copy()
        
        # Split code
        o = wm.split(";")[0] # ocean
        d = wm.split(";")[1] # depth layer
        w = wm.split(";")[2] # water mass code
        
        # prepare to keep track of values in each water mass to set y-limits
        vals_track = list()
        vals_track2 = dict()
        
        ax_com3 = fig_com3.add_subplot(nr, nc, i + 1)
        for ia, a in enumerate(ages_res):
            
            # Subset values for tracer a
            vals = rss.loc[rss.x_tracer==a].slope
            vals_track.extend(vals) # keep track to set limits
            
            # Keep track of values for Kruskal tests (do not append empty lists)
            if vals.to_list(): vals_track2[a] = vals.to_list()
            
            # Colour to differentiate tracers
            bx_col = ages_pal[a]
            
            # Do boxplot
            ax_com3.boxplot(vals,
                            positions=[ia],
                            widths=.4,
                            boxprops={'color': bx_col,
                                      'linewidth': .7},
                            medianprops={'color': bx_col,
                                         'linewidth': 1.2},
                            flierprops={'marker': 'o',
                                        'markersize': 3,
                                        'markeredgewidth': .2, 
                                        'markerfacecolor': bx_col, 
                                        'markeredgecolor': 'w'},
                            capprops={'color': bx_col,
                                      'linewidth': .7},
                            whiskerprops={'color': bx_col,
                                          'linewidth': .7},
                            capwidths=.1,
                            tick_labels=[ages_res_labs[a]]
                            )
            
            
        # Customise axes etc.
        ax_com3.set(xticks=range(0, len(ages_res)),
                    xticklabels=ages_res_labs.values())
        ax_com3.tick_params(axis='both', labelsize=5, length=1.8)
    
        # Extend y-limits to make room for labels
        if vals_track:
            ax_com3.set(ylim=[0, (np.max(vals_track) + np.max(vals_track)*.4)])
            ax_com3.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax_com3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        else:
            ax_com3.text(.5, .5, "No valid results",
                         color='#777',
                         fontsize=8, fontweight='bold',
                         ha='center', va='center',
                         transform=ax_com3.transAxes)
            ax_com3.set(yticks=[])
    
        #--------------------------------------------------------------------------
        # 
        # Add number of data points
        # 
        # Apparently the transform does not work until .set() is applied to the 
        # axis. So I can't do it within the plotting loop. 
        # But it's not something about the limits necessarily, which I would expect
        # because doing this after setting the xticks also seems to do the trick??
        
        for ia, a in enumerate(ages_res):
            # Skip if tracer has no data
            if a not in vals_track2: continue
            v = vals_track2[a]
            if v:
                ypos = ax_com3.transLimits.transform((ia, max(v)))[1] + .05
                ypos = ax_com3.transLimits.inverted().transform((ia, ypos))[1]
                ax_com3.text(ia, ypos, len(v),
                             fontsize=5, color='#333',
                             ha='center', va='baseline')
        
        #--------------------------------------------------------------------------
    
        # Do Kruskal-Wallis test
        if len(vals_track) > 1: # only if more than 1 group (tracer) has data
            
            # Test
            kwt = kruskal(*list(vals_track2.values()))
            
            # Text to add to plot
            txt = "$\mathit{p}$"
            if kwt[1] < .001:
                txt = txt + " < 0.001 "
            elif kwt[1] < .01:
                txt = txt + " < 0.01 "
            else:
                txt = txt + " = " + str(round(kwt[1], 2))
             
            ax_com3.text(.97, .91, txt,
                         fontsize=5, color='#333',
                         ha='right', va='baseline',
                         transform=ax_com3.transAxes)
            
        # Water mass label
        ax_com3.text(.03, .91, wm_labels_b[w],
                     color='#222',
                     fontsize=6,
                     ha='left', va='baseline',
                     transform=ax_com3.transAxes)
    
        # Y-axis labels
        if i in first_col:
            ax_com3.set_ylabel(ylab, fontsize=5, labelpad=3)
        
            
    fpath = ('figures/hansell_glodap/global/rates/_our_rates_aou_per_tracer_water_mass_boxplot_kruskal.svg')
    fig_com3.savefig(fpath, format='svg', bbox_inches='tight')
    plt.close(fig_com3)
    
    
    #%%% Water mass intercomparison
    
    # Compare rates across water masses
    
    # ----------------------------- Prepare data -------------------------------- #
    
    
    # Subset OURs based on AOU, ONLY THOSE WHICH MEET THE PVALUE AND R2 CRITERIA!
    idx = ((reg.y_var=='AOU_RES') &
           (reg.x_tracer.isin(ages_res)) &
           (reg.slope >= 0) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit))
    ss = reg.loc[idx, :].reset_index(drop=True)
    
    # Get the median values per water mass, to sort the plot based on them.
    # Use values from CFC-11 for this.
    ss_agg = ss.loc[ss.x_tracer=='AGE_CFC11_RES', :]
    ss_agg = ss_agg.groupby('water_mass').agg(median_our=('slope', 'median'),
                                              mean_depth=('CTD_PRESSURE', 'mean'))
    ss_agg = ss_agg.reset_index() # water_mass to column
    
    # Split water mass code and convert ocean and depth to categorical
    ss_agg[['o', 'd', 'w']] = ss_agg.water_mass.str.split(";", expand=True)
    ss_agg.o = pd.Categorical(ss_agg.o, categories=['atlantic', 'indian', 'pacific', 'mediterranean'])
    ss_agg.d = pd.Categorical(ss_agg.d, categories=['central', 'intermediate', 'deep'])
    
    # Sort the data based on ocean, depth and lastly rate
    ss_agg = (ss_agg
              .sort_values(['d', 'o', 'median_our'], 
                           ascending=[True, False, False])
              .reset_index(drop=True))
    
        
    # Convert age_res labels into bold
    ages_res_labs_b = {k:v for k, v in zip(ages_res,
                                           [("$\mathbf{" +
                                             ages_res_labs[g].
                                             replace("$", "").
                                             replace("\mathregular", "") +
                                             "}$")
                                             for g in ages_res])}
    
    # Color palette for oceans
    ocean_pal = {k: v for k, v in zip(ocean_polys.keys(),
                                      ['#44AA99', '#88CCEE', '#DDCC77',
                                       '#EE8866'])}
    
    
    
    ## Plot
    
    nr = len(ss_agg)
    nc = len(ages)
    fig_rdg, ax_rdg = plt.subplots(nrows=nr, ncols=nc,
                                   figsize=(4*cm*nc, (1*cm*nr)))
    for ir, r in ss_agg.iterrows():
        
        o = r['water_mass'].split(";")[0] # ocean
        d = r['water_mass'].split(";")[1] # depth layer
        w = r['water_mass'].split(";")[2] # water mass code
        
        for ia, a in enumerate(ages_res):
            idx = (ss.x_tracer==a) & (ss.water_mass==r['water_mass'])
            vals = ss.loc[idx, 'slope']
            
            # Plot density curves
            if len(vals) > 1:
                density = gaussian_kde(vals)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                xs = np.arange(0, 40, .1)
                ys = density(xs)
                # ax_rdg[ir, ia].plot(xs, ys)
                ax_rdg[ir, ia].fill_between(xs, ys, color=ocean_pal[o],
                                            zorder=0)
            # Add location of underlying data
            ax_rdg[ir, ia].scatter(vals, [max(ys) + max(ys)*.25]*len(vals),
                                   marker='v', facecolor='#555', edgecolor='none',
                                   s=3,
                                   zorder=2)
            
            # Add reference line for median
            ax_rdg[ir, ia].axvline(np.median(vals), 
                                   linewidth=.5, color='#888',
                                   zorder=1)
            
            ## Customise axes
            
            # No y-axis labels
            ax_rdg[ir, ia].tick_params(axis='y', left=False, labelleft=False)
            
            # Add water mass name to the left of the first column
            if ia==0:
                ax_rdg[ir, ia].set_ylabel(wm_labels_b[w],
                                          color='#222',
                                          fontsize=6,
                                          rotation=0,
                                          ha='right', va='center')
                
            # Add x-axis title in the last row
            if ir==(ss_agg.shape[0]-1):
                ax_rdg[ir, ia].tick_params(axis='x', labelsize=6,
                                           length=1.5, width=.6, color='#555')
                ax_rdg[ir, ia].set_xlabel("OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]",
                                          fontsize=6, labelpad=2)
            else:
                ax_rdg[ir, ia].tick_params(axis='x', labelbottom=False,
                                           length=1.5, width=.6, color='#555')
                
            # Add column title in first row
            if ir==0:
                ax_rdg[ir, ia].set_title(ages_res_labs[a], fontsize=7)
            
            # Remove spines, only leave the bottom one (x-axis)
            spines = ['top', 'right', 'left']
            for s in spines:
                ax_rdg[ir, ia].spines[s].set_visible(False)
            ax_rdg[ir, ia].spines['bottom'].set_color('#555')
            ax_rdg[ir, ia].spines['bottom'].set_linewidth(.6)
            
            # Adjust shared x-axis
            ax_rdg[ir, ia].set(xlim=[0, 40], xticks=range(0, 50, 10),
                               ylim=[0, max(ys) + max(ys)*.3])
            
            print(w + " -> " + str(np.median(vals)))
        
        
    
    fpath = ('figures/hansell_glodap/global/rates/hansell_glodap_rates_aou_water_mass_density.pdf')
    fig_rdg.savefig(fpath, format='pdf', bbox_inches='tight')
    
    
    
    ## Redo figure but using boxplots so that the figure is more compact
    
    fig_bxp, ax_bxp = plt.subplots(nrows=1, ncols=nc,
                                   figsize=(4*cm*nc, (.35*cm*nr)))
    for ia, a in enumerate(ages_res):
        
        ax_bxp[ia].grid(axis='x', color='w', lw=.3)
    
        for ir, r in ss_agg.iterrows():
        
            o = r['water_mass'].split(";")[0] # ocean
            d = r['water_mass'].split(";")[1] # depth layer
            w = r['water_mass'].split(";")[2] # water mass code
        
            # Subset data for tracer a in water_mass
            idx = (ss.x_tracer==a) & (ss.water_mass==r['water_mass'])
            vals = ss.loc[idx, 'slope']
            
            # Plot boxplots
            if len(vals) >= 5:
                
                bplot = ax_bxp[ia].boxplot(vals,
                                           positions=[ir],
                                           widths=.6,
                                           boxprops={'color': ocean_pal[o],
                                                     'facecolor': ocean_pal[o] + '11',
                                                     'linewidth': .5},
                                           medianprops={'color': ocean_pal[o],
                                                        'linewidth': .5},
                                           flierprops={'marker': 'o',
                                                       'markersize': 1.5,
                                                       'markeredgewidth': .2, 
                                                       'markerfacecolor': ocean_pal[o], 
                                                       'markeredgecolor': 'none'},
                                           capprops={'color': ocean_pal[o],
                                                     'linewidth': .5},
                                           whiskerprops={'color': ocean_pal[o],
                                                         'linewidth': .5},
                                           patch_artist=True,
                                           capwidths=.1,
                                           vert=False,
                                           tick_labels=[wm_labels_b[w]])
            else:
                
                # If less than 5 data values, just add points        
                ax_bxp[ia].scatter(vals, [ir]*len(vals),
                                   marker='o',
                                   facecolor=ocean_pal[o], edgecolor='none',
                                   s=2.5,
                                   zorder=2)
                # Apparently if I don't add a placeholder empty "boxplot" in this
                # positions too, matplotlib saves some space and the axis gets a
                # bit squished? (but only if the first value in the axis has no
                # boxplot). In any case, this workaround also serves to add labels
                # in all positions.
                ax_bxp[ia].boxplot([np.nan], positions=[ir],
                                   vert=False, tick_labels=[wm_labels_b[w]])
    
                
            # Add text indicating number of data points?
            add_n = True
            if add_n & (len(vals) > 0):
                xpos = .98
                ypos_ir = ir
                if ((a=="AGE_CFC11_RES") & (w=="SPEW")):
                    ypos_ir += .3
                elif ((a=="AGE_CFC12_RES") & (w=="SPEW")):
                    xpos = .875
                # if ((a=="AGE_CFC12_RES") & (w=="SPEW")):
                #     xpos = .87
                # # elif ((a=="AGE_CFC12_RES") & (w=="PSUW")):
                # #     xpos = .24
                # else:
                #     xpos = .98
                if len(vals)==1:
                    txt = str('{:.1f}'.format(np.mean(vals)))
                else:
                    txt = (str('{:.1f}'.format(np.mean(vals))) + " ± " + 
                           str('{:.1f}'.format(np.std(vals))))
                    # " (" + str(len(vals)) + ")"
                ax_bxp[ia].text(xpos, ypos_ir, txt,
                                fontsize=2.5,
                                weight='bold',
                                color='#555',
                                ha='right', va='center',
                                transform=ax_bxp[ia].get_yaxis_transform())
            
            ## Customise axes
            
            # labels=[wm_labels_b[w]]
            
            # Y-axis labels in first subplot only
            if ia==0:
                ax_bxp[ia].tick_params(axis='y', left=False, labelsize=4.5,
                                       labelcolor='#555')
            else:
                ax_bxp[ia].tick_params(axis='y', left=False, labelleft=False)
            
                
            # Add x-axis title
            ax_bxp[ia].tick_params(axis='x', labelsize=5,
                                   length=1.5, width=.7, color='#333',
                                   pad=2)
            ax_bxp[ia].set_xlabel("OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]",
                                  fontsize=5, labelpad=2)
                
            # Add column title in first row
            if ir==0:
                ax_bxp[ia].set_title(ages_res_labs_b[a],
                                     fontsize=5, color='#333')
            
            # Remove spines, only leave the bottom one (x-axis)
            spines = ['top', 'right', 'left']
            for s in spines:
                ax_bxp[ia].spines[s].set_visible(False)
            ax_bxp[ia].spines['bottom'].set_color('#333')
            ax_bxp[ia].spines['bottom'].set_linewidth(.7)
    
            # Adjust shared x-axis
            xtcks = range(0, 35, 5)
            xtcklabs = [x if x%10==0 else '' for x in xtcks]
            ax_bxp[ia].set(xlim=[0, 30], xticks=xtcks, xticklabels=xtcklabs)
        
        # Add patches delimiting the water depth layers
        ax_bxp[ia].axhspan(ymin=-0.5, ymax=16.5, facecolor='#f9f9f9')
        ax_bxp[ia].axhspan(ymin=16.5, ymax=26.5, facecolor='#eee')
        ax_bxp[ia].axhspan(ymin=26.5, ymax=30.5, facecolor='#ddd')
        
        # Invert y axis to match the desired order (roughly from higher to lower)
        ax_bxp[ia].invert_yaxis()
        
    
            
    fpath = 'figures/hansell_glodap/global/rates/hansell_glodap_rates_aou_water_mass_boxplot.pdf'
    fig_bxp.savefig(fpath, format='pdf', bbox_inches='tight')
    fpath = fpath.replace("pdf", "svg")
    fig_bxp.savefig(fpath, format='svg', bbox_inches='tight')
    
    
    
    #%% MAP RATES
    
    
    wm_depths_idx = {k:v for k, v in zip(wm_depths, [0, 1, 2])}
    wm_depths_lab = {k:v for k, v in zip(wm_depths,
                                         ['Central', 'Interm.', 'Deep'])}
    
    # Plot SAIW and MW separately in another figure, include them in the supp. info
    # (because they overlap with other water masses)
    in_supinf = ['SAIW', 'MW']
    
    nr = len(wm_depths_idx) # + 1
    nc = len(ages_res)
    for v in vres:
        
        # Initialise plot
        mproj = ccrs.Mollweide(central_longitude=-160)
        fig_m, ax_m = plt.subplots(nrows=nr, ncols=nc,
                                   figsize=(5*cm*nc, 3*cm*nr),
                                   subplot_kw={'projection': mproj})
    
        # Add land to each map (to avoid overplotting in each loop)
        for i in range(nr):
            for ig, g in enumerate(ages_res):
                ax_m[i, ig].add_feature(cfeature.LAND, facecolor='#eee',
                                        edgecolor='k',
                                        linewidth=.1, zorder=1)
                # Add labels of layers
                if ig==0:
                    d_lab = wm_depths_lab[wm_depths[i]] 
                    ax_m[i, ig].text(-.1, .5, "$\mathbf{" + d_lab + "}$",
                                     c="#444", size=5,
                                     rotation=90,
                                     va='center',
                                     transform=ax_m[i, ig].transAxes)
                if i==0:
                    ax_m[i, ig].text(.5, 1.11, ages_res_labs_b[g],
                                     c="#444", size=6,
                                     ha='center',
                                     transform=ax_m[i, ig].transAxes)
    
         
        # Create normalising function to map rate values between [0,1] to
        # then assign to the colourmap.
        # Share it across water masses so that colourmap limits are equal.
        # Also:
        #   - set rounding precission for labels (if used) 
        #   - set labels
        #   - set ticks
        if v=='AOU_RES':
           
            # For AOU, extend colourmap for positive values
            cap_val1 = 0
            cap_val2 = 20
            col_norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
            px_pal = mpl.colormaps.get_cmap('YlGnBu')
            rnd = 1 # rounding precission for labels
            cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
            # Create colorbar tickmarks, ensure cap_val2 is included
            inc = np.ceil((cap_val2 - cap_val1) / 5)
            tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                                   [cap_val2]))
            tcks = np.unique(tcks)
            # Set colourbar extension
            ext = 'max'
            
        else:
            
            # (Plot rates based on oxygen as absolute values, even if
            # technically negative.)
            cap_val1 = 0
            cap_val2 = 20 
            col_norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
            px_pal = mpl.colormaps.get_cmap('YlGnBu')
            rnd = 1
            cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
            inc = np.ceil((cap_val2 - cap_val1) / 5)
            tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                                   [cap_val2]))
            tcks = np.unique(tcks)
            ext = 'max'
    
                
        for ig, g in enumerate(ages_res):
    
            # Go through water masses
            for ik1, k1 in enumerate(wm_polys_flat):
                
                # Split water mass key
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                
                # Water masses will be group by depth layer in maps.
                # Get the corresponding map index for depth layer d
                i = wm_depths_idx[d]
                
                # Special case: to avoid overlaps, MW and SAIW will be mapped
                # separately
                if w in in_supinf:
                    continue # i = 3
                
                # Overlay water mass border as reference
                ax_m[i, ig].add_geometries(wm_polys_plot_flat[k1],
                                           facecolor='none',
                                           edgecolor='k',
                                           linewidth=.4,
                                           crs=ccrs.PlateCarree(),
                                           zorder=2)
                
                # Subset regression results for (v vs g) in water mass w
                # (Even if we want to represent only significant results, we also 
                # want to plot as such the pixels where the regression was *not* 
                # significant. This way, we are able to tell apart those areas with
                # non-significant results from those where there was no data)
                idx = ((reg.y_var==v) & (reg.x_tracer==g)) 
                reg_ss_k1 = reg.loc[idx & (reg.water_mass==k1)].copy()
                
                # Iterate through results to plot the pixels with the rate values
                for ir, r in reg_ss_k1.iterrows():
                    
                    # If acceptable regression, give colour; otherwise grey
                    # Consider pval, r2 and enforce sign for AOU/OXYGEN
                    if v=='AOU_RES':
                        boo = ((r.spvalue < pval_limit) & 
                               (r.r2_AN > r2_limit) &
                               (r.slope >= 0))
                    elif v=='OXYGEN_RES':
                        boo = ((r.spvalue < pval_limit) & 
                               (r.r2_AN > r2_limit) &
                               (r.slope <= 0))
                      
                    if boo:
                        
                        # If conditions acceptable for pixel r:
                        # 
                        # Colour of pixel based on rate value
                        # 
                        # (Plot rates based on oxygen as absolute values, even if
                        # technically negative.)
                        val = abs(r.slope) if v=='OXYGEN_RES' else r.slope
                        pxc = px_pal(col_norm(val))
                        
                        # Add filled pixel polygon
                        px = pixel_polys[k1][r.pixel]
                        pix_pol = ax_m[i, ig].add_geometries(px,
                                                             facecolor=pxc,
                                                             edgecolor='#777',
                                                             linewidth=.1,
                                                             crs=ccrs.PlateCarree(),
                                                             zorder=0)
                        
                        # Add rate value on top?
                        add_labels = False
                        if add_labels:
                            
                            # Get luminance of colour to add text in black or white
                            # This is rough approximation, see details:
                            # https://stackoverflow.com/a/56678483/9271401
                            lum = 0.2126 * pxc[0] + 0.7152 * pxc[1] + 0.0722 * pxc[2]
                            txt_color = 'k' if lum >.3 else 'w'
                            ax_m[i, ig].text(px.centroid.x, px.centroid.y,
                                              str(round(r.slope, rnd)),
                                              size=1,
                                              color=txt_color,
                                              ha='center', va='center',
                                              transform=ccrs.PlateCarree(),
                                              zorder=1)
                        
                    else:
                        
                        # Otherwise pixel in grey
                        pxc = '#aaa'
                        px = pixel_polys[k1][r.pixel]
                        pix_pol = ax_m[i, ig].add_geometries(px,
                                                             facecolor=pxc,
                                                             edgecolor='#777',
                                                             linewidth=.1,
                                                             crs=ccrs.PlateCarree(),
                                                             zorder=0)
    
        # Add shared colour bar
        axins1 = ax_m[nr-1, 1].inset_axes([.025, -.25, .95, .1])
        cbar = fig_m.colorbar(mpl.cm.ScalarMappable(norm=col_norm, cmap=px_pal),
                              cax=axins1,
                              extend=ext,
                              ticks=tcks,
                              orientation='horizontal')
        cbar.ax.set_xlabel(cbar_lab, size=6, labelpad=2)
        cbar.ax.tick_params(labelsize=6)     
        
        # Adjust subplots
        fig_m.subplots_adjust(wspace=.13, hspace=-.2)
        
        
        # Export
        # Figure background transparent
        fig_m.set_facecolor('none')
        fpath = ('figures/hansell_glodap/global/rates/maps/hansell_glodap_global' +
                 '_rate_' + v + '_pixel_map.pdf')
        fig_m.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False, 
                      dpi=300) # low dpi can affect colourbar rendering in pdfs
        fpath = fpath.replace("pdf", "svg")
        fig_m.savefig(fpath, format='svg', bbox_inches='tight', transparent=False, 
                      dpi=300)
        plt.close(fig_m)
        
    
    
    #### Quick comparison of density anomalies in WSACW and ESACW to check whether
    #    samples came from notably different sigma ranges (within their general
    #    boundaries) and that could account for the differences in the magnitude
    #    of rates between those two water masses.
    
    idx = ((reg.x_tracer=="AGE_CFC11_RES") &
           (reg.y_var=='AOU_RES') &
           (reg.water_mass=="atlantic;central;WSACW"))
    mean_sigma0_WSACW = round(np.nanmean(reg.SIGMA0.loc[idx]), 3)
    idx = ((reg.x_tracer=="AGE_CFC11_RES") &
           (reg.y_var=='AOU_RES') &
           (reg.water_mass=="atlantic;central;ESACW"))
    mean_sigma0_ESACW = round(np.nanmean(reg.SIGMA0.loc[idx]), 3)
    
    print("Mean SIGMA0 for WSACW: " + str(mean_sigma0_WSACW) + 
          "\n" +
          "Mean SIGMA0 for ESACW: " + str(mean_sigma0_ESACW))
    
    #%%% SAIW & MW
    
    # Repeat mapping but only for SAIW and MW
    
    for v in vres:
        
        mproj = ccrs.Orthographic(central_longitude=-30, central_latitude=50)
        fig_msi, ax_msi = plt.subplots(nrows=1, ncols=nc,
                                       figsize=(5*cm*nc, 6*cm),
                                       subplot_kw={'projection': mproj})
    
        for ig, g in enumerate(ages_res):
            ax_msi[ig].add_feature(cfeature.LAND, facecolor='#eee',
                                   edgecolor='k',
                                   linewidth=.1, zorder=1)
    
            if i==0:
                ax_msi[ig].text(.5, 1.07, ages_res_labs_b[g],
                                 c="#444", size=6,
                                 ha='center',
                                 transform=ax_msi[ig].transAxes)
    
        if v=='AOU_RES':
           
            cap_val1 = 0
            cap_val2 = 20
            col_norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
            px_pal = mpl.colormaps.get_cmap('YlGnBu')
            rnd = 1 
            cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
            inc = np.ceil((cap_val2-cap_val1)/5)
            tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                                   [cap_val2]))
            tcks = np.unique(tcks)
            ext = 'max'
            
        else:
         
            cap_val1 = 0
            cap_val2 = 20
            col_norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
            px_pal = mpl.colormaps.get_cmap('YlGnBu')
            rnd = 1
            cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
            inc = np.ceil((cap_val2-cap_val1)/5)
            tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                                   [cap_val2]))
            tcks = np.unique(tcks)
            ext = 'max'
                
        for ig, g in enumerate(ages_res):
            for ik1, k1 in enumerate(wm_polys_flat):
                
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                
                # Special case: only do MW and SAIW (skip otherwise)
                if w not in in_supinf:
                    continue 
    
                ax_msi[ig].add_geometries(wm_polys_plot_flat[k1],
                                           facecolor='none',
                                           edgecolor='k',
                                           linewidth=.4,
                                           crs=ccrs.PlateCarree(),
                                           zorder=2)
    
                idx = ((reg.y_var==v) & (reg.x_tracer==g)) 
                reg_ss_k1 = reg.loc[idx & (reg.water_mass==k1)].copy()
                for ir, r in reg_ss_k1.iterrows():
                    
                    if v=='AOU_RES':
                        boo = ((r.spvalue < pval_limit) & 
                               (r.r2_AN > r2_limit) &
                               (r.slope >= 0))
                    elif v=='OXYGEN_RES':
                        boo = ((r.spvalue < pval_limit) & 
                               (r.r2_AN > r2_limit) &
                               (r.slope <= 0))
                        
                    if boo:
                        
                        # Colour of pixel based on rate value
                        val = abs(r.slope) if v=='OXYGEN_RES' else r.slope
                        pxc = px_pal(col_norm(val))
                        
                        # Add filled pixel polygon
                        px = pixel_polys[k1][r.pixel]
                        pix_pol = ax_msi[ig].add_geometries(px,
                                                             facecolor=pxc,
                                                             edgecolor='#777',
                                                             linewidth=.1,
                                                             crs=ccrs.PlateCarree(),
                                                             zorder=0)
            
                        
                    else:
                        
                        pxc = '#aaa'
                        px = pixel_polys[k1][r.pixel]
                        pix_pol = ax_msi[ig].add_geometries(px,
                                                             facecolor=pxc,
                                                             edgecolor='#777',
                                                             linewidth=.1,
                                                             crs=ccrs.PlateCarree(),
                                                             zorder=0)
    
            
        axins1 = ax_msi[1].inset_axes([.025, -.25, .95, .1])
        cbar = fig_msi.colorbar(mpl.cm.ScalarMappable(norm=col_norm, cmap=px_pal), 
                                cax=axins1,
                                extend=ext,
                                ticks=tcks,
                                orientation='horizontal')
        cbar.ax.set_xlabel(cbar_lab, size=6)
        cbar.ax.tick_params(labelsize=6)
        
            
        fig_msi.subplots_adjust(wspace=.13, hspace=.1)
        fpath = ('figures/hansell_glodap/global/rates/maps/' +
                 'hansell_glodap_global_rate_' + v + '_pixel_map_SI.svg')
        fig_msi.savefig(fpath, format='svg', bbox_inches='tight',
                      transparent=True, dpi=300)
        # plt.close(fig_msi)
    
    
    
    
    #%%% PATTERNS IN WATER MASSES
    
    # Function to quickly check location of pixels in a regression result table
    def map_pixels(df):
        
        mproj = ccrs.Mollweide(central_longitude=-160)
        fig, ax = plt.subplots(figsize=(8*cm, 6*cm), 
                               subplot_kw={'projection': mproj})
        ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                    name='land',
                                                    scale='110m'),
                       facecolor='#ccc',
                       edgecolor='black',
                       linewidth=.1,
                       zorder=0)
        
        for ir, r in df.iterrows():
            
            px = pixel_polys[r.water_mass][r.pixel]
            ax.add_geometries(px,
                              facecolor='firebrick',
                              edgecolor='#777',
                              linewidth=.1,
                              crs=ccrs.PlateCarree(),
                              zorder=0)
    
    
    #%%%%% High rates in the NPIW close to the Sea of Okhotsk
    
    idx1 = ((reg.y_var=='AOU_RES') &
            (reg.x_tracer.isin(ages_res)) &
            (reg.water_mass.str.contains('NPIW')) &
            (reg.spvalue < pval_limit) &
            (reg.r2_AN > r2_limit) &
            (reg.slope >= 0))
    
    # Only get pixel whose upper bound is 50 or 60, and max longitude west of 180
    # (note these are nominal, in the case of the NPIW the 60ºN bound is actually
    #  like 52º, see map)
    lat_max = [float(x[3]) for x in reg.pixel.str.split(';')]
    lon_max = [float(x[2]) for x in reg.pixel.str.split(';')]
    idx2 = (pd.Series([(x in [50.0, 60.0]) for x in lat_max]) &
            pd.Series([(x in [150.0, 160.0, 170.0, 180.0]) for x in lon_max]))
    idx = (idx1) & (idx2)
    ss = reg.loc[idx, :]
    map_pixels(ss)
    
    for a in ages_res:
        
        print("NPIW, northwest, " + a + " mean OUR: " + 
              str(round(np.mean(ss.slope[ss.x_tracer==a]), 1)))
        
    ## Outside of the northwest area
    
    idx = (idx1) & (~idx2)
    ss = reg.loc[idx, :]
    map_pixels(ss)
    for a in ages_res:
        
        print("NPIW, outside northwest, " + a + " mean OUR: " + 
              str(round(np.mean(ss.slope[ss.x_tracer==a]), 1)))
        
        
    
    #%%%%% AAIW_P patterns
    
    # Map only AAIW_P rates with rescaled colourmap
    # Do only with CFCs because SF6 has poor coverage for AAIW_P
    idx = ((reg.y_var=='AOU_RES') &
           (reg.x_tracer.isin(['AGE_CFC11_RES', 'AGE_CFC12_RES'])) &
           (reg.water_mass.str.contains('AAIW_P')) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit) &
           (reg.slope >= 0))
    
    ss = reg.loc[idx, :]
    # map_pixels(ss)
    
    k = 'pacific;intermediate;AAIW_P'
    p = wm_polys_plot_flat[k]
    mproj = ccrs.NearsidePerspective(central_longitude=p.centroid.x,
                                     central_latitude=p.centroid.y)
    nr = 2 + 1 # 2 OURs plots + 1 circualtion diagram
    nc = 1
    fig, ax = plt.subplots(nrows=nr, ncols=nc, 
                           figsize=(10*cm, 18*cm), 
                           subplot_kw={'projection': mproj})
    for iv, v in enumerate(['AGE_CFC11_RES', 'AGE_CFC12_RES']):
        
        ax[iv].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                        name='land',
                                                        scale='110m'),
                           facecolor='#ccc',
                           edgecolor='black',
                           linewidth=.1,
                           zorder=1)
    
        cap_val1 = 2
        cap_val2 = 8
        col_norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('YlGnBu')
        cbar_lab = "OUR [$\mathregular{\mu mol\ kg^{-1}\ yr^{-1}}$]"
        # Create colorbar tickmarks, ensure cap_val2 is included
        inc = np.ceil((cap_val2 - cap_val1) / 10)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        # Set colourbar extension
        ext = 'both'
        
        # Overlay water mass border as reference
        ax[iv].add_geometries(p,
                              facecolor='none',
                              edgecolor='k',
                              linewidth=.4,
                              crs=ccrs.PlateCarree(),
                              zorder=2)
        
        ax[iv].text(-.09, .5, ages_res_labs_b[v],
                    ha='right', va='center',
                    rotation=90,
                    size=8, weight='bold', color='#444',
                    transform=ax[iv].transAxes)
    
        # Unlike in the global map, add only pixels that meet the required criteria
        # (that is, no grey pixels). This helps visualisation.
        
        # Iterate through results to plot the pixels with the rate values
        for ir, r in ss.loc[ss.x_tracer==v,:].iterrows():
            
            # Colour of pixel based on rate value
            val = abs(r.slope)
            pxc = px_pal(col_norm(val))
            
            # Add filled pixel polygon
            px = pixel_polys[k][r.pixel]
            pix_pol = ax[iv].add_geometries(px,
                                            facecolor=pxc,
                                            edgecolor='#777',
                                            linewidth=.1,
                                            crs=ccrs.PlateCarree(),
                                            zorder=0)
    
    # Add shared colour bar
    axins1 = ax[nr-2].inset_axes([1.12, .6, .08, .7])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=col_norm, cmap=px_pal),
                      cax=axins1,
                      extend=ext,
                      ticks=tcks)
    cbar.ax.set_ylabel(cbar_lab, size=6, labelpad=2)
    cbar.ax.tick_params(labelsize=6)
    
    
    ## Now do the diagram
    
    # Add land and AAIPW polygon
    ax[2].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                   name='land',
                                                   scale='110m'),
                      facecolor='#ccc',
                      edgecolor='black',
                      linewidth=.1,
                      zorder=1)
    ax[2].add_geometries(p,
                         facecolor='none',
                         edgecolor='#444',
                         linewidth=1,
                         crs=ccrs.PlateCarree(),
                         zorder=2)
    ax[2].add_geometries(p,
                          facecolor='#444',
                          edgecolor='none',
                          alpha=.08,
                          crs=ccrs.PlateCarree(),
                          zorder=0)
    
    # Mark formation area
    p = geometry.Polygon([(-76, -45), (-76, -58), 
                          (-90, -58), (-90, -45)])
    ax[2].add_geometries(p,
                         facecolor='none',
                         edgecolor='#444',
                         linewidth=.5,
                         hatch='||||||',
                         crs=ccrs.PlateCarree(),
                         zorder=2)
    
    # Mark water mass path
    arrow_list = [[(-110, -30), (-92, -47), '0.05'],
                  [(-120, -32), (-100, -55), '0.05'],
                  [(-85, -30), (-95, -41), '-0.2'],
                  [(-145, -23), (-113, -30), '0.3'],
                  [(-174, -23), (-148, -23), '0.05'],
                  [(-195, -20), (-183, -23), '-0.05'],
                  [(-170, -48), (-176, -22), '0.3'],
                  [(-130, -53), (-169, -48), '0.05']]
    for a in arrow_list:
        asty = 'fancy,tail_width=0.25,head_width=0.3,head_length=0.3'
        ax[2].annotate("", a[0], a[1], 
                       arrowprops=dict(
                           arrowstyle=asty,
                           connectionstyle=('arc3,rad=' + a[2]),
                           color='#555', #shrinkA=5, shrinkB=5,
                           linewidth=.1),
                       transform=ccrs.PlateCarree())
    
    
    fig.set_facecolor('none')
    
    fig.subplots_adjust(hspace=-.1)
    fpath = ('figures/hansell_glodap/global/rates/maps/hansell_glodap_global' +
             '_rate_AOU_RES_pixel_map_AAIWP.svg')
    fig.savefig(fpath, format='svg', bbox_inches='tight', transparent=False, 
                dpi=300)
    
    
    #%%%% PEqIW & AAIWA
    
    #### Show some values
    
    idx = ((reg.water_mass.str.contains('PEqIW')) &
           (reg.y_var=='AOU_RES') &
           (reg.x_tracer.isin(ages_res)) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit) &
           (reg.slope >= 0))
    ss = reg.loc[idx, :]
    for a in ages_res:
        
        print("PEqIW, " + a + " mean OUR: " + 
              str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))
        
        
    idx = ((reg.water_mass.str.contains('PEqIW')) &
           (reg.y_var=='AOU_RES') &
           (reg.x_tracer.isin(ages_res)) &
           (reg.LATITUDE > -10) & (reg.LATITUDE < 10) &
           (reg.spvalue < pval_limit) &
           (reg.r2_AN > r2_limit) &
           (reg.slope >= 0))
    ss = reg.loc[idx, :]
    for a in ages_res:
        
        print("AAIWA, lats (-10, 10), " + a + " mean OUR: " + 
              str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))
    
    
    
        
    
    #%% WATER MASS AVERAGES
    
    # Compute the average values of rates per water mass
    reg_av_w = []
    for iw, w in enumerate(wm_polys_flat):
        for v in vres:    
                for ix, x in enumerate(xvrs):
                    
                    # Subset regression for w + y + x
                    # (i.e., all the water mass values)
                    idx = ((reg['water_mass']==w) &
                           (reg['y_var']==v) &
                           (reg['x_tracer']==x))
                    reg_ss = reg.loc[idx]
                    
                    # Consider only regression deemed acceptable in terms of pval 
                    # and R2
                    if v=='AOU_RES':
                        idx2 = ((reg_ss['spvalue'] < pval_limit) &
                                (reg_ss['r2_AN'] > r2_limit) &
                                (reg_ss['slope'] >= 0))
                    elif v=='OXYGEN_RES':
                        idx2 = ((reg_ss['spvalue'] < pval_limit) &
                                (reg_ss['r2_AN'] > r2_limit) &
                                (reg_ss['slope'] <= 0))
                    else:
                        idx2 = ((reg_ss['spvalue'] < pval_limit) &
                                (reg_ss['r2_AN'] > r2_limit))
                            
                    reg_ss = reg_ss.loc[idx2]
                    
                    # Average rate (slope)
                    slope_av = round(np.mean(reg_ss['slope']), 3)
                    slope_sd = round(np.std(reg_ss['slope']), 3)
                    
                    # Store results
                    reg_av_w.append([w, v, x, slope_av, slope_sd])
    
    
    # Create dataframe with results
    reg_av_w = pd.DataFrame(reg_av_w, columns=['water_mass', 'y_var', 'x_tracer', 
                                               'slope_mean', 'slope_sd'])
    
    
    
    
    # Get the average OUR values based on each tracer for relevant water masses
    # mentioned in the text.
    def our_water_mass_av(w):
        idx = ((reg_av_w.water_mass.str.contains(w)) &
               (reg_av_w.y_var=='AOU_RES') &
               (reg_av_w.x_tracer.isin(ages_res)))
        return reg_av_w.loc[idx, ['y_var', 'x_tracer', 'slope_mean', 'slope_sd']]
    
    our_water_mass_av('SPEW')
    our_water_mass_av('WSACW')
    our_water_mass_av('IEW')
    our_water_mass_av('PSUW')
    our_water_mass_av('LSW')
    our_water_mass_av('UNADW')
    
    
    #%% EXPORT
    
    # Export result table
    reg_ss1 = reg.loc[(reg.x_tracer.isin(ages_res)) &
                      (reg.y_var=='AOU_RES'), :]
    fpath = 'deriveddata/hansell_glodap/global/OURs.csv'
    reg_ss1.to_csv(fpath, sep=',', na_rep='-9999', header=True, index=False)

    
    ## Export pixel polygons
    for w in pixel_polys:
        for p in pixel_polys[w]:
            
            fpath = Path('deriveddata/hansell_glodap/global/regression_pixel_polys') / f'{w},{p}_polygon.geojson'
            fpath.write_text(to_geojson(pixel_polys[w][p]))
    
