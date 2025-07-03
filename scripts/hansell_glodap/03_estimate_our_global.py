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
from shapely import geometry, from_geojson, polygonize, to_geojson
import warnings
import datetime as dt
from pathlib import Path

# plotting & mapping
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeoaxes
from scipy.stats import gaussian_kde, kruskal, spearmanr

# regressions
import statsmodels.formula.api as smf
import scripts.modules.RegressConsensus as rc


#%% DIRECTORIES FOR FIGURES

ds = ['figures/hansell_glodap/global/',
      'figures/hansell_glodap/global/helper/',
      'figures/hansell_glodap/global/helper/latitudinal/',
      'figures/hansell_glodap/global/regressions/',
      'figures/hansell_glodap/global/rates/',
      'figures/hansell_glodap/global/rates/profiles/',
      'figures/hansell_glodap/global/rates/maps/',
      'figures/hansell_glodap/global/stoichiometry/profiles/',
      'figures/hansell_glodap/global/stoichiometry/maps/',
      'deriveddata/hansell_glodap/global/',
      'deriveddata/hansell_glodap/global/regression_pixel_polys/']
for d in ds:
    if not os.path.exists(d):
        os.makedirs(d)
    
    
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
    fpath = [*pathlib.Path(str(d) + "\\ocean").glob("*.geojson")][0]
    
    # Read and store it
    o = str(d).split("\\")[-1]
    ocean_polys[o] = from_geojson(fpath.read_text())
    

## Water masses

wm_polys = {}
wm_polys_plot = {} # load uncut version of polygons too (for mapping)
wm_depths = ['central', 'intermediate', 'deep']
for d in dlist:
    
    o = str(d).split("\\")[-1]
    wm_polys[o] = {}
    wm_polys_plot[o] = {}
    
    for z in wm_depths:
        
        # Get wm paths at depth z and ocean d
        flist = [*pathlib.Path(str(d) + "\\wms\\" + z).glob("*.geojson")]
        
        # Skip iteration if certain depth is absent (i.e. flist is empty)
        # (to skip 'deep' in Indian and Pacific)
        if not flist: continue
        
        wm_polys[o][z] = {}
        wm_polys_plot[o][z] = {}
        for f in flist:
            
            # Get wm name (accounts for when the name itself has underscore, 
            # e.g. AAIW_P)            
            w = "_".join(str(f).split("\\")[-1].split("_")[0:-1])

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
                
                # Set side lines of pixel
                lo1 = lo[0]
                lo2 = lo[1]
                la1 = la[0]
                la2 = la[1]
                px_borders = [geometry.LineString([(lo1, la1), (lo1, la2)]),
                              geometry.LineString([(lo2, la1), (lo2, la2)]),
                              geometry.LineString([(lo1, la1), (lo2, la1)]),
                              geometry.LineString([(lo1, la2), (lo2, la2)])]
                    
                # Create pixel polygon
                pxp = polygonize(px_borders)
            
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
    norm = mpl.colors.Normalize(vmin=1, vmax=len(pixel_polys[k]))
    
    # Map water mass pixels with distinct colour just to aid visualisation
    for ip, p in enumerate(pixel_polys[k]):
        ax_px.add_geometries(pixel_polys[k][p],
                             facecolor=px_pal(norm(ip + 1)),
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
np.random.seed(1934)
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
        
        offset = np.random.choice([-1, 1], size=1)[0] * .001
        rlon = r['LONGITUDE'] + offset if (r['LONGITUDE'] / 10).is_integer() else r['LONGITUDE']
        rlat = r['LATITUDE'] + offset if (r['LATITUDE'] / 10).is_integer() else r['LATITUDE']
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
        norm = mpl.colors.Normalize(vmin=1, vmax=len(pixel_polys[k]))
        
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
                          facecolor=px_pal(norm(ip + 1)),
                          transform=ccrs.PlateCarree(),
                          zorder=2)
            # Map pixel
            ax_px.add_geometries(pixel_polys[k][p],
                                 facecolor='none',
                                 edgecolor=px_pal(norm(ip + 1)),
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


# Compute squared PT and S
tbl['PT2'] = tbl['PT'] ** 2
tbl['S2'] = tbl['CTD_SALINITY'] ** 2

# Set variables for which residuals need to be estimated
vrs = ['OXYGEN', 'AOU', 'DOC',
       'AGE_CFC11', 'AGE_CFC12', 'AGE_SF6',
       'NITRATE', 'PHOSPHATE', 'SILICIC_ACID']
vrs_flags = {'OXYGEN': 'OXYGEN_FLAG_W',
             'AOU': 'OXYGEN_FLAG_W',
             'DOC': 'DOC_FLAG_W',
             'AGE_CFC11': 'CFC_11_FLAG_W',
             'AGE_CFC12': 'CFC_12_FLAG_W',
             'AGE_SF6': 'SF6_FLAG_W',
             'NITRATE': 'NITRATE_FLAG_W',
             'PHOSPHATE': 'PHOSPHATE_FLAG_W',
             'SILICIC_ACID': 'SILICIC_ACID_FLAG_W'}

for v in vrs:
    
    # Create new column for residual variable
    vres = v + '_RES'
    tbl[vres] = np.nan
    
    for k in wm_polys_flat:
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
            
        print(v + " ~ " + w)
        # Subset samples of water mass w, with valid flags
        boo = ((tbl['WATER_MASS']==w) &
               (tbl[vrs_flags[v]].isin([0, 2, 6])) &
               (~np.isnan(tbl[v])))
        ss = tbl.loc[boo, :].copy()
            
        # Perform corrections separately for each cruise
        uc = ss.EXPOCODE.unique()
        for c in uc:
            
            # Subset samples for cruise c
            ssc = ss.loc[ss.EXPOCODE==c, :]
            
            # If there is enough data for v in b in w in c
            # (at least 5 non-nan values)
            min_obs = 5
            if sum(~np.isnan(ssc[v]))>=min_obs:
                
                # Perform regression
                frml = v + ' ~ CTD_SALINITY + PT + S2 + PT2'
                md1 = smf.ols(frml, data=ssc).fit()
                
                # Introduce residual values in table using the index
                tbl.loc[md1.resid.index, vres] = md1.resid.astype('float64')


#%% // OPTIONAL //

# Plot variables (and residuals) against latitude (to aid a better 
# understanding of the data)

plot_vars_vs_lat = False

if plot_vars_vs_lat:

    #%%% PLOT VARIABLES AGAINST LATITUDE
    
    # Plot selected variables within each water mass agains latitude
    
    # Select variables
    vrs = ['OXYGEN', 'AOU', 'DOC', 'NITRATE', 'AGE_CFC11', 'AGE_CFC12', 'AGE_SF6']
    vrs_flags = {'OXYGEN': 'OXYGEN_FLAG_W',
                 'AOU': 'OXYGEN_FLAG_W',
                 'DOC': 'DOC_FLAG_W',
                 'NITRATE': 'NITRATE_FLAG_W',
                 'AGE_CFC11': 'CFC_11_FLAG_W',
                 'AGE_CFC12': 'CFC_12_FLAG_W',
                 'AGE_SF6': 'SF6_FLAG_W'}
    
    v1_lab = ["O$_2$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "AOU [$\mathregular{\mu}$mol kg$^{-1}$]",
              "DOC [$\mathregular{\mu}$mol kg$^{-1}$]",
              "NO$_3^{-}$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "Age$_\mathregular{CFC\u201011}$ [y]",
              "Age$_\mathregular{CFC\u201012}$ [y]",
              "Age$_\mathregular{SF_6}$ [y]"]
    v1_lab = dict(zip(vrs, v1_lab)) 
    
    
    for ik, k in enumerate(pixel_polys):
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        if o=='mediterranean': continue
    
        print("Plotting " + w + " ...")
        
        # Subset data for water mass w
        ssw = tbl.loc[tbl.WATER_MASS==w, :]
        
        # Get extention of water mass and create segments based on it
        kys = pixel_polys[k].keys()
        kys_lonmin = pd.Series([float(x.split(";")[0]) for x in kys])
        lonr = pd.Series(lonlines[o])
        lonr = lonr[pd.Series([x[0] for x in lonlines[o]]).isin(kys_lonmin)]
        latmin = min([float(x.split(";")[1]) for x in kys])
        latmax = max([float(x.split(";")[3]) for x in kys])
        latr = np.arange(latmin, latmax, 5)
        latr = [(la1, la2) for la1, la2 in zip(latr[:-1], latr[1:])]
        if o=='pacific':
            # Prune and reorder lons so that geographically it makes sense 
            # (for Pacific)
            lonr = pd.Series(lonr)
            lonr = lonr[lonr.isin(pd.Series(lonlines[o]))]
            lonr = pd.concat((lonr[[x[0]>=0 for x in lonr]],
                              lonr[[x[0]<0 for x in lonr]])).to_list()
        if w=='MW':
            # Special case for MW
            lonr = [(float(kys_lonmin.iloc[0]), 
                     float(list(kys)[0].split(";")[2]))]
            
        
        # Initiate plot
        nr = len(vrs)
        nc = len(lonr)
        fig_s, ax_s = plt.subplots(nrows=nr, ncols=nc,
                                   figsize=(5*cm*nc, 3.2*cm*nr),
                                   squeeze=False)
        fig_s.suptitle("$" + w + "$",
                       fontsize=15, fontweight='bold',
                       x=.1, y=.93, ha='left')
        
        delta_lat = np.max(latr) - np.min(latr)
        xmin = np.min(latr) - delta_lat * .05
        xmax = np.max(latr) + delta_lat * .05
        for iv, v in enumerate(vrs):
            
            vmin = np.min(ssw.loc[ssw[vrs_flags[v]].isin([0, 2, 6]), v])
            vmax = np.max(ssw.loc[ssw[vrs_flags[v]].isin([0, 2, 6]), v])
            delta_var = vmax - vmin
            vmin = vmin - delta_var * .05
            vmin = vmin if vmin > 0 else 0
            vmax = vmax + delta_var * .05
            vmax = vmax if not np.isnan(vmax) else 1
            for ilo, lo in enumerate(lonr):
                
                # Customise subplot axes etc.
                ax_s[iv, ilo].set(xlim=[xmin, xmax],
                                  ylim=[vmin, vmax])
                ax_s[iv, ilo].xaxis.set_minor_locator(mticker.MultipleLocator(5))
                ax_s[iv, ilo].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax_s[iv, ilo].tick_params(which='major', axis='both',
                                          labelsize=5, pad=2,
                                          direction='in', length=2.5,
                                          top=True, right=True)
                ax_s[iv, ilo].tick_params(which='minor', axis='both',
                                          direction='in', length=1.5,
                                          top=True, right=True)
                if ilo==0:
                    ax_s[iv, ilo].set_ylabel(v1_lab[v], fontsize=6, labelpad=1.5)
                if iv==(len(vrs)-1):
                    ax_s[iv, ilo].set_xlabel('Latitude [$\degree$N]',
                                             fontsize=6,
                                             labelpad=2)
                    
                # add small title on top of each column
                if iv==0:
                    ax_s[iv, ilo].set_title(str(lo),
                                            fontsize=7,
                                            fontweight='bold',
                                            c='#777')
                
                # Add data points for each longitude segment
                ssl = ssw.loc[(ssw.LONGITUDE >= lo[0]) &
                              (ssw.LONGITUDE < lo[1]) &
                              (ssw[vrs_flags[v]].isin([0, 2, 6])) &
                              (~np.isnan(ssw[v])), :]
                # adding np.isnan is relevant for ages because there might be valid
                # measurements but when estimating age it might be NaN because
                # it fell out of the accepted date range. There are some cases
                # (e.g. UNADW (10,20) for SF6) where that is true for all samples,
                # and that needs to be considered as "no data" and be skipped.
                
                if ssl.empty:
                    ax_s[iv, ilo].text(.5, .5, "No data",
                                       size=6, fontweight='bold',
                                       c='#999',
                                       ha='center', va='center',
                                       transform=ax_s[iv, ilo].transAxes)
                    continue
                
                ax_s[iv, ilo].scatter(ssl.LATITUDE, ssl[v],
                                      marker='o', s=2, c='#ccc',
                                      zorder=0)
                
                for la in latr:
                    
                    # Estimate average and sd values for each latitude segment
                    
                    # Subset data for water mass w in the º5 range
                    ssv = ssl.loc[(ssl.LATITUDE >= la[0]) &
                                  (ssl.LATITUDE < la[1]), v].copy()
                    
    
                    # Estimate mean and sd
                    if (len(ssv)==0) | (all(np.isnan(ssv))):
                        # if latr increment < px_h could be no data in latr segment
                        # but pixels exists (e.g., latr each 5, px_h=10, one 
                        # 5-segment could have no data and the other yes)
                        ssv_mean = np.nan
                        ssv_sd = np.nan
                    else:
                        ssv_mean = np.nanmean(ssv)
                        ssv_sd = np.nanstd(ssv)
                    
    
                    # show 5-deg averages with sd bars
                    ax_s[iv, ilo].errorbar(np.mean(la), ssv_mean,
                                           ssv_sd,
                                           c='none', # do not plot lines
                                           marker='o',
                                           markersize=4,
                                           markerfacecolor='w', 
                                           markeredgecolor='#222',
                                           markeredgewidth=1,
                                           ecolor='#222',
                                           elinewidth=1,
                                           zorder=1)   
                    
        fpath = ('figures/hansell_glodap/global/helper/latitudinal/hansell_glodap_' + w +
                 '_latitudinal_data.pdf')
        fig_s.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
        plt.close(fig_s)
    
    #%%% PLOT RESIDUALS AGAINST LATITUDE
    
    # Select variables to include in plotS
    vrs_res = ['OXYGEN_RES', 'AOU_RES', 'DOC_RES', 'NITRATE_RES', 
               'AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
    v2_lab = ["O$_{2\Delta}$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "AOU$_\Delta$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "DOC$_\Delta$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "NO$_3^{-}$$_\Delta$ [$\mathregular{\mu}$mol kg$^{-1}$]",
              "Age$_\mathregular{CFC\u201011}$$_\Delta$ [y]",
              "Age$_\mathregular{CFC\u201012}$$_\Delta$ [y]",
              "Age$_\mathregular{SF_6}$$_\Delta$ [y]"]
    v2_lab = dict(zip(vrs_res, v2_lab)) 
    
    
    
    for ik, k in enumerate(pixel_polys):
        
        # Split water mass key
        o = k.split(";")[0] # ocean
        d = k.split(";")[1] # depth layer
        w = k.split(";")[2] # water mass code
        
        if o=='mediterranean': continue
    
        print("Plotting " + w + " ...")
        
        # Subset data for water mass w
        ssw = tbl.loc[tbl.WATER_MASS==w, :]
        
        # Get extention of water mass and create segments based on it
        kys = pixel_polys[k].keys()
        kys_lonmin = pd.Series([float(x.split(";")[0]) for x in kys])
        lonr = pd.Series(lonlines[o])
        lonr = lonr[pd.Series([x[0] for x in lonlines[o]]).isin(kys_lonmin)]
        latmin = min([float(x.split(";")[1]) for x in kys])
        latmax = max([float(x.split(";")[3]) for x in kys])
        latr = np.arange(latmin, latmax, 5)
        latr = [(la1, la2) for la1, la2 in zip(latr[:-1], latr[1:])]
        if o=='pacific':
            # Prune and reorder lons so that geographically it makes sense 
            # (for Pacific)
            lonr = pd.Series(lonr)
            lonr = lonr[lonr.isin(pd.Series(lonlines[o]))]
            lonr = pd.concat((lonr[[x[0]>=0 for x in lonr]],
                              lonr[[x[0]<0 for x in lonr]])).to_list()
        if w=='MW':
            # Special case for MW
            lonr = [(float(kys_lonmin.iloc[0]), 
                     float(list(kys)[0].split(";")[2]))]
            
        
        # Initiate plot
        nr = len(vrs_res)
        nc = len(lonr)
        fig_r, ax_r = plt.subplots(nrows=nr, ncols=nc,
                                   figsize=(5*cm*nc, 3.2*cm*nr),
                                   squeeze=False)
        fig_r.suptitle("$" + w + "$",
                       fontsize=15, fontweight='bold',
                       x=.1, y=.93, ha='left')
        
        delta_lat = np.max(latr) - np.min(latr)
        xmin = np.min(latr) - delta_lat * .05
        xmax = np.max(latr) + delta_lat * .05
        for iv, v in enumerate(vrs_res):
            
            vmin = np.min(ssw.loc[~np.isnan(ssw[v]), v])
            vmax = np.max(ssw.loc[~np.isnan(ssw[v]), v])
            delta_var = vmax - vmin
            vmin = vmin - delta_var * .05
            vmin = vmin if not np.isnan(vmin) else -1
            vmax = vmax + delta_var * .05
            vmax = vmax if not np.isnan(vmax) else 1
            for ilo, lo in enumerate(lonr):
                
                # Customise subplot axes etc.
                ax_r[iv, ilo].set(xlim=[xmin, xmax],
                                  ylim=[vmin, vmax])
                ax_r[iv, ilo].xaxis.set_minor_locator(mticker.MultipleLocator(5))
                ax_r[iv, ilo].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax_r[iv, ilo].tick_params(which='major', axis='both',
                                          labelsize=5, pad=2,
                                          direction='in', length=2.5,
                                          top=True, right=True)
                ax_r[iv, ilo].tick_params(which='minor', axis='both',
                                          direction='in', length=1.5,
                                          top=True, right=True)
                if ilo==0:
                    ax_r[iv, ilo].set_ylabel(v2_lab[v], fontsize=6, labelpad=1.5)
                if iv==(len(vrs)-1):
                    ax_r[iv, ilo].set_xlabel('Latitude [$\degree$N]',
                                             fontsize=6,
                                             labelpad=2)
                    
                # add small title on top of each column
                if iv==0:
                    ax_r[iv, ilo].set_title(str(lo),
                                            fontsize=7,
                                            fontweight='bold',
                                            c='#777')
                
                # Add data points for each longitude segment
                ssl = ssw.loc[(ssw.LONGITUDE >= lo[0]) &
                              (ssw.LONGITUDE < lo[1]) &
                              (~np.isnan(ssw[v])), :]
                # flags were accounted for when estimating residuals
                
                if ssl.empty:
                    ax_r[iv, ilo].text(.5, .5, "No data",
                                       size=6, fontweight='bold',
                                       c='#999',
                                       ha='center', va='center',
                                       transform=ax_r[iv, ilo].transAxes)
                    continue
                
                ax_r[iv, ilo].scatter(ssl.LATITUDE, ssl[v],
                                      marker='o', s=2, c='#ccc',
                                      zorder=0)
                # add reference line for 0
                ax_r[iv, ilo].axhline(0, c='#bbb', linestyle=':', linewidth=.7,
                                      zorder=1)
                
                for la in latr:
                    
                    # Estimate average and sd values for each latitude segment
                    
                    # Subset data for water mass w in the º5 range
                    ssv = ssl.loc[(ssl.LATITUDE >= la[0]) &
                                  (ssl.LATITUDE < la[1]), v].copy()
                    
    
                    # Estimate mean and sd
                    if (len(ssv)==0) | (all(np.isnan(ssv))):
                        # if latr increment < px_h could be no data in latr segment
                        # but pixels exists (e.g., latr each 5, px_h=10, one 
                        # 5-segment could have no data and the other yes)
                        ssv_mean = np.nan
                        ssv_sd = np.nan
                    else:
                        ssv_mean = np.nanmean(ssv)
                        ssv_sd = np.nanstd(ssv)
                    
    
                    # show 5-deg averages with sd bars
                    ax_r[iv, ilo].errorbar(np.mean(la), ssv_mean,
                                           ssv_sd,
                                           c='none', # do not plot lines
                                           marker='o',
                                           markersize=4,
                                           markerfacecolor='w', 
                                           markeredgecolor='#222',
                                           markeredgewidth=1,
                                           ecolor='#222',
                                           elinewidth=1,
                                           zorder=1)   
                    
        fpath = ('figures/hansell_glodap/global/helper/latitudinal/hansell_glodap_' + w +
                 '_latitudinal_data_residuals.pdf')
        fig_r.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
        plt.close(fig_r)
    


#%% REGRESSIONS: OXYGEN, DOC UTILISATION RATES

# Perform linear regressions with residuals to estimate OUR and DOCUR. Do so
# individually for each water mass pixel.
#
# ALSO: use this loop to estimate stoichiometric relationships.


#-------------------

# When estimating the regressions, skip plots?
# (doing and exporting plots increases running time from like 5 min to multiple
#  hours, ~10h, so it is advisable to skip them if working in the 
#  script/testing)
skip_plots = False # True | False

#-------------------


# Set the x variables (including tracer ages) to iterate
xvrs = ['AGE_CFC11_RES',    # for rates
        'AGE_CFC12_RES',    # for rates
        'AGE_SF6_RES',      # for rates
        'NITRATE_RES',      # for stoichiometry
        'PHOSPHATE_RES',    # for stoichiometry
        'SILICIC_ACID_RES', # for stoichiometry
        'DOC_RES'           # for stoichiometry
        ]

# And the dependent variables
vres = ['OXYGEN_RES', 'AOU_RES', 'DOC_RES', 'NITRATE_RES']


# Set labels
xvrs_labs = ["Age$_\mathregular{CFC\u201011}$$_\Delta$ [y]",
             "Age$_\mathregular{CFC\u201012}$$_\Delta$ [y]",
             "Age$_\mathregular{SF_6}$$_\Delta$ [y]",
             "Nitrate$_\Delta$ [$\mu$mol kg$^{-1}$]",
             "Phosphate$_\Delta$ [$\mu$mol kg$^{-1}$]",
             "Silicic acid$_\Delta$ [$\mu$mol kg$^{-1}$]",
             "DOC$_\Delta$ [$\mu$mol kg$^{-1}$]"]
xvrs_labs = dict(zip(xvrs, xvrs_labs))
vres_labs = ["O$_{2\Delta}$ [$\mu$mol kg$^{-1}$]",
             "AOU$_\Delta$ [$\mu$mol kg$^{-1}$]",
             "DOC$_\Delta$ [$\mu$mol kg$^{-1}$]",
             "Nitrate$_\Delta$ [$\mu$mol kg$^{-1}$]"]
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


# Set variables to store results etc.        
reg = []
reg_obs = []
skipped_instances = []
land = cfeature.NaturalEarthFeature('physical', 'land', '110m')
start_time = dt.datetime.now()

for v in vres:
    for x in xvrs:
        1
        # Note that at some point vres==xvrs (for DOC and NITRATE), but leave
        # it as is a do no skip, as a way to check that regressions are done
        # properly (such cases must return perfect 1:1 relationships)
        
        # Iterate through each water mass
        for k1 in pixel_polys:
            
            # Split water mass key
            o = k1.split(";")[0] # ocean
            d = k1.split(";")[1] # depth layer
            w = k1.split(";")[2] # water mass code
            
            # Initiate plot for water mass w
            n_px = len(pixel_polys[k1])
            nc = nr = int(np.ceil(n_px ** .5))
            fig_reg = plt.figure(figsize=(5*cm*nc, 3.5*cm*nr))
            
            # Iterate through pixels within each water mass
            for ik2, k2 in enumerate(pixel_polys[k1]):
                
                # Create pixel key
                px_k = k1 + ";" + k2
                
                # Subset correspoding data, only samples with valid values for
                # both target variables of the regression
                idx = ((tbl.PIXEL==px_k) &
                       (~np.isnan(tbl[v])) &
                       (~np.isnan(tbl[x])))
                ssp = tbl.loc[idx, :]
                
                # set minimum number of observations required to do regression
                min_obs = 5
                nobs = ssp.shape[0]
                

                #### Do linear regression
                
                # Select independent and dependent variables
                X = ssp[x]
                Y = ssp[v]

                # Skip variable pair if no data available OR less than min_obs
                # keep track of skipped
                if nobs < min_obs:
                    msg = (v + ", " + x + ", " + px_k +
                           ": not enough observations (" + str(nobs) +
                           ", at least " + str(min_obs) + " required)")
                    skipped_instances.append(msg)
                    print(msg)

                else: # otherwise do regression
                    
                    # Perform linear regression (model II)
                    md2 = rc.RegressConsensusW(X, Y, Wx=.5)
                    
                    # Extract and store results
                    # NOTE THAT np.nanmean() behaves correctly when all 
                    # elements are nan (returns a nan), but still displays a 
                    # warning. It has no use here, so supress it to avoid 
                    # confussion.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        tmp = [v,    # dependent variable
                               x,    # independent variable (age or other)
                               k1,   # water mass
                               k2,   # pixel
                               px_k, # pixel key
                               md2['slope'],     # slope (OUR / DOCUR)
                               md2['spvalue'],   # slope p-value
                               md2['sse'],       # slope standard error
                               md2['sci95'],     # slope 95% confidence interval
                               md2['intercept'], # intercept
                               md2['ipvalue'],   # intercept p-value
                               md2['ise'],       # intercept standard error
                               md2['ici95'],     # intercept 95% confidence interval
                               md2['r2'],        # coefficient of determination, R^2
                               md2['r2_adj'],    # adjusted R2
                               md2['n'],         # number of samples in the regression
                               np.nanmean(ssp['LONGITUDE']),     # mean actual lon of samples
                               np.nanstd(ssp['LONGITUDE']),      # sd of lon of samples
                               np.nanmean(ssp['LATITUDE']),      # mean actual lat of samples
                               np.nanstd(ssp['LATITUDE']),       # sd of lat of samples
                               np.nanmean(ssp['CTD_PRESSURE']),  # mean depth of samples
                               np.nanstd(ssp['CTD_PRESSURE']),   # sd of depth of samples
                               np.nanmean(ssp['PT']),            # etc.
                               np.nanstd(ssp['PT']),
                               np.nanmean(ssp['SIGMA0']),
                               np.nanstd(ssp['SIGMA0']),
                               np.nanmean(ssp['OXYGEN']),
                               np.nanstd(ssp['OXYGEN']),
                               np.nanmean(ssp['OXYGEN_RES']),
                               np.nanstd(ssp['OXYGEN_RES']),
                               np.nanmean(ssp['AOU']),
                               np.nanstd(ssp['AOU']),
                               np.nanmean(ssp['AOU_RES']),
                               np.nanstd(ssp['AOU_RES']),
                               np.nanmean(ssp['DOC']),
                               np.nanstd(ssp['DOC']),
                               np.nanmean(ssp['DOC_RES']),
                               np.nanstd(ssp['DOC_RES']),
                               np.nanmean(ssp['NITRATE']),
                               np.nanstd(ssp['NITRATE']),
                               np.nanmean(ssp['NITRATE_RES']),
                               np.nanstd(ssp['NITRATE_RES']),
                               np.nanmean(ssp['PHOSPHATE']),
                               np.nanstd(ssp['PHOSPHATE']),
                               np.nanmean(ssp['PHOSPHATE_RES']),
                               np.nanstd(ssp['PHOSPHATE_RES']),
                               np.nanmean(ssp['SILICIC_ACID']),
                               np.nanstd(ssp['SILICIC_ACID']),
                               np.nanmean(ssp['SILICIC_ACID_RES']),
                               np.nanstd(ssp['SILICIC_ACID_RES']),
                               np.nanmean(ssp['NPP_EPPL']),
                               np.nanstd(ssp['NPP_EPPL']),
                               np.nanmean(ssp['NPP_CBPM']),
                               np.nanstd(ssp['NPP_CBPM']),
                               np.nanmean(ssp['B']),
                               np.nanstd(ssp['B']),
                               np.nanmean(ssp['POC3D']),
                               np.nanstd(ssp['POC3D']),
                               np.nanmean(ssp['AGE_CFC11']),
                               np.nanstd(ssp['AGE_CFC11']),
                               np.nanmean(ssp['AGE_CFC11_RES']),
                               np.nanstd(ssp['AGE_CFC11_RES']),
                               np.nanmean(ssp['AGE_CFC12']),
                               np.nanstd(ssp['AGE_CFC12']),
                               np.nanmean(ssp['AGE_CFC12_RES']),
                               np.nanstd(ssp['AGE_CFC12_RES']),
                               np.nanmean(ssp['AGE_SF6']),
                               np.nanstd(ssp['AGE_SF6']),
                               np.nanmean(ssp['AGE_SF6_RES']),
                               np.nanstd(ssp['AGE_SF6_RES'])]
                    
                    # Append to previous
                    reg.append(tmp)
                
                if skip_plots:
                    continue
                
                print("Plotting " + v + " vs. " + x + " (" + px_k + ")")
                #### Plot results
                
                # Initiate subplot for v and x regression in px_k
                # fig_reg = plt.figure(figsize=(5*cm*nc, 3.5*cm*nr))
                ax_reg = fig_reg.add_subplot(nr, nc, ik2 + 1)
                
                
                ## Add data points and regression result
                if nobs < min_obs:
                    
                    # Include clarification if not enough data
                    ax_reg.scatter(X, Y,
                                   marker='o',
                                   s=7,
                                   linewidth=.2,
                                   facecolor='#444',
                                   edgecolor='#222',
                                   zorder=0)
                    ax_reg.text(.5, .55,
                                "Not enough data (" + str(nobs) + ")",
                                size=6, fontweight='bold',
                                c='#999',
                                ha='center', va='center',
                                transform=ax_reg.transAxes,
                                zorder=1)

                else:
                    ax_reg.scatter(X, Y,
                                   marker='o',
                                   s=7,
                                   linewidth=.2,
                                   facecolor='#444',
                                   edgecolor='#fff',
                                   zorder=0)
                    # If regression was significant, add line
                    if md2['spvalue']<.05:
                        x0 = np.nanmin(X)
                        x1 = np.nanmax(X)
                        y0 = md2['intercept'] + x0 * md2['slope']
                        y1 = md2['intercept'] + x1 * md2['slope']
                        ax_reg.plot([x0, x1], [y0, y1],
                                    c='#444', lw=1.2,
                                    zorder=1)
                    # Add regression parameters
                    txt_slope = ("Slope = " + str(round(md2['slope'], 2)) +
                                 " ± " + str(round(md2['sci95'], 2)))
                    txt_spval = p_label(md2['spvalue'])
                    txt_r2 = ("$R^2$ = " + '{:.3f}'.format(md2['r2']))
                    ax_reg.text(.98, .84,
                                txt_slope,
                                size=3,
                                c='#555',
                                ha='right', va='bottom',
                                transform=ax_reg.transAxes,
                                zorder=1)
                    ax_reg.text(.98, .76,
                                txt_spval,
                                size=3,
                                c='#555',
                                ha='right', va='bottom',
                                transform=ax_reg.transAxes,
                                zorder=1)
                    ax_reg.text(.98, .68,
                                txt_r2,
                                size=3,
                                c='#555',
                                ha='right', va='bottom',
                                transform=ax_reg.transAxes,
                                zorder=1)
                    # Include line and text for slope, slope pval and R2
                    
                
                ## Customise axes
                
                # Make space for inset map
                if nobs==0:
                    xmin = -1
                    xmax = +1
                elif nobs==1:
                    xmin = X.iat[0] - 1
                    xmax = X.iat[0] + 1
                else:
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
            
            if skip_plots:
                continue
            
            ptitle = (w + ": " + vres_labs[v].split(" ")[0] + 
                      " vs. " + xvrs_labs[x].split(" ")[0])
            fig_reg.suptitle(ptitle, x=.5, y = .92, weight='bold', color='#222')

            fpath = ('figures/hansell_glodap/global/regressions/' +
                     'hansell_glodap_regressions_' +
                     v + '_' + x + '_' + w + '.pdf')
            fig_reg.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
            
            plt.close(fig_reg)
                


# Convert reg into dataframe

nms = ['y_var', 'x_tracer', 'water_mass', 'pixel', 'pixel_key', 
       'slope', 'spvalue', 'sse', 'sci95',
       'intercept', 'ipvalue', 'ise', 'ici95',
       'r2', 'r2_adj', 'n',
       'LONGITUDE', 'LONGITUDE_SD', 'LATITUDE', 'LATITUDE_SD',
       'CTD_PRESSURE', 'CTD_PRESSURE_SD',
       'PT', 'PT_SD', 'SIGMA0', 'SIGMA0_SD',
       'OXYGEN', 'OXYGEN_SD', 'OXYGEN_RES', 'OXYGEN_SD_RES',
       'AOU', 'AOU_SD', 'AOU_RES', 'AOU_RES_SD',
       'DOC', 'DOC_SD', 'DOC_RES', 'DOC_RES_SD',
       'NITRATE', 'NITRATE_SD', 'NITRATE_RES', 'NITRATE_RES_SD',
       'PHOSPHATE', 'PHOSPHATE_SD', 'PHOSPHATE_RES', 'PHOSPHATE_RES_SD',
       'SILICIC_ACID', 'SILICIC_ACID_SD', 'SILICIC_ACID_RES', 'SILICIC_ACID_RES_SD',
       'NPP_EPPL', 'NPP_EPPL_SD', 'NPP_CBPM', 'NPP_CBPM_SD',
       'B', 'B_SD',
       'POC3D', 'POC3D_SD',
       'AGE_CFC11', 'AGE_CFC11_SD', 'AGE_CFC11_RES', 'AGE_CFC11_RES_SD',
       'AGE_CFC12', 'AGE_CFC12_SD', 'AGE_CFC12_RES', 'AGE_CFC12_RES_SD',
       'AGE_SF6', 'AGE_SF6_SD', 'AGE_SF6_RES', 'AGE_SF6_RES_SD']
reg = pd.DataFrame(reg, columns=nms)

end_time = dt.datetime.now()
print('Duration: {}'.format(end_time - start_time))


#%% SET QUALITY OF REGRESSION RESULTS

# for AOU rates with values < 0 (for OXYGEN those > 0)
# do not make sense biologically...* but there are only a few outliers.
# Get the actual number of such outliers. 
#
# * Could be a sporious result due to scarce data or some left over influence
# of water mass mixing (with some input of a water mass with higher O2)

## Check range of results for AOURs
ages_res = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
idx = ((reg.y_var=='AOU_RES') & 
       (reg.x_tracer.isin(ages_res)))
all_aour = reg.loc[idx, :]

print("Negative aOURs are " + str(sum(all_aour.slope < 0)) +
      " out of " + str(len(all_aour)))


### Plot R2 distribution across all regressions

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


fpath = ('figures/hansell_glodap/global/regressions/_hist_regressions_R2.svg')
fig_hist.savefig(fpath, format='svg', bbox_inches='tight')
plt.close(fig_hist)
# plt.scatter(all_aour.r2, all_aour.n)



#------------------------------------------------------------------------------
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

#### Set  pval and R2 limits to accept regressions...

pval_limit = .001 # acceptable if lower than
r2_limit = .15   # acceptable if greater than

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
#------------------------------------------------------------------------------



# On top of R2 and p, also for AOU discard rates with values < 0; for OXYGEN 
# those > 0. Get the actual number of such outliers

## Check range of results when filtering based on the said criteria
idx = ((reg.y_var=='AOU_RES') & 
       (reg.x_tracer.isin(ages_res)) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit))
all_aour_f = reg.loc[idx, :]

neg_aour = all_aour_f.loc[all_aour_f.slope < 0, :]
len(neg_aour)


print("\nFiltering rates based on p-value and\nminimum R2 "
      "reduces aOURs from " + str(len(all_aour)) +
      " to " + str(len(all_aour_f)))

print("\nAfter filtering rates based on p-value and\nminimum R2, "
      "negative aOURs are " + str(len(neg_aour)) +
      " out of " + str(len(all_aour_f)) + 
      " (" + str(round(100*len(neg_aour)/len(all_aour_f), 1)) + " %)")


#%% PLOT RATES

#### Range of results among those positive

idx = ((reg.y_var=='AOU_RES') & 
       (reg.x_tracer.isin(ages_res)) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit) &
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
               (ss_joined.r2_aou > r2_limit) &
               (ss_joined.slope_aou >= 0))
        ss_k1 = ss_joined.loc[idx, :]
        
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
    ax_bx.set(xticks=range(0, len(ages_res)),
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
               (ss_joined.r2_aou > r2_limit) &
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
vres = ['OXYGEN_RES', 'AOU_RES', 'DOC_RES']
ages_res = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']

rate_labs = ["OUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]",
             "aOUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]",
             "DOCUR [$\mathregular{\mu}$mol L$^{-1}$ yr$^{-1}$]"]
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
                      "aOUR [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
                      ],
          'DOC_RES': [[-5, 5],
                      np.arange(-5, 7.5, 2.5),
                      1.25,
                      "aDOCUR [$\mathregular{\mu}$mol L$^{-1}$ yr$^{-1}$]"
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
                                   xerr=ss['sci95'],
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


#### Regressions against each other

comparisons = [['AGE_CFC11_RES', 'AGE_CFC12_RES'],
               ['AGE_CFC11_RES', 'AGE_SF6_RES'],
               ['AGE_CFC12_RES', 'AGE_SF6_RES']]
vres_lims = [[-40, 5], [-5, 40], [-6, 4]]

fig_com, ax_com = plt.subplots(nrows=len(vres), ncols=len(comparisons),
                               figsize=(5*cm * len(vres),
                                        5*cm * len(comparisons)))
for iv, v in enumerate(vres):
    for ic, c in enumerate(comparisons):
        
        # Subset data for rate with v, and tracer ages in c
        idx = ((reg['y_var']==v) &
               (reg.spvalue < pval_limit) &
               (reg.r2 > r2_limit))
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
               str(round(np.mean(r_01["slope_" + c[1]]), 1)) + " and " +
               c[0] + " rates = " +
               str(round(np.mean(r_01["slope_" + c[0]]), 1)))
        print(txt)
        txt = ("Thus, on average " + c[1] + " rates are " +
               str(round(100*np.mean(r_01["slope_" + c[1]])/np.mean(r_01["slope_" + c[0]]), 1)) +
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
            txt_slope = ("Slope = " + str(round(md_com['slope'], 2)) +
                         " ± " + str(round(md_com['sci95'], 2)))
            txt_spval = p_label(md_com['spvalue'])
            txt_r2 = ("$R^2$ = " + '{:.3f}'.format(md_com['r2']))
            ax_com[iv, ic].text(.05, .93,
                                txt_slope,
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
           (reg.r2 > r2_limit))
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
    txt_slope = ("Slope = " + str(round(md_com['slope'], 2)) +
                 " ± " + str(round(md_com['sci95'], 2)))
    txt_spval = p_label(md_com['spvalue'])
    txt_r2 = ("$R^2$ = " + '{:.3f}'.format(md_com['r2']))
    ax_com[ic].text(.05, .93,
                    txt_slope,
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
               (reg.r2 > r2_limit))
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
                            labels=list(ages_res_labs.values())
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
       (reg.r2 > r2_limit) &
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
                        labels=[ages_res_labs[a]]
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
       (reg.r2 > r2_limit))
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
                                       labels=[wm_labels_b[w]])
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
                               vert=False, labels=[wm_labels_b[w]])

            
        # Add text indicating number of data points?
        add_n = True
        if add_n & (len(vals) > 0):
            if ((a=="AGE_CFC12_RES") & (w=="SPEW")):
                xpos = .87
            # elif ((a=="AGE_CFC12_RES") & (w=="PSUW")):
            #     xpos = .24
            else:
                xpos = .98
            if len(vals)==1:
                txt = str('{:.1f}'.format(np.mean(vals)))
            else:
                txt = (str('{:.1f}'.format(np.mean(vals))) + " ± " + 
                       str('{:.1f}'.format(np.std(vals))))
                # " (" + str(len(vals)) + ")"
            ax_bxp[ia].text(xpos, ir, txt,
                            fontsize=3,
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
        ax_bxp[ia].set(xlim=[0, 45], xticks=range(0, 50, 10))
    
    # Add patches delimiting the water depth layers
    ax_bxp[ia].axhspan(ymin=-.5, ymax=16.5, facecolor='#f9f9f9')
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
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
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
        
    elif v=='OXYGEN_RES':
        
        # (Plot rates based on oxygen as absolute values, even if
        # technically negative.)
        cap_val1 = 0
        cap_val2 = 20 
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('YlGnBu')
        rnd = 1
        cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
        inc = np.ceil((cap_val2 - cap_val1) / 5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        ext = 'max'
        
    else:
         
        # For DOC, have both + and - values (make it symmetric)
        cap_val1 = -2
        cap_val2 = +2
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('coolwarm')
        rnd = 2
        cbar_lab = "DOCUR [$\mu$mol L$^{-1}$ yr$^{-1}$]"
        inc = round((cap_val2 - 0) / 2, 1)
        tcks = np.concatenate((-np.arange(0, cap_val2+inc, inc),
                               np.arange(0, cap_val2+inc, inc)))
        tcks = np.sort(np.unique(tcks))
        ext = 'both'

            
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
                           (r.r2 > r2_limit) &
                           (r.slope >= 0))
                elif v=='OXYGEN_RES':
                    boo = ((r.spvalue < pval_limit) & 
                           (r.r2 > r2_limit) &
                           (r.slope <= 0))
                else:
                    boo = ((r.spvalue < pval_limit) & 
                           (r.r2 > r2_limit))
                  
                if boo:
                    
                    # If conditions acceptable for pixel r:
                    # 
                    # Colour of pixel based on rate value
                    # 
                    # (Plot rates based on oxygen as absolute values, even if
                    # technically negative.)
                    val = abs(r.slope) if v=='OXYGEN_RES' else r.slope
                    pxc = px_pal(norm(val))
                    
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
    cbar = fig_m.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=px_pal),
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
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('YlGnBu')
        rnd = 1 
        cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
        inc = np.ceil((cap_val2-cap_val1)/5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        ext = 'max'
        
    elif v=='OXYGEN_RES':
     
        cap_val1 = 0
        cap_val2 = 20
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('YlGnBu')
        rnd = 1
        cbar_lab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
        inc = np.ceil((cap_val2-cap_val1)/5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        ext = 'max'
        
    else:
         
        cap_val1 = -2
        cap_val2 = +2
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = mpl.colormaps.get_cmap('coolwarm')
        rnd = 2
        cbar_lab = "DOCUR [$\mu$mol L$^{-1}$ yr$^{-1}$]"
        inc = round((cap_val2-0)/2, 1)
        tcks = np.concatenate((-np.arange(0, cap_val2+inc, inc),
                               np.arange(0, cap_val2+inc, inc)))
        tcks = np.sort(np.unique(tcks))
        ext = 'both'
        
            
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
                           (r.r2 > r2_limit) &
                           (r.slope >= 0))
                elif v=='OXYGEN_RES':
                    boo = ((r.spvalue < pval_limit) & 
                           (r.r2 > r2_limit) &
                           (r.slope <= 0))
                else:
                    boo = ((r.spvalue < pval_limit) & 
                           (r.r2 > r2_limit))
                    
                if boo:
                    
                    # Colour of pixel based on rate value
                    val = abs(r.slope) if v=='OXYGEN_RES' else r.slope
                    pxc = px_pal(norm(val))
                    
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
    cbar = fig_msi.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=px_pal), 
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
        (reg.r2 > r2_limit) &
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
       (reg.r2 > r2_limit) &
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
    norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
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
        pxc = px_pal(norm(val))
        
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
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=px_pal),
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
       (reg.r2 > r2_limit) &
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
       (reg.r2 > r2_limit) &
       (reg.slope >= 0))
ss = reg.loc[idx, :]
for a in ages_res:
    
    print("AAIWA, lats (-10, 10), " + a + " mean OUR: " + 
          str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))


#%% // OPTIONAL 2 //

#%%% PLOT STOICHIOMETRY

plot_stoichiometry = True

if plot_stoichiometry:

    #%%%% vs depth
    
    # Plot stoichiometric ratios, i.e., the slopes of the regressions of variables
    # against each other
    vres1 = ['OXYGEN_RES', 'AOU_RES', 'DOC_RES', 'NITRATE_RES']
    vres2 = ['DOC_RES', 'NITRATE_RES', 'PHOSPHATE_RES', 'SILICIC_ACID_RES']
    
    
    #------------------------------------------------------------------------------
    
    # Create labels like "VAR:VAR [mol:mol]"
    lb1 = ["O$_{2\Delta}$", "AOU$_\Delta$", "DOC$_\Delta$", "NO$_3^{-}$$_\Delta$"]
    lb1 = dict(zip(vres1, lb1))
    lb2 = ["DOC$_\Delta$", "NO$_3^{-}$$_\Delta$", 
           "PO$_4^{3-}$$_\Delta$", "SiO$_4$H$_4$$_\Delta$", ]
    lb2 = dict(zip(vres2, lb2))
    mol2 = "[mol:mol]"
    vres_labs = []
    vres_tit = []
    kys = []
    for v1 in vres1:
        for v2 in vres2:
            
            # create label
            lb = lb1[v1] + ":" + lb2[v2] + " " + mol2
            vres_labs.append(lb)
            
            # create title
            tt = lb1[v1] + ":" + lb2[v2]
            vres_tit.append(tt)
            
            # create key for dict
            ky = v1 + "_" + v2
            kys.append(ky)
            
    vres_labs = dict(zip(kys, vres_labs))
    vres_tit = dict(zip(kys, vres_tit))
    
    #------------------------------------------------------------------------------
    
    # Create reference stoichiometric ratios 
    # (based on Anderson & Sarmiento 1994, Anderson 1995)
    stoirat = [
               -1/0.72, # O2:DOC; not really useful, the ratio makes sense with DOC change
               -170/16, # O2:NITRATE
               -170/1,  # O2:PHOSPHATE
               np.nan,  # O2:SILICIC_ACID
               1/0.72,  # AOU:DOC; not really useful, the ratio makes sense with DOC change
               170/16,  # AOU:NITRATE
               170/1,   # AOU:PHOSPHATE
               np.nan,  # AOU:SILICIC_ACID
               1,       # DOC:DOC; PLACEHOLDER
               117/16,  # DOC:NITRATE
               117/1,   # DOC:PHOSPHATE
               np.nan,  # DOC:SILICIC_ACID
               16/117,  # NITRATE:DOC
               1,       # NITRATE:NITRATE; PLACEHOLDER
               16/1,    # NITRATE:PHOSPHATE
               1        # NITRATE:SILICIC_ACID
               ]
    stoirat = dict(zip(kys, stoirat))
    
    #------------------------------------------------------------------------------
    
    # Set y axis for each case
    yaxpar = {'OXYGEN_RES_DOC_RES': [[-40, 60],             # limits
                                     range(-40, 80, 20),    # major ticks
                                     10                     # minor ticks
                                     ],
              'OXYGEN_RES_NITRATE_RES': [[-15, -5], range(-15, -2, 3), 1.5],
              'OXYGEN_RES_PHOSPHATE_RES': [[-200, 200], range(-200, 300, 100), 50],
              'OXYGEN_RES_SILICIC_ACID_RES': [[-40, 5], range(-40, 10, 10), 5],
              'AOU_RES_DOC_RES': [[-50, 50], range(-50, 75, 25), 12.5],
              'AOU_RES_NITRATE_RES': [[0, 20], range(0, 24, 4), 2],
              'AOU_RES_PHOSPHATE_RES': [[-200, 200], range(-200, 300, 100), 50],
              'AOU_RES_SILICIC_ACID_RES': [[-5, 40], range(0, 50, 10), 5],
              'DOC_RES_DOC_RES': [[0, 2], np.arange(0, 2.5, .5), .25],
              'DOC_RES_NITRATE_RES': [[-10, 10], range(-10, 15, 5), 2.5],
              'DOC_RES_PHOSPHATE_RES': [[-100, 100], range(-100, 100, 50), 25],
              'DOC_RES_SILICIC_ACID_RES': [[-6, 6], range(-6, 9, 3), 1.5],
              'NITRATE_RES_DOC_RES': [[-6, 7], range(-6, 9, 3), 1.5],
              'NITRATE_RES_NITRATE_RES': [[0, 2], np.arange(0, 2.5, .5), .25],
              'NITRATE_RES_PHOSPHATE_RES': [[0, 25], range(5, 30, 5), 2.5],
              'NITRATE_RES_SILICIC_ACID_RES': [[-1, 6], range(0, 8, 2), 1],
              }
    
    #------------------------------------------------------------------------------
    
    # Set axis parameters for each variable. Borrow from previous and append label
    xaxpar = yaxpar # then y axis, now x axis in profiles
    for k in kys:
        xaxpar[k].append(vres_labs[k]) # access entry and append axis label
    
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
    
    #------------------------------------------------------------------------------
    
    
    for v1 in vres1:   
        
        nr = len(vres2)
        nc = len(depth_as)
        fig_ps, ax_ps = plt.subplots(nrows=nr, ncols=nc,
                                     figsize=(5*cm*nc, 6*cm*nr))
            
        for i2, v2 in enumerate(vres2):
            
            # set key of variable combination v1+v2
            ky = v1 + "_" + v2
        
            for iz, z in enumerate(depth_as):
            
                # unpack axis params for v and d
                xl, xtck, xtck_mi, xlab = xaxpar[ky]
                yl, ytck, ytck_mi, ylab = yaxpar[z]
            
                
                # subset data for tracer age g
                ss = reg.loc[(reg['y_var']==v1) &
                             (reg['x_tracer']==v2), :].copy()
                
                # plot only significant rates that have a reasonably good fit
                ss = ss.loc[(ss.spvalue<pval_limit) &
                            (ss.r2>r2_limit), :]
                
                # plot scatter and error bars
                ax_ps[i2, iz].errorbar(x=ss['slope'],
                                       y=ss[z],
                                       xerr=ss['sci95'],
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
                        
                ax_ps[i2, iz].set(xlim=xl, xticks=xtck, ylim=yl, yticks=ytck)
                ax_ps[i2, iz].xaxis.set_minor_locator(mticker.MultipleLocator(xtck_mi))
                ax_ps[i2, iz].yaxis.set_minor_locator(mticker.MultipleLocator(ytck_mi))
                ax_ps[i2, iz].tick_params(which='major', axis='both',
                                          labelsize=6, pad=2,
                                          direction='in', length=2.5,
                                          top=True, right=True)
                ax_ps[i2, iz].tick_params(which='minor', axis='both',
                                          direction='in', length=1.5,
                                          top=True, right=True)
                ax_ps[i2, iz].set_xlabel(xlab, fontsize=6.5, labelpad=2)
                ax_ps[i2, iz].set_ylabel(ylab, fontsize=6.5, labelpad=1.5)
                
                # add reference line for 0
                ax_ps[i2, iz].axvline(0, c='#999', linestyle=':', linewidth=1, zorder=1)
                
                # flip y axis
                ax_ps[i2, iz].invert_yaxis()
                
                
        # adjust spacing between subplots
        fig_ps.subplots_adjust(wspace=.5, hspace=.3)
        
        # save figure
        fpath = ('figures/hansell_glodap/global/stoichiometry/profiles/' +
                 'hansell_glodap_stoichiometry_profile_' + v1 + '.pdf')
        fig_ps.savefig(fpath, format='pdf', bbox_inches='tight')
        plt.close(fig_ps)
    
    
    
    #%%% MAP STOICHIOMETRY
    
    # Column labels
    columns_labs = {}
    
    nr = len(wm_depths_idx) + 1
    nc = len(vres2)
    
    for v1 in vres1:
        
        # Initialise plot
        mproj = ccrs.Mollweide(central_longitude=-160)
        fig_m2, ax_m2 = plt.subplots(nrows=nr, ncols=nc,
                                     figsize=(5*cm*nc, 3*cm*nr),
                                     subplot_kw={'projection': mproj})
    
        # Add land to each map (to avoid overplotting in each loop)
        for i in range(nr):
            for i2, v2 in enumerate(vres2):
                
                ky = v1 + "_" + v2
                
                ax_m2[i, i2].add_feature(cfeature.LAND, facecolor='#eee',
                                         edgecolor='k',
                                         linewidth=.1, zorder=1)
                # Add labels
                if i2==0:
                    if i in [0, 1, 2]: d_code = wm_depths[i]
                    depth_lab = wm_depths_lab[d_code] if i in [0, 1, 2] else ""
                    ax_m2[i, i2].text(-.1, .5, "$\mathbf{" + depth_lab + "}$",
                                      c="#444", size=5,
                                      rotation=90,
                                      va='center',
                                      transform=ax_m2[i, i2].transAxes)
                if i==0:
                    ax_m2[i, ig].text(.5, 1.07, lb2[v2],
                                      c="#444", size=6,
                                      ha='center',
                                      transform=ax_m2[i, i2].transAxes)
                    
        for i2, v2 in enumerate(vres2):
            
            # Subset regression data for variable v and tracer g
            idx = ((reg.y_var==v1) & (reg.x_tracer==v2)) 
            reg_ss = reg.loc[idx, :].copy()
            
            # Keep only significant regressions. This will be used to set colourmap
            # bounds.
            reg_ss = reg_ss.loc[(reg_ss.spvalue < pval_limit) & 
                                (reg_ss.r2 > r2_limit)]
            
            # Create normalising function to map values between [0,1] to
            # then assign to colourmap.
            # Share it across water masses so that colourmap limits are equal.
            # Cap it at .1 & .9 quantiles to avoid outliers distorting the 
            # colourmap range (round it to have nicer numbers at bounds)
            mgn = np.log10(np.std(reg_ss.slope)) if np.std(reg_ss.slope)>0 else 0
            if mgn>=1:
                rnd_lims = 0
            elif mgn>=0:
                rnd_lims = 1
            else:
                rnd_lims = int(abs(np.floor(mgn)))
                
            if v1==v2: # for NITRATE:NITRATE, this is just for checking
                cap_val1 = 0
                cap_val2 = 2
            else:
                cap_val1 = round(np.quantile(reg_ss.slope, .1), rnd_lims)
                cap_val2 = round(np.quantile(reg_ss.slope, .9), rnd_lims)
            
            norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
            px_pal = mpl.colormaps.get_cmap('viridis')
                
            # rounding precission for labs
            # rnd_labs = rnd_lims + 1
            
            # Colorbar label
            cbar_lab = lb1[v1] + ":" + lb2[v2]
            
            # Create colorbar tickmarks, ensure cap_val2 is included
            inc = round((cap_val2-cap_val1)/5, rnd_lims)
            tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                                   [cap_val2]))
            tcks = np.unique(tcks)
            
            # Go through water masses
            for ik1, k1 in enumerate(wm_polys_flat):
                
                # Split water mass key
                o = k1.split(";")[0] # ocean
                d = k1.split(";")[1] # depth layer
                w = k1.split(";")[2] # water mass code
                
                # Water masses will be group by depth layer in maps.
                # Get the corresponding map index for depth layer d
                j = wm_depths_idx[d]
                
                # Special case: to avoid overlaps, MW and SAIW will be mapped
                # separately
                if w in ['MW', 'SAIW']:
                    j = 3
                
                # Create subplot for water mass w
                # Overlay water mass border as reference
                ax_m2[j, i2].add_geometries(wm_polys_plot_flat[k1],
                                            facecolor='none',
                                            edgecolor='k',
                                            linewidth=.4,
                                            crs=ccrs.PlateCarree(),
                                            zorder=2)
                
                # Subset again regression results for (v1 vs v2) in water mass w
                # (see explantion in the %% MAP RATES section)
                idx = ((reg.y_var==v1) & (reg.x_tracer==v2)) 
                reg_ss_k1 = reg.loc[idx & (reg.water_mass==k1)].copy()
                
                # Iterate through results to plot the pixels with the rate values
                for ir, r in reg_ss_k1.iterrows():
                    
                    # If acceptable regression, give colour; otherwise empty
                    if (r.spvalue < pval_limit) & (r.r2 > r2_limit):
                        
                        # Colour of pixel based on rate value
                        pxc = px_pal(norm(r.slope))
                        
                        # Add filled pixel polygon
                        px = pixel_polys[k1][r.pixel]
                        pix_pol = ax_m2[j, i2].add_geometries(px,
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
                        pix_pol = ax_m2[j, i2].add_geometries(px,
                                                              facecolor=pxc,
                                                              edgecolor='#777',
                                                              linewidth=.1,
                                                              crs=ccrs.PlateCarree(),
                                                              zorder=0)
        
            axins1 = ax_m2[3, i2].inset_axes([.025, -.25, .95, .1])
            cbar = fig_m2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=px_pal), 
                                   cax=axins1,
                                   extend='both',
                                   ticks=tcks,
                                   orientation='horizontal')
            cbar.ax.set_xlabel(cbar_lab, size=6)
            cbar.ax.tick_params(labelsize=6)
                    
                    
        fig_m2.subplots_adjust(wspace=.13, hspace=.1)
        fpath = ('figures/hansell_glodap/global/stoichiometry/maps/' +
                 'hansell_glodap_global_' + v1 + 'ratio_pixel_map.pdf')
        fig_m2.savefig(fpath, format='pdf', bbox_inches='tight',
                       transparent=True, dpi=300)
        plt.close(fig_m2)
    

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
                            (reg_ss['r2'] > r2_limit) &
                            (reg_ss['slope'] >= 0))
                elif v=='OXYGEN_RES':
                    idx2 = ((reg_ss['spvalue'] < pval_limit) &
                            (reg_ss['r2'] > r2_limit) &
                            (reg_ss['slope'] <= 0))
                else:
                    idx2 = ((reg_ss['spvalue'] < pval_limit) &
                            (reg_ss['r2'] > r2_limit))
                        
                reg_ss = reg_ss.loc[idx2]
                
                # Average rate/ratio (slope)
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

## Export OUR & DOCUR regression results
reg_ss1 = reg.loc[reg.x_tracer.isin(ages_res), :]
fpath = 'deriveddata/hansell_glodap/global/o2_aou_doc_age_regressions.csv'
reg_ss1.to_csv(fpath, sep=',', na_rep='-9999', header=True, index=False)

## Export stoichiometry regression results
reg_ss2 = reg.loc[~(reg.x_tracer.isin(ages_res)), :]
reg_ss2 = reg_ss2.rename(columns={'x_tracer': 'x_var'})
fpath = 'deriveddata/hansell_glodap/global/stoichiometry_regressions.csv'
reg_ss2.to_csv(fpath, sep=',', na_rep='-9999', header=True, index=False)


## Export pixel polygons
for w in pixel_polys:
    for p in pixel_polys[w]:
        
        fpath = Path('deriveddata/hansell_glodap/global/' +
                     'regression_pixel_polys/' + w + ',' + p + 
                     '_polygon.geojson')
        fpath.write_text(to_geojson(pixel_polys[w][p]))
# fpath = 'deriveddata/hansell_glodap/global/regression_pixels.npy'
# np.save(fpath, pixel_polys) 


