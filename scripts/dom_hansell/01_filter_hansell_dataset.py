# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:19:39 2023

@author: Markel Gómez Letona

## OceanICU ~ Hansell DOM dataset ##

Search for stations containing:
    - Oxygen
    - CFC-11, CFC-12 and SF6
    
Also, estimate derived variables:
    - Apparent oxygen utilisation ([O2_solubility] - [O2_measured])
    - Potential density anomaly
    
"""

#%% IMPORTS

import pandas as pd
import numpy as np
import pathlib
import os
from shapely import geometry, from_geojson
import xarray as xr 

import gsw
import scripts.modules.longhurst as lh

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.colors import ListedColormap


#%% LOAD DATA

## Note that in the data table:
#  - First row is the header
#  - Second row are the corresponding units of each variable
#  - Rest are rows with actual data

fpath = 'rawdata/dom_hansell/All_Basins_Data_Merged_Hansell_2022.xlsx'

# Get header:
hdr = pd.read_excel(fpath, header=None, nrows=1).values.flatten().tolist()
hdr = [h.replace(' ', '_') for h in hdr]
hdr = [h.replace('-', '_') for h in hdr]


# Get units:
var_units = pd.read_excel(fpath, header=None, nrows=1, skiprows=1).values.flatten().tolist()

# Get the data:
# (some variables have mixed numbers and strings so I better specify dtypes) 
hns = pd.read_excel(fpath, header=None, names=hdr, skiprows=2,
                    dtype={'EXPOCODE': 'string', 
                           'STATION': 'string', 
                           'CAST': 'string', 
                           'BOTTLE': 'string', 
                           'BOTTOM_DEPTH': 'string', 
                           'CTD_OXYGEN_FLAG_W': 'string'})


#%% DATA HANDLING

# Some EXPOCODES have whitespaces, so remove them:
hns['EXPOCODE'] = hns['EXPOCODE'].replace(" ", "")
# list(set(hns['EXPOCODE']))
# Apparently this still leaves two EXPOCODES with whitespaces? Doesn't make 
# sense, they must have hidden characters that are not strictly speaking white
# spaces.
hns['EXPOCODE'] = hns['EXPOCODE'].replace({"    31DSCG94_1": "31DSCG94_1",
                                           " 33RO20050111": "33RO20050111",
                                           "33RO20071215 ": "33RO20071215"})


# read_excel() reads "#N/A" as NA directly. This is relevant for BOTTOM_DEPTH
# hns['BOTTOM_DEPTH'] = hns['BOTTOM_DEPTH'].replace({'#N/A': '-999'})

# Modify "CTD_OXYGEN_FLAG_W" to remove the faulty "5+R267270" flag and convert 
# to int:
hns['CTD_OXYGEN_FLAG_W'] = pd.to_numeric(hns['CTD_OXYGEN_FLAG_W']
                                         .replace("5+R267270", "9"))

# Modify cruise 'A22 (2021)       ' to remove trailing whitespaces
hns['CRUISE'] = hns['CRUISE'].replace("A22 (2021)       ", "A22 (2021)")

# Replace -999 values with np.nan otherwise might have issues when deriving 
# variables (see below):
hns.replace(-999, np.nan, inplace=True)


# Most O2 values are given with 1 decimal of precission, so apply that to all
hns['CTD_OXYGEN'] = round(hns['CTD_OXYGEN'], 1)


#### Correct issues:

## AR01 (1998): 
#  CTD_SALINITY_FLAG_W and CTD_OXYGEN VALUES ARE SWITCHED <->
idx = hns['CRUISE']=="AR01 (1998)"
# Get values and switch them to the proper column:
OXYVALS = hns.loc[idx, 'CTD_SALINITY_FLAG_W']
SALFLAG = hns.loc[idx, 'CTD_OXYGEN']
hns.loc[idx, 'CTD_SALINITY_FLAG_W'] = SALFLAG
hns.loc[idx, 'CTD_OXYGEN'] = OXYVALS


#%% DERIVED VARIABLES

# Based on the Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10. 
# 
# See documentation:
# https://teos-10.github.io/GSW-Python/gsw_flat.html

## Prerequisite variables for our variables of interest:

#### ABSOLUTE SALINITY (SA, g/kg):
hns['SA'] = round(gsw.conversions.SA_from_SP(SP=hns['CTD_SALINITY'],
                                             p=hns['CTD_PRESSURE'],
                                             lon=hns['LONGITUDE'],
                                             lat=hns['LATITUDE']),
                  4) # limit decimals according to source variable

#### CONSERVATIVE TEMPERATURE (CT, deg C):
hns['CT'] = round(gsw.conversions.CT_from_t(SA=hns['SA'],
                                            t=hns['CTD_TEMPERATURE'],
                                            p=hns['CTD_PRESSURE']),
                  4)

#### POTENTIAL TEMPERATURE (pt, deg C; reference pressure = 0 dbar):
hns['PT'] = round(gsw.conversions.pt0_from_t(SA=hns['SA'],
                                             t=hns['CTD_TEMPERATURE'],
                                             p=hns['CTD_PRESSURE']),
                  4)

#### OXYGEN SOLUBILITY (O2SOL, umol/kg; O2 at equilibrium with air):
# This function uses the solubility coefficients derived from the data of Benson and Krause (1984), as fitted by Garcia and Gordon (1992, 1993).
hns['O2SOL'] = round(gsw.O2sol_SP_pt(SP=hns['CTD_SALINITY'],
                                     pt=hns['PT']),
                     1)
# checked that gives same as gsw.O2sol(hns['SA'][0], hns['CT'][0], hns['CTD_PRESSURE'][0], hns['LONGITUDE'][0], hns['LATITUDE'][0])

#### OXYGEN SATURATION (O2SAT, %):
hns['O2SAT'] = round(100*hns['CTD_OXYGEN']/hns['O2SOL'],
                     2)

#### APPARENT OXYGEN UTILISATION (AOU, umol/kg):
hns['AOU'] = round(hns['O2SOL'] - hns['CTD_OXYGEN'], 1)

#### POTENTIAL DENSITY ANOMALY (SIGMA0, kg/m^3; reference pressure = 0 dbar):
hns['SIGMA0'] = round(gsw.density.sigma0(SA=hns['SA'],
                                         CT=hns['CT']),
                      3)

#### POTENTIAL DENSITY ANOMALY (SIGMA1, kg/m^3; reference pressure = 1000 dbar):
hns['SIGMA1'] = round(gsw.density.sigma1(SA=hns['SA'],
                                         CT=hns['CT']),
                      3)

    
#%% FILTER DATASET

# Filter data based on the desired criteria:

# Variables we are interested in to estimate O2 utilisation rates:
# Must have:
#   - O2.
#   - At least 1 tracer of CFC-11, CFC-12 & SF6.
#   - DOC will be complementary when assesing O2 utilisation rates, so do not
#     enforce it as a requirement for the time being. We will further subset 
#     the table when needed. 
var_names1 = ['CTD_OXYGEN'] # variables that ALL must me valid
var_flags1 = [nm + '_FLAG_W' for nm in var_names1]
var_names2 = ['CFC_11', 'CFC_12', 'SF6'] # vars that AT LEAST ONE must be valid
var_flags2 = [nm + '_FLAG_W' for nm in var_names2] 

# Acceptable quality flags (see WOCE BOTTLE flag codes):
# NOTE: there are some samples flagged as 1 ("sample for this measurement was 
# drawn from water bottle but analysis not received"), but there *IS* a value. 
# This must have been overlooked, but I cannot be sure they are acceptable
# values that should have being flagged as 2 or 6....
aflags = [2, 6]

# Retain bottles with acceptable measurements in all the target variables (that
# are found in the depth range of interest):
# Variables that ALL must have acceptable values:
hns = hns.assign(is_aflag1 = hns[var_flags1].isin(aflags).apply(all, axis=1))
# Variables from which AT LEAST ONE must have acceptable values:
hns = hns.assign(is_aflag2 = hns[var_flags2].isin(aflags).apply(any, axis=1))
# Filter
idx = ((hns['is_aflag1']) & 
       (hns['is_aflag2']))
hns_ss = hns[idx].copy()


# Check that the subset has no missing values (especially just in case those 
# '1' flags had some missing values)
print((hns_ss[var_names1]==-999).any(axis=None)) # none of the compulsory variables should be =-999. should be False...
print((hns_ss[var_names2]==-999).all(axis=None)) # none of the samples should have ALL tracers =-999. should be False...



#%% ASSIGN OCEAN

# Loads polygons created in:
# 
# scripts/ocean_wm_defs/ocean_water_mass_definitions.py


#%%% LOAD POLYGONS

fpath = pathlib.Path('deriveddata/ocean_wm_defs/').glob('*')
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


#%%% ASSIGNMENT

# Set 'NO_OCEAN' as default
hns_ss = hns_ss.assign(OCEAN='NO_OCEAN')

# Convert coordinates of samples into geometry points (precompute to avoid
# doing it each iteration)
pts = [geometry.Point([r['LONGITUDE'], r['LATITUDE']]) for i, r in hns_ss.iterrows()]

# Perform assignment
for o in ocean_polys:
    
    # Get polygon of ocean o
    op = ocean_polys[o]
    
    # Assess whether each point (sample) is contained within polygon (ocean) o
    boo = ((op.contains(pts)) | (op.touches(pts)))
    
    # Assign ocean code for those samples that fall within o
    hns_ss.loc[boo, 'OCEAN'] = o.capitalize()



#%% ASSIGN WATER MASS

# Loads polygons created in:
# 
# scripts/ocean_wm_defs/ocean_water_mass_definitions.py 
# 
# and also uses SIGMA0 ranges gathered from the literature.


#%%% SET SIGMA0 RANGES

## Load SIGMA0 ranges
fpath = 'deriveddata/ocean_wm_defs/water_mass_sigma0_definitions.txt'
wm_sigma0 = pd.read_csv(fpath, sep='\t')
wm_sigma0 = {k:[v1, v2] for k, v1, v2 in zip(wm_sigma0.water_mass,
                                             wm_sigma0.sigma0_min,
                                             wm_sigma0.sigma0_max)}


#%%% LOAD POLYGONS

wm_polys = {}
wm_depths = ['central', 'intermediate', 'deep']
for d in dlist:
    for z in wm_depths:
        
        # Get wm paths at depth z and ocean d
        flist = [*pathlib.Path(str(d) + "\\wms\\" + z).glob("*.geojson")]
        
        # Skip iteration if certain depth is absent (i.e. flist is empty)
        if not flist: continue
        
        for f in flist:
            
            # Get wm name (accounts for when the name itself has underscore, 
            # e.g. AAIW_P)            
            w = "_".join(str(f).split("\\")[-1].split("_")[0:-1])

            # Load polygon
            wm_polys[w] = from_geojson(f.read_text())
            
            
    
#%%% ASSIGNMENT

wm_boo = {}
hns_ss = hns_ss.assign(WATER_MASS='NO_WATER_MASS')
for w in wm_polys:
    
    # Get the indices of the samples that fulfill the conditions of 
    # water mass w.
    # When assessing the geographical position, also do .touches() to include 
    # points exactly in the polygon boundary [.contains() leaves out points 
    # that fall exactly on the boundary]
    # (in some case this might coindice with the boundary of two water masses,
    # and the point will be assigned to the last one, but those will be
    # excepcions relative to the total data volume)

    idx = ((hns_ss['SIGMA0'] > wm_sigma0[w][0]) &
           (hns_ss['SIGMA0'] <= wm_sigma0[w][1]) &
           (wm_polys[w].contains(pts) | wm_polys[w].touches(pts)))
    wm_boo[w] = idx # keep track (will be used to tune MW vs UNADW case)
    
    # Assign water mass string to those samples:
    hns_ss.loc[idx, 'WATER_MASS'] = w

# There is a small area~sigma0 range where MW and UNADW overlap due to how
# definitions have been kept as simple as reasonably possible. In those 
# overlaps, make sure to assign the samples as MW
mw_unadw_overlap = wm_boo['MW'] & wm_boo['UNADW']
sum(mw_unadw_overlap)
hns_ss.loc[mw_unadw_overlap, 'WATER_MASS'] = 'MW'



#%% ADD LONGHURST PROVINCES

###
### [takes ~ <1 min]
###

# Finding the Longhurst province to which a sample belongs is quite time costly
# (relatively). Thus, considering the very high number of samples, it is more
# efficient to do it once per station and then map the results to all the 
# samples per station.

# Create a temporary variable to uniquely identify stations across cruises
hns_ss = hns_ss.assign(STID = lambda x: x['EXPOCODE'] + x['STATION'])


# Keep unique instances of stations
colnames = ['STID', 'LONGITUDE', 'LATITUDE']
hns_ss2 = hns_ss.loc[~hns_ss.duplicated('STID'), colnames]


### Assign stations Longhurst provinces

hns_ss2['LP'] = 'nan'
for ir, r in hns_ss2.iterrows():
    hns_ss2.loc[ir, 'LP'] = lh.find_longhurst(r['LONGITUDE'], r['LATITUDE'])


### Map results to sample table
hns_ss = hns_ss.merge(hns_ss2[['STID', 'LP']], on='STID')


# Remove temporary variables/column
hns_ss.drop('STID', axis=1, inplace=True)
del hns_ss2


#%% ASSIGN NPP

# Assign net primary production values to each sample based on its geographic
# location. NPP values are from a climatology (average) of monthly data 
# provided by the OSU. Two climatologies (CbPM and Eppley-VGPM) are available
# here.
# 
# The climatologies are computed in: scripts/npp/npp_create_climatology.py

# To assign values, match sample coordinates to the NPP-pixel they are 
# contained in.

# Load data
fpath = 'deriveddata/npp/climatology/eppley-vgpm.npz'
with np.load(fpath) as npz:
    lat = npz['lat']
    lon = npz['lon']
    eppl = npz['npp']
    
fpath = 'deriveddata/npp/climatology/cbpm.npz'
with np.load(fpath) as npz:
    cbpm = npz['npp']
    


# Find pixel and assign value
res = 1/12
hns_ss['NPP_EPPL'] = np.nan
hns_ss['NPP_CBPM'] = np.nan
for ir, r in hns_ss.iterrows():
    
    # Get coordinates and find the closest value in NPP data
    lo = r.LONGITUDE
    la = r.LATITUDE
    dlo = abs(lon - lo)
    dla = abs(lat - la)
    ilon = np.argmin(dlo)
    ilat = np.argmin(dla)
    
    # Assign NPP values to sample.
    # Account for instances with no close pixel -> leave nan (allow at most 
    # 2 pixels away)
    if ((min(dlo) > 2*res) | (min(dla) > 2*res)):
        continue
    else:
        hns_ss.loc[ir, 'NPP_EPPL'] = eppl[ilat, ilon]
        hns_ss.loc[ir, 'NPP_CBPM'] = cbpm[ilat, ilon]
        
    
#%% ASSIGN B

# Assign b parameter (of the Martin curve), following Marsay et al. (2015),
# which parameterise it based on the temperature of the upper 500 m. In Fig. 3
# they use the World Ocean Atlas. Do the same.

fpath = 'rawdata/woa/woa23_decav_t00mn01.csv.gz'
woa = pd.read_csv(fpath, compression='gzip', sep=',', skiprows=1)

# In the table, each row is a pair of lat-lon, and each column a depth.
# 
# Extract latitudes, longitudes, depths, and values
lat = np.array(woa.iloc[:, 0])
lon = np.array(woa.iloc[:, 1])
z = np.array([0] + [int(x) for x in woa.columns[3:]])
t = np.array(woa.iloc[:, 2:])

# Interpolate to have a regularly spaced grid so that median is not skewed to
# surface waters (WOA has more resolution there)
zq = np.arange(0, 501, 5)
ti = np.empty((t.shape[0], len(zq)))
for i in range(t.shape[0]):
    ti[i, :] = np.interp(zq, z, t[i, :])


# Following Marsay et al. (2015), Fig. 2A and 3, estimate the median
# temperature in the upper 500 m (to then use it to estimate the b parameter)
t_m500 = np.nanmedian(ti, axis=1)

# Estimate b parameter
b = (0.062 * t_m500) + 0.303  # b = (0.062 × T) + 0.303

# Distributions of t_m500 and b are shown in 01_woa_b_visualisation.py


# Assign b to regression pixels: get b points within the boundary
res = 1
hns_ss['B'] = np.nan
for ir, r in hns_ss.iterrows():
    
    # Get coordinates and find the closest value in b data
    lo = r.LONGITUDE
    la = r.LATITUDE
    dlo = abs(lon - lo)
    dla = abs(lat - la)
    dlo_min = min(dlo)
    dla_min = min(dla)
    idx = (dlo==dlo_min) & (dla==dla_min)
    
    # There are edge cases where the closest value in terms of lat,lon has no
    # value in b (because its land, or ice. In those cases (b[idx].size==0), 
    # subset values close-by (will be averaged when assigning). 
    if b[idx].size==0:
        idx = (dlo < res) & (dla < res)
        if b[idx].size==0: # extend to 2*res at most
            idx = (dlo < 2*res) & (dla < 2*res)
            if b[idx].size==0:
                continue
    # Assign b values to sample.
    # Account for instances with no close pixel -> leave nan (allow at most 
    # 2 pixels away)
    if ((dlo_min > 2*res) | (dla_min > 2*res)):
        continue
    else:
        hns_ss.loc[ir, 'B'] = np.nanmean(b[idx])
        # ^ do mean because there are exceptions were data point is equally 
        # distant from 2 b values, + the exception when b[idx].size==0 (see
        # above)
        #if b[idx].size==0: print("NaN in b -> " + str(ir))


# quick check
# plt.scatter(hns_ss.LONGITUDE, hns_ss.LATITUDE, c=hns_ss.B)

#%% ASSIGN POC

# POC values from the Copernicus data product. Note that data is only available
# for the upper 1000 m, but that is almost all of the depth range in which the
# study focuses (and actually covers the 200-1000m, which is the *main* focus).

# Load climatological file
fpath = 'deriveddata/poc3d/cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg-climatology_P1M-m_MEAN_P202411_____.nc'
poc3d = xr.open_dataset(fpath)


# Find closest coordinates, then interpolate vertical profile
res = 1/4
hns_ss['POC3D'] = np.nan
for ir, r in hns_ss.iterrows():
    
    # Get coordinates and find the closest one
    lo = r.LONGITUDE
    la = r.LATITUDE
    p = r.CTD_PRESSURE
    dlo = abs(poc3d.longitude.values - lo)
    dla = abs(poc3d.latitude.values - la)
    ilon = np.argmin(dlo)
    ilat = np.argmin(dla)
    
    # Get closest profile and interpolate it to the sample depth
    # Account for instances with no close pixel -> leave nan (allow at most 
    # 2 pixels away)
    if ((min(dlo) > 2*res) | (min(dla) > 2*res)):
        continue
    else:
        poc_profile = poc3d.poc.values[:, ilat, ilon]
        ipoc = np.interp(p, poc3d.depth.values, poc_profile,
                         left=np.nan, right=np.nan)
        hns_ss.loc[ir, 'POC3D'] = ipoc


#%% PLOT MAPS

dpath = 'figures/dom_hansell/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    

#%%% STATIONS

# Map available stations after filtering the dataset. For that we will only 
# keep unique instance of each station:
st = hns_ss[~hns_ss.duplicated(subset=['CRUISE', 'STATION'], keep='first')]

# Add number of samples per station:
st = st.merge(hns_ss.groupby(['CRUISE', 'STATION']).size().to_frame().reset_index().rename(columns={0: 'nsample'}),
              how='inner', on=['CRUISE', 'STATION'])


# Map:
cm = 1/2.54
fig_n, ax_n = plt.subplots(figsize=(15*cm, 8*cm), 
                           subplot_kw={'projection': ccrs.Mollweide()})
ax_n.add_feature(cfeature.LAND, facecolor='#444444')
ax_n.add_feature(cfeature.OCEAN, facecolor='#e6ecf2')
s = ax_n.scatter(x=st['LONGITUDE'],
                 y=st['LATITUDE'],
                 c=st['nsample'],
                 cmap='viridis',
                 vmin=0, vmax=25,
                 s=4,
                 alpha=1,
                 transform=ccrs.PlateCarree())
cbar = fig_n.colorbar(s, ax=ax_n,
                      extend='max',
                      shrink=.7)
cbar.set_label("No. of samples")

fpath = 'figures/dom_hansell/map_filter_o2_doc_cfc11_cfc12_sf6_nsample.pdf'
plt.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%%% STATIONS BY CRUISE

# Plot second map where points are coloured by cruise:

# Create custom colormap to deal with the large amount of cruises:
tabextra= ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors + plt.cm.Set2.colors + plt.cm.tab10.colors)

# Factorise the cruise variable and assign colors:
levels, categories = pd.factorize(st['CRUISE'])
colors = [tabextra(i) for i in levels] # using our custom colormap

# handles for the legend:
handles = [matplotlib.patches.Patch(color=tabextra(i), label=c) for i, c in enumerate(categories)]

# Plot:
fig_c, ax_c = plt.subplots(figsize=(15*cm, 8*cm), 
                           subplot_kw={'projection': ccrs.Mollweide()})
ax_c.add_feature(cfeature.LAND, facecolor='#444444')
ax_c.add_feature(cfeature.OCEAN, facecolor='#e6ecf2')
s = ax_c.scatter(x=st['LONGITUDE'], 
                 y=st['LATITUDE'],
                 c=colors,
                 s=4,
                 alpha=1,
                 transform=ccrs.PlateCarree())
fig_c.legend(handles=handles, title="Cruise", ncol=3, prop={'size': 7},
             bbox_to_anchor=(.5, -.1), loc='upper center')

fpath = 'figures/dom_hansell/map_filter_o2_doc_cfc11_cfc12_sf6_cruise.pdf'
fig_c.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%%% WATER MASSES

nc = 3
nwm = len(wm_polys)
nr = int(nwm / nc) if (nwm % nc)==0 else int(nwm / nc + 1)

fig_w = plt.figure(figsize=(5*cm * nc, 3*cm * nr))
for iw, w in enumerate(wm_polys):
    
    # Subset samples for each w, g combination
    idx = hns_ss['WATER_MASS']==w
    ss = hns_ss.loc[idx, :].copy()
    
    # Initialise subplot
    mproj = ccrs.Mollweide(central_longitude=-160)
    ax_w = fig_w.add_subplot(nr, nc, iw + 1, projection=mproj)
    
    # Plot land, water mass polygon and sample points
    ax_w.add_feature(cfeature.LAND,
                     facecolor='#ccc',
                     edgecolor='#444',
                     linewidth=.2,
                     zorder=0)
    ax_w.add_geometries(wm_polys[w],
                                facecolor='none',
                                edgecolor='k',
                                linewidth=.7,
                                crs=ccrs.PlateCarree(),
                                zorder=1)
    ax_w.scatter(ss.LONGITUDE, ss.LATITUDE,
                 s=.5,
                 linewidth=.05,
                 transform=ccrs.PlateCarree(),
                 zorder=2)
    ax_w.text(.01, 1.04,
              "$\mathbf{" + w + "}$",
              size=4,
              transform=ax_w.transAxes)
    ax_w.set_global()
    
    
fpath = 'figures/dom_hansell/map_filter_o2_doc_cfc11_cfc12_sf6_water_mass.pdf'
fig_w.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%% EXPORT 
   
# Only export variables of our interest:

# vars with no flags:    
var_meta = ['EXPOCODE', 'CRUISE', 'STATION', 'CAST', 'BOTTLE', 'BOTTLE_FLAG_W',
            'DATE', 'LATITUDE', 'LONGITUDE', 'OCEAN', 'WATER_MASS', 'LP',
            'BOTTOM_DEPTH', 'CTD_PRESSURE', 'CTD_TEMPERATURE']

# vars that have flags:
var_names = ['CTD_SALINITY', 'CTD_OXYGEN', 'CHLOROPHYLL_A',
             'SILICIC_ACID', 'NITRATE', 'PHOSPHATE', 'NITRITE', 'AMMONIUM', 
             'CFC_11', 'CFC_12', 'CFC_113', 'SF6', 'CCL4', 
             'DELTA_14DIC', 'TRITIUM', 'HELIUM',
             'DIC', 'ALKALINITY', 'PH_TOT', 
             'DOC', 'TDN', 'POC', 'PON']
# derived vars:
dev_names = ['SA', 'CT', 'PT', 'O2SOL', 'O2SAT', 'AOU', 'SIGMA0', 'SIGMA1',
             'NPP_EPPL', 'NPP_CBPM', 
             'B', 'POC3D']
var_flags = [nm + "_FLAG_W" for nm in var_names]
var_names = [val for pair in zip(var_names, var_flags) for val in pair] # this interleaves vars and flags to maintain column order

all_vars = var_meta + var_names + dev_names
hns_export = hns_ss[all_vars]


dpath = 'deriveddata/dom_hansell/'
if not os.path.exists(dpath):
    os.makedirs(dpath)

fpath = 'deriveddata/dom_hansell/Hansell_2022_o2_doc_cfc11_cfc12_sf6.csv'
hns_export.to_csv(fpath, sep=',', header=True, index=False, na_rep='-9999')

## Also export the unfiltered dataset with the derived variables:
# hns_export2 = hns[all_vars]
# fpath = 'deriveddata/dom_hansell/Hansell_2022_with_derived_vars.csv'
# hns_export2.to_csv(fpath, sep=',', header=True, index=False)
