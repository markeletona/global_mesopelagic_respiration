# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:00:37 2024

@author: Markel Gómez Letona

## OceanICU ~ GLODAPv2.2023 DOM dataset ##

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


#%% READ DATA

# Read all data as strings so that pandas does not have to guess the datatype
# (it is a very large dataset and adds more time to the loading process).
# Then we will transform to floats the columns that need to be transformed.
fpath = 'rawdata/glodap/GLODAPv2.2023_Merged_Master_File.csv.zip'
gv2 = pd.read_csv(fpath, sep=',', header=0, 
                  compression='zip',
                  dtype={'G2expocode': str,
                         'G2bottle': str})


#%% FILTER DATASET

# Lighten the glodap table to retain only the necessary variables.
vrs = ['G2expocode', 'G2cruise', 'G2station', 
       'G2year', 'G2month', 'G2day',
       'G2latitude', 'G2longitude', 'G2bottle',
       'G2pressure', 'G2depth', 'G2theta', 'G2salinity',
       'G2sigma0', 'G2sigma1',
       'G2oxygen', 'G2oxygenf', 'G2aou', 'G2aouf',
       'G2cfc11', 'G2cfc11f', 'G2cfc12', 'G2cfc12f', 'G2sf6', 'G2sf6f',
       'G2doc', 'G2docf', 'G2don', 'G2donf', 'G2tdn', 'G2tdnf',
       'G2nitrate', 'G2nitratef', 'G2nitrite', 'G2nitritef',
       'G2phosphate', 'G2phosphatef', 'G2silicate', 'G2silicatef',
       'G2chla', 'G2chlaf']
gv2 = gv2.loc[:, vrs]

# Replace bad values with nan:
gv2.replace(-9999, np.nan, inplace=True)

# Convert to float the target columns:
for f in [v for v in vrs if v not in ['G2expocode']]:
    
    # Check first 1000 values to decide to convert to float or integer:
    # (with this, if all tested values are nan -> column is converted to float 
    #  as safeguard)
    var_is_float = any([not float(i).is_integer() for i in gv2.loc[:1000, f]])
    
    # Exception:
    if f in ['G2station', 'G2depth']: var_is_float = True
    
    # Convert columns
    if var_is_float:
        gv2[f] = gv2[f].astype('float64')
    else:
        gv2[f] = gv2[f].astype('float64').astype('Int64')
    # Note that integer type *needs* to be 'Int64' (not int64, int or whatever)
    # to accept nans within it. 
    # https://pandas.pydata.org/docs/user_guide/integer_na.html#construction


# Glodap has some longitude coords in [0 360]. Turn them into [-180, 180]
# np.nanmax(gv2.G2longitude)
over180 = gv2.G2longitude > 180
gv2.loc[over180, 'G2longitude'] = gv2.loc[over180, 'G2longitude'] - 360


# Filter flags (accepted -> 0 & 2 ; 0=interpolated, 2=ok), retaining samples 
# with acceptable flags in O2 and at least one tracer:
aflags = [0, 2]
var_flags1 = ['G2oxygenf'] # all
var_flags2 = ['G2cfc11f', 'G2cfc12f', 'G2sf6f'] # any
gv2 = gv2.assign(is_aflag1 = gv2[var_flags1].isin(aflags).apply(all, axis=1))
gv2 = gv2.assign(is_aflag2 = gv2[var_flags2].isin(aflags).apply(any, axis=1))
gv2 = gv2[(gv2['is_aflag1']) & 
          (gv2['is_aflag2'])]
gv2.drop(columns=['is_aflag1', 'is_aflag2'], inplace=True)


# As with the Hansell dataset, many O2 values are given with 1 decimal of
# precission, so harmonise everything
gv2['G2oxygen'] = round(gv2['G2oxygen'], 1)


#%% DERIVED VARIABLES

# Based on the Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10. 
# 
# See documentation:
# https://teos-10.github.io/GSW-Python/gsw_flat.html


# Glodap already has an AOU variable, but derive it myself for consistency
# with the Hansell dataset.

#### OXYGEN SOLUBILITY (O2SOL, umol/kg; O2 at equilibrium with air):
# This function uses the solubility coefficients derived from the data of Benson and Krause (1984), as fitted by Garcia and Gordon (1992, 1993).
gv2['G2oxygensolubility'] = round(gsw.O2sol_SP_pt(SP=gv2['G2salinity'],
                                                  pt=gv2['G2theta']),
                                  1)

#### APPARENT OXYGEN UTILISATION (AOU, umol/kg):
gv2['G2aou'] = round(gv2['G2oxygensolubility'] - gv2['G2oxygen'], 1)


#%% ASSIGN OCEAN

# Loads polygons created in:
# 
# scripts/ocean_wm_defs/ocean_water_mass_definitions.py


#%%% LOAD POLYGONS

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


#%%% ASSIGNMENT

# Set 'NO_OCEAN' as default
gv2 = gv2.assign(G2ocean='NO_OCEAN')

# Convert coordinates of samples into geometry points (precompute to avoid
# doing it each iteration)
pts = [geometry.Point([r['G2longitude'], r['G2latitude']]) for i, r in gv2.iterrows()]

# Perform assignment
for o in ocean_polys:
    
    # Get polygon of ocean o
    op = ocean_polys[o]
    
    # Assess whether each point (sample) is contained within polygon (ocean) o
    boo = ((op.contains(pts)) | (op.touches(pts)))
    
    # Assign ocean code for those samples that fall within o
    gv2.loc[boo, 'G2ocean'] = o.capitalize()


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
gv2 = gv2.assign(G2watermass='NO_WATER_MASS')
for w in wm_polys:
    
    # Get the indices of the samples that fulfill the conditions of 
    # water mass w.
    # When assessing the geographical position, also do .touches() to include 
    # points exactly in the polygon boundary [.contains() leaves out points 
    # that fall exactly on the boundary]
    # (in some case this might coindice with the boundary of two water masses,
    # and the point will be assigned to the last one, but those will be
    # excepcions relative to the total data volume)

    idx = ((gv2['G2sigma0'] > wm_sigma0[w][0]) &
           (gv2['G2sigma0'] <= wm_sigma0[w][1]) &
           (wm_polys[w].contains(pts) | wm_polys[w].touches(pts)))
    wm_boo[w] = idx # keep track (will be used to tune MW vs UNADW case)
    
    # Assign water mass string to those samples:
    gv2.loc[idx, 'G2watermass'] = w

# There is a small area~sigma0 range where MW and UNADW overlap due to how
# definitions have been kept as simple as reasonably possible. In those 
# overlaps, make sure to assign the samples as MW
mw_unadw_overlap = wm_boo['MW'] & wm_boo['UNADW']
sum(mw_unadw_overlap)
gv2.loc[mw_unadw_overlap, 'G2watermass'] = 'MW'



#%% ADD LONGHURST PROVINCES

###
### [takes ~ 2 min]
###

# Finding the Longhurst province to which a sample belongs is quite time costly
# (relatively). Thus, considering the very high number of samples, it is more
# efficient to do it once per station and then map the results to all the 
# samples per station.

# Create a temporary variable to uniquely identify stations across cruises
gv2 = gv2.assign(STID = lambda x: x['G2expocode'] + x['G2station'].astype(str))


# Keep unique instances of stations
colnames = ['STID', 'G2longitude', 'G2latitude']
gv2_stcoords = gv2.loc[~gv2.duplicated('STID'), colnames]


#### Assign stations Longhurst provinces
gv2_stcoords['G2lp'] = 'nan'
for ir, r in gv2_stcoords.iterrows():
    gv2_stcoords.loc[ir, 'G2lp'] = lh.find_longhurst(r['G2longitude'],
                                                     r['G2latitude'])


# Check sample that have not been assigned to any province (e.g. coastal areas)
# Manually assign them.
idx = gv2_stcoords['G2lp']=='NOT_IN_PROV'
gv2_stcoords_not = gv2_stcoords.loc[idx,:]


#### Map results to sample table
gv2 = gv2.merge(gv2_stcoords[['STID', 'G2lp']], on='STID')


# Remove temporary variables/column
gv2.drop('STID', axis=1, inplace=True)
del gv2_stcoords, gv2_stcoords_not


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
gv2['G2npp_eppl'] = np.nan
gv2['G2npp_cbpm'] = np.nan
for ir, r in gv2.iterrows():
    
    # Get coordinates and find the closest value in NPP data
    lo = r.G2longitude
    la = r.G2latitude
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
        gv2.loc[ir, 'G2npp_eppl'] = eppl[ilat, ilon]
        gv2.loc[ir, 'G2npp_cbpm'] = cbpm[ilat, ilon]
        
        
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
gv2['G2b'] = np.nan
for ir, r in gv2.iterrows():
    
    # Get coordinates and find the closest value in b data
    lo = r.G2longitude
    la = r.G2latitude
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
        gv2.loc[ir, 'G2b'] = np.nanmean(b[idx])
        # ^ do mean because there are exceptions were data point is equally 
        # distant from 2 b values, + the exception when b[idx].size==0 (see
        # above)
        if b[idx].size==0: print("NaN in b -> " + str(ir))


# quick check
# plt.scatter(gv2.G2longitude, gv2.G2latitude, c=gv2.G2b)


#%% ASSIGN POC

# POC values from the Copernicus data product. Note that data is only available
# for the upper 1000 m, but that is almost all of the depth range in which the
# study focuses (and actually covers the 200-1000m, which is the *main* focus).

# Load climatological file
fpath = 'deriveddata/poc3d/cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg-climatology_P1M-m_MEAN_P202411_____.nc'
poc3d = xr.open_dataset(fpath)


# Find closest coordinates, then interpolate vertical profile
res = 1/4
gv2['G2poc3d'] = np.nan
for ir, r in gv2.iterrows():
    
    # Get coordinates and find the closest one
    lo = r.G2longitude
    la = r.G2latitude
    p = r.G2pressure
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
        gv2.loc[ir, 'G2poc3d'] = ipoc
        

#%% PLOT MAP

dpath = 'figures/glodap/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    

#%%% STATIONS

# Map available stations after filtering the dataset. For that we will only 
# keep unique instance of each station:
st = gv2[~gv2.duplicated(subset=['G2expocode', 'G2station'], keep='first')]

# Add number of samples per station:
st = st.merge(gv2.groupby(['G2expocode', 'G2station']).size().to_frame().reset_index().rename(columns={0: 'nsample'}),
              how='inner', on=['G2expocode', 'G2station'])


# Map:
cm = 1/2.54
fig_n, ax_n = plt.subplots(figsize=(15*cm, 8*cm), 
                           subplot_kw={'projection': ccrs.Mollweide()})
ax_n.add_feature(cfeature.LAND, facecolor='#444444')
ax_n.add_feature(cfeature.OCEAN, facecolor='#e6ecf2')
s = ax_n.scatter(x=st['G2longitude'],
                 y=st['G2latitude'],
                 c=st['nsample'],
                 cmap='viridis',
                 vmin=0, vmax=30,
                 s=4,
                 alpha=1,
                 transform=ccrs.PlateCarree())
cbar = fig_n.colorbar(s, ax=ax_n,
                      extend='max',
                      shrink=.7)
cbar.set_label("No. of samples")

fpath = 'figures/glodap/map_filter_o2_cfc11_cfc12_sf6_nsample.pdf'
fig_n.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)

# To many cruises to do the plot colouring per cruise...


#%%% WATER MASSES

nc = 3
nwm = len(wm_polys)
nr = int(nwm / nc) if (nwm % nc)==0 else int(nwm / nc + 1)

fig_w = plt.figure(figsize=(5*cm * nc, 3*cm * nr))
for iw, w in enumerate(wm_polys):
    
    # Subset samples for each w, g combination
    idx = gv2['G2watermass']==w
    ss = gv2.loc[idx, :].copy()
    
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
    ax_w.scatter(ss.G2longitude, ss.G2latitude,
                 s=.5,
                 linewidth=.05,
                 transform=ccrs.PlateCarree(),
                 zorder=2)
    ax_w.text(.01, 1.04,
              "$\mathbf{" + w + "}$",
              size=4,
              transform=ax_w.transAxes)
    ax_w.set_global()
    
    
fpath = 'figures/glodap/map_filter_o2_doc_cfc11_cfc12_sf6_water_mass.pdf'
fig_w.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%% EXPORT 
   
# Only export variables of our interest:

# vars with no flags:    
var_meta = ['G2expocode', 'G2cruise', 'G2station',
            'G2year', 'G2month', 'G2day',
            'G2latitude', 'G2longitude',
            'G2bottle', 'G2pressure',
            'G2ocean', 'G2watermass', 'G2lp',
            'G2theta', 'G2salinity', 'G2sigma0', 'G2sigma1']

# vars that have flags:
var_names = ['G2oxygen', 'G2cfc11', 'G2cfc12', 'G2sf6', 
             'G2doc', 'G2don', 'G2tdn',
             'G2nitrate', 'G2nitrite', 'G2phosphate', 'G2silicate',
             'G2chla']
var_flags = [nm + 'f' for nm in var_names]
var_names = [val for pair in zip(var_names, var_flags) for val in pair] # this interleaves vars and flags to maintain column order

# derived vars:
dev_names = ['G2aou', 'G2npp_eppl', 'G2npp_cbpm',
             'G2b', 'G2poc3d']

all_vars = var_meta + var_names + dev_names
gv2_export = gv2[all_vars]


dpath = 'deriveddata/glodap/'
if not os.path.exists(dpath):
    os.makedirs(dpath)


fpath = 'deriveddata/glodap/GLODAPv2.2023_o2_cfc11_cfc12_sf6.csv'
gv2_export.to_csv(fpath, sep=',', header=True, index=False, na_rep='-9999')


