# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:48:42 2025

@author: Markel
"""

#%% IMPORTS

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo


#%% LOAD DATA

fpath = 'rawdata/woa/woa23_decav_t00mn01.csv.gz'
df = pd.read_csv(fpath, compression='gzip', sep=',', skiprows=1)

# In the table, each row is a pair of lat-lon, and each column a depth.


#%% DATA HANDLING

# Extract latitudes, longitudes, depths, and values
lat = np.array(df.iloc[:, 0])
lon = np.array(df.iloc[:, 1])
z = np.array([0] + [int(x) for x in df.columns[3:]])
t = np.array(df.iloc[:, 2:])

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


#%% MAP DATA

# Create regular grid out of irregular values (gaps will be NaNs)
ulon = np.unique(lon)
ulat = np.unique(lat)
z_array = np.nan * np.empty((len(ulat), len(ulon)))

# Convert to dataframes to set coords as indexes to use them to allocate values
t_m500_r = pd.DataFrame(z_array)
t_m500_r.columns = ulon
t_m500_r.index = ulat
b_r = t_m500_r.copy()
for i in range(len(t_m500)):
    t_m500_r.loc[lat[i], lon[i]] = t_m500[i]
    b_r.loc[lat[i], lon[i]] = b[i]


# Create grid for plot
LON, LAT = np.meshgrid(ulon, ulat)


## Plotting

cm = 1/2.54

pcm = [0, 1]
cbar_titles = ["Temperature [$\\degree$C]",
               "$\mathbf{b}$"]
cbar_ticks = [range(0, 21, 4), np.arange(.3, 1.6, .3)]

mproj = ccrs.Mollweide(central_longitude=-160)
fig_m, ax_m = plt.subplots(nrows=2, ncols=1,
                           figsize=(10*cm, 10*cm),
                           subplot_kw={'projection': mproj})

pcm[0] = ax_m[0].pcolormesh(LON, LAT, t_m500_r,
                            vmin=0, vmax=20,
                            cmap='Spectral_r',
                            rasterized=True,
                            transform=ccrs.PlateCarree(),
                            zorder=0)
pcm[1] = ax_m[1].pcolormesh(LON, LAT, b_r,
                            vmin=.3, vmax=1.5,
                            cmap=cmo.cm.tarn,
                            rasterized=True,
                            transform=ccrs.PlateCarree(),
                            zorder=0)

for i in range(2):
    ax_m[i].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                     name='land',
                                                     scale='110m'),
                        facecolor='#eee',
                        edgecolor='k',
                        linewidth=.1,
                        zorder=1)
    cb = fig_m.colorbar(pcm[i], ax=ax_m[i],
                        ticks=cbar_ticks[i],
                        extend='both',
                        pad=.04,
                        shrink=.75)
    cb.ax.tick_params(labelsize=5)
    cb.ax.set_ylabel(cbar_titles[i], fontsize=5)
    ax_m[i].set_global()

txt = "$b = 0.062 \\times T + 0.303$   (Marsay et al., 2015)"
ax_m[1].text(.5, -.1, txt, fontsize=4.5, ha='center',
             transform=ax_m[1].transAxes)


fpath = 'figures/hansell_glodap/global/rates/maps/helper/helper_woa_t500_b.svg'
fig_m.savefig(fpath, format='svg', bbox_inches='tight', transparent=False,
              dpi=300)
