 # -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:22:16 2024

Show range of data with valid tracer ages.

@author: Markel
"""

#%% IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


#%% LOAD DATA

# Filtered, merged Hansell+GLODAP dataset:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
tbl = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                   'CRUISE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})
tbl.replace(-9999, np.nan, inplace=True)


#%% PLOT LONGITUDINAL BANDS

# Plot age data availability in three selected longitudinal bands (one per 
# ocean): map of band selection + the band sections with the data availability.

# Select longitudinal bands to use as examples of the data availability in each
# ocean.
#                        [t,   r,    b,   l] 
band_lims = {'Atlantic': [60, -15,  -70, -35],
             'Indian'  : [30,  65, -70,  45],
             'Pacific' : [60, -145, -70, -165]}

band_polys = {}
poly_res = 1
for b in band_lims:

    # Depending on the projection used when mapping, polygon bands with only 
    # the four vertices are displayed not as smooth curved lines, but as if 
    # with low resolution. So forcing more vertices improves visualisation
    poly_vertices = [[band_lims[b][1], band_lims[b][0]],
                     [band_lims[b][1], band_lims[b][2]],
                     [band_lims[b][3], band_lims[b][2]],
                     [band_lims[b][3], band_lims[b][0]]]

    new_vertices_all = []
    for i in range(len(poly_vertices)-1):
        
        # Interpolation when going straigh up/down in latitude
        if poly_vertices[i][0]==poly_vertices[i+1][0]:
            
            # Make sure that the range is created with the proper increment sign
            if poly_vertices[i][1]>poly_vertices[i+1][1]: 
                pr = -poly_res
            else:
                pr = poly_res
            
            # Create new query points (note that here "x" is really our "y" as we
            # are moving along latitude)
            qx = [*np.arange(poly_vertices[i][1], poly_vertices[i+1][1], pr)]
            
            # Interpolate
            x = [poly_vertices[i][1], poly_vertices[i+1][1]]
            y = [poly_vertices[i][0], poly_vertices[i+1][0]]
            qy = np.interp(qx, x, y)
            
            new_vertices = [[vx, vy] for vx, vy in zip(qy, qx)]
        
        # Interpolation when going straigh left/right in longitude
        elif poly_vertices[i][1]==poly_vertices[i+1][1]:
            
            if poly_vertices[i][0]>poly_vertices[i+1][0]: 
                pr = -poly_res
            else:
                pr = poly_res
                
            qx = [*np.arange(poly_vertices[i][0], poly_vertices[i+1][0], pr)]
            x = [poly_vertices[i][0], poly_vertices[i+1][0]]
            y = [poly_vertices[i][1], poly_vertices[i+1][1]]
            qy = np.interp(qx, x, y)
            new_vertices = [[vx, vy] for vx, vy in zip(qx, qy)]
        
        new_vertices_all.extend(new_vertices)
    
    
    band_polys[b] = geometry.Polygon(new_vertices_all)


## Set up plot

cm = 1/2.54
ages = ['AGE_CFC11', 'AGE_CFC12', 'AGE_SF6']
ages_tracer = {k: v for k, v in zip(ages, ['pCFC11', 'pCFC12', 'pSF6'])}
ocean_pal = {k: v for k, v in zip(band_lims.keys(),
                                  ['#44AA99', '#88CCEE', '#EE8866'])}
age_labels = {k: v for k, v in zip(ages, ["$\mathbf{Age_{CFC\u201011}}$",
                                          "$\mathbf{Age_{CFC\u201012}}$",
                                          "$\mathbf{Age_{SF_6}}$"])}

fig_band = plt.figure(figsize=(6*cm*len(ages), (8 + 5*len(band_lims))*cm))
gs = GridSpec(4, 3, figure=fig_band)


#%%% BAND MAP

# Get the position of the stations within the defined bands
tblu = tbl.drop_duplicates(subset=['EXPOCODE', 'STATION'])

idx_bands = band_lims.copy()
for b in band_lims:
    
    # Get the indices of samples in each band
    idx_bands[b] = ((tblu.LATITUDE < band_lims[b][0]) &
                    (tblu.LONGITUDE < band_lims[b][1]) &
                    (tblu.LATITUDE > band_lims[b][2]) &
                    (tblu.LONGITUDE > band_lims[b][3]))
    
# Combine all booleans so that the rest of the data can be mapped separately
idx_bands_list = [value for value in idx_bands.values()]
idx_bands_all = [any(l) for l in zip(*idx_bands_list)]


# Map the band locations 
ax_map = fig_band.add_subplot(gs[0, :], projection=ccrs.Mollweide(
    central_longitude=-160))
ax_map.add_feature(cfeature.LAND, 
                   facecolor='#ccc', 
                   edgecolor='k', linewidth=.3,
                   zorder=1)
s1 = ax_map.scatter(x=tblu.LONGITUDE.loc[[not i for i in idx_bands_all]],
                    y=tblu.LATITUDE.loc[[not i for i in idx_bands_all]],
                    c='#aaa',
                    s=.5,
                    linewidth=.5,
                    transform=ccrs.PlateCarree(),
                    zorder=2)
for b in idx_bands:
    idx_bands[b]
    s2 = ax_map.scatter(x=tblu.LONGITUDE.loc[idx_bands[b]],
                        y=tblu.LATITUDE.loc[idx_bands[b]],
                        c=ocean_pal[b],
                        s=.5,
                        linewidth=.5,
                        transform=ccrs.PlateCarree(),
                        zorder=2)
    ax_map.add_geometries(band_polys[b],
                          crs=ccrs.PlateCarree(),
                          facecolor=ocean_pal[b],
                          alpha=.2,
                          zorder=0)
ax_map.set_global()
    


#%%% BAND SECTION

brks_y = [*range(0, 1750, 250)]
for i, b in enumerate(band_lims):
    for j, a in enumerate(ages):
        
        # Get the samples within the band
        in_band = ((tbl.LATITUDE < band_lims[b][0]) &
                   (tbl.LONGITUDE < band_lims[b][1]) &
                   (tbl.LATITUDE > band_lims[b][2]) &
                   (tbl.LONGITUDE > band_lims[b][3]))
                
        # Get the samples with a valid tracer age
        valid_age = ~np.isnan(tbl.loc[in_band, a])
        
        # Get the samples with a tracer measurement
        with_tracer = ~np.isnan(tbl.loc[in_band, ages_tracer[a]])
        
        # Within the band and with valid age
        idx = (in_band & valid_age)
        
        # Within the band and with tracer measurement but no valid age
        idx2 = (in_band & (~valid_age) & with_tracer)
        
        # Set up subplot
        ax_s = fig_band.add_subplot(gs[i+1, j])
        
        # Plot datapoints with valid ages
        ax_s.scatter(tbl.LATITUDE.loc[idx],
                     tbl.CTD_PRESSURE.loc[idx],
                     marker='o',
                     edgecolor='w',
                     facecolor=ocean_pal[b],
                     s=8,
                     linewidth=.1,
                     zorder=1)
        
        # And in light grey those that had a tracer measurement but not valid
        # age (do to being out of the accepted time period for each tracer)
        ax_s.scatter(tbl.LATITUDE.loc[idx2],
                     tbl.CTD_PRESSURE.loc[idx2],
                     marker='o',
                     edgecolor='#ccc',
                     facecolor='none',
                     s=5,
                     linewidth=.2,
                     zorder=0)
        
        # Add labels:
        ax_s.text(.03, .94, age_labels[a],
                  size=5,
                  ha='left', va='center',
                  transform=ax_s.transAxes)
        ax_s.text(.98, .94, b,
                  size=5,
                  ha='right', va='center',
                  weight='bold',
                  transform=ax_s.transAxes)

        
        # Adjust axes etc.
        ax_s.axhline(y=1000, color='k', linestyle='dotted')
        ax_s.set(xlim=[-70, 60],
                 ylim=[0, 1500],
                 xticks=np.arange(-60, 80, 20),
                 yticks=brks_y)
        ax_s.tick_params(which='major', axis='both',
                         labelsize=5, pad=2,
                         direction='in', length=2.5,
                         top=True, right=True)
        if i==max(range(len(band_lims))): # add x axis label on last row only
            ax_s.set_xlabel('Latidude [$\degree$N]', size=6)
        if j==0: # add y label on first column only
            ax_s.set_ylabel('Depth [dbar]', size=6)
            
        ax_s.invert_yaxis()

fig_band.subplots_adjust(hspace=.1)

fpath = 'figures/hansell_glodap/tracer_ages/hansell_glodap_tracer_age_dist_sect.pdf'
fig_band.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig_band.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%% MAP AT DEPTHS

# Map availabilty of ages at specific depths:
#   - 200 m
#   - 500 m
#   - 1000 m
#   - 1500 m

depths = [200, 500, 800, 1000, 1500]
fig_map2, ax_map2 = plt.subplots(nrows=len(depths), ncols=len(ages),
                                 subplot_kw={'projection': ccrs.Mollweide(
                                     central_longitude=-160)},
                                 figsize=(15*cm, 15*cm))
for i, d in enumerate(depths):
    
    delta = 10 if d==200 else 50
    
    for ia, a in enumerate(ages):
        
        # Get samples at depth d (~)
        at_depth = ((tbl.CTD_PRESSURE >= (d - delta)) &
                    (tbl.CTD_PRESSURE <= (d + delta)))
        
        # Get the samples with a valid tracer age
        valid_age = ~np.isnan(tbl.loc[:, a])
        
        # Get the samples with a tracer measurement
        with_tracer = ~np.isnan(tbl.loc[:, ages_tracer[a]])
        
        # At depth and with valid age
        idx = (at_depth & valid_age)
        
        # At depth and with tracer measurement but no valid age
        idx2 = (at_depth & ~valid_age & with_tracer)
        

        ax_map2[i, ia].add_feature(cfeature.LAND,
                                   facecolor='#ccc',
                                   edgecolor='black', linewidth=.3,
                                   zorder=0)
        s1 = ax_map2[i, ia].scatter(x=tbl.LONGITUDE.loc[idx],
                                    y=tbl.LATITUDE.loc[idx],
                                    c='#3cb371',
                                    label="tracer_and_age",
                                    s=.5,
                                    linewidth=.5,
                                    alpha=.7,
                                    transform=ccrs.PlateCarree(),
                                    zorder=2)
        s2 = ax_map2[i, ia].scatter(x=tbl.LONGITUDE.loc[idx2],
                                    y=tbl.LATITUDE.loc[idx2],
                                    c='#cb410b',
                                    label="tracer_not_age",
                                    s=.5,
                                    linewidth=.5,
                                    alpha=.2,
                                    transform=ccrs.PlateCarree(),
                                    zorder=1)
        if i==0:
            ax_map2[i, ia].text(.5, 1.15, age_labels[a],
                                size=6,
                                ha='center', va='center',
                                transform=ax_map2[i, ia].transAxes)
        if ia==0:
            ax_map2[i, ia].text(-.03, .5, str(d) + " dbar",
                                size=5,
                                ha='right', va='center',
                                weight='bold',
                                transform=ax_map2[i, ia].transAxes)
            
        ax_map2[i, ia].set_global()
        
        if (i==(len(depths)-1)) & (ia==1):
            leg = fig_map2.legend(handles=[Line2D([0], [0], 
                                                  color='none', 
                                                  marker='o',
                                                  markerfacecolor='#3cb371',
                                                  markeredgecolor='none',
                                                  markersize=7,
                                                  label="Measured tracer, and valid age"),
                                           Line2D([0], [0],
                                                  color='none', 
                                                  marker='o',
                                                  markerfacecolor='#cb410b',
                                                  markeredgecolor='none',
                                                  markersize=7,
                                                  label="Measured tracer, but invalid age")],
                                  loc='upper center',
                                  bbox_to_anchor=[.5, .13],
                                  ncol=2,
                                  prop={'size': 7},
                                  frameon=False,
                                  handletextpad=0)


fig_map2.subplots_adjust(hspace=.1)

fpath = 'figures/hansell_glodap/tracer_ages/hansell_glodap_tracer_age_dist_map.pdf'
fig_map2.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig_map2.savefig(fpath, format='png', bbox_inches='tight', dpi=600)
    
