# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:54:33 2024

Definitions of ocean and water mass polygons and grids.
Defines geographical extents and creates shapefiles with boundaries.
    
@author: Markel
"""

#%% IMPORTS

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import geometry, to_geojson, unary_union
import scripts.modules.geoutils as gu
import scripts.modules.polylabel as pl

from pathlib import Path
import os
from copy import deepcopy


#%% SET UP BOUNDARIES


#%%% SET UP HELPER BASE MAP

cm = 1/2.54
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 13*cm),
                         subplot_kw={'projection': ccrs.PlateCarree()})

# # Longitude tags
# for j in range(-90, 91):
#     for i in range(-180, 180):
        
#         ax1.text(i, j, str(i), size=.01,
#                   ha='center', va='center',
#                   zorder=1)
        
# Land
ax1.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                             name='land',
                                             scale='50m'),
                facecolor='#ccc',
                edgecolor='black',
                linewidth=.25,
                zorder=0)



#%%% OCEANS

# Boundaries for oceans are stored in text files. The geoutils module has a 
# function to load text files with lon, lat point pairs:
# 
# read_boundary()


#%%%% ATLANTIC

# Read Atlantic boundary points from file
fpath = 'deriveddata/ocean_wm_defs/atlantic/ocean/atlantic_boundary_points.txt'
atlantic_coords = gu.read_boundary(fpath)

# Map the boundary points to check result
atlantic_lon = [i[0] for i in atlantic_coords]
atlantic_lat = [i[1] for i in atlantic_coords]
ax1.scatter(atlantic_lon, atlantic_lat, s=.5, c='red', zorder=2)


#%%%% INDIAN

# Read Indian boundary points from file
fpath = 'deriveddata/ocean_wm_defs/indian/ocean/indian_boundary_points.txt'
indian_coords = gu.read_boundary(fpath)

# Map the boundary points to check result
indian_lon = [i[0] for i in indian_coords]
indian_lat = [i[1] for i in indian_coords]
ax1.scatter(indian_lon, indian_lat, s=.5, c='green', zorder=2)


#%%%% PACIFIC

# Read Pacific boundary points from file
fpath = 'deriveddata/ocean_wm_defs/pacific/ocean/pacific_boundary_points.txt'
pacific_coords = gu.read_boundary(fpath)

# Map the boundary points to check result
pacific_lon = [i[0] for i in pacific_coords]
pacific_lat = [i[1] for i in pacific_coords]
ax1.scatter(pacific_lon, pacific_lat, s=.5, c='blue', zorder=2)


#%%%% MEDITERRANEAN

# Read Western Mediterranean boundary points from file
fpath = 'deriveddata/ocean_wm_defs/mediterranean/ocean/wmed_boundary_points.txt'
wmed_coords = gu.read_boundary(fpath)

# Map the boundary points to check result
wmed_lon = [i[0] for i in wmed_coords]
wmed_lat = [i[1] for i in wmed_coords]
ax1.scatter(wmed_lon, wmed_lat, s=.5, c='pink', zorder=2)


# Same for Eastern Mediterranean 
fpath = 'deriveddata/ocean_wm_defs/mediterranean/ocean/emed_boundary_points.txt'
emed_coords = gu.read_boundary(fpath)

emed_lon = [i[0] for i in emed_coords]
emed_lat = [i[1] for i in emed_coords]
ax1.scatter(emed_lon, emed_lat, s=.5, c='gold', zorder=2)


#%%%% EXPORT MAP

ax1.set_global()

dpath = 'figures/ocean_wm_defs/helpers/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    
fpath = 'figures/ocean_wm_defs/helpers/helper_ocean_boundary_points.pdf'
fig1.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%%% WATER MASSES

# Set up geographical boundaries of water masses: [[lon, lat], [lon, lat], ...]
# They are meant to be (partly) rough boundaries -> excess at continents will
# be cropped with the OCEAN polygons.

# Geographical areas are based on literature. See:
# OceanICU/references/water_masses/water_mass_definitions.docx

# Store values in separate dicts for oceans, and central/intermediate/deep
# water masses
wm_box = {}


#%%%% ATLANTIC

atlantic_wm = {}
atlantic_wm['c'] = {}
atlantic_wm['c']['WNACW'] = [[-85, 15], [-85, 45], [-65, 48], [-55, 48],
                             [-25, 48], [-25, 15]]
atlantic_wm['c']['ENACW'] = [[-25, 60], [-5, 60], [-5, 58], [-5.2, 50],
                             [-3, 48], [8, 43], [8, 21], [-17, 21],
                             [-25, 15]]
atlantic_wm['c']['WSACW'] = [[-40, -18], [-4, -24], [7, -34], [7, -46],
                             [-69, -46]]
atlantic_wm['c']['ESACW'] = [[-67, 15], [-67, -18], [-40, -18], [-4, -24], 
                             [7, -34], [7, -46], [25, -46], [25, 21], 
                             [-17, 21], [-25, 15]]
atlantic_wm['i'] = {}
atlantic_wm['i']['SPMW'] = [[-3.5, 58.5], [-14.2, 65.4], [-21.7, 65.4],
                            [-29.7, 65.9], [-44, 65.9], [-44, 60], [-61, 54],
                            [-82, 48], [-82, 25], [0, 25], [0, 45], [-10, 50],
                            [-10, 54]]
atlantic_wm['i']['SAIW'] = [[-29.7, 65.9], [-43, 69], [-70, 69], [-70, 54],
                           [-54.3, 48], [-54.3, 45], [-15, 45], [-15, 55],
                           [-19.1, 63.4]]
atlantic_wm['i']['AAIW_A'] = [[-80, 25], [22, 25], [22, -65], [-80, -65]]
atlantic_wm['d'] = {}
atlantic_wm['d']['LSW'] = [[-29.7, 65.9], [-43, 69], [-70, 69], [-70, 54],
                           [-54.3, 48], [-54.3, 45], [-15, 45], [-15, 55],
                           [-19.1, 63.4]]
atlantic_wm['d']['MW'] = [[-17, 42.5], [-17, 30], [-4, 30], [-4, 42.5]]
atlantic_wm['d']['UNADW'] = [[-85, 45], [-85, -40], [22, -40], [22, 45]]
atlantic_wm['d']['CDW'] = [[-85, -40], [-85, -80], [22, -80], [22, -40]]

# Store in Atlantic water masses in dict for all oceans
wm_box['atlantic'] = atlantic_wm


#%%%% INDIAN

indian_wm = {}
indian_wm['c'] = {}
indian_wm['c']['ASW'] = [[50, 8.2], [40, 7], [40, 25], [55, 35], [75, 28],
                         [77.5, 8.2]]
indian_wm['c']['IUW'] = [[116, -22], [100, -22], [100, -18], [85, -18],
                         [85, -13], [70, -13], [70, -6], [85, -6], [85, -2],
                         [100, -2], [120, -2], [130, -7], [130, -22]]
indian_wm['c']['IEW'] = [[116, -25], [116, -22], [100, -22], [100, -18], 
                         [85, -18], [85, -13], [70, -13], [70, -6], [85, -6], 
                         [85, -2], [100, -2], [110, -2], [110, 29],
                         [77.5, 29], [77.5, 8.2], [50, 8.2], [34, 8.2],
                         [34, -15], [47, -18], [90, -25]]
indian_wm['c']['SICW'] = [[34, -15], [47, -18], [90, -25], [116, -25],
                          [150, -30], [150, -46], [18, -46], [18, -15]]
indian_wm['i'] = {}
indian_wm['i']['IIW'] = [[116, -22], [100, -22], [100, -18], [85, -18],
                         [85, -13], [70, -13], [70, -6], [85, -6], [85, -2],
                         [100, -2], [120, -2], [130, -7], [130, -22]]
indian_wm['i']['AAIW_I'] = [[116, -22], [100, -22], [100, -18], [85, -18],
                            [85, -13], [70, -13], [70, -6], [50, -6], [38, -7],
                            [20, -20], [20, -65], [150, -65], [150, -30]]
indian_wm['i']['RSPGIW'] = [[50, -6], [70, -6], [85, -6], [85, -2],
                            [100, -2], [105, -2], [105, 35], [55, 35],
                            [40, 25], [40, 7], [38, -7]]

wm_box['indian'] = indian_wm


#%%%% PACIFIC

pacific_wm = {}
pacific_wm['c'] = {}
pacific_wm['c']['WNPCW'] = [[140, 42], [179, 42], [-179, 42], [-160, 42],
                            [-170, 10], [117, 10], [117, 35]]
pacific_wm['c']['ENPCW'] = [[-160, 42], [-170, 10], [-113, 10], [-132, 28],
                            [-138, 42]]
pacific_wm['c']['CCST'] = [[-113, 10], [-132, 28], [-138, 42], [-126, 50],
                           [-114, 50], [-81, 20], [-81, 10]]
pacific_wm['c']['NPEW'] = [[117, 10], [117, 0], [-74, 0], [-74, 10]]
pacific_wm['c']['SPEW'] = [[124, 0], [124, -15], [-90, -15], [-80, -5],
                           [-77, 0]]
pacific_wm['c']['WSPCW'] = [[141, -15], [141, -46], [177, -46], [-155, -26],
                            [-155, -15]]
pacific_wm['c']['ESPCW'] = [[177, -46], [-155, -26], [-155, -15], [-90, -15],
                            [-110, -46]]
pacific_wm['c']['PCCST'] = [[-110, -46], [-90, -15], [-80, -5], [-77, 0],
                            [-60, 0], [-70, -46]]
pacific_wm['c']['PSUW'] = [[-138, 42], [-126, 50], [-126, 60], [135, 60],
                           [135, 42]]
pacific_wm['i'] = {}
pacific_wm['i']['NPIW'] = [[118, 18], [118, 24], [144, 44], [144, 44],
                           [157, 52], [-112, 52], [-101, 18]]
pacific_wm['i']['PEqIW'] = [[118, 18], [118, -6], [145, -6], [-171, -15],
                            [-135, -15], [-94, -22], [-70, -22], [-75, 18]]
pacific_wm['i']['AAIW_P'] = [[143, -6], [145, -6], [-171, -15], [-135, -15],
                             [-94, -22], [-65, -22], [-65, -65],
                             [143, -65]]
wm_box['pacific'] = pacific_wm


#%%%% MEDITERRANEAN

# As the are not that many data points, the Mediterannean will be kept simple,
# with a single central, intermediate and deep water mass taking up each of the
# two basins.
# The basins are defined separately to make it easier the defintion of the 
# water masses. They will be later joined to export the full mediterranean
# polygon.
wmed_wm = {}
wmed_wm['i'] = {}
wmed_wm['i']['WMIW'] = [[-8, 34], [-8, 46], [17, 46], [17, 34]]

wmed_wm['d'] = {}
wmed_wm['d']['WMDW'] = [[-8, 34], [-8, 46], [17, 46], [17, 34]]

wm_box['wmed'] = wmed_wm


emed_wm = {}
emed_wm['i'] = {}
emed_wm['i']['LIW'] = [[7, 29], [7, 47], [37, 47], [37, 29]]

emed_wm['d'] = {}
emed_wm['d']['EMDW'] = [[7, 29], [7, 47], [37, 47], [37, 29]]

wm_box['emed'] = emed_wm


#%% CREATE POLYGONS

# Due to crossing the dateline (or antimeridian), polygons in the Pacific will
# create issues.
#
# The official GeoJSON standard advocates splitting geometries along the 
# antimeridian: https://datatracker.ietf.org/doc/html/rfc7946#section-3.1.9
# 
# | «In representing Features that cross the antimeridian, interoperability is 
# |  improved by modifying their geometry. Any geometry that crosses the 
# |  antimeridian SHOULD be represented by cutting it in two such that neither 
# |  part's representation crosses the antimeridian.
# |
# |  [...]
# |
# |  A rectangle extending from 40 degrees N, 170 degrees E across the 
# |  antimeridian to 50 degrees N, 170 degrees W should be cut in two and 
# |  represented as a MultiPolygon.»
# 
# 
# This is a good option when doing operations with polygons, but for plotting 
# purposes this is not ideal as the cutting line will be visible. To avoid 
# this, it will be enough to shift the longitudes, converting [-180, 180] into
# [0, 360].
#
# Functions to deal with such issues have been created in the custom geoutils 
# module:
# 
#   shift_lon_180_180(),
#   shift_lon_0_360(), 
#   split_poly_in_antimeridian()
# 
# 
# Additional considerations:
# 
# Points set manually for water masses will be relatively sparse and, when 
# mapping the resulting polygons, lines joining points (i.e. polygon 
# boundaries) can be represented in a bit undexpected ways depending of 
# projection (straigh line where an arched line would be expected and so on).
# To minimise this, interpolating more points in between can help "force" the
# expected paths. The geoutils module has a function for this:
# 
# interpolate_coords()



#%%% OCEANS

# Create the polygon for the oceans based on the coordinates
poly_oceans = {}
poly_oceans['atlantic'] = geometry.Polygon(atlantic_coords)
poly_oceans['indian'] = geometry.Polygon(indian_coords)
poly_oceans['pacific'] = gu.split_poly_in_antimeridian(pacific_coords)
poly_oceans['wmed'] = geometry.Polygon(wmed_coords)
poly_oceans['emed'] = geometry.Polygon(emed_coords)

# geometry.Polygon(pacific_coords) # to check how it does not work otherwise


# Store the unsplitted versions for mapping purposes
uncut_polys = [poly_oceans['atlantic'],
               poly_oceans['indian'],
               geometry.Polygon(gu.shift_lon_0_360(pacific_coords)),
               poly_oceans['wmed'],
               poly_oceans['emed']]
poly_oceans_map = {k: v for k, v in zip(poly_oceans.keys(), uncut_polys)}


#%%%% MAP

# Plot cut or uncut polygons?
plot_cut_polygons = True
if plot_cut_polygons:
    polys_to_plot = deepcopy(poly_oceans)
else:
    polys_to_plot = deepcopy(poly_oceans_map)


# Set colours
ocean_pal = {k: v for k, v in zip(poly_oceans.keys(),
                                  ['#44AA99', '#88CCEE', '#EE8866',
                                   '#DDCC77', '#999933'])}

# Map
kws = {'projection': ccrs.Mollweide(central_longitude=-160)}
fig_p, ax_p = plt.subplots(nrows=1, ncols=1, figsize=(15*cm, 8*cm),
                           subplot_kw=kws)
for p in polys_to_plot:
    if isinstance(polys_to_plot[p], geometry.polygon.Polygon):
        ax_p.add_geometries(polys_to_plot[p],
                            edgecolor=ocean_pal[p],
                            facecolor=(ocean_pal[p] + '55'),
                            crs=ccrs.PlateCarree(),
                            zorder=0)
    elif isinstance(polys_to_plot[p], geometry.collection.GeometryCollection):
        for pp in polys_to_plot[p].geoms:
            ax_p.add_geometries(pp,
                                edgecolor=ocean_pal[p],
                                facecolor=(ocean_pal[p] + '55'),
                                crs=ccrs.PlateCarree(),
                                zorder=0)
ax_p.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                              name='land',
                                              scale='50m'),
                 facecolor='#ccc',
                 edgecolor='black',
                 linewidth=.25,
                 zorder=1)
ax_p.set_global()

fpath = 'figures/ocean_wm_defs/ocean_polygons.pdf'
fig_p.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%%% WATER MASSES

# Create polygons of water masses based on the established geographical 
# boundaries, trimming them if necessary based on the boundaries set for the
# oceans
wm_poly = deepcopy(wm_box)
wm_poly_map = deepcopy(wm_box)

# Iterate through oceans
for o in wm_box:
    
    # Get associated ocean polygon
    po = poly_oceans[o]
    
    # Iterate through water mass depths (central, intermediate, deep)
    for d in wm_box[o]:
        
        # Iterate through water masses
        for w in wm_box[o][d]:
        
            # Get boundary coordinates
            w_coord = wm_box[o][d][w]
            
            # Evaluate whether it crosses the antimeridian
            # if difference value between subsequent longitudes is > 180 it 
            # means that the shortest path is crossing the antimeridian (which
            # it does, because that is how we have designed the polygons, 
            # taking the long way makes no sense).
            lon = [i[0] for i in w_coord]
            does_cross_antimeridian = any([abs(lon[i+1]-lon[i]) > 180 for i in range(len(lon)-1)])
            
            # Interpolate coordinates and create polygon 
            # (split polygon at antimeridian if it crosses it)
            if does_cross_antimeridian:
                
                # To interpolate, shift coords to [0, 360] and then back
                w_coord_i = gu.interpolate_coords(gu.shift_lon_0_360(w_coord))
                w_coord_i = gu.shift_lon_180_180(w_coord_i)
                
                # Create poly and split it
                w_bound = gu.split_poly_in_antimeridian(w_coord_i)
                
            else:
                
                # Interpolate and create polygon
                w_coord_i = gu.interpolate_coords(w_coord)
                w_bound = geometry.Polygon(w_coord_i)
    
            # Shed excess area that falls out of the ocean polygon
            w_bound2 = w_bound.difference(po.symmetric_difference(w_bound))

            # Create the unsplitted versions too, for mapping purposes
            pom = poly_oceans_map[o]
            if does_cross_antimeridian:
                # With lons shifted to [0, 360]
                w_coord_i = gu.interpolate_coords(gu.shift_lon_0_360(w_coord))
                w_bound_unsplitted = geometry.Polygon(w_coord_i)
                w_bound_unsplitted2 = w_bound_unsplitted.difference(pom.symmetric_difference(w_bound_unsplitted))
            else:
                w_coord_i = gu.interpolate_coords(w_coord)
                w_bound_unsplitted = geometry.Polygon(w_coord_i)
                w_bound_unsplitted2 = w_bound_unsplitted.difference(po.symmetric_difference(w_bound_unsplitted))          
            
            # Store resulting polygon for water mass w
            wm_poly[o][d][w] = w_bound2
            wm_poly_map[o][d][w] = w_bound_unsplitted2


# Put together the water masses of the wmed and emed dicts
wm_poly['mediterranean'] = {}
wm_poly['mediterranean']['i'] = (wm_poly['wmed']['i'] | wm_poly['emed']['i'])
wm_poly['mediterranean']['d'] = (wm_poly['wmed']['d'] | wm_poly['emed']['d'])
wm_poly_map['mediterranean'] = {}
wm_poly_map['mediterranean']['i'] = (wm_poly_map['wmed']['i'] | wm_poly_map['emed']['i'])
wm_poly_map['mediterranean']['d'] = (wm_poly_map['wmed']['d'] | wm_poly_map['emed']['d'])

# Remove the separate wmed and emed
wm_poly.pop('wmed')
wm_poly.pop('emed')
wm_poly_map.pop('wmed')
wm_poly_map.pop('emed')


#%%%% MAP

# Plot separate maps for each water mass depth
depth_labels = {'c': 'Central', 'i': 'Intermediate', 'd': 'Deep'}
nc = 1
nr = len(depth_labels)


wm_depths_idx = {k:v for k, v in zip(['c', 'i', 'd'], [0, 1, 2])}


# Set colours
ocean_pal = {k: v for k, v in zip(wm_poly_map.keys(),
                                  ['#44AA99', '#88CCEE', '#EE8866',
                                   '#DDCC77'])}


#### Map for checking purposes (plot splitted polygons)

# Initialise figure and iterate through water masses
kws = {'projection': ccrs.Mollweide(central_longitude=-160)}
fig_w, ax_w = plt.subplots(nrows=nr, ncols=nc,
                           figsize=(14*cm, 6*cm * nr),
                           subplot_kw=kws)
for io, o in enumerate(wm_poly):
    for d in wm_poly[o]:
        for w in wm_poly[o][d]:
            
            # Map water mass polygon
            wmp = wm_poly[o][d][w]
            i = wm_depths_idx[d]
            ax_w[i].add_geometries(wmp,
                                   edgecolor=ocean_pal[o],
                                   facecolor=(ocean_pal[o] + '55'),
                                   linewidth=.2,
                                   crs=ccrs.PlateCarree(),
                                   zorder=0)
    
            # Add w label
            if isinstance(wmp, geometry.polygon.Polygon):
                p_coords = [[c[0], c[1]] for c in list(wmp.exterior.coords)]
                x, y = pl.polylabel(p_coords)
                ax_w[i].text(x, y, w, size=3, fontweight='bold', color='k',
                             ha='center', va='center',
                             transform=ccrs.PlateCarree())
            elif (isinstance(wmp, geometry.collection.GeometryCollection) |
                  isinstance(wmp, geometry.multipolygon.MultiPolygon)):
                for g in wmp.geoms:
                    p_coords = [[c[0], c[1]] for c in list(g.exterior.coords)]
                    x, y = pl.polylabel(p_coords)
                    ax_w[i].text(x, y, w, size=3, fontweight='bold', color='k',
                                 ha='center', va='center',
                                 transform=ccrs.PlateCarree())


for i, d in enumerate(depth_labels):
    
    # Add water mass layer label
    # (can't do it in the previos loop because one would be add every time it
    # goes through an ocean)
    ax_w[i].text(.005, .975, depth_labels[d], size=6, fontweight='bold',
                 ha='left',
                 transform=ax_w[i].transAxes)
    
    # Add land (do it separately here to avoid plotting over the same)
    ax_w[i].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                      name='land',
                                                      scale='50m'),
                        facecolor='#ccc',
                        edgecolor='black',
                        linewidth=.25,
                        zorder=1)
    
    # Make sure to show all globe
    ax_w[i].set_global()


fpath = 'figures/ocean_wm_defs/watermass_polygons_checking.pdf'
fig_w.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#### Proper map


# Get all wm codes
wm_labels = [w for o in wm_poly for d in wm_poly[o] for w in wm_poly[o][d]]
wm_labels = {k:("$\mathregular{" + v + "}$") for k, v in zip(wm_labels, wm_labels)}

# Most are already valid labels, except for the AAIWs
# Do it bold manually as setting fontweight='bold' leaves subscripts of AAIWs
# not being bold...
wm_labels_b = {k:("$\mathbf{" + v + "}$") for k, v in zip(wm_labels, wm_labels)}



# Initialise figure and iterate through water masses
kws = {'projection': ccrs.Mollweide(central_longitude=-160)}
fig_w, ax_w = plt.subplots(nrows=nr, ncols=nc,
                           figsize=(14*cm, 6*cm * nr),
                           subplot_kw=kws)
for io, o in enumerate(wm_poly_map):
    for d in wm_poly_map[o]:
        for w in wm_poly_map[o][d]:
            
            
            # When plotting polygons, do different border line for the two
            # overlapping exceptions
            lstyle = ':' if w in ['SAIW', 'MW'] else '-'
            wmp = wm_poly_map[o][d][w]
            i = wm_depths_idx[d]
            ax_w[i].add_geometries(wmp,
                                   edgecolor=ocean_pal[o],
                                   facecolor=(ocean_pal[o] + '55'),
                                   linewidth=.5,
                                   linestyle=lstyle,
                                   crs=ccrs.PlateCarree(),
                                   zorder=0)
    
            # Add w label
            # Use polylabel() in general but in some exceptions the centroid
            # position is actually visually better in this case.
            wm_excep = ['ENACW','WSACW', 'SICW', 
                        'SAIW', 'SPMW',
                        'NPEW',
                        'PEqIW', 'AAIW_P',
                        'LSW', 'CDW']
            if w not in wm_excep:
                p_coords = [[c[0], c[1]] for c in list(wmp.exterior.coords)]
                x, y = pl.polylabel(p_coords)
                ax_w[i].text(x, y, wm_labels_b[w], 
                             size=3, color='k',
                             ha='center', va='center',
                             transform=ccrs.PlateCarree())
            else:
                x, y = wmp.centroid.x, wmp.centroid.y
                ax_w[i].text(x, y, wm_labels_b[w],
                             size=3, fontweight='bold', color='k',
                             ha='center', va='center',
                             transform=ccrs.PlateCarree())


for i, d in enumerate(depth_labels):
    
    # Add water mass layer label
    # (can't do it in the previos loop because one would be add every time it
    # goes through an ocean)
    ax_w[i].text(0, .98, depth_labels[d], size=8, fontweight='bold',
                 color='#222',
                 ha='left',
                 transform=ax_w[i].transAxes)
    
    # Plot background to white
    ax_w[i].set_facecolor('w')
    
    # Add land
    ax_w[i].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                      name='land',
                                                      scale='50m'),
                        facecolor='#ccc',
                        edgecolor='black',
                        linewidth=.25,
                        zorder=1)
    
    # Make sure to show all globe
    ax_w[i].set_global()

# But figure background transparent
fig_w.set_facecolor('none')


fpath = 'figures/ocean_wm_defs/watermass_polygons.pdf'
fig_w.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)

fpath = fpath.replace('pdf', 'png')
fig_w.savefig(fpath, format='png', bbox_inches='tight', transparent=False, dpi=600)


#### Proper map with annotations

## First load SIGMA0 ranges to add this information to the figure
fpath = 'deriveddata/ocean_wm_defs/water_mass_sigma0_definitions.txt'
wm_sigma0 = pd.read_csv(fpath, sep='\t')
wm_sigma0 = {k:[v1, v2] for k, v1, v2 in zip(wm_sigma0.water_mass,
                                             wm_sigma0.sigma0_min,
                                             wm_sigma0.sigma0_max)}


# Initialise figure and iterate through water masses

# As maps and "plots" are alternated, projection cannot be set for all subplots
# and thus subplots need to be added one at a time.

nc = 2
mproj = ccrs.Mollweide(central_longitude=-160)
cntr = 0

fig_w_a = plt.figure(figsize=(7*cm * nc, 6*cm * nr))
gs = fig_w_a.add_gridspec(nrows=nr, ncols=nc, 
                          hspace=.2, wspace=.05,
                          width_ratios=(3, 1))
axs = []
for i in range(nr):
    for j in range(nc):
        if j==0:
            axs.append(fig_w_a.add_subplot(gs[i, j], projection=mproj))
        else:
            axs.append(fig_w_a.add_subplot(gs[i, j]))

# Map water mass polygons

max_char = max([len(wm_labels[w]) for w in wm_labels])

# Control additions of labels
cntr_c = 0
cntr_i = 0
cntr_d = 0
for io, o in enumerate(wm_poly_map):
    for d in wm_poly_map[o]:
        for iw, w in enumerate(wm_poly_map[o][d]):
            
            
            # When plotting polygons, do different border line for the two
            # overlapping exceptions
            lstyle = ':' if w in ['SAIW', 'MW'] else '-'
            wmp = wm_poly_map[o][d][w]
            i = wm_depths_idx[d]
            ax = axs[i*2]
            ax.add_geometries(wmp,
                              edgecolor=ocean_pal[o],
                              facecolor=(ocean_pal[o] + '55'),
                              linewidth=.5,
                              linestyle=lstyle,
                              crs=ccrs.PlateCarree(),
                              zorder=0)
    
            # Add w label
            # Use polylabel() in general but in some exceptions the centroid
            # position is actually visually better in this case. Also, adjust
            # font size of labels in some cases
            small_labs = ['ASW', 'CCST', 'PCCST', 'ENACW', 'RSPGIW', 'MW',
                          'WMIW', 'WMDW', 'LIW', 'EMDW']
            fs = 3 if w in small_labs else 3.5
            wm_excep = ['ENACW','WSACW', 'SICW', 
                        'SAIW', 'SPMW',
                        'NPEW',
                        'PEqIW', 'AAIW_P',
                        'LSW', 'CDW',
                        'WMIW', 'WMDW', 'LIW', 'EMDW']
            if w not in wm_excep:
                p_coords = [[c[0], c[1]] for c in list(wmp.exterior.coords)]
                x, y = pl.polylabel(p_coords)
                if w=='WSPCW': x = x + 7
                if w=='PSUW': 
                    y = y - 5
                    x = x - 15
                ax.text(x, y, wm_labels_b[w],
                        size=fs, color='k',
                        ha='center', va='center',
                        transform=ccrs.PlateCarree())
            else:
                x, y = wmp.centroid.x, wmp.centroid.y
                if w=='SPMW': y = y - 5
                if w=='ENACW': y = y + 3
                if (w=='WMIW') | (w=='WMDW'):
                    x = x - 1
                    y = y - 6
                if (w=='LIW') | (w=='EMDW'): 
                    x = x + 20
                    y = y - 3
                ax.text(x, y, wm_labels_b[w],
                        size=fs, fontweight='bold', color='k',
                        ha='center', va='center',
                        transform=ccrs.PlateCarree())
                    
            ax = axs[i*2 + 1]
            ax.set(xticks=[], yticks=[])
            
            txt = ('{:.1f}'.format(wm_sigma0[w][0]) +
                   " – " +
                   '{:.1f}'.format(wm_sigma0[w][1]))
            x = .15
            x2 = x + .45
            y_top = .87
            fs = 4.5
            va = 'center_baseline'
            if d=='c':
                y = y_top - cntr_c * .05
                ax.text(x, y, wm_labels_b[w], size=fs, color=ocean_pal[o],
                        ha='left', va=va, transform=ax.transAxes)
                ax.text(x2, y, txt, size=fs, color='k',
                        ha='left', va=va, transform=ax.transAxes)
                cntr_c += 1
            elif d=='i':
                y = y_top - cntr_i * .05
                ax.text(x, y, wm_labels_b[w], size=fs, color=ocean_pal[o],
                        ha='left', va=va, transform=ax.transAxes)
                ax.text(x2, y, txt, size=fs, color='k',
                        ha='left', va=va, transform=ax.transAxes)
                cntr_i += 1
            else:
                y = y_top - cntr_d* .05
                ax.text(x, y, wm_labels_b[w], size=fs, color=ocean_pal[o],
                        ha='left', va=va, transform=ax.transAxes)
                ax.text(x2, y, txt, size=fs, color='k',
                        ha='left', va=va, transform=ax.transAxes)
                cntr_d += 1
                

# Add land and annotations
for i, d in enumerate(depth_labels):
    
    ## Maps
    
    ax = axs[i*2]
    
    # Add land
    ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                      name='land',
                                                      scale='50m'),
                         facecolor='#ccc',
                         edgecolor='black',
                         linewidth=.25,
                         zorder=1)
    
    # Make sure to show all globe
    ax.set_global()
    
    # Add water mass layer label
    # (can't do it in the previos loop because one would be add every time it
    # goes through an ocean)
    ax.text(0, .98, depth_labels[d], size=8, fontweight='bold',
                 color='#222',
                 ha='left',
                 transform=ax.transAxes)
        
    # Map background to white
    ax.set_facecolor('w')
    
    
    ## Tables
    
    ax = axs[i*2 + 1]
    
    # Add headers
    y = y_top + .05
    ax.text(x, y, "$\mathbfit{Water\ mass}$", size=fs, color='#555',
            ha='left', va='baseline', transform=ax.transAxes)
    ax.text(x2, y, ("$\mathbfit{\sigma_{\\theta}}$ " + 
                    "$\mathregular{[kg\ m^{-3}]}$"),
            size=fs, color='#555',
            ha='left', va='baseline', transform=ax.transAxes)
    
    # Remove spines and background
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_facecolor('none')
    

# Figure background transparent
fig_w_a.set_facecolor('none')


# Export
fpath = 'figures/ocean_wm_defs/watermass_polygons_annot.pdf'
fig_w_a.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)

fpath = fpath.replace('pdf', 'png')
fig_w_a.savefig(fpath, format='png', bbox_inches='tight', transparent=False, dpi=600)

fpath = fpath.replace('png', 'svg')
fig_w_a.savefig(fpath, format='svg', bbox_inches='tight', transparent=False, dpi=600)


#%% EXPORT

#%%% OCEANS


# Unite Med polygons into a single one (and get rid of basin polygons)
poly_oceans['mediterranean'] = unary_union([poly_oceans['wmed'], 
                                            poly_oceans['emed']])
poly_oceans.pop('wmed')
poly_oceans.pop('emed')


# Save polygons in geojson format. To use them, load with from_geojson()
for o in poly_oceans:
    
    dpath = 'deriveddata/ocean_wm_defs/' + o + '/ocean/'
    if not os.path.exists(dpath): os.makedirs(dpath)
        
    fpath = Path(dpath + o + '_polygon.geojson')
    fpath.write_text(to_geojson(poly_oceans[o]))
    
    if o=='pacific':
        
        dpath = 'deriveddata/ocean_wm_defs/' + o + '/ocean/uncut/'
        if not os.path.exists(dpath): os.makedirs(dpath)
            
        fpath = Path(dpath + o + '_polygon_uncut.geojson')
        fpath.write_text(to_geojson(poly_oceans_map[o]))


#%%% WATER MASSES

# Save water mass polygons
for o in wm_poly:
    for d in wm_poly[o]:
        for w in wm_poly[o][d]:
            
            dpath = 'deriveddata/ocean_wm_defs/' + o + '/wms/' + depth_labels[d].lower() + '/'
            if not os.path.exists(dpath): os.makedirs(dpath)
            
            fpath = Path(dpath + w + '_polygon.geojson')
            fpath.write_text(to_geojson(wm_poly[o][d][w]))
            
            # Save uncut versions for the Pacific
            if o=='pacific':
                
                dpath = 'deriveddata/ocean_wm_defs/' + o + '/wms/uncut/' + depth_labels[d].lower() + '/'
                if not os.path.exists(dpath): os.makedirs(dpath)
                
                fpath = Path(dpath + w + '_polygon_uncut.geojson')
                fpath.write_text(to_geojson(wm_poly_map[o][d][w]))


