# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:26:56 2023

@author: Markel Gómez Letona


Testing method to check whether a point (station/sample) is within a polygon
(Longhurst province).

see:
    - https://pypi.org/project/pyshp/#reading-shapefiles
    - https://stackoverflow.com/a/36400130
    - https://stackoverflow.com/a/23453678
 

"""


#%% IMPORTS

import shapefile
import shapely as sy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

#%% LOAD DATA

# shapefile with longhurst provinces:
fpath = 'rawdata/longhurst/Longhurst_world_v4_2010'
sf = shapefile.Reader(fpath)

# Check shapetype:
sf.shapeTypeName

# Check the bounding box area that the shapefile covers in total:
sf.bbox

#### Read shapefile's geometry

# It's the collection of points or shapes made from vertices and implied arcs 
# representing physical locations.
shapes = sf.shapes()
len(shapes)

# list of tuples containing an (x,y) coordinate for each point in a shape:
pts = shapes[1].points


#### Read records

# It contains the attributes for each shape in the collection of geometries.
# They link the shape with the metadata (i.e. POLYGON <-> LONGHURST PROV).

# They are defined by a series of field names:
sf.fields # [(NAME, TYPE, LENGTH, DECIMALS)] // TYPE==C -> CHARACTER

# To get the records:
records = sf.records()

# Can access all the fields of an specific record:
r1 = records[1]

# Or to get all the records of an specific field:
provcodes = [x['ProvCode'] for x in records]


#### Load some oceanographic data for testing
    
# Filtered Hansell dataset:
fpath = 'deriveddata/dom_hansell/Hansell_2022_o2_doc_cfc11_cfc12_sf6_with_ages.csv'
hns = pd.read_csv(fpath, sep=",", header=0, dtype={'BOTTLE': str})
hns.replace(-999, np.nan, inplace=True)
# Subset, no need to have entire dataset for testing:
hns = hns.loc[hns['CRUISE']=='A16N (2003)',:]


#%% PLOT LONGHURST PROVINCES

# I really hate that there seems not to be a relatively easy way to map
# a "Shape" class object in Cartopy... or at least any way to convert it.

# Creater shapefile reader and read records:
fpath = 'rawdata/longhurst/Longhurst_world_v4_2010'
reader = shpreader.Reader(fpath)
longprovs = reader.records()

# Initialise figure with desired projection
subplot_kw = {'projection': ccrs.Robinson()}
cm = 1/2.54
fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(14*cm, 10*cm))

# Add land:
ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                            name='land',
                                            scale='50m'),
               facecolor='#aaa', edgecolor='k',
               linewidth=.2)

# Dict with coords for province labels (see note within loop):
prov_lab_coords = {
    ## Arctic
    'BPLR': (-145, 78),
    ## Atlantic
    'ARCT': (-9.5, 69.5),
    'SARC': (37, 72),
    'NADR': (-25.6, 49.6),
    'GFST': (-52, 40.8),
    'NASW': (-52.8, 32.2),
    'NATR': (-44.2, 18.7),
    'WTRA': (-33.3, 3.9),
    'ETRA': (-5.7, -2.9),
    'SATL': (-18.1, -25.5),
    'NECS': (3, 55.8),
    'CNRY': (-13.9, 21),
    'GUIN': (3.1, 5),
    'GUIA': (-55.3, 4.1),
    'NWCS': (-67, 43),
    'MEDI': (19.2, 34.6),
    'CARB': (-72, 15),
    'NASE': (-18, 35.1),
    'BRAZ': (-46.2, -27),
    'FKLD': (-59, -52.5),
    'BENG': (12.1, -25.7),
    ## Indian
    'MONS': (77.4, -2.1),
    'ISSG': (76.8, -24.7),
    'EAFR': (39, -25.7),
    'REDS': (45, 23.5),
    'ARAB': (56, 13.5),
    'INDE': (90, 18.9),
    'INDW': (72.3, 15.4),
    ## Pacific
    'AUSW': (112, -30.7),
    'BERS': (170, 55.3),
    'PSAE': (-148, 50.5),
    'PSAW': (168, 47.4),
    'KURO': (147, 38),
    'NPPF': (-155, 39.1),
    'NPSW': (155, 22.8),
    'TASM': (165, -37),
    'SPSG': (-125, -23.9),
    'NPTG': (-148, 22.2),
    'PNEC': (-115, 8.6),
    'PEQD': (-128, -1.5),
    'WARM': (155, 5),
    'ARCH': (158, -15),
    'ALSK': (-140, 59),
    'CCAL': (-122.5, 34.2),
    'CAMR': (-94.7, 14.4),
    'CHIL': (-74, -36),
    'CHIN': (121.1, 33),
    'SUND': (122.6, -1.2),
    'AUSE': (152.2, -25.9),
    'NEWZ': (175, -45.2),
    ## Antarctic // circumpolar
    'SSTC': (-1, -42.7),
    'SANT': (-145, -50),
    'ANTA': (-2, -61.1),
    'APLR': (-160, -74)
    }

# Dict for colours:
prov_cols = {
    ## Coastal Atlantic
    'BENG': '#007c55',
    'BRAZ': '#2f8c69',
    'CNRY': '#4c9b7d',
    'FKLD': '#66ab91',
    'GUIA': '#7fbba5',
    'GUIN': '#98cbb9',
    'NECS': '#b2dbcd',
    'NWCS': '#ccebe1',
    ## Coastal Indian
    'AUSW': '#007c55',
    'EAFR': '#2f8c69',
    'INDE': '#4c9b7d',
    'INDW': '#66ab91',
    'REDS': '#7fbba5',
    ## Coastal Pacific
    'ALSK': '#007c55',
    'AUSE': '#2f8c69',
    'CAMR': '#4c9b7d',
    'CCAL': '#66ab91',
    'CHIL': '#7fbba5',
    'CHIN': '#98cbb9',
    'NEWZ': '#b2dbcd',
    'SUND': '#ccebe1',
    ## Polar Antarctic
    'ANTA': '#a1ced7',
    'APLR': '#ecf5f7',
    ## Polar Arctic
    'BPLR': '#ecf5f7',
    ## Polar Atlantic
    'ARCT': '#a1ced7',
    'SARC': '#c7e2e7',
    ## Polar Pacific
    'BERS': '#a1ced7',
    ## Trade wind Atlantic
    'CARB': '#005b96',
    'ETRA': '#3671a5',
    'NATR': '#5688b3',
    'SATL': '#759ec2',
    'WTRA': '#93b6d1',
    ## Trade wind Indian
    'ISSG': '#005b96',
    'MONS': '#759ec2',
    ## Trade wind Pacific
    'ARCH': '#005b96',
    'NPTG': '#3671a5',
    'PEQD': '#5688b3',
    'PNEC': '#759ec2',
    'SPSG': '#93b6d1',
    'WARM': '#b3cde0',
    ## Westerly Antarctic 
    'SANT': '#e5ac37',
    'SSTC': '#f7ce7d',
    ## Westerly Atlantic
    'GFST': '#e5ac37',
    'MEDI': '#eab54a',
    'NADR': '#eab54a',
    'NASE': '#f2c66c',
    'NASW': '#f7ce7d',
    ## Westerly Indian
    'ARAB': '#e5ac37',
    ## Westerly Pacific
    'KURO': '#e5ac37',
    'NPPF': '#eab54a',
    'NPSE': '#eebd5b',
    'NPSW': '#f2c66c',
    'PSAE': '#f7ce7d',
    'PSAW': '#fbd78d',
    'TASM': '#ffdf9e',
    }


# Add each polygon individually:
# (this allows to plot only some, or with different colors, and so on)
for lp in longprovs:
    
    # if lp.attributes['ProvCode']=="NPPF":
    #     print(lp.attributes['ProvDescr'])
    # Plot polygon:
    ax.add_geometries(lp.geometry, ccrs.PlateCarree(),
                      facecolor=prov_cols[lp.attributes['ProvCode']],
                      edgecolor='none')
    # Add label:
    # (using polygon centroids [lp.geometry.centroid.x, lp.geometry.centroid.y]
    #  does not always give good results for label positioning, depends on 
    #  shape)
    # Thus, I've created a table with desired coords for labels.
    # x = lp.geometry.centroid.x  
    # y = lp.geometry.centroid.y
    x, y = prov_lab_coords[lp.attributes['ProvCode']]
    lb = lp.attributes['ProvCode']
    ax.text(x, y, lb, color='k', size=5,
            ha='center', va='center', 
            transform=ccrs.PlateCarree())
    # ax.scatter(x,y, transform=ccrs.PlateCarree())
    


fpath = "figures/longhurst/map_longhurst_provinces.pdf"
fig.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%% POINTS IN POLYGONS


## Find in which Longhurst provinces are samples located.

#### Test to find if samples are in a specific province (e.g. NASE):
idx = np.where(['NASE'==x['ProvCode'] for x in records])[0][0]
bshp = shapes[idx] # bounding shape
hns['in_NASE'] = np.nan
for ir, r in hns.iterrows():
    
    # Get coordinates of sample:
    scoords = (r['LONGITUDE'], r['LATITUDE']) # x,y tuple
    
    # Check if samples is in province:
    hns.loc[ir, 'in_NASE'] = sy.geometry.Point(scoords).within(sy.geometry.shape(bshp))



## Plot results to check:
    
# Reduce to one sample per station to avoid overcrowding with useless points
hns2 = hns.loc[~hns.duplicated(subset='STATION'),:].copy()

# Create vector with colours based on whether points are in or out of the prov
boo_cols = {True: '#C41E3A', False: '#36648B'}
in_NASE_colours = [boo_cols[x] for x in hns2['in_NASE']]

# Plot station points underlied by the province:
subplot_kw = {'projection': ccrs.Robinson()}
cm = 1/2.54
fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(10*cm, 10*cm))
ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                            name='land',
                                            scale='50m'),
               facecolor='#aaa', edgecolor='k',
               linewidth=.2,
               zorder=0)
ax.set_extent([-70, 20, -10, 80], crs=ccrs.PlateCarree())
longprovs = reader.records() # need to call again as generators can only be used once
for lp in longprovs:
    if lp.attributes['ProvCode']=='NASE':
        ax.add_geometries(lp.geometry, ccrs.PlateCarree(),
                          facecolor='#e7a5b0',
                          edgecolor='none',
                          zorder=1)
        ax.text(-14.9, 34, 'NASE', color='#4e0c17', size=5, weight='bold',
                ha='center', va='center', 
                transform=ccrs.PlateCarree(),
                zorder=2)
    else:
        ax.add_geometries(lp.geometry, ccrs.PlateCarree(),
                          facecolor='none',
                          edgecolor='k',
                          linewidth=.2,
                          zorder=1)
ax.scatter(x=hns2['LONGITUDE'], y=hns2['LATITUDE'],
           facecolor=in_NASE_colours, edgecolor='k',
           s=6, linewidth=.2,
           transform=ccrs.PlateCarree(),
           zorder=2)

fpath = "figures/longhurst/nase_a16n_test.pdf"
fig.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#### Generalise to find in which polygon each sample is

# This is quite time costly. Do it only with stations, and then propagate that
# information to all samples.
# Generalised for the whole dataset:
hns['STID'] = np.nan # unique id for stations across cruises
for ir, r in hns.iterrows(): 
    hns.loc[ir, 'STID'] = r['EXPOCODE'] + str(r['STATION'])
hns2 = hns.loc[~hns.duplicated(subset='STID'),:].copy()

hns2['LONGPROV'] = np.nan
for ir, r in hns2.iterrows():
    
    is_in_poly = False
    i = 0
    while not is_in_poly:
        
        # Get shape (province) i:
        bshp = shapes[i]
        
        # Get coordinates of sample:
        scoords = (r['LONGITUDE'], r['LATITUDE']) # x,y tuple
        
        # Check if sample is in province:
        is_in_poly = sy.geometry.Point(scoords).within(sy.geometry.shape(bshp))
        
        # Add up to continue search:
        i+=1

    # When the province is found, the while loop is existed.
    # The province of the sample will be i-1 (because we add 1 at the end of
    # each iteration)
    hns2.loc[ir, 'LONGPROV'] = records[i-1][0]

# Map results to the dataset:
hns = hns.merge(hns2[['STID', 'LONGPROV']], how='left', on='STID')


## Re-subset stations to make sure assignation was done properly:
hns3 = hns.loc[~hns.duplicated(subset='STID'),:].copy()

# Create vector with colours based on provs
cprov_points = {'SARC': '#90b9c1', 'NADR': '#954567', 'NASE': '#36802d',
                'NATR': '#d32a2d', 'WTRA': '#c0a73a'}
cprov_poly = {'SARC': '#ecf5f7', 'NADR': '#dfc7d1', 'NASE': '#aeccab',
              'NATR': '#f6d4d5', 'WTRA': '#f9edb6'}
cp_labcoords = {'SARC': (-13.2, 61.5), 'NADR': (-27, 50), 'NASE': (-14.9, 34),
              'NATR': (-34.5, 18.5), 'WTRA': (-31.5, 3)}

LONGPROV_colours = [cprov_points[x] for x in hns3['LONGPROV']]

# Plot station points underlied by the province:
subplot_kw = {'projection': ccrs.Robinson()}
cm = 1/2.54
fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(10*cm, 10*cm))
ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                            name='land',
                                            scale='50m'),
               facecolor='#aaa', edgecolor='k',
               linewidth=.2,
               zorder=0)
ax.set_extent([-70, 20, -10, 80], crs=ccrs.PlateCarree())
longprovs = reader.records() # need to call again as generators can only be used once
for lp in longprovs:
    if lp.attributes['ProvCode'] in cprov_points.keys():
        ax.add_geometries(lp.geometry, ccrs.PlateCarree(),
                          facecolor=cprov_poly[lp.attributes['ProvCode']],
                          edgecolor='none',
                          zorder=1)
        x, y = cp_labcoords[lp.attributes['ProvCode']]
        ax.text(x, y, lp.attributes['ProvCode'], 
                color=cprov_points[lp.attributes['ProvCode']], 
                size=5, weight='bold',
                ha='center', va='center', 
                transform=ccrs.PlateCarree(),
                zorder=2)
    else:
        ax.add_geometries(lp.geometry, ccrs.PlateCarree(),
                          facecolor='none',
                          edgecolor='k',
                          linewidth=.2,
                          zorder=1)
ax.scatter(x=hns3['LONGITUDE'], y=hns3['LATITUDE'],
           facecolor=LONGPROV_colours, edgecolor='k',
           s=6, linewidth=.2,
           transform=ccrs.PlateCarree(),
           zorder=2)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=.5, color='grey', alpha=0.5, linestyle='--',
                  xlabel_style = {'size': 7}, ylabel_style = {'size': 7})
gl.right_labels = False
gl.top_labels = False


fpath = "figures/longhurst/longprovs_a16n_test.pdf"
fig.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)

