# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:17:40 2023

@author: Markel Gómez Letona

Module with utility functions to work with Longhurst provinces. 

Depends on the pyshp, shapely, pandas packages.
cartopy too, if mapping utility functions are used
_______________________________________________________________________________

"""

import shapefile # this is 'pyshp'
import shapely as sy
from pandas import read_csv
import cartopy.io.shapereader as shpreader


# -----------------------------------------------------------------------------


## Read Longhurst province shapefile
fpath = 'rawdata/longhurst/Longhurst_world_v4_2010'
try:
    sf = shapefile.Reader(fpath)
except Exception as e: # ShapefileException is not recognised?
    if 'Unable to open' in str(e):
        print("Shapefile of Longhurst provinces (Longhurst_world_v4_2010) not " 
              "found!"
              "\n\nCheck that it is correctly named or that it is in the "
              "correct location (rawdata/longhurst/)."
              "\n\nOtherwise, download data from:"
              "\n\nhttps://www.marineregions.org/downloads.php#longhurst"
              "\n\nPlace file (longhurst_v4_2010.zip) in the "
              "rawdata/longhurst/ folder of the project and unzip it.")
    else:
        print("Error when reading Longhurst shapefile:\n\n" + str(e))
            

## Read shapefile's geometry
shapes = sf.shapes()

## Get the records:
records = sf.records()

## Extract province codes
provcodes = [i[0] for i in records]

## Convert shapes to shapely polygons
lh_polys = []
for i in range(len(shapes)):
    lh_polys.append(sy.geometry.shape(shapes[i]))
    
    
## Read shapefile with Cartopy's reader so that easily plottable shapes can be
## retrieved
cartopy_reader = shpreader.Reader(fpath)
longprovs = cartopy_reader.records()
cartopy_polygons = []
cartopy_polygons_attributes = []
for i in longprovs:
    cartopy_polygons.append(i.geometry)
    cartopy_polygons_attributes.append(i.attributes['ProvCode'])

## Read metadata of Longhurst provinces

# The source file is provided in: https://www.marineregions.org/sources.php#longhurst
# It's .xls file. Here the .csv file is the same but without the pre-header 
# lines and without the empty columns between data columns.
fpath2 = 'rawdata/longhurst/Longhurst_Province_Summary.csv'
try:
    lmeta = read_csv(fpath2, sep=';', header=0)
except FileNotFoundError:
    print("No such file or directory: " + fpath2,
          "\n\nThe source file can be obtained in: https://www.marineregions.org/sources.php#longhurst"
          "\n\nRemove pre-header lines and empty columns between data columns,"
          " and convert to ';'-separated .csv")


## Utility functions


def find_longhurst(x, y, out='code'):   
    """
    Find the Longhurst province to which a point of x,y coordinates belongs.
    
    https://www.marineregions.org/downloads.php#longhurst

    Parameters
    ----------
    x : FLOAT
        LONGITUDE of point. Must be a float or coercible to float.
    y : FLOAT
        LATITUDE of point. Must be a float or coercible to float.
    out : STRING, optional
        Output type, one of 'code' (province code) or 'name' (full name of 
        province). The default is 'code'.

    Returns
    -------
    The Longhurst province code or full name for coordinates x,y. If a point
    does not belong to any province (e.g., land) 'NOT_IN_PROV' is returned.

    """
    x = float(x)
    y = float(y)

    # Get coordinates of point
    scoords = (x, y)
        
    # Check if point is in province polygons
    is_in_poly = sy.geometry.Point(scoords).within(lh_polys)
        
    # Get the province that returned True
    if sum(is_in_poly)==1:
        i = [i for i, x in enumerate(is_in_poly) if x][0]
        lp = {'code': records[i][0], 'name': records[i][1]}
    elif sum(is_in_poly)==0:
        lp = {'code': 'NOT_IN_PROV', 'name': 'Point not in Longhurst province. Likely land, or unassigned area close to Antarctica.'}
    else:
        lp = {'code': 'MULTIPLE_MATCHES', 'name': 'Edge case? Or something likely went wrong...'}
    return(lp[out])


def longhurst_meta(prov=None):
    """
    Returns metadata associated to an specific Longhurst province. If no input
    is provided, the entire metadata table is returned.
    
    Summary values by Mathias Taeger and David Lazarus, Museum für Naturkunde, 
    Berlin. 26.3.2010. https://www.marineregions.org/sources.php#longhurst

    Parameters
    ----------
    prov : STRING, optional
        Longhurst province code for which metadata is to be retrieved. The
        default is None, i.e., returns a dataframe with the metadata of all 
        provinces.

    Returns
    -------
    Dictionary with the metadata for the specified Longhurst province:
        'PROVCODE' : Longhurst province code.
        'PROVDESCR' : Longhurst province full name.
        'Biome' : 'C'=coastal, 'P'=polar, 'T'=trade winds, 'W'=westerlies.
        'productivity_gC_m2_d' : integrated primary productivity, gC·m-2·d-1.
        'prod_class' : productivity class, 1-5:
                       very low (1) = <0.4
                       low (2) = <0.8
                       middle (3) = <1.2
                       high (4) = <1.6
                       very high (5) = >1.6
         'chl_mg_m2' : integrated Chl-a, mg·m-2.
         'chl_class' : Chl-a class, 1-5:
                       very low (1) = <5
                       low (2) = <10
                       middle (3) = <15
                       high (4) = <20
                       very high (5) = >25
         'photic_depth_m' : depth of photic layer, m.
         'photic_class' : photic layer class, 1-5:
                          very low (1) = <30
                          low (2) = <40
                          middle (3) = <50
                          high (4) = <60
                          very high (5) = >60            
         'mld_sigma_m' : mixed layer depth, m.
         'mld_class' : mixed layer depth class, 1-5:
                       very low (1) = <20
                       low (2) = <40
                       middle (3) = <60
                       high (4) = <80
                       very high (5) = >80                
         'temp_0_celsius' : temperature at 0 m depth, celsius.
         'temp_50_celsius' : temperature at 50 m depth, celsius.
         'temp_diff' : temperature difference between 0 and 50 m depth, celsius.
         
         If no input is provided, a dataframe is returned with the same
         variables for all provinces.

    """
    if prov==None:
        # If no input, return entire table
        provmeta = lmeta
    else:
        # Return metadata for specified province
        if not prov in provcodes:
            raise ValueError("invalid province code") 
        else:
            provmeta = dict(lmeta.loc[lmeta['PROVCODE']==prov, :].squeeze())
        
    return(provmeta)

def longhurst_shape(prov=None):
    """
    Returns Shape associated to an specific Longhurst province. If no input
    is provided, Shapes of all provinces are returned.

    Parameters
    ----------
    prov : STRING, optional
        Longhurst province code for which the shape is to be retrieved. The
        default is None, i.e., returns a Shapes object with shapes of all 
        provinces.

    Returns
    -------
    Shapes object for requested province, or list of all of them.

    """
    if prov==None:
        # If no input, return all shapes (+ records to identify provinces)
        shp = {'shapes': shapes, 'records': records}
    else:
        # Return shape for specified province
        if not prov in provcodes:
            raise ValueError("invalid province code")
        else:
            # get position of province
            idx = [i for i, x in enumerate([p==prov for p in provcodes]) if x]
            if len(idx)>1:
                raise ValueError("input matches multiple province codes...")
            else:
                shp = shapes[idx[0]]
    return(shp)

def longhurst_shape_cartopy(prov=None):
    """
    Returns Shape associated to an specific Longhurst province, ready to plot
    with cartopy. If no input is provided, Shapes of all provinces are 
    returned.

    Parameters
    ----------
    prov : STRING, optional
        Longhurst province code for which the shape is to be retrieved. The
        default is None, i.e., returns a Shapes object with shapes of all 
        provinces.

    Returns
    -------
    Polygon or Multypoligon object for requested province, or list with all
    of them.

    """

    if prov==None:
        # If no input, return all shapes
        shp = {'province': cartopy_polygons_attributes,
               'polygons': cartopy_polygons}
    else:
        # Return shape for specified province
        if not prov in provcodes:
            raise ValueError("invalid province code")
        else:
            # get position of province
            idx = [i for i, x in enumerate([p==prov for p in provcodes]) if x]
            if len(idx)>1:
                raise ValueError("input matches multiple province codes...")
            else:
                shp = cartopy_polygons[idx[0]]
    return(shp)