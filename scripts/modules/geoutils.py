# -*- coding: utf-8 -*-
"""
Created on 2024-10-21

@author: Markel Gómez Letona

Utility functions to deal with various common operations when dealing with
geographic polygons, geojson coordinates, etc.

Depends on numpy and shapely.

_______________________________________________________________________________

"""

import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import split, unary_union
from shapely.affinity import translate
from shapely.validation import make_valid


# Load text files with coordiantes in lon, lat point pairs in columns
def read_boundary(fpath: str, sep: str = '\t', header: bool = True,
                  coord_order = 'lonlat'):
    """
    Load text file with lon, lat coordinates pairs in columns.

    Parameters
    ----------
    fpath : str
        Path to text file.
    sep : str, optional
        Column separator (delimiter). The default is '\t'.
    header : bool, optional
        Whether the text file has a header. The default is True.
    coord_order : str, optional
        Order of coordinate columns in the input file. Either 'lonlat' (first
        longitude, then latitude) or 'latlon' (opposite). The default is 
        'lonlat'.

    Returns
    -------
    b : list
        Coordinate list in geojson format: 
        [[lon, lat], [lon, lat], [lon, lat], ...].

    """
    
    # Open file and read lines
    with open(fpath) as f:
        lines = f.readlines()
        if header:
            lines = lines[1:] # remove header

    # Set in which order the lon lat columns are in the file
    if coord_order=='lonlat':
        clon = 0; clat = 1
    elif coord_order=='latlon':
        clon = 1; clat = 0
    else:
        raise ValueError("'" + coord_order + 
                         "' is not a valid option for coord_order.")
     
    # Convert into list of float coords with structure as
    # [[lon, lat], [lon, lat], [lon, lat], ....]        
    b = [] # create list for coords 
    for l in lines:
        a = [] # create sublist for lon,lat pair
        # remove newline and split by sep
        for i in l.replace('\n', '').split(sep):
            a.append(float(i))
        # Make sure to order properly based on input file so that it always
        # ends up in lon, lat order.
        a = [a[clon], a[clat]]
        b.append(a)
        
    return b



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
# For plotting purposes this is not ideal as the cutting line will be visible.
# To avoid this, it will be enough to shift the longitudes, converting
# [-180, 180] into [0, 360]. Function to perform this change.
#
# Create functions to deal with such issues:
    
def shift_lon_0_360(coord_pair_list: list):
    """
    Given a list of coordinates with the GeoJSON format 
    ([[lon, lat], [lon, lat], ...]), convert longitudes from [-180, 180] into 
    [0, 360].

    Parameters
    ----------
    coord_pair_list : list
        List of coordinates with longitudes in the range [-180, 180].

    Returns
    -------
    new_coord_pair_list : list
        List of coordinates with longitudes converted into the range [0, 360].

    """
    new_coord_pair_list = []
    for i in range(len(coord_pair_list)):
        ilon = coord_pair_list[i][0]
        if ilon < 0:
            ilon = ilon + 360    
        new_coord_pair_list.append([ilon, coord_pair_list[i][1]])      
    return new_coord_pair_list


def shift_lon_180_180(coord_pair_list: list):
    """
    Given a list of coordinates with the GeoJSON format 
    ([[lon, lat], [lon, lat], ...]), convert longitudes from [0, 360] into 
    [-180, 180].

    Parameters
    ----------
    coord_pair_list : list
        List of coordinates with longitudes in the range [0, 360].

    Returns
    -------
    new_coord_pair_list : list
        List of coordinates with longitudes converted into the range [-180, 180].

    """
    new_coord_pair_list = []
    for i in range(len(coord_pair_list)):
        ilon = coord_pair_list[i][0]
        if ilon > 180:
            ilon = ilon - 360    
        new_coord_pair_list.append([ilon, coord_pair_list[i][1]])      
    return new_coord_pair_list


def shift_lon_a(coord_pair_list: list, a=0):
    """
    Given a list of coordinates with the GeoJSON format 
    ([[lon, lat], [lon, lat], ...]), convert longitudes from [-180, 180] into 
    [a, a+360].

    Parameters
    ----------
    coord_pair_list : list
        List of coordinates with longitudes in the range [-180, 180].
    a: float (or coercible to float)
        New starting longitude. Defaults to 0.

    Returns
    -------
    new_coord_pair_list : list
        List of coordinates with longitudes converted into the range [a, a+360].

    """
    a = float(a)
    
    new_coord_pair_list = []
    for i in range(len(coord_pair_list)):
        ilon = coord_pair_list[i][0]
        if ilon < a:
            ilon = ilon + 360  
        new_coord_pair_list.append([ilon, coord_pair_list[i][1]])      
    return new_coord_pair_list



def split_poly_in_antimeridian(boundary: list, meridian: float = 180,
                               shift_lon: bool = True,
                               output_type: str = "GeometryCollection"):
    """
    Given a list of coordinates with the GeoJSON format 
    ([[lon, lat], [lon, lat], ...]), split the corresponding polygon in the
    antimeridian.
    
    Created for the particular case of the antimeridian, but admits setting any
    meridian as splitting longitude.

    Parameters
    ----------
    boundary : list
        List of coordinates of the polygon boundary.
    meridian : float, optional
        Longitude to split the polygon. The default is 180.
    shift_lon : bool, optional
        Whether to shift the longitudes to [0, 360] to properly split the 
        polygons in the antimeridian. The default is True.
    output_type : str, optional
        Type of the output, either a GeometryCollection of a list of polygons
        contained in it ("PolygonList"). The default is "GeometryCollection".

    Returns
    -------
    out : GeometryCollection, list
        Resulting GeometryCollection or list of polygons from split.

    """
    
    # Create linestring with meridian value (defaults to antimeridian)
    split_meridian = LineString([[meridian, -90.0], [meridian, 90.0]])
    
    # Create polygon from boundary points (allowing shifted longitudes)
    if shift_lon:
        p = Polygon(shift_lon_0_360(boundary))
    else:
        p = Polygon(boundary)
        
    # Split polygon
    splitted_p = split(p, split_meridian)
    
    # Shift polygon coordinates back to [-180, 180]
    poly_list = []
    for i, p in enumerate(splitted_p.geoms):
        if shift_lon:
            if p.centroid.x > 180:
                poly_list.append(translate(p, xoff=-360))
            else:
                poly_list.append(p)
        else: poly_list.append(p)
    
    # Return a GeometryCollection, or the individual polygons in a list
    if output_type=="GeometryCollection":
        out = GeometryCollection(poly_list)
    elif output_type=="PolygonList":
        out = poly_list
    else:
        raise ValueError("'" + output_type + 
                         "' is not a valid option for output_type.")
    
    return out




# Points set manually for polygons can be relatively sparse, and when 
# mapping the resulting polygon lines joining points (i.e. polygon boundaries)
# can be represented in a bit undexpected ways depending of projection 
# (straight line where an arched line would be expected and so on). To minimise
# this, interpolating more points in between can help "force" the expected 
# paths. 
# Create function for this:

def interpolate_coords(coord_pair_list: list,
                       delta: float = 1,
                       close: bool = True,
                       shift_lon: bool = False):
    """
    Given a list of coordinates with the GeoJSON format 
    ([[lon, lat], [lon, lat], ...]), interpolate coordinates to a given
    resolution 'delta'.

    Parameters
    ----------
    coord_pair_list : list
        List of coordinates.
    delta : float, optional
        Resolution of the new coordinates, in degrees. The default is 1.
    close : bool, optional
        Whether to close the coordinate path (start=end). The default is True.
    shift_lon : bool, optional
        Whether to shift longitudes to [0, 360] range. The default is False.

    Returns
    -------
    new_coords : list
        List of interpolated coordinates.

    """
    
    # Shift longitudes to [0, 360] if requested
    if shift_lon:
        coord_pair_list = shift_lon_0_360(coord_pair_list)
        
    # Close boundary if necessary (start and end need to be the same point)
    if close & (not str(coord_pair_list[0])==str(coord_pair_list[-1])):
        coord_pair_list.append(coord_pair_list[0])
        
    # Convert list of coordinates into an array
    coords = np.array(coord_pair_list)
    
    # Given the coordinate list represents a polygon boundary, it is
    # non-monotonical. Interpolation needs to be done along another variable 
    # that it's not x -> the distance between coordinates.
    # Estimate linear distance along the boundary
    dist = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0) # introduce 0 at start
    
    # Set up array with query values; try to keep query as regular as possible
    # (with whole numbers in straigh lines). Make sure to include original 
    # coords.
    q = [np.arange(dist[i], dist[i+1], delta) for i in range(len(dist)-1)]
    q = np.concatenate(q)
    q = np.unique(np.concatenate((dist, q))) # unique() returns sorted
    
    # Interpolate longitudes and latitudes
    newx = np.interp(q, dist, coords[:,0])
    newy = np.interp(q, dist, coords[:,1]) 
    
    # Gather into coord list (rounding to avoid imprecissions when converting
    # from np.float64 to float)
    new_coords = [[round(x, 6), round(y, 6)] for x, y in zip(newx, newy)]
    
    return new_coords



def shift_polygon(p: Polygon, a: float = 0):
    """
    
    Shifts an individual polygon from [-180, 180] coordinates to [a, a+360].
    
    Parameters
    ----------
    p
        Input polygon.
    a : float, optional
        Start of new shifted coordinates. Resulting range will be [a, a+360]. 
        The default is 0, i.e., [0, 360].

    Returns
    -------
    new_shifted_poly : Polygon
        Resulting shifted polygon.

    """
    
    if not isinstance(p, Polygon):
        raise TypeError(f"Expected Polygon, got {type(p).__name__}")
    
    # Get the coordiantes of the polygon boundary
    xy = p.exterior.coords.xy
    p_coords = [[xy[0][i], xy[1][i]] for i in range(len(xy[0]))]
    
    # Avoid points exactly at the new antimeridian
    # Be aware of the side of the antimeridian in which the coords are
    lon = [i[0] for i in p_coords]
    lat = [i[1] for i in p_coords]
    side = np.sign(np.median(np.array(lon) - a))
    new_lon = [i + side * 0.0001 if i == a else i for i in lon]
    new_p_coords = [[lon, lat] for lon, lat in zip(new_lon, lat)]
    
    # Shift coords
    shifted_coords = shift_lon_a(new_p_coords, a=a)
    
    # Add extra bit so that coords close to 180 overlap from each side, so that
    # the unary union joins them properly.
    lon = [i[0] for i in shifted_coords]
    lat = [i[1] for i in shifted_coords]
    offset_up = [.11 if ((i<=180) & (i>179.9)) else 0 for i in lon]
    offset_down = [-.11 if ((i>=180) & (i<180.1)) else 0 for i in lon]
    lon_offset = [i + o for i, o in zip(lon, offset_up)]
    lon_offset = [i + o for i, o in zip(lon_offset, offset_down)]
    new_shifted_coords = [[lon, lat] for lon, lat in zip(lon_offset, lat)]
    
    # Convert back to polygon
    new_shifted_poly = Polygon(new_shifted_coords)
    
    # In some edge cases "holes" of islands tend to create issues so workaround
    # to fix that
    if not new_shifted_poly.is_valid:
        new_shifted_poly = make_valid(new_shifted_poly)
    
    return new_shifted_poly



def shift_polygon_to_map(mp, a=0):
    
    """
    Shift polygons to avoid the antimeridian split line showing up in the map.
    The intended shift is that which moves the antimeridian to the edge of the 
    map so that is no longer visible. This is most useful e.g. for maps of the 
    Pacific, or global maps centered around the Pacific.
    
    Its intented input are MultiPolygon class objects (polygons splitted at the
    antimeridian are effectively MultiPolygons), but also accepts 
    GeometryCollection or simple Polygon, in the latter case it just returns 
    them shifted.
    
    Internally uses shift_polygon().
    """
    
    # If input is an individual polygon, coerce to MultiPolygon class so that
    # the function code is generally applicable
    if isinstance(mp, Polygon):
        
        mp = MultiPolygon([mp])
    
    if (isinstance(mp, MultiPolygon) | isinstance(mp, GeometryCollection)):
        
        shifted_polys = []
        for p in mp.geoms:
            
            # Slipt polygon in new antimeridian (a).
            # This is need for polygons that wrap around (at least most of) 
            # the globe. Polygon coordinates maintain their order, so when
            # shifthing them to the new range [a, a+360], they jump back/forth
            # at the new antimeridian. As creating general code to reorder them
            # is complicated, an easy workaround is to split the polygons by 
            # the new antimeridian, shift coordinates of both sides, and then
            # do a unary_union back together. If a polygon does not cross the 
            # new antimeridian, the split just returns a single polygon, so 
            # there is no issue.
            
            # Create linestring with meridian value
            split_meridian = LineString([[a, -90.0], [a, 90.0]])
            
            # Split polygon
            splitted_p = split(p, split_meridian)
            
            # Shift individual polygons
            for sp in splitted_p.geoms:
                
                # Shift polygon
                shifted_p = shift_polygon(sp, a=a)
            
                # Gather resulting polyons
                shifted_polys.append(shifted_p)

        # Join them to achieve new shifted polygon
        try:
            new_poly = unary_union(shifted_polys)
            
        except Exception as e:
            # If union fails,
            print(f"unary_union() failed: {e}.\n Returning list of individual polygons to assess.")
            new_poly = shifted_polys
    
        return new_poly

    else:
        raise TypeError(f"Expected Polygon, MultiPolygon or GeometryCollection, got {type(mp).__name__}.")
