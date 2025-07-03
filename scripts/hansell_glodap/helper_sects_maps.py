# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:59:11 2025

Helper maps and sections to contextualise oxygen utilisation rate patterns.

@author: Markel
"""

#%% IMPORTS

# general
import numpy as np
import pandas as pd
import pathlib
from scipy.interpolate import RBFInterpolator

from shapely import from_geojson

# plotting & mapping
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeoaxes
import matplotlib.patheffects as pe



#%% LOAD DATA

# Filtered, merged Hansell+GLODAP dataset, with ages:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
tbl = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                   'CRUISE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})

# Original Glodap dataset, with all data
fpath = 'rawdata/glodap/GLODAPv2.2023_Merged_Master_File.csv.zip'
gv2 = pd.read_csv(fpath, sep=',', header=0, 
                  compression='zip',
                  dtype={'G2expocode': str,
                         'G2bottle': str})

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
wm_polys_plot = {} # load uncut version of polygons too (for plotting)
wm_depths = ['central', 'intermediate', 'deep']
for d in dlist:
    
    o = str(d).split("\\")[-1]
    wm_polys[o] = {}
    wm_polys_plot[o] = {}
    
    for z in wm_depths:
        
        # Get wm paths at depth z and ocean d
        flist = [*pathlib.Path(str(d) + "\\wms\\" + z).glob("*.geojson")]
        
        # Skip iteration if certain depth is absent (i.e. flist is empty)
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
            
            # For Pacific, replace with uncut version
            if o=='pacific':
                fp = list(f.parts)
                fp[-1] = fp[-1].replace("polygon", "polygon_uncut") 
                f2 = (pathlib.Path(*fp[:fp.index(z)]).
                      joinpath('uncut').
                      joinpath(*fp[fp.index(z):]))
                wm_polys_plot[o][z][w] = from_geojson(f2.read_text())
                

# Create a second version of the water mass dict, which instead of nested has
# composite keys following the pattern -> ocean-depth_layer-watermass
# 
# Create function to flatten nested dictionaries, resulting in a single dict
# with structure {'key1_key2_key': value} (where key1, key2,... are the keys 
# of the nested dicts)
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
gv2 = gv2.replace(-9999, np.nan)

# Glodap has some longitude coords in [0 360]. Turn them into [-180, 180]
# np.nanmax(gv2.G2longitude)
over180 = gv2.G2longitude > 180
gv2.loc[over180, 'G2longitude'] = gv2.loc[over180, 'G2longitude'] - 360


#%% STATION MAP

cm = 1/2.54

mproj = ccrs.Mollweide(central_longitude=-160)
fig_st, ax_st = plt.subplots(figsize=(10*cm, 6*cm),
                             subplot_kw={'projection': mproj})

gv2_u = gv2.drop_duplicates(subset=['G2cruise', 'G2station'])
# gv2_u = gv2_u.loc[gv2_u.G2cruise==246,:]
ax_st.scatter(gv2_u.G2longitude, gv2_u.G2latitude,
              s=3,
              edgecolor='none',
              transform=ccrs.PlateCarree())
ax_st.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                               name='land',
                                               scale='50m'),
                  facecolor='#ccc',
                  edgecolor='black',
                  linewidth=.25,
                  zorder=2)
ax_st.set_global()
# ax_st.set_extent([150, -60, 10, 0])


#%% MAPS


# Select SIGMA0 horizons and variables to plot
SIGMA0 = [26.4, 26.7, 27.0, 27.2, 27.7]
VARS = ['G2oxygen', 'G2salinity', 'G2doc']

# Set SIGMA0 layer thinckness
t = .05

# Set variable limits and labels
VARS_lims = {'G2oxygen': [0, 320, 15],
             'G2salinity': [33, 36.3, .1],
             'G2doc': [36, 54, 1]} # min, max, step size between
VARS_labs = {'G2oxygen': "O$_2$ [$\mu$mol kg$^{-1}$]",
             'G2salinity': "Salinity",
             'G2doc': "DOC [$\mu$mol L$^{-1}$]"}
VARS_extend = {'G2oxygen': 'max',
               'G2salinity': 'both',
               'G2doc': 'both'}
VARS_pal = {'G2oxygen': 'PuOr',
            'G2salinity': 'BrBG_r',
            'G2doc': 'magma'}

# Set interpolation grid resolution
deg_res_lon = 1
deg_res_lat = 1

nr = len(SIGMA0)
nc = len(VARS)
mproj = ccrs.Mollweide(central_longitude=-160)
fig_m, ax_m = plt.subplots(nrows=nr, ncols=nc,
                           squeeze=False,
                           figsize=(6*cm * nc, 4*cm * nr),
                           subplot_kw={'projection': mproj})
for iv, v in enumerate(VARS):
    for ig, g in enumerate(SIGMA0):
        
        #### Interpolation
        
        # If the variable is DOC, take the Hansel+Glodap merged dataset
        if not v=='G2doc':
            
            # Subset data for the g SIGMA0 layer, without NaNs for variable v
            idx = ((gv2.G2sigma0 > (g - t)) &
                   (gv2.G2sigma0 < (g + t)) &
                   (~np.isnan(gv2[v])))
            ss = gv2.loc[idx, :].copy()

        else:
            
            # Rename variables to match glodap naming (to make code universal)
            ss = tbl.copy()
            ss.columns = ['G2' + x.lower() for x in ss.columns]
            
            idx = ((ss.G2sigma0 > (g - t)) &
                   (ss.G2sigma0 < (g + t)) &
                   (~np.isnan(ss[v])))
            ss = ss.loc[idx, :]

        
        # Wrap around antimeridian by 'offset' degrees, duplicating data, to 
        # aid interpolation in such edge case (will be clipped after 
        # interpolation). (basically copies patches of data left and right of 
        # data edges)
        # Copy data and transform longitude to put it to the right of the 
        # antimeridian.
        offset = 150 if g==26 else 20
        idx1 = (ss.G2longitude < (-180 + offset)) & (ss.G2longitude > (-180))
        ss_copy1 = ss.loc[idx1, :].copy()
        ss_copy1.G2longitude = ss_copy1.G2longitude + 360
        
        # Same to put it left
        idx2 = (ss.G2longitude > (180 - offset)) & (ss.G2longitude < (180))
        ss_copy2 = ss.loc[idx2, :].copy()
        ss_copy2.G2longitude = ss_copy2.G2longitude - 360
        
        # Gather "duplicated" data with original
        ss_wrapped = pd.concat([ss, ss_copy1, ss_copy2])
        
        # Average duplicated coordinates (otherwise interpolation breaks)
        # (round a bit because when interpolating superclose points with 
        # different values return weird interpolation patterns locally)
        ss_wrapped.G2longitude = round(ss_wrapped.G2longitude, 1)
        ss_wrapped.G2latitude = round(ss_wrapped.G2latitude, 1)
        ss_wrapped = (ss_wrapped.groupby(['G2latitude', 'G2longitude']).
                      agg({v: 'mean'}).
                      reset_index())
        
        # Smooth data averaging values around point 
        # (because when interpolating superclose points with different values 
        # interpolation returns weird patterns locally)
        radius = 2
        ss_wrapped[v + '_S'] = np.nan
        for i, r in ss_wrapped.iterrows():
            
            # Estimate distance from sample r to the other samples
            dist = ((ss_wrapped.G2longitude - r.G2longitude)**2 + 
                    (ss_wrapped.G2latitude - r.G2latitude)**2) ** .5
            
            # Retain only samples within set radius
            within_radius = dist < radius
            dist = dist[within_radius]
            
            # Estimate weighted average depending on distance from sample
            dist_frac = ((radius - dist) / radius)**2 # **1 = linear
            weights = dist_frac / sum(dist_frac)
            values = ss_wrapped.loc[within_radius, v]
            ss_wrapped.loc[i, v + '_S'] = np.nansum(values * weights)
            
        
        # Set the desired X and Y variables:
        ix = ss_wrapped.G2longitude
        iy = ss_wrapped.G2latitude
        xmin = min(ix)
        xmax = max(ix)
        ymin = min(iy)
        ymax = max(iy)
        xsd = np.std(ix)
        ysd = np.std(iy)
        
        # Create the interpolation grid fitting to the requirements of 
        # RBFInterpolator:
        ndim_lon = int(round((360 + offset*2) / deg_res_lon)) + 1
        ndim_lat = int(round(180 / deg_res_lat)) + 1
        xq = np.linspace(-180 - offset, 180 + offset, ndim_lon)
        yq = np.linspace(-90, 90, ndim_lat)
        yq = yq[(yq >= ymin) & (yq <= ymax)]
        ndim_lat = len(yq)
        rx, ry = np.meshgrid(xq/xsd, yq/ysd)
        meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)
        
        # Assemble the data point coordinates and values as required:
        dpc = np.array([ix/xsd, iy/ysd]).T
        dpv = ss_wrapped.loc[:, v + '_S']
                
        # Exclude missing values:
        notnan = ~np.isnan(dpv)
        dpc = dpc[notnan]
        dpv = dpv[notnan]
                
        # Perform interpolation with RBF
        rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                               neighbors=100, smoothing=0)
        rv = rbfi(meshflat)
        rv = rv.reshape(ndim_lat, ndim_lon)
        
        # Enforce positive values for relevant variables
        if v in ['G2oxygen', 'G2salinity', 'G2doc']:
            rv[rv < 0] = 0            
        
        # Clip the duplicated data for edge cases
        idx = (ix >= -180) & (ix < 180)
        ix2 = ix[idx]
        iy2 = iy[idx]
        ndim_lon2 = int(round(360 / deg_res_lon)) + 1
        idx = (rx*xsd >= -180) & (rx*xsd <= 180)
        rx2 = rx[idx].reshape(ndim_lat, ndim_lon2)
        ry2 = ry[idx].reshape(ndim_lat, ndim_lon2)
        rv2 = rv[idx].reshape(ndim_lat, ndim_lon2)
        
        
        #### Plotting
        
        ## Map results
        if not v=='G2doc':
            lvs = np.arange(VARS_lims[v][0],
                            VARS_lims[v][1] + .01,
                            VARS_lims[v][2])
            vmap = ax_m[ig, iv].contourf(rx2*xsd,
                                         ry2*ysd,
                                         rv2,
                                         levels=lvs,
                                         extend=VARS_extend[v],
                                         vmin=VARS_lims[v][0], 
                                         vmax=VARS_lims[v][1],
                                         cmap=VARS_pal[v],
                                         transform_first=True, # *
                                         transform=ccrs.PlateCarree(), 
                                         zorder=0)
        else:
            lvs = np.arange(VARS_lims[v][0],
                            VARS_lims[v][1] + .01,
                            VARS_lims[v][2])
            norm = mpl.colors.BoundaryNorm(lvs, 
                                           plt.get_cmap(VARS_pal[v]).N)
            vmap = ax_m[ig, iv].scatter(dpc[:,0]*xsd,
                                        dpc[:,1]*ysd,
                                        c=dpv,
                                        marker='o',
                                        s=2,
                                        cmap=VARS_pal[v],
                                        norm=norm,
                                        edgecolor='#222',
                                        linewidth=.1,
                                        transform=ccrs.PlateCarree(),
                                        zorder=1)
        
        ax_m[ig, iv].add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                              name='land',
                                                              scale='110m'),
                                 facecolor='#ccc',
                                 edgecolor='black',
                                 linewidth=.1,
                                 zorder=2)
        ax_m[ig, iv].set_global()
        
        if iv==0:
            txt = "$\mathbf{\sigma_\\theta}$ = " + '{:.1f}'.format(g)
            ax_m[ig, iv].text(-.03, .5, txt,
                              size=7,
                              rotation=90,
                              ha='right', va='center',
                              transform=ax_m[ig, iv].transAxes)
        

    # Add colorbar at bottom line of each variable
    axins1 = ax_m[nr-1, iv].inset_axes([.025, -.25, .95, .1])
    if not v=='G2doc':
        cbar = fig_m.colorbar(vmap,
                              cax=axins1,
                              # ticks=tcks,
                              orientation='horizontal')
    else:
        cbar = fig_m.colorbar(vmap,
                              cax=axins1,
                              extend=VARS_extend[v],
                              # ticks=tcks,
                              orientation='horizontal')
    cbar.ax.tick_params(labelsize=5, length=2, width=.6)
    cbar.ax.set_xlabel(VARS_labs[v], fontsize=6, labelpad=2)
        
fig_m.subplots_adjust(hspace=-.45, wspace=.1)
fpath = ('figures/hansell_glodap/global/helper/' +
         'help_map_' + '_'.join(VARS) + '.svg')
fig_m.savefig(fpath, format='svg', bbox_inches='tight')


# * I encountered a bug for salinity in SIGMA0=27.2, this is a workaround. 
# See: https://github.com/SciTools/cartopy/issues/2176



#%% SECTIONS


#%%% Pacific North

# Pacific North -> cruise 299 / 1053 / 502

lon_w = 150
lon_e = -120
cruise_code = 299
p_max = 1600

# idx = ((gv2.G2latitude > (lat - .1)) &
#        (gv2.G2latitude < (lat + .1)) &
#        ((gv2.G2longitude < lon_e) | (gv2.G2longitude > lon_w)) &
#        (gv2.G2pressure <= p_max))
idx = ((gv2.G2cruise==cruise_code) &
       (gv2.G2pressure <= p_max))
gv2_299 = gv2.loc[idx, :]
gv2_299_u = gv2_299.drop_duplicates(subset=['G2cruise', 'G2station'])


#------------------------------------------------------------------------------

# Select variables to plot
VARS = ['G2oxygen', 'G2salinity']


# Set variable limits and labels
VARS_lims = {'G2oxygen': [0, 330, 15],
             'G2salinity': [32.6, 34.6, .1]} # min, max, step size between
VARS_labs = {'G2oxygen': "O$_2$ [$\mu$mol kg$^{-1}$]",
             'G2salinity': "Salinity"}
VARS_extend = {'G2oxygen': 'max',
               'G2salinity': 'both'}
VARS_pal = {'G2oxygen': 'PuOr',
            'G2salinity': 'BrBG_r'}

#------------------------------------------------------------------------------


clon = 180 - np.mean(gv2_299_u.G2longitude)
clat = np.mean(gv2_299_u.G2latitude)
mproj = ccrs.NearsidePerspective(central_longitude=clon, central_latitude=clat)



deg_res_lon = .2
deg_res_p = 5

nr = len(VARS)
fig_s, ax_s = plt.subplots(nrows=nr, ncols=1,
                           squeeze=False,
                           figsize=(10*cm, 4*cm * nr))
for iv, v in enumerate(VARS):
        
    #### Interpolation
    
    # Subset data for the g SIGMA0 layer, without NaNs for variable v
    idx = ((~np.isnan(gv2_299[v])))
    ss = gv2_299.loc[idx, :].copy()
    
    
    lons = ss.G2longitude
    if np.sign(np.min(lons))*np.sign(np.max(lons)) < 0: # if section start/end at different sides of antimeridian
        
        # Given that sections will be limited to a fraction of the globe,
        # transform longitude to avoid antimeridian 'jump'
        below0 = (ss.G2longitude < 0)
        ss.loc[below0, 'G2longitude'] = ss.G2longitude[below0] + 360
        
    
    # Average duplicated coordinates (otherwise interpolation breaks)
    # (round a bit because when interpolating superclose points with 
    # different values return weird interpolation patterns locally)
    ss.G2longitude = deg_res_lon * round(ss.G2longitude / deg_res_lon)
    #ss.G2latitude = round(ss.G2latitude, 1)
    ss.G2pressure = round(ss.G2pressure, 0)
    l = ['G2longitude', 'G2pressure'] # 'G2latitude', 
    ss = (ss.groupby(l).
          agg({v: 'mean', 'G2sigma0': 'mean'}).
          reset_index())
    
    # Set the desired X and Y variables:
    ix = ss.G2longitude
    iy = ss.G2pressure
    xmin = min(ix)
    xmax = max(ix)
    ymin = min(iy)
    ymax = max(iy)
    xsd = np.std(ix)
    ysd = np.std(iy)
    
    # Create the interpolation grid fitting to the requirements of 
    # RBFInterpolator:
    ndim_lon = int(round((lon_w - lon_e) / deg_res_lon))
    ndim_p = int(round(p_max / deg_res_p))
    xseq = np.linspace(xmin, xmax, ndim_lon)
    yseq = np.linspace(ymin, ymax, ndim_p)
    rx, ry = np.meshgrid(xseq / xsd, yseq / ysd)
    meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)
    

    # Assemble the data point coordinates and values as required:
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, v]
            
    # Exclude missing values:
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
            
    # Perform interpolation with RBF
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv = rbfi(meshflat)
    rv = rv.reshape(ndim_p, ndim_lon)
    
    
    # Enforce positive values for relevant variables
    if v in ['G2oxygen', 'G2salinity']:
        rv[rv < 0] = 0            
    
    # Interpolate SIGMA0 too, to add as contour
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, 'G2sigma0']
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv2 = rbfi(meshflat)
    rv2 = rv2.reshape(ndim_p, ndim_lon)
    
    
    #### Plotting
    
    ## Map results
            
    # Sample locations
    # ax_s[iv, 0].scatter(dpc[:, 0]*xsd,
    #                       dpc[:, 1]*ysd,
    #                       c='k',
    #                       marker='o',
    #                       s=.2,
    #                       edgecolor='none',
    #                       zorder=2)
    
    # Stations on top
    ax_s[iv, 0].scatter(ix.unique(), [-30]*len(ix.unique()),
                        c='k',
                        marker='v',
                        s=3,
                        edgecolor='none',
                        clip_on=False,
                        zorder=2)
    # Interpolated contourf
    cfvar = ax_s[iv, 0].contourf(rx*xsd,
                                 ry*ysd,
                                 rv,
                                 levels=np.arange(VARS_lims[v][0],
                                                  VARS_lims[v][1] + .01,
                                                  VARS_lims[v][2]),
                                 extend=VARS_extend[v],
                                 vmin=VARS_lims[v][0], 
                                 vmax=VARS_lims[v][1],
                                 cmap=VARS_pal[v], 
                                 zorder=0)
    
    # SIGMA0 as contour
    csig = ax_s[iv, 0].contour(rx*xsd,
                                 ry*ysd,
                                 rv2,
                                 colors='#aaa',
                                 linewidths=.2,
                                 levels=[26, 26.4, 26.7, 27.2, 27.7],
                                 zorder=1)
    csig.set(path_effects=[pe.withStroke(linewidth=.4, foreground='k')])
    
    clbls = ax_s[iv, 0].clabel(csig, csig.levels, fontsize=2.7, inline_spacing=2)
    plt.setp(clbls, path_effects=[pe.withStroke(linewidth=.2, foreground='k')])
    
    # Customise axes
    ax_s[iv, 0].set(ylim=[0, p_max - 100],
                    yticks=range(0, 1501, 250))
    xt = [*range(150, 240, 10)]
    xl = [str(x - 360) if x > 180 else str(x) for x in xt]
    xl = [x.replace("180", "±180") for x in xl]
    ax_s[iv, 0].set_xticks(xt, xl)
    ax_s[iv, 0].xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_s[iv, 0].set_ylabel("Depth [dbar]", fontsize=5, labelpad=2)
    ax_s[iv, 0].set_xlabel("Longitude [$\degree$E]", fontsize=5, labelpad=2)
    ax_s[iv, 0].tick_params(axis='both', which='major', length=2, labelsize=5,
                            pad=2)
    ax_s[iv, 0].invert_yaxis()
    
    ## Add inset map
    axins = inset_axes(ax_s[iv, 0], width="25%", height="25%",
                       loc="lower right",
                       bbox_to_anchor=(.1, -.04, .985, 1),
                       bbox_transform=ax_s[iv, 0].transAxes,
                       axes_class=cgeoaxes.GeoAxes,
                       axes_kwargs={'projection': mproj})
    # Add stations
    axins.scatter(gv2_299_u.G2longitude, gv2_299_u.G2latitude,
                  s=.2,
                  edgecolor='none',
                  transform=ccrs.PlateCarree())
    axins.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                   name='land',
                                                   scale='110m'),
                      facecolor='#ccc',
                      edgecolor='black',
                      linewidth=.1,
                      zorder=2)
    axins.set_global()
    axins.spines['geo'].set_linewidth(.3) # thiner border
        

    # Add colorbar at bottom line of each variable
    cbar = fig_s.colorbar(cfvar, ax=ax_s[iv, 0])
    cbar.ax.tick_params(labelsize=5, length=2, width=.6)
    cbar.ax.set_ylabel(VARS_labs[v], fontsize=5, labelpad=2)
        
fig_s.subplots_adjust(hspace=.3)
fpath = ('figures/hansell_glodap/global/helper/' +
         'help_section_NP_' +  str(cruise_code) + '_' + '_'.join(VARS) + 
         '.svg')
fig_s.savefig(fpath, format='svg', bbox_inches='tight')



#%%% Pacific South I

# Pacific South -> cruise 273 or 243 (around 30ºS), or cruise 246?


lon_w = 150
lon_e = -70
cruise_code = 273
p_max = 1600

# idx = ((gv2.G2latitude > (lat - .1)) &
#        (gv2.G2latitude < (lat + .1)) &
#        ((gv2.G2longitude < lon_e) | (gv2.G2longitude > lon_w)) &
#        (gv2.G2pressure <= p_max))
idx = ((gv2.G2cruise==cruise_code) &
       (gv2.G2pressure <= p_max))
gv2_273 = gv2.loc[idx, :]
gv2_273_u = gv2_273.drop_duplicates(subset=['G2cruise', 'G2station'])


#------------------------------------------------------------------------------

# Select variables to plot
VARS = ['G2oxygen', 'G2salinity']


# Set variable limits and labels
VARS_lims = {'G2oxygen': [0, 270, 10],
             'G2salinity': [34, 35.6, .1]} # min, max, step size between
VARS_labs = {'G2oxygen': "O$_2$ [$\mu$mol kg$^{-1}$]",
             'G2salinity': "Salinity"}
VARS_extend = {'G2oxygen': 'max',
               'G2salinity': 'both'}
VARS_pal = {'G2oxygen': 'PuOr',
            'G2salinity': 'BrBG_r'}

#------------------------------------------------------------------------------


clon = 180 - np.mean(gv2_273_u.G2longitude)
clat = np.mean(gv2_273_u.G2latitude)
mproj = ccrs.NearsidePerspective(central_longitude=clon, central_latitude=clat)



deg_res_lon = .2
deg_res_p = 5

nr = len(VARS)
fig_s, ax_s = plt.subplots(nrows=nr, ncols=1,
                           squeeze=False,
                           figsize=(10*cm, 4*cm * nr))
for iv, v in enumerate(VARS):
        
    #### Interpolation
    
    # Subset data for the g SIGMA0 layer, without NaNs for variable v
    idx = ((~np.isnan(gv2_273[v])))
    ss = gv2_273.loc[idx, :].copy()
    
    
    lons = ss.G2longitude
    if np.sign(np.min(lons))*np.sign(np.max(lons)) < 0: # if section start/end at different sides of antimeridian
        
        # Given that sections will be limited to a fraction of the globe,
        # transform longitude to avoid antimeridian 'jump'
        below0 = (ss.G2longitude < 0)
        ss.loc[below0, 'G2longitude'] = ss.G2longitude[below0] + 360
        
    
    # Average duplicated coordinates (otherwise interpolation breaks)
    # (round a bit because when interpolating superclose points with 
    # different values return weird interpolation patterns locally)
    ss.G2longitude = deg_res_lon * round(ss.G2longitude / deg_res_lon)
    #ss.G2latitude = round(ss.G2latitude, 1)
    ss.G2pressure = round(ss.G2pressure, 0)
    l = ['G2longitude', 'G2pressure'] # 'G2latitude', 
    ss = (ss.groupby(l).
          agg({v: 'mean', 'G2sigma0': 'mean'}).
          reset_index())
    
    # Set the desired X and Y variables:
    ix = ss.G2longitude
    iy = ss.G2pressure
    xmin = min(ix)
    xmax = max(ix)
    ymin = min(iy)
    ymax = max(iy)
    xsd = np.std(ix)
    ysd = np.std(iy)
    
    # Create the interpolation grid fitting to the requirements of 
    # RBFInterpolator:
    ndim_lon = int(round((lon_w - lon_e) / deg_res_lon))
    ndim_p = int(round(p_max / deg_res_p))
    xseq = np.linspace(xmin, xmax, ndim_lon)
    yseq = np.linspace(ymin, ymax, ndim_p)
    rx, ry = np.meshgrid(xseq / xsd, yseq / ysd)
    meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)
    

    # Assemble the data point coordinates and values as required:
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, v]
            
    # Exclude missing values:
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
            
    # Perform interpolation with RBF
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv = rbfi(meshflat)
    rv = rv.reshape(ndim_p, ndim_lon)
    
    
    # Enforce positive values for relevant variables
    if v in ['G2oxygen', 'G2salinity']:
        rv[rv < 0] = 0            
    
    # Interpolate SIGMA0 too, to add as contour
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, 'G2sigma0']
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv2 = rbfi(meshflat)
    rv2 = rv2.reshape(ndim_p, ndim_lon)
    
    
    #### Plotting
    
    ## Map results
            
    # Sample locations
    # ax_s[iv, 0].scatter(dpc[:, 0]*xsd,
    #                       dpc[:, 1]*ysd,
    #                       c='k',
    #                       marker='o',
    #                       s=.2,
    #                       edgecolor='none',
    #                       zorder=2)
    
    # Stations on top
    ax_s[iv, 0].scatter(ix.unique(), [-30]*len(ix.unique()),
                        c='k',
                        marker='v',
                        s=3,
                        edgecolor='none',
                        clip_on=False,
                        zorder=2)
    # Interpolated contourf
    cfvar = ax_s[iv, 0].contourf(rx*xsd,
                                 ry*ysd,
                                 rv,
                                 levels=np.arange(VARS_lims[v][0],
                                                  VARS_lims[v][1] + .01,
                                                  VARS_lims[v][2]),
                                 extend=VARS_extend[v],
                                 vmin=VARS_lims[v][0], 
                                 vmax=VARS_lims[v][1],
                                 cmap=VARS_pal[v], 
                                 zorder=0)
    
    # SIGMA0 as contour
    csig = ax_s[iv, 0].contour(rx*xsd,
                               ry*ysd,
                               rv2,
                               colors='#aaa',
                               linewidths=.2,
                               levels=[26, 26.4, 26.7, 27.2, 27.7],
                               zorder=1)
    csig.set(path_effects=[pe.withStroke(linewidth=.4, foreground='k')])
    
    clbls = ax_s[iv, 0].clabel(csig, csig.levels, fontsize=2.7, inline_spacing=2)
    plt.setp(clbls, path_effects=[pe.withStroke(linewidth=.2, foreground='k')])
    
    # Customise axes
    ax_s[iv, 0].set(ylim=[0, p_max - 100],
                    yticks=range(0, 1501, 250))
    xt = [*range(160, 290, 20)]
    xl = [str(x - 360) if x > 180 else str(x) for x in xt]
    xl = [x.replace("180", "±180") for x in xl]
    ax_s[iv, 0].set_xticks(xt, xl)
    ax_s[iv, 0].xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_s[iv, 0].set_ylabel("Depth [dbar]", fontsize=5, labelpad=2)
    ax_s[iv, 0].set_xlabel("Longitude [$\degree$E]", fontsize=5, labelpad=2)
    ax_s[iv, 0].tick_params(axis='both', which='major', length=2, labelsize=5,
                            pad=2)
    ax_s[iv, 0].invert_yaxis()
    
    ## Add inset map
    axins = inset_axes(ax_s[iv, 0], width="25%", height="25%",
                       loc="lower right",
                       bbox_to_anchor=(.1, -.04, .985, 1),
                       bbox_transform=ax_s[iv, 0].transAxes,
                       axes_class=cgeoaxes.GeoAxes,
                       axes_kwargs={'projection': mproj})
    # Add stations
    axins.scatter(gv2_273_u.G2longitude, gv2_273_u.G2latitude,
                  s=.2,
                  edgecolor='none',
                  transform=ccrs.PlateCarree())
    axins.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                   name='land',
                                                   scale='110m'),
                      facecolor='#ccc',
                      edgecolor='black',
                      linewidth=.1,
                      zorder=2)
    axins.set_global()
    axins.spines['geo'].set_linewidth(.3) # thiner border
        

    # Add colorbar at bottom line of each variable
    cbar = fig_s.colorbar(cfvar, ax=ax_s[iv, 0])
    cbar.ax.tick_params(labelsize=5, length=2, width=.6)
    cbar.ax.set_ylabel(VARS_labs[v], fontsize=5, labelpad=2)
        
fig_s.subplots_adjust(hspace=.3)
fpath = ('figures/hansell_glodap/global/helper/' +
         'help_section_SP_' +  str(cruise_code) + '_' + '_'.join(VARS) + 
         '.svg')
fig_s.savefig(fpath, format='svg', bbox_inches='tight')



#%%% Pacific South II

# Pacific South -> cruise 273 or 243 (around 30ºS), or cruise 246?


lon_w = 150
lon_e = -70
cruise_code = 246
p_max = 1600

# idx = ((gv2.G2latitude > (lat - .1)) &
#        (gv2.G2latitude < (lat + .1)) &
#        ((gv2.G2longitude < lon_e) | (gv2.G2longitude > lon_w)) &
#        (gv2.G2pressure <= p_max))
idx = ((gv2.G2cruise==cruise_code) &
       (gv2.G2latitude > -53.5) &
       (gv2.G2pressure <= p_max))
gv2_246 = gv2.loc[idx, :]
gv2_246_u = gv2_246.drop_duplicates(subset=['G2cruise', 'G2station'])


#------------------------------------------------------------------------------

# Select variables to plot
VARS = ['G2oxygen', 'G2salinity']


# Set variable limits and labels
VARS_lims = {'G2oxygen': [170, 300, 5],
             'G2salinity': [34.1, 34.6, .02]} # min, max, step size between
VARS_labs = {'G2oxygen': "O$_2$ [$\mu$mol kg$^{-1}$]",
             'G2salinity': "Salinity"}
VARS_extend = {'G2oxygen': 'both',
               'G2salinity': 'both'}
VARS_pal = {'G2oxygen': 'PuOr',
            'G2salinity': 'BrBG_r'}

#------------------------------------------------------------------------------


clon = 180 - np.mean(gv2_246_u.G2longitude)
clat = np.mean(gv2_246_u.G2latitude)
mproj = ccrs.NearsidePerspective(central_longitude=clon, central_latitude=clat)



deg_res_lon = .2
deg_res_p = 5

nr = len(VARS)
fig_s, ax_s = plt.subplots(nrows=nr, ncols=1,
                           squeeze=False,
                           figsize=(10*cm, 4*cm * nr))
for iv, v in enumerate(VARS):
        
    #### Interpolation
    
    # Subset data for the g SIGMA0 layer, without NaNs for variable v
    idx = ((~np.isnan(gv2_246[v])))
    ss = gv2_246.loc[idx, :].copy()
    
    
    lons = ss.G2longitude
    if np.sign(np.min(lons))*np.sign(np.max(lons)) < 0: # if section start/end at different sides of antimeridian
        
        # Given that sections will be limited to a fraction of the globe,
        # transform longitude to avoid antimeridian 'jump'
        below0 = (ss.G2longitude < 0)
        ss.loc[below0, 'G2longitude'] = ss.G2longitude[below0] + 360
        
    
    # Average duplicated coordinates (otherwise interpolation breaks)
    # (round a bit because when interpolating superclose points with 
    # different values return weird interpolation patterns locally)
    ss.G2longitude = deg_res_lon * round(ss.G2longitude / deg_res_lon)
    #ss.G2latitude = round(ss.G2latitude, 1)
    ss.G2pressure = round(ss.G2pressure, 0)
    l = ['G2longitude', 'G2pressure'] # 'G2latitude', 
    ss = (ss.groupby(l).
          agg({v: 'mean', 'G2sigma0': 'mean'}).
          reset_index())
    
    # Set the desired X and Y variables:
    ix = ss.G2longitude
    iy = ss.G2pressure
    xmin = min(ix)
    xmax = max(ix)
    ymin = min(iy)
    ymax = max(iy)
    xsd = np.std(ix)
    ysd = np.std(iy)
    
    # Create the interpolation grid fitting to the requirements of 
    # RBFInterpolator:
    ndim_lon = int(round((lon_w - lon_e) / deg_res_lon))
    ndim_p = int(round(p_max / deg_res_p))
    xseq = np.linspace(xmin, xmax, ndim_lon)
    yseq = np.linspace(ymin, ymax, ndim_p)
    rx, ry = np.meshgrid(xseq / xsd, yseq / ysd)
    meshflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)
    

    # Assemble the data point coordinates and values as required:
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, v]
            
    # Exclude missing values:
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
            
    # Perform interpolation with RBF
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv = rbfi(meshflat)
    rv = rv.reshape(ndim_p, ndim_lon)
    
    
    # Enforce positive values for relevant variables
    if v in ['G2oxygen', 'G2salinity']:
        rv[rv < 0] = 0            
    
    # Interpolate SIGMA0 too, to add as contour
    dpc = np.array([ix / xsd, iy / ysd]).T
    dpv = ss.loc[:, 'G2sigma0']
    notnan = ~np.isnan(dpv)
    dpc = dpc[notnan]
    dpv = dpv[notnan]
    rbfi = RBFInterpolator(y=dpc, d=dpv, kernel='linear',
                           neighbors=100, smoothing=0)
    rv2 = rbfi(meshflat)
    rv2 = rv2.reshape(ndim_p, ndim_lon)
    
    
    #### Plotting
    
    ## Map results
            
    # Sample locations
    # ax_s[iv, 0].scatter(dpc[:, 0]*xsd,
    #                       dpc[:, 1]*ysd,
    #                       c='k',
    #                       marker='o',
    #                       s=.2,
    #                       edgecolor='none',
    #                       zorder=2)
    
    # Stations on top
    ax_s[iv, 0].scatter(ix.unique(), [-30]*len(ix.unique()),
                        c='k',
                        marker='v',
                        s=3,
                        edgecolor='none',
                        clip_on=False,
                        zorder=2)
    # Interpolated contourf
    cfvar = ax_s[iv, 0].contourf(rx*xsd,
                                 ry*ysd,
                                 rv,
                                 levels=np.arange(VARS_lims[v][0],
                                                  VARS_lims[v][1] + .01,
                                                  VARS_lims[v][2]),
                                 extend=VARS_extend[v],
                                 vmin=VARS_lims[v][0], 
                                 vmax=VARS_lims[v][1],
                                 cmap=VARS_pal[v], 
                                 zorder=0)
    
    # SIGMA0 as contour
    csig = ax_s[iv, 0].contour(rx*xsd,
                               ry*ysd,
                               rv2,
                               colors='#aaa',
                               linewidths=.2,
                               levels=[26, 26.4, 26.7, 27.2, 27.7],
                               zorder=1)
    csig.set(path_effects=[pe.withStroke(linewidth=.4, foreground='k')])
    
    clbls = ax_s[iv, 0].clabel(csig, csig.levels, fontsize=2.7, inline_spacing=2)
    plt.setp(clbls, path_effects=[pe.withStroke(linewidth=.2, foreground='k')])
    
    # Customise axes
    ax_s[iv, 0].set(ylim=[0, p_max - 100],
                    yticks=range(0, 1501, 250))
    # xt = [*range(160, 290, 20)]
    # xl = [str(x - 360) if x > 180 else str(x) for x in xt]
    # xl = [x.replace("180", "±180") for x in xl]
    # ax_s[iv, 0].set_xticks(xt, xl)
    ax_s[iv, 0].xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_s[iv, 0].set_ylabel("Depth [dbar]", fontsize=5, labelpad=2)
    ax_s[iv, 0].set_xlabel("Longitude [$\degree$E]", fontsize=5, labelpad=2)
    ax_s[iv, 0].tick_params(axis='both', which='major', length=2, labelsize=5,
                            pad=2)
    ax_s[iv, 0].invert_yaxis()
    
    ## Add inset map
    axins = inset_axes(ax_s[iv, 0], width="25%", height="25%",
                       loc="lower right",
                       bbox_to_anchor=(.1, -.04, .985, 1),
                       bbox_transform=ax_s[iv, 0].transAxes,
                       axes_class=cgeoaxes.GeoAxes,
                       axes_kwargs={'projection': mproj})
    # Add stations
    axins.scatter(gv2_246_u.G2longitude, gv2_246_u.G2latitude,
                  s=.2,
                  edgecolor='none',
                  transform=ccrs.PlateCarree())
    axins.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                                   name='land',
                                                   scale='110m'),
                      facecolor='#ccc',
                      edgecolor='black',
                      linewidth=.1,
                      zorder=2)
    axins.set_global()
    axins.spines['geo'].set_linewidth(.3) # thiner border
        

    # Add colorbar at bottom line of each variable
    cbar = fig_s.colorbar(cfvar, ax=ax_s[iv, 0])
    cbar.ax.tick_params(labelsize=5, length=2, width=.6)
    cbar.ax.set_ylabel(VARS_labs[v], fontsize=5, labelpad=2)
        
fig_s.subplots_adjust(hspace=.3)
fpath = ('figures/hansell_glodap/global/helper/' +
         'help_section_SP_' +  str(cruise_code) + '_' + '_'.join(VARS) + 
         '.svg')
fig_s.savefig(fpath, format='svg', bbox_inches='tight')

