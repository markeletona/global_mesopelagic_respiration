# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 09:54:35 2025

Analyse trends in OURs.


@author: Markel
"""

#%% IMPORTS

import pandas as pd
import numpy as np
import pathlib
import os
from shapely import from_geojson
import scripts.modules.RegressConsensus as rc
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import cmocean as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature



#%% LOAD DATA

# OURs
fpath = 'deriveddata/hansell_glodap/global/o2_aou_doc_age_regressions.csv'
reg = pd.read_csv(fpath, sep=',')


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



## Polygons of pixels used in the regressions

# fpath = 'deriveddata/hansell_glodap/global/regression_pixels.npy'
# pixel_polys = np.load(fpath, allow_pickle='TRUE').item()

# Set directory and get list of files.
fpath = pathlib.Path('deriveddata/hansell_glodap/global/regression_pixel_polys/').glob("*.geojson")
flist = [x for x in fpath]

# Preallocate space and read files geojson files
pixel_polys = {}
for w in wm_polys_flat: pixel_polys[w] = {}
for f in flist:
    
    # Read and store it
    pcode = str(f).split("\\")[-1].replace("_polygon.geojson", "").split(",")
    w = pcode[0]
    p = pcode[1]
    pixel_polys[w][p] = from_geojson(f.read_text())
    

#%% DATA HANDLING

# Replace bad values with nan
reg = reg.replace(-9999, np.nan)


#%% SETUP

## Set up fun for smoothing
def lowess_with_confidence_bounds(
    x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y,
                                       xvals=eval_x,
                                       **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    # make sure to reset indexes of x and y so that x[sample], y[sample]
    # work properly (safeguard if provided x,y have already some index)
    smoothed_values = np.empty((N, len(eval_x)))
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, 
            **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top



#------------------------------------------------------------------------------
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

#### Set  pval and R2 limits to accept regressions...

pval_limit = .001 # acceptable if lower than
r2_limit = .15   # acceptable if greater than

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
#------------------------------------------------------------------------------



# This will only be done for Atlantic, Indian and Pacific
oceans = ['atlantic', 'indian', 'pacific']

# Color palette for oceans
ocean_pal = {k: v for k, v in zip(oceans, ['#44AA99', '#88CCEE', '#EE8866'])}


# Age res labels
ages_res = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
ages_res_labs = ["Age$_\mathregular{CFC\u201011\Delta}$",
                 "Age$_\mathregular{CFC\u201012\Delta}$",
                 "Age$_\mathregular{SF_6\Delta}$"]
ages_res_labs = dict(zip(ages_res, ages_res_labs))

ages_pal = {'AGE_CFC11_RES': '#DDAA33',
            'AGE_CFC12_RES': '#BB5566',
            'AGE_SF6_RES':   '#004488'}

# Convert age_res labels into bold
ages_res_labs_b = {k:v for k, v in zip(ages_res,
                                       [("$\mathbf{" +
                                         ages_res_labs[g].
                                         replace("$", "").
                                         replace("\mathregular", "") +
                                         "}$")
                                         for g in ages_res])}

ylab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"


#%% ESTIMATE CARBON FLUX 

# Estimate carbon flux at water mass depth based on NPP and Martin curve.

def cflux(z, f0, z0=1, b=.858):
    return f0 * (z / z0) ** (-b)


# Test different z0:
plt.figure()
for y in [1, 20, 50]:
    zs = range(y, 1500)
    plt.scatter([cflux(f0=500, z=x, z0=y) for x in zs], zs, 
                label=("z0 = " + str(y)))

plt.gca().invert_yaxis()
plt.legend()


# Test different b:
plt.figure()
for b in [.6, .8, 1]:
    zs = range(20, 1500)
    plt.scatter([cflux(f0=500, z=x, z0=20, b=b) for x in zs], zs, 
                label=("b = " + str(b)))

plt.gca().invert_yaxis()
plt.legend()



reg['CFLUX_EPPL'] = cflux(z=reg.CTD_PRESSURE, 
                          f0=reg.NPP_EPPL,
                          z0=10,
                          b=reg.B)

reg['CFLUX_CBPM'] = cflux(z=reg.CTD_PRESSURE, 
                          f0=reg.NPP_CBPM,
                          z0=10,
                          b=reg.B)


#%%% MAP NPP, B AND CFLUX IN PIXELS

dpath = 'figures/hansell_glodap/global/rates/maps/helper/'
if not os.path.exists(dpath): os.makedirs(dpath)

cm = 1/2.54

wm_depths_idx = {k:v for k, v in zip(wm_depths, [0, 1, 2])}
wm_depths_lab = {k:v for k, v in zip(wm_depths,
                                     ['Central', 'Interm.', 'Deep'])}

# Do not include SAIW and MW (because they overlap with other water masses)
in_supinf = ['SAIW', 'MW']

nr = len(wm_depths_idx) # + 1
nc = len(ages_res)
for v in ['NPP_EPPL', 'NPP_CBPM', 'B', 'CFLUX_EPPL', 'CFLUX_CBPM']:

    
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

     
    if v=='NPP_EPPL':
       
        cap_val1 = 200
        cap_val2 = 1000
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = cmo.cm.algae
        rnd = 1 # rounding precission for labels
        cbar_lab = "NPP$_{Eppley}$ [mg C m$^{-2}$ d$^{-1}$]"
        # Create colorbar tickmarks, ensure cap_val2 is included
        inc = np.ceil((cap_val2 - cap_val1) / 5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        # Set colourbar extension
        ext = 'both'
        
    elif v=='NPP_CBPM':
       
        cap_val1 = 200
        cap_val2 = 1000
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = cmo.cm.algae
        rnd = 1 # rounding precission for labels
        cbar_lab = "NPP$_{CbPM}$ [mg C m$^{-2}$ d$^{-1}$]"
        # Create colorbar tickmarks, ensure cap_val2 is included
        inc = np.ceil((cap_val2 - cap_val1) / 5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        # Set colourbar extension
        ext = 'both'
        
    elif v=='B':
         
        cap_val1 = .3
        cap_val2 = 1.5
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = cmo.cm.tarn #mpl.colormaps.get_cmap('RdBu_r')
        rnd = 2
        cbar_lab = "b"
        inc = .3
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.sort(np.unique(tcks))
        ext = 'both'
        
    elif v=='CFLUX_EPPL':
       
        cap_val1 = 0
        cap_val2 = 60
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = cmo.cm.matter
        rnd = 1 # rounding precission for labels
        cbar_lab = "C$_{flux\u2010Eppley}$ [mg C m$^{-2}$ d$^{-1}$]"
        # Create colorbar tickmarks, ensure cap_val2 is included
        inc = np.ceil((cap_val2 - cap_val1) / 5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        # Set colourbar extension
        ext = 'max'
        
    elif v=='CFLUX_CBPM':
       
        cap_val1 = 0
        cap_val2 = 60
        norm = mpl.colors.Normalize(vmin=cap_val1, vmax=cap_val2)
        px_pal = cmo.cm.matter
        rnd = 1 # rounding precission for labels
        cbar_lab = "C$_{flux\u2010CbPM}$ [mg C m$^{-2}$ d$^{-1}$]"
        # Create colorbar tickmarks, ensure cap_val2 is included
        inc = np.ceil((cap_val2 - cap_val1) / 5)
        tcks = np.concatenate((np.arange(cap_val1, cap_val2, inc),
                               [cap_val2]))
        tcks = np.unique(tcks)
        # Set colourbar extension
        ext = 'max'
        
    else:
        print("No settings for " + v + ", skipping map!")
        continue

            
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
            idx = ((reg.y_var=='AOU_RES') & (reg.x_tracer==g)) 
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
                    val = r[v]
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
    fpath = ('figures/hansell_glodap/global/rates/maps/helper/' +
             'hansell_glodap_global_rate_pixel_map_aux_' + v + '.svg')
    fig_m.savefig(fpath, format='svg', bbox_inches='tight', transparent=False, 
                  dpi=300) # low dpi can affect colourbar rendering in pdfs
    # fpath = fpath.replace("pdf", "svg")
    # fig_m.savefig(fpath, format='svg', bbox_inches='tight', transparent=False, 
    #               dpi=300)
    plt.close(fig_m)



#%% Central water masses

# Subset OURs for central water mass (meeting required criteria)
# (do it preemptively here to set shared y axis of subplots)
idx = ((reg.water_mass.str.contains('central')) &
       (reg.y_var=='AOU_RES') &
       (reg.x_tracer.isin(ages_res)) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit) &
       (reg.slope >= 0))
r_central = reg.loc[idx, :]



#%%% Latitudinal trends

dpath = 'figures/hansell_glodap/global/rates/vs_latitude/'
if not os.path.exists(dpath): os.makedirs(dpath)

cm = 1/2.54

# Set seed
np.random.seed(123)


smo_our = {}
smo_our_lat = {}

nr = len(ages_res)
nc = len(oceans)
fig_lat, ax_lat = plt.subplots(nrows=nr, ncols=nc, 
                               figsize=(5*cm * nc, 5*cm * nr))
for io, o in enumerate(oceans):

    # Subset OURs for central water mass of ocean o  
    rss = r_central.loc[r_central.water_mass.str.contains(o), :]

    
    for ia, a in enumerate(ages_res):
        
        # Subset rates for age a
        rss2 = rss.loc[rss.x_tracer==a, :]
        
        # compute smoothing and plot it
        x = rss2.LATITUDE
        y = rss2.slope
        eval_x = np.linspace(np.min(x), np.max(x), 200)
        # Due to the distribution of data for SF6 rates, the bootstrapping 
        # gives issues (creates edge cases with large outliers) if frac is set
        # below aprox. .4 (which is ok for the rest). So do 
        # exception and increase frac for SF6 rates.
        fr = .6 if (a=='AGE_SF6_RES') else .4
        smoothed, bottom, top = lowess_with_confidence_bounds(
            x, y, eval_x, N=1000,
            conf_interval=.95,
            lowess_kw={'frac': fr}
            )
        smo_our[a + '-' + o + '-' + v] = smoothed
        smo_our_lat[a + '-' + o + '-' + v] = eval_x
        
        # Plot results
        ax_lat[ia, io].scatter(x, y,
                               s=10, lw=.1,
                               facecolor=ocean_pal[o],
                               edgecolor='w', alpha=.5,
                               )
        ax_lat[ia, io].plot(eval_x, smoothed,
                            color=ocean_pal[o], alpha=1,
                            zorder=2)
        ax_lat[ia, io].fill_between(eval_x, bottom, top,
                            alpha=.25, facecolor=ocean_pal[o],
                            zorder=1)
        
        # Customise axes
        ax_lat[ia, io].set(xlim=[-60, 60],
                           ylim=[0, 5*np.ceil(max(r_central.slope)/5)],
                           xticks=range(-60, 80, 30))
        ax_lat[ia, io].tick_params(axis='both', which='major',
                                   direction='in', length=2,
                                   labelsize=7,
                                   right=True, top=True)
        ax_lat[ia, io].tick_params(axis='both', which='minor',
                                   direction='in', length=1.5,
                                   labelsize=7,
                                   right=True, top=True)
        ax_lat[ia, io].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_lat[ia, io].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_lat[ia, io].text(.06, .91, ages_res_labs_b[a],
                            ha='left', va='center',
                            rotation=0,
                            size=5, weight='bold', color='#444',
                            transform=ax_lat[ia, io].transAxes)
        if io==0:
            ax_lat[ia, io].set_ylabel(ylab, fontsize=7)
        if io==(nc-1):
            ax_lat[ia, io].tick_params(axis='both', which='major', 
                                 labelleft=False, labelright=True)

        if io not in [0, nc-1]:
            ax_lat[ia, io].set(yticklabels=[])
        if ia==(nr-1):
            ax_lat[ia, io].set_xlabel("Latitude [$\degree$N]", fontsize=7)
        if ia==0:
            ax_lat[ia, io].text(.5, 1.05, o.capitalize(),
                                ha='center', va='baseline',
                                size=7, weight='bold', color='#444',
                                transform=ax_lat[ia, io].transAxes)
            
    # Plot background to white
    ax_lat[ia, io].set_facecolor('w')

# Figure background transparent
fig_lat.set_facecolor('none')


fig_lat.subplots_adjust(wspace=.13, hspace=.18)
fpath = ('figures/hansell_glodap/global/rates/vs_latitude/our_by_latitude_central.svg')
fig_lat.savefig(fpath, format='svg', bbox_inches='tight', transparent=False)



#### Show some values

idx = ((reg.water_mass.str.contains('central')) &
       (reg.y_var=='AOU_RES') &
       (reg.x_tracer.isin(ages_res)) &
       (reg.water_mass.str.contains('atlantic')) &
       (reg.LATITUDE > 5) & (reg.LATITUDE < 15) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit) &
       (reg.slope >= 0))
ss = reg.loc[idx, :]
for a in ages_res:
    
    print("Atlantic central, lats (5, 15), " + a + " mean OUR: " + 
          str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))
    
idx = ((reg.water_mass.str.contains('central')) &
       (reg.y_var=='AOU_RES') &
       (reg.x_tracer.isin(ages_res)) &
       (reg.water_mass.str.contains('SPEW')) &
       # (reg.water_mass.str.contains('pacific')) &
       # (reg.LATITUDE > -10) & (reg.LATITUDE < 0) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit) &
       (reg.slope >= 0))
ss = reg.loc[idx, :]
for a in ages_res:
    
    print("Pacific central, SPEW, " + a + " mean OUR: " + 
          str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))
    
idx = ((reg.water_mass.str.contains('central')) &
       (reg.y_var=='AOU_RES') &
       (reg.x_tracer.isin(ages_res)) &
       (reg.water_mass.str.contains('IUW')) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit) &
       (reg.slope >= 0))
ss = reg.loc[idx, :]
for a in ages_res:
    
    print("Indian central, IUW, " + a + " mean OUR: " + 
          str(round(np.nanmean(ss.slope[ss.x_tracer==a]), 1)))
    
    
#%%%% temp, oxygen, DOC


VARS = ['PT', 'OXYGEN', 'DOC']
round_val = {'PT': 3,
             'DOC': 5,
             'OXYGEN': 30}
VARS_lab = {'PT': "$\\theta$ [$\degree$C]",
            'DOC': "DOC [$\mu$mol L$^{-1}$]",
            'OXYGEN': "O$_2$ [$\mu$mol kg$^{-1}$]"}

for a in ages_res:
    
    # do it separately for values associated to rates based on each age 
    r_central_a = r_central.loc[r_central.x_tracer==a, :]
    
    nr = len(VARS)
    nc = len(oceans)
    fig_lat, ax_lat = plt.subplots(nrows=nr, ncols=nc, 
                                   figsize=(5*cm * nc, 5*cm * nr))
    for io, o in enumerate(oceans):
    
        # Subset OURs for central water mass of ocean o  
        rss = r_central_a.loc[r_central_a.water_mass.str.contains(o), :]
        
        for iv, v in enumerate(VARS):
                        
            # Compute smoothing and plot it
            x = rss.LATITUDE
            y = rss[v]
            eval_x = np.linspace(np.min(x), np.max(x), 200)
            # Again, exception due to the distribution of data
            fr = .6 if (v=='DOC') | (a=='AGE_SF6_RES') else .4
            smoothed, bottom, top = lowess_with_confidence_bounds(
                x, y, eval_x, N=1000,
                conf_interval=.95,
                lowess_kw={'frac': fr}
                )
    
            
            # Plot results
            ax_lat[iv, io].scatter(x, y,
                                   s=10, lw=.1,
                                   facecolor=ocean_pal[o],
                                   edgecolor='w', alpha=.5,
                                   )
            ax_lat[iv, io].plot(eval_x, smoothed,
                                color=ocean_pal[o], alpha=1,
                                zorder=2)
            ax_lat[iv, io].fill_between(eval_x, bottom, top,
                                alpha=.25, facecolor=ocean_pal[o],
                                zorder=1)
            
            # Customise axes
            if v=='OXYGEN':
                ymin = 0
            else:
                ymin = round_val[v] * np.floor(np.min(r_central_a[v]) / round_val[v])
            ymax = round_val[v] * np.ceil(np.max(r_central_a[v]) / round_val[v])
            ax_lat[iv, io].set(xlim=[-60, 60],
                               ylim=[ymin, ymax],
                               xticks=range(-60, 80, 30))
            ax_lat[iv, io].tick_params(axis='both', which='major',
                                       direction='in', length=2,
                                       labelsize=7,
                                       right=True, top=True)
            ax_lat[iv, io].tick_params(axis='both', which='minor',
                                       direction='in', length=1.5,
                                       labelsize=7,
                                       right=True, top=True)
            ax_lat[iv, io].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax_lat[iv, io].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
            if io==0:
                ax_lat[iv, io].set_ylabel(VARS_lab[v], fontsize=7)
            if io==(nc-1):
                ax_lat[iv, io].tick_params(axis='both', which='major', 
                                     labelleft=False, labelright=True)
            if io not in [0, nc-1]:
                ax_lat[iv, io].set(yticklabels=[])
            if iv==(nr-1):
                ax_lat[iv, io].set_xlabel("Latitude [$\degree$N]", fontsize=7)
            if iv==0:
                ax_lat[iv, io].text(.5, 1.05, o.capitalize(),
                                    ha='center', va='baseline',
                                    size=7, weight='bold', color='#444',
                                    transform=ax_lat[iv, io].transAxes)
                
        # Plot background to white
        ax_lat[iv, io].set_facecolor('w')
    
    # Figure background transparent
    fig_lat.set_facecolor('none')
    
    
    fig_lat.subplots_adjust(wspace=.13, hspace=.18)
    fpath = ('figures/hansell_glodap/global/rates/vs_latitude/envvar_by_latitude_' +
             'central_for_OURs_of_' + a + '.svg')
    fig_lat.savefig(fpath, format='svg', bbox_inches='tight',
                    transparent=False)


#%%%% NPP, CFLUX


VARS = ['NPP_EPPL', 'NPP_CBPM', 'CFLUX_EPPL', 'CFLUX_CBPM']

round_val = {'NPP_EPPL': 300,
             'NPP_CBPM': 300,
             'CFLUX_EPPL': 20,
             'CFLUX_CBPM': 20}
VARS_lab = {'NPP_EPPL': "NPP$_{Eppley}$ [mg C m$^{-2}$ d$^{-1}$]",
            'NPP_CBPM': "NPP$_{CbPM}$ [mg C m$^{-2}$ d$^{-1}$]",
            'CFLUX_EPPL': "C$_{flux\u2010Eppley}$ [mg C m$^{-2}$ d$^{-1}$]",
            'CFLUX_CBPM': "C$_{flux\u2010CbPM}$ [mg C m$^{-2}$ d$^{-1}$]"}

smo_v = {}
smo_v_lat = {}
for a in ages_res:
    
    # do it separately for values associated to rates based on each age 
    r_central_a = r_central.loc[r_central.x_tracer==a, :]
    
    nr = len(VARS)
    nc = len(oceans)
    fig_lat, ax_lat = plt.subplots(nrows=nr, ncols=nc, 
                                   figsize=(5*cm * nc, 5*cm * nr))
    for io, o in enumerate(oceans):
    
        # Subset OURs for central water mass of ocean o  
        rss = r_central_a.loc[r_central_a.water_mass.str.contains(o), :]
        
        for iv, v in enumerate(VARS):
                        
            # Compute smoothing and plot it
            x = rss.LATITUDE
            y = rss[v]
            eval_x = np.linspace(np.min(x), np.max(x), 200)
            fr = .5 if (a=='AGE_SF6_RES') else .4
            smoothed, bottom, top = lowess_with_confidence_bounds(
                x, y, eval_x, N=1000,
                conf_interval=.95,
                lowess_kw={'frac': fr}
                )
            smo_v[a + '-' + o + '-' + v] = smoothed
            smo_v_lat[a + '-' + o + '-' + v] = eval_x

            # Plot results
            ax_lat[iv, io].scatter(x, y,
                                   s=10, lw=.1,
                                   facecolor=ocean_pal[o],
                                   edgecolor='w', alpha=.5,
                                   )
            ax_lat[iv, io].plot(eval_x, smoothed,
                                color=ocean_pal[o], alpha=1,
                                zorder=2)
            ax_lat[iv, io].fill_between(eval_x, bottom, top,
                                alpha=.25, facecolor=ocean_pal[o],
                                zorder=1)
            
            # Customise axes
            ymin = round_val[v] * np.floor(np.min(r_central_a[v]) / round_val[v])
            if 'NPP' in v: 
                # There is a single outlier in the Atlantic (>4000)* that
                # distorts the visualisation it is included in the LOWESS
                # computation but not included in the plot.
                # *To be precise at x,y: (-6.65, 4161)
                ymax = 2000
            elif 'EP_' in v:
                # Similar for export production
                # x,y: (-6.65, 1091)
                ymax = 350
            else:
                ymax = round_val[v] * np.ceil(np.max(r_central_a[v]) / round_val[v])
            ax_lat[iv, io].set(xlim=[-60, 60],
                               ylim=[ymin, ymax],
                               xticks=range(-60, 80, 30))
            ax_lat[iv, io].tick_params(axis='both', which='major',
                                       direction='in', length=2,
                                       labelsize=7,
                                       right=True, top=True)
            ax_lat[iv, io].tick_params(axis='both', which='minor',
                                       direction='in', length=1.5,
                                       labelsize=7,
                                       right=True, top=True)
            ax_lat[iv, io].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax_lat[iv, io].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
            if io==0:
                ax_lat[iv, io].set_ylabel(VARS_lab[v], fontsize=7)
            if io==(nc-1):
                ax_lat[iv, io].tick_params(axis='both', which='major', 
                                     labelleft=False, labelright=True)
            if io not in [0, nc-1]:
                ax_lat[iv, io].set(yticklabels=[])
            if iv==(nr-1):
                ax_lat[iv, io].set_xlabel("Latitude [$\degree$N]", fontsize=7)
            if iv==0:
                ax_lat[iv, io].text(.5, 1.05, o.capitalize(),
                                    ha='center', va='baseline',
                                    size=7, weight='bold', color='#444',
                                    transform=ax_lat[iv, io].transAxes)
                
        # Plot background to white
        ax_lat[iv, io].set_facecolor('w')
    
    # Figure background transparent
    fig_lat.set_facecolor('none')
    
    
    fig_lat.subplots_adjust(wspace=.13, hspace=.18)
    fpath = ('figures/hansell_glodap/global/rates/vs_latitude/production_by_latitude_' +
             'central_for_OURs_of_' + a + '.svg')
    fig_lat.savefig(fpath, format='svg', bbox_inches='tight',
                    transparent=False)


#%%% Correlate OURs and NPP

# Select variables
VARS = ['NPP_EPPL', 'NPP_CBPM', 'CFLUX_EPPL', 'CFLUX_CBPM']

# Transform values?
log10_transform = True

VARS_lab_log10 = {'NPP_EPPL': 'Log$_{10}$(NPP$_{Eppley}$) [mg C m$^{-2}$ d$^{-1}$]',
                  'NPP_CBPM': 'Log$_{10}$(NPP$_{CbPM}$) [mg C m$^{-2}$ d$^{-1}$]',
                  'CFLUX_EPPL': 'Log$_{10}$(C$_{flux\u2010Eppley}$) [mg C m$^{-2}$ d$^{-1}$]',
                  'CFLUX_CBPM': 'Log$_{10}$(C$_{flux\u2010CbPM}$) [mg C m$^{-2}$ d$^{-1}$]'}

for a in ages_res:
    
    # do it separately for values associated to rates based on each age 
    r_central_a = r_central.loc[r_central.x_tracer==a, :]
    
    nr = len(VARS)
    nc = len(oceans)
    fig_cor, ax_cor = plt.subplots(nrows=nr, ncols=nc, 
                                   figsize=(5*cm * nc, 5*cm * nr))

    for iv, v in enumerate(VARS):

        # Set shared limits
        x_shared = r_central_a[v]
        y_shared = r_central_a.slope

        if log10_transform:
            x_shared = np.log10(x_shared)
            y_shared = np.log10(y_shared)
            xmin = .2 * np.floor(np.min(x_shared) / .2) - .1
            xmax = .2 * np.ceil(np.max(x_shared) / .2) + .1
            ymin = .2 * np.floor(np.min(y_shared) / .2) - .1
            ymax = .2 * np.ceil(np.max(y_shared) / .2) + .1
        else:
            xmin = 0
            xmax = round_val[v] * np.ceil(np.max(x_shared) / round_val[v])
            ymin = 0
            ymax = 5 * np.ceil(np.max(y_shared) / 5)
            
        for io, o in enumerate(oceans):
        
            # Subset OURs for central water mass of ocean o  
            rss = r_central_a.loc[r_central_a.water_mass.str.contains(o), :]
            
            # Do regression of npp vs our
            x = rss[v]
            y = rss.slope
            if log10_transform:
                x = np.log10(x)
                y = np.log10(y)
            md_npp = rc.RegressConsensusW(x, y)
            
            # Plot results
            ax_cor[iv, io].scatter(x, y,
                                   s=10, lw=.1,
                                   facecolor=ocean_pal[o],
                                   edgecolor='w', alpha=.5,
                                   )
            # If regression was significant, add line
            if md_npp['spvalue'] < .05:
                x0 = np.nanmin(x)
                x1 = np.nanmax(x)
                y0 = md_npp['intercept'] + x0 * md_npp['slope']
                y1 = md_npp['intercept'] + x1 * md_npp['slope']
                ax_cor[iv, io].plot([x0, x1], [y0, y1],
                                    c=ocean_pal[o], lw=1.2,
                                    zorder=1)
                # add results
                ast = "(*)" if md_npp['ipvalue']>.05 else ""
                if md_npp['spvalue'] < .001:
                    pval = "$\mathit{p}$ < 0.001"
                elif md_npp['spvalue'] < .01:
                    pval = "$\mathit{p}$ < 0.01"
                else:
                    pval = "$\mathit{p}$ = " + '{:.2f}'.format(md_npp['spvalue'])
                isign = "$-$" if md_npp['intercept'] < 0 else "+"
                txt = ("y = " + 
                       '{:.5f}'.format(md_npp['slope']) + 
                       "x " + isign + " " +
                       '{:.1f}'.format(abs(md_npp['intercept'])) + 
                       ast + " ; " + pval +
                       " ; R$^2$ = " + '{:.2f}'.format(md_npp['r2']))
                ax_cor[iv, io].text(.98, .04, txt,
                                    size=3,
                                    ha='right', va='baseline',
                                    transform=ax_cor[iv, io].transAxes)
                ax_cor[iv, io].set(xlim=[xmin, xmax],
                                   ylim=[ymin, ymax])
            if log10_transform:
                ax_cor[iv, io].set_xlabel(VARS_lab_log10[v], fontsize=6, labelpad=2)

            else:
                ax_cor[iv, io].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax_cor[iv, io].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax_cor[iv, io].set_xlabel(VARS_lab[v], fontsize=6, labelpad=2)

            ax_cor[iv, io].tick_params(axis='both', which='major',
                                       direction='in', length=2,
                                       labelsize=6,
                                       right=True, top=True)
            ax_cor[iv, io].tick_params(axis='both', which='minor',
                                       direction='in', length=1.5,
                                       right=True, top=True)

            if io==0:
                ylab = "Log$_{10}$(OUR) [$\mathregular{\mu}$mol kg$^{-1}$ yr$^{-1}$]"
                ax_cor[iv, io].set_ylabel(ylab, fontsize=6)
                
            if iv==0:
                ax_cor[iv, io].text(.5, 1.05, o.capitalize(),
                                    ha='center', va='baseline',
                                    size=7, weight='bold', color='#444',
                                    transform=ax_cor[iv, io].transAxes)
            # Plot background to white
            ax_cor[iv, io].set_facecolor('w')
            

        # Figure background transparent
        fig_cor.set_facecolor('none')
        
        
        fig_cor.subplots_adjust(wspace=.3, hspace=.3)
        fpath = ('figures/hansell_glodap/global/rates/vs_latitude/' +
                 'npp_vs_OUR_central_for_OURs_of_' + a + '.svg')
        fig_cor.savefig(fpath, format='svg', bbox_inches='tight',
                        transparent=False)
        




#%% Deep Atlantic

# Subset OURs for central water mass (meeting required criteria)
# (do it preemptively to set shared y axis of subplots)
idx = ((reg.water_mass.str.contains('LSW|UNADW|CDW')) &
       (reg.y_var=='AOU_RES') &
       (reg.x_tracer.isin(ages_res)) &
        # (reg.spvalue < pval_limit) &
        # (reg.r2 > r2_limit) &
        (reg.slope >= 0))
r_deep = reg.loc[idx, :].copy()

# Set =0 the slopes of regressions that do NOT meet any of the required criteria
r_deep.loc[((r_deep.spvalue >= pval_limit) | 
            (r_deep.r2 <= r2_limit)), 'slope'] = 0


#%%% Latitudinal trends

# Subset OURs for Atlantic 
o = 'atlantic'
io = 0
rss = r_deep.loc[[o in x for x in r_deep.water_mass], :]


VARS = ['slope', 'PT', 'DOC', 'OXYGEN', 'NPP_EPPL']
VARS_lab = {'slope' : "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]",
            'PT': "$\\theta$ [$\degree$C]",
            'DOC': "DOC [$\mu$mol L$^{-1}$]",
            'OXYGEN': "O$_2$ [$\mu$mol kg$^{-1}$]",
            'NPP_EPPL': "NPP$_{Eppley}$ [mg C m$^{-2}$ d$^{-1}$]",
            'NPP_CBPM': "NPP$_{CbPM}$ [mg C m$^{-2}$ d$^{-1}$]"}
VARS_tit = {'slope' : "$\mathbf{OUR}$",
            'PT': "$\mathbf{\\theta}$",
            'DOC': "$\mathbf{DOC}$",
            'OXYGEN': "$\mathbf{O_2}$",
            'NPP_EPPL': "$\mathbf{NPP_{Eppley}}$",
            'NPP_CBPM': "$\mathbf{NPP_{CbPM}}$"}
# Values to round to (multiples of)
VARS_r = {'slope' : 2,
          'PT': 2,
          'DOC': 2,
          'OXYGEN': 20,
          'NPP_EPPL': 200,
          'NPP_CBPM': 200}

nr = len(ages_res)
nc = len(VARS)
fig_lat, ax_lat = plt.subplots(nrows=nr, ncols=nc, 
                               figsize=(6*cm * nc, 5*cm * nr),
                               squeeze=False)

for ia, a in enumerate(ages_res):

    # Subset rates for age a
    rss2 = rss.loc[rss.x_tracer==a, :]
    
    for iv, v in enumerate(VARS):

        # compute smoothing and plot it
        x = rss2.LATITUDE
        y = rss2[v]
        eval_x = np.linspace(np.min(x), np.max(x), 200)
        # Due to the distribution of data for SF6 rates, the bootstrapping 
        # gives issues (creates edge cases with large outliers) if frac is set
        # below aprox. .4 (which is ok for the rest). So do 
        # exception and increase frac for SF6 rates.
        fr = 1 if (a=='AGE_SF6_RES') else .4
        smoothed, bottom, top = lowess_with_confidence_bounds(
            x, y, eval_x, N=1000,
            conf_interval=.95,
            lowess_kw={'frac': fr}
            )
    
        
        # Plot results
        ax_lat[ia, iv].scatter(x, y,
                               s=10, lw=.1,
                               facecolor=ocean_pal[o],
                               edgecolor='w', alpha=.5,
                               )
        ax_lat[ia, iv].plot(eval_x, smoothed,
                            color=ocean_pal[o], alpha=1,
                            zorder=2)
        ax_lat[ia, iv].fill_between(eval_x, bottom, top,
                            alpha=.25, facecolor=ocean_pal[o],
                            zorder=1)
        
        # Customise axes
        ax_lat[ia, iv].set(xlim=[-80, 60],
                           ylim=[VARS_r[v]*np.floor(np.min(r_deep[v])/VARS_r[v]), 
                                 VARS_r[v]*np.ceil(np.max(r_deep[v])/VARS_r[v])],
                           xticks=range(-60, 80, 30))
        ax_lat[ia, iv].tick_params(axis='both', which='major',
                                   direction='in', length=2,
                                   labelsize=6,
                                   right=True, top=True)
        ax_lat[ia, iv].tick_params(axis='both', which='minor',
                                   direction='in', length=1.5,
                                   labelsize=6,
                                   right=True, top=True)
        ax_lat[ia, iv].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_lat[ia, iv].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_lat[ia, iv].set_ylabel(VARS_lab[v], fontsize=6)
        
        if iv==(nc-1):
            ax_lat[ia, iv].text(1.13, .5, ages_res_labs_b[a],
                                ha='left', va='center',
                                rotation=270,
                                size=7, weight='bold', color='#444',
                                transform=ax_lat[ia, iv].transAxes)
        if io not in [0, nc-1]:
            ax_lat[ia, iv].set(yticklabels=[])
        if ia==(nr-1):
            ax_lat[ia, iv].set_xlabel("Latitude [$\degree$N]", fontsize=6)
        if ia==0:
            ax_lat[ia, iv].text(.5, 1.10, VARS_tit[v],
                                ha='center', va='baseline',
                                size=7, weight='bold', color='#444',
                                transform=ax_lat[ia, iv].transAxes)
            
    # Plot background to white
    ax_lat[ia, iv].set_facecolor('w')

# Figure background transparent
fig_lat.set_facecolor('none')


fig_lat.subplots_adjust(wspace=.5, hspace=.18)
fpath = ('figures/hansell_glodap/global/rates/vs_latitude/our_by_latitude_deep_atlantic.svg')
fig_lat.savefig(fpath, format='svg', bbox_inches='tight', transparent=False)


#%%% OUR vs DOC

nr = len(ages_res)
nc = 1
fig_our_vs_doc, ax_our_vs_doc = plt.subplots(nrows=nr, ncols=nc, 
                                             figsize=(6*cm * nc, 5*cm * nr),
                                             squeeze=False)

for ia, a in enumerate(ages_res):

    # Subset rates for age a
    rss2 = rss.loc[(rss.x_tracer==a) & 
                   ~(rss.water_mass.str.contains('CDW')), :]
    
    # Regress OUR vs [DOC]
    x = rss2.slope
    y = rss2.DOC
    md_our_doc = rc.RegressConsensusW(x, y)
            
    # Plot results
    ax_our_vs_doc[ia, 0].scatter(x, y,
                                 s=10, lw=.1,
                                 facecolor=ocean_pal['atlantic'],
                                 edgecolor='w', alpha=.5,
                                 )
    
    # If regression was significant, add line
    if md_our_doc['spvalue'] < .05:
        x0 = np.nanmin(x)
        x1 = np.nanmax(x)
        y0 = md_our_doc['intercept'] + x0 * md_our_doc['slope']
        y1 = md_our_doc['intercept'] + x1 * md_our_doc['slope']
        ax_our_vs_doc[ia, 0].plot([x0, x1], [y0, y1],
                                  c=ocean_pal[o], lw=1.2,
                                  zorder=1)
        # add results
        ast = "(*)" if md_our_doc['ipvalue']>.05 else ""
        if md_our_doc['spvalue'] < .001:
            pval = "$\mathit{p}$ < 0.001"
        elif md_our_doc['spvalue'] < .01:
            pval = "$\mathit{p}$ < 0.01"
        else:
            pval = "$\mathit{p}$ = " + '{:.2f}'.format(md_our_doc['spvalue'])
        isign = "$-$" if md_our_doc['intercept'] < 0 else "+"
        txt = ("DOC = " + '{:.2f}'.format(md_our_doc['slope']) + 
               " $\\times$ OUR " + isign + " " +
               '{:.1f}'.format(abs(md_our_doc['intercept'])) + 
               ast + " ; " + pval +
               " ; R$^2$ = " + '{:.2f}'.format(md_our_doc['r2']))
        ax_our_vs_doc[ia, 0].text(.98, .04, txt,
                                  size=3,
                                  ha='right', va='baseline',
                                  transform=ax_our_vs_doc[ia, 0].transAxes)

    
    ax_our_vs_doc[ia, 0].set(xlim=[0, 7], ylim=[35, 50])
    ax_our_vs_doc[ia, 0].set_ylabel(VARS_lab['DOC'], fontsize=7)
    ax_our_vs_doc[ia, 0].set_xlabel(VARS_lab['slope'], fontsize=7)
    ax_our_vs_doc[ia, 0].tick_params(axis='both', which='major',
                                         direction='in', length=2,
                                         labelsize=6,
                                         right=True, top=True)
    ax_our_vs_doc[ia, 0].tick_params(axis='both', which='minor',
                                         direction='in', length=1.5,
                                         labelsize=6,
                                         right=True, top=True)
    ax_our_vs_doc[ia, 0].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_our_vs_doc[ia, 0].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    

fig_our_vs_doc.subplots_adjust(hspace=.4)

fpath = ('figures/hansell_glodap/global/rates/vs_latitude/our_vs_doc_deep_atlantic.svg')
fig_our_vs_doc.savefig(fpath, format='svg', bbox_inches='tight', transparent=False)


#%%% DOCUR: show some values

idx = ((reg.water_mass.str.contains('SPMW|LSW|UNADW|CDW')) &
       (reg.y_var=='DOC_RES') &
       (reg.x_tracer.isin(ages_res)) &
       (reg.spvalue < pval_limit) &
       (reg.r2 > r2_limit))
r_deep_docur = reg.loc[idx, :].copy()

for w in r_deep_docur.water_mass.unique():
    for a in ages_res:
        
        ss = r_deep_docur.loc[((r_deep_docur.water_mass==w) &
                               (r_deep_docur.x_tracer==a))]
        print(w.split(";")[2] + " -> " + a + " mean DOCUR: " + 
              str(round(np.nanmean(ss.slope), 2)) +
              " [" + str(round(np.min(ss.slope), 2)) + ", " +
              str(round(np.max(ss.slope), 2)) + "] "
              " (" + str(len(ss)) + ")")
