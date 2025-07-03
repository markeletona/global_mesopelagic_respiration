# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:07:36 2024

Visualise literature review of OURs.

@author: Markel Gómez Letona
"""


#%% IMPORTS

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import gaussian_kde

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
 

#%% LOAD DATA

fpath = 'rawdata/litrev/OUR_estimates_literature.csv'
tb = pd.read_table(fpath, sep=',', header=0,
                   dtype={'article': str,
                          'method': str,
                          'our': float,
                          'our_rangeminus': float,
                          'our_rangeplus': float,
                          'depth': float,
                          'depth_range': float,
                          'water_mass': str,
                          'latitude': float,
                          'longitude': float,
                          'ocean': str})

# Replace invalid values (-9999) with nan
tb = tb.replace(-9999, np.nan)
tb = tb.replace('-9999', np.nan)

# Load our aOUR estimates too, for comparison
fpath = 'deriveddata/hansell_glodap/global/o2_aou_doc_age_regressions.csv'
reg1 = pd.read_csv(fpath, sep=',')
reg1 = reg1.replace(-9999, np.nan)


#%% MAP OF DATA LOCATION AND DEPTH

# Figure distribution: one row, map on the left occupying most and depth 
# distribution to the right, narrow
#
# Create corresponding grid.
# Note that the shape of the map makes it quite low in height compared to the
# axis itself, so the depth plot is taller. But, to be of similar height we 
# will need to tweak the grid rows too.
cm = 1/2.54
fig_map = plt.figure(figsize=(12*cm, 8*cm))
nc = 6
nr = 17
gs = GridSpec(nr, nc, figure=fig_map)


## Map 

# Create axis with projection
mproj = ccrs.Mollweide(central_longitude=-160)
ax_map = fig_map.add_subplot(gs[:, 0:(nc-1)], projection=mproj)

# Map land
feat = cfeature.NaturalEarthFeature(category='physical',
                                    name='land',
                                    scale='50m')
ax_map.add_feature(feat,
                   facecolor='#ddd', edgecolor='#555',
                   linewidth=.2,
                   zorder=0)

# Add points
ax_map.scatter(tb['longitude'], tb['latitude'],
               marker='o', facecolor='#222', edgecolor='w',
               s=10, linewidth=.2,
               transform=ccrs.PlateCarree())
ax_map.set_global()


## Depth distribution


# Compute density distribution
vals = tb.depth[~np.isnan(tb.depth)]
density = gaussian_kde(vals)
density.covariance_factor = lambda : .25
density._compute_covariance()
xs = np.arange(0, max(vals), 10)
ys = density(xs)

# Add axis for the depth distribution
ax_dep = fig_map.add_subplot(gs[3:(nr-3), (nc-1)])

# Plot density distribution
ax_dep.fill_betweenx(xs, ys,
                     facecolor='#bbb',
                     zorder=1)
# ax_dep.plot(ys, xs,
#             color='#555',
#             lw=.5,
#             zorder=1)

# Plot locations of points
ax_dep.scatter([-.0001]*len(tb), tb.depth,
                marker='o', facecolor='#222', edgecolor='w',
                s=10, linewidth=.1,
                zorder=1)
ax_dep.set(ylim=[0, 5000], yticks=range(0, 5500, 1000))
# Remove spines and adjust axes to give a clean look
spines = ['top', 'right', 'bottom', 'left']
for s in spines:
    ax_dep.spines[s].set_visible(False)
    
ax_dep.tick_params(axis='x', top=False, bottom=False, labelbottom=False)
ax_dep.tick_params(axis='y', right=False, left=False, 
                   labelright=True, labelleft=False,
                   labelsize=5, labelcolor='#555')
ax_dep.set_axisbelow(True) # so that grid is below other plot elements
ax_dep.grid(axis='y', color='#ccc', linewidth=.2, zorder=0)
ax_dep.set_ylabel("Depth [m, dbar]", fontsize=5, labelpad=4, color='#555')
ax_dep.yaxis.set_label_position('right')
ax_dep.invert_yaxis()

# Figure background transparent
fig_map.set_facecolor('none')

fpath = 'figures/litrev/our_literate_review_map.pdf'
fig_map.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)
fpath = fpath.replace("pdf", "svg")
fig_map.savefig(fpath, format='svg', bbox_inches='tight', transparent=False)



#%% OUR PROFILES

# Plot OUR vs depth scatters, overlying a LOWESS smoothing on top.
# 
# Use this wrapper function to estimate the LOWESS and its confidence 
# intervals (with bootstrap resampling) 
# https://www.statsmodels.org/dev/examples/notebooks/generated/lowess.html#Confidence-interval

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


# Set seed
np.random.seed(321)


#%%% Represent results separated by ocean/sea

xlab = "OUR [$\mu$mol kg$^{-1}$ yr$^{-1}$]"
ylab = "Depth [m, dbar]"

basins = ['Atlantic', 'Indian', 'Pacific', 'Arctic', 'Southern', 
          'Mediterranean', 'Global']
basins_lab = {k:v for k, v in zip(basins, basins)}
nc = 2
nr = int(np.ceil(len(basins) / 2))
n_b = len(basins)

fig_prf = plt.figure(figsize=(12*cm, 15*cm))
for i, v in enumerate(basins):
    
    # Subset data
    idx_oce = (tb['ocean']==v)
    tbss = tb.loc[idx_oce,:]
    x = tbss['our']
    y = tbss['depth']
    
    # Plot data
    ax_prf = fig_prf.add_subplot(nr, nc, i + 1)
    ax_prf.scatter(x, y,
                      marker='o', color='none', edgecolor='steelblue',
                      s=10, linewidth=.5,
                      zorder=1)
    
    # Customise axes etc.
    ax_prf.set(xlim=[.01, 2000],
               ylim=[0, 5000])
    ax_prf.set_xscale('log')
    ax_prf.tick_params(which='major', direction='in',
                          top=True, right=True,
                          length=2.5, labelsize=5, pad=2)
    ax_prf.tick_params(which='minor', direction='in',
                          top=True, right=True,
                          length=1.5, labelsize=5, pad=2)
    ax_prf.text(.95, .07, basins_lab[v], size=6,
                color='#222', weight='bold',
                horizontalalignment='right',
                path_effects=[pe.withStroke(linewidth=2, foreground='w')],
                transform=ax_prf.transAxes)
    ax_prf.grid(axis='x', color='#ccc', linewidth=.2, zorder=0)
    ax_prf.set_axisbelow(True) # so that grid is below other plot elements
    
    # Y-axis labels only on the left
    if (i % nc)==0:
        ax_prf.set_ylabel(ylab, fontsize=5, labelpad=2)
    
    # X-axis labels in last row of each column
    last_rows = [*range(n_b - (nc - 1), n_b + 1)]
    if (i + 1) in last_rows:
        ax_prf.set_xlabel(xlab, fontsize=5, labelpad=2)
    
    ax_prf.invert_yaxis()


fpath = "figures/litrev/our_literate_review_profiles.svg"
fig_prf.savefig(fpath, format='svg', bbox_inches='tight', transparent=True)



#%%% Compare with our aOURs

#------------------------------------------------------------------------------

#### Set  pval and R2 limits to accept regressions...

pval_limit = .001 # acceptable if lower than
r2_limit = .15   # acceptable if greater than


#------------------------------------------------------------------------------


# Create column for ocean to match regression and litrev data
reg1['ocean'] = reg1.water_mass.str.split(";", expand=True)[0]


# Exclude incubation results from the literature
incub = ['Aristegui et al. (2005)', 'Cohn et al. (2024)',
         'Mazuecos et al. (2015)', 'Odate et al. (2002)',
         'Weinbauer et al. (2013)']
idx_exc = ~(tb['article'].isin(incub))

# Parameters for plotting...
nms_tracers = ['AGE_CFC11_RES', 'AGE_CFC12_RES', 'AGE_SF6_RES']
mrk_tracers = dict(zip(nms_tracers, ['o', '^',  's']))
pal_tracers = dict(zip(nms_tracers, ['#DDAA33', '#BB5566', '#004488']))
txt_tracers = dict(zip(nms_tracers,
                       ["OUR$_\mathregular{CFC\u201011}$",
                        "OUR$_\mathregular{CFC\u201012}$",
                        "OUR$_\mathregular{SF_6}$"]))

ocs = ['Atlantic', 'Indian', 'Pacific']
nc = len(ocs)


#%%%% PLOT ALL POINTS


fig_ic, ax_ic = plt.subplots(1, nc, figsize=(5*cm*nc, 5*cm))
for i, v in enumerate(ocs):
    
    # Subset literature values above 1500 m
    idx_oce = ((tb.ocean==v) & (tb.depth <= 1500))
    tbss = tb.loc[idx_oce & idx_exc,:]
    x = tbss.our
    y = tbss.depth
    
    # Subset our aOURs, enforcing conditions
    idx = ((reg1.ocean==v.lower()) &
           (reg1.spvalue < pval_limit) &
           (reg1.r2 > r2_limit) &
           (reg1.slope >= 0) &
           (reg1.y_var=='AOU_RES'))
    reg1ss = reg1.loc[idx, :]

    # Plot lit.
    ax_ic[i].scatter(x, y,
                     marker='o', facecolor='#222', edgecolor='w',
                     s=10, linewidth=.2,
                     zorder=2)
    
    # compute smoothing and plot it
    # eval_y = np.linspace(np.min(y), np.max(y), 200)
    # smoothed, left, right = lowess_with_confidence_bounds(
    #     y, np.log10(x), eval_y, N=1000,
    #     lowess_kw={"frac": .2}
    #     )
    # xs = smoothed[:, 1]
    # ys = smoothed[:, 0]
    # ax_ic.plot(10**xs, ys, color="orange", alpha=.8, zorder=2)
    # ax_ic.fill_betweenx(eval_y, 10**left, 10**right,
    #                     alpha=.25, color="orange",
    #                     zorder=2)
    
    # Add our aOURs
    what_to_plot = 'scatter' # scatter | errorbar
    for a in nms_tracers:
        reg1ssa = reg1ss.loc[reg1ss.x_tracer==a, :]
        if what_to_plot=='errorbar':
            ax_ic[i].errorbar(x=reg1ssa['slope'],
                              y=reg1ssa['CTD_PRESSURE'],
                              xerr=reg1ssa['sci95'],
                              yerr=reg1ssa['CTD_PRESSURE_SD'],
                              color=pal_tracers[a],
                              markersize=3,
                              marker=mrk_tracers[a],
                              markeredgecolor='#222',
                              markeredgewidth=.5,
                              linestyle='none',
                              capsize=1.25,
                              ecolor=pal_tracers[a],
                              elinewidth=.75,
                              zorder=1)
        else:
            ax_ic[i].scatter(reg1ssa['slope'], 
                             reg1ssa['CTD_PRESSURE'],
                             marker=mrk_tracers[a],
                             edgecolor=pal_tracers[a],
                             facecolor='w',
                             s=8,
                             alpha=.8,
                             linewidth=.6,
                             zorder=1)
        
    # Add legend (only in one plot)
    if i==0:
        hdl = [Line2D([0], [0],
                      color='none',
                      marker=mrk_tracers[k],
                      markeredgecolor=kv,
                      markerfacecolor='w',
                      markersize=4,
                      markeredgewidth=.7,
                      label=txt_tracers[k]) for k, kv in pal_tracers.items()]
        # Add legend point por literature values
        hdl = hdl + [Line2D([0], [0],
                            color='none',
                            marker='o',
                            markeredgecolor='none',
                            markerfacecolor='#222',
                            markersize=4,
                            markeredgewidth=.1,
                            label='OUR$_{Lit.}$')]
        legi = ax_ic[i].legend(handles=hdl,
                               handletextpad=.1,
                               labelspacing=.6,
                               prop={'size': 4.5},
                               frameon=False,
                               loc='upper left',
                               borderaxespad=1)
        legi.get_frame().set_edgecolor('none')
    
    # Add label
    ax_ic[i].text(.95, .07, v, size=6,
                  color='#222', weight='bold',
                  horizontalalignment='right',
                  path_effects=[pe.withStroke(linewidth=2, foreground='w')],
                  transform=ax_ic[i].transAxes)
    
    # Prepare axes
    ax_ic[i].set(xlim=[.01, 200],
                 ylim=[0, 1500],
                 yticks=range(0, 1800, 300))
    ax_ic[i].yaxis.set_minor_locator(mticker.MultipleLocator(150))
    ax_ic[i].set_xscale('log')
    ax_ic[i].tick_params(axis='both', which='major', 
                         direction='in',
                         length=2.5, labelsize=5,
                         top=True, right=True)
    ax_ic[i].tick_params(axis='both', which='minor', 
                         direction='in',
                         length=1.5,
                         top=True, right=True)
    # y tick labels only on first and last column (left and right, respect.)
    if i==(nc-1):
        ax_ic[i].tick_params(axis='both', which='major', 
                             labelleft=False, labelright=True)
    if i not in [0, nc-1]:
        ax_ic[i].set(yticklabels=[])
        
    ax_ic[i].set_axisbelow(True) # so that grid is below other plot elements
    ax_ic[i].grid(axis='x', color='#ccc', linewidth=.2, zorder=0)

    if i==0:
        ax_ic[i].set_ylabel(ylab, fontsize=6, labelpad=2)
    ax_ic[i].set_xlabel(xlab, fontsize=6, labelpad=2)
    ax_ic[i].invert_yaxis()
    
    # Plot background to white
    ax_ic[i].set_facecolor('w')

# But figure background transparent
fig_ic.set_facecolor('none')

# Adjust spacing...
fig_ic.subplots_adjust(wspace=.07)

# export
fpath = ('figures/litrev/our_literate_review_profiles_compare_' + 
         what_to_plot + '.pdf')
fig_ic.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)
fpath = fpath.replace("pdf", "svg")
fig_ic.savefig(fpath, format='svg', bbox_inches='tight', transparent=False)    


#%%%% SMOOTH PROFILES

fig_ic2, ax_ic2 = plt.subplots(1, nc, figsize=(5*cm*nc, 5*cm))
for i, v in enumerate(ocs):
    
    # Subset literature values above 1500 m
    idx_oce = ((tb.ocean==v) & (tb.depth <= 1500))
    tbss = tb.loc[idx_oce & idx_exc,:]
    x = tbss.our
    y = tbss.depth
    
    # Subset our aOURs, enforcing conditions
    idx = ((reg1.ocean==v.lower()) &
           (reg1.spvalue < pval_limit) &
           (reg1.r2 > r2_limit) &
           (reg1.slope >= 0) &
           (reg1.y_var=='AOU_RES'))
    reg1ss = reg1.loc[idx, :]

    # Plot lit.
    ax_ic2[i].scatter(x, y,
                     marker='o', facecolor='#aaa', edgecolor='w',
                     s=10, linewidth=.2,
                     zorder=1)
    
    # Add smoothed line of literature values?
    smooth_lit = True
    if smooth_lit:
        if v in ['Atlantic', 'Pacific']: # exclude Indian (too few points)
            eval_y = np.linspace(np.min(y), np.max(y), 200)
            fr = .2 # 1 if (v=='Indian') else .2
            smoothed, left, right = lowess_with_confidence_bounds(
                y, np.log10(x), eval_y, N=1000,
                conf_interval=.95,
                lowess_kw={'frac': fr}
                )
            ax_ic2[i].plot(10**smoothed, eval_y, color='#222', alpha=.8, 
                           zorder=2)
            ax_ic2[i].fill_betweenx(eval_y, 10**left, 10**right,
                                alpha=.25, facecolor='#222',
                                zorder=1)
    
    # Add our aOURs (as smoothed lines of underlying data)
    for a in nms_tracers:
        
        # Subset data
        reg1ssa = reg1ss.loc[reg1ss.x_tracer==a, :]
        
        # compute smoothing and plot it
        x = reg1ssa.slope
        y = reg1ssa.CTD_PRESSURE
        eval_y = np.linspace(np.min(y), np.max(y), 200)
        # Due to the distribution of data for Indian SF6 rates, the 
        # bootstrapping gives issues (creates edge cases with large outliers)
        # if frac is set below aprox. .3 (which is ok for the rest). So do 
        # exception and increase frac for Indian SF6 rates.
        # (see testing in the section below)
        fr = .35 if ((v=='Indian') & (a=='AGE_SF6_RES')) else .2
        smoothed, left, right = lowess_with_confidence_bounds(
            y, np.log10(x), eval_y, N=1000,
            conf_interval=.95,
            lowess_kw={'frac': fr}
            )
        ax_ic2[i].plot(10**smoothed, eval_y, color=pal_tracers[a], alpha=.8, 
                       zorder=2)
        ax_ic2[i].fill_betweenx(eval_y, 10**left, 10**right,
                            alpha=.25, facecolor=pal_tracers[a],
                            zorder=1)
    
        
    # Add legend (only in one plot)
    if i==0:
        hdl = [Line2D([0], [0],
                      color=kv,
                      label=txt_tracers[k]) for k, kv in pal_tracers.items()]
        # Add legend point por literature values
        hdl = hdl + [Line2D([0], [0],
                            color='#222',
                            marker='o',
                            markeredgecolor='none',
                            markerfacecolor='#aaa',
                            markersize=3.2,
                            markeredgewidth=.1,
                            label='OUR$_{Lit.}$')]
        legi = ax_ic2[i].legend(handles=hdl,
                                handletextpad=1,
                                handlelength=1,
                                labelspacing=.6,
                                prop={'size': 4.5},
                                frameon=False,
                                loc='upper left',
                                borderaxespad=1)
        legi.get_frame().set_edgecolor('none')
        for t in legi.get_texts(): t.set_va('center')
        
    # Add label
    ax_ic2[i].text(.95, .07, v, size=6,
                  color='#222', weight='bold',
                  ha='right',
                  path_effects=[pe.withStroke(linewidth=2, foreground='w')],
                  transform=ax_ic2[i].transAxes)
    
    # Prepare axes
    ax_ic2[i].set(xlim=[.01, 200],
                  ylim=[0, 1500],
                  yticks=range(0, 1800, 300))
    ax_ic2[i].yaxis.set_minor_locator(mticker.MultipleLocator(150))
    ax_ic2[i].set_xscale('log')
    ax_ic2[i].tick_params(axis='both', which='major', 
                         direction='in',
                         length=2.5, labelsize=5,
                         top=True, right=True)
    ax_ic2[i].tick_params(axis='both', which='minor', 
                         direction='in',
                         length=1.5,
                         top=True, right=True)
    # y tick labels only on first and last column (left and right, respect.)
    if i==(nc-1):
        ax_ic2[i].tick_params(axis='both', which='major', 
                             labelleft=False, labelright=True)
    if i not in [0, nc-1]:
        ax_ic2[i].set(yticklabels=[])
        
    ax_ic2[i].set_axisbelow(True) # so that grid is below other plot elements
    ax_ic2[i].grid(axis='x', color='#ccc', linewidth=.2, zorder=0)

    if i==0:
        ax_ic2[i].set_ylabel(ylab, fontsize=6, labelpad=2)
    ax_ic2[i].set_xlabel(xlab, fontsize=6, labelpad=2)
    ax_ic2[i].invert_yaxis()
    
    # Plot background to white
    ax_ic2[i].set_facecolor('w')

# But figure background transparent
fig_ic2.set_facecolor('none')

# Adjust spacing...
fig_ic2.subplots_adjust(wspace=.07)

# export
fpath = ('figures/litrev/our_literate_review_profiles_compare_smooth.pdf')
fig_ic2.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)
fpath = fpath.replace("pdf", "png")
fig_ic2.savefig(fpath, format='png', bbox_inches='tight', transparent=False,
                dpi=600) 



#%%%% Testing smoothing of Indian sf6

# This subset of data seems to give problems when bootstraping.
# Apparently this is due to the distribution of data? It seem of be very
# sensible to the fraction of values included in the lowess. E.g., below ~.2
# some of the bootstraps create large outliers... After testing, for this case
# subset of data frac needs to be set to .5, or at least .4

# -----------------------------------------------------------------------------

# Test results changing the frac value:
fr = .5

# -----------------------------------------------------------------------------

# Subset data
idx = ((reg1.ocean=='indian') &
       (reg1.spvalue < pval_limit) &
       (reg1.r2 > r2_limit) &
       (reg1.slope >= 0) &
       (reg1.y_var=='AOU_RES') &
       (reg1.x_tracer=='AGE_SF6_RES'))
reg1ss = reg1.loc[idx, :]   

x = reg1ss.CTD_PRESSURE
y = np.log10(reg1ss.slope)
eval_x = np.linspace(np.min(x), np.max(x), 100)
N = 30
conf_interval = .95
np.random.seed(123)

# Lowess smoothing
smoothed = sm.nonparametric.lowess(exog=x, endog=y,
                                   xvals=eval_x,
                                   frac=fr)

# Bootstraping
smoothed_values = np.empty((N, len(eval_x)))
bootstrapped_points = np.empty((N, len(x)))
x.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
for i in range(N):
    sample = np.random.choice(len(x), len(x), replace=True)
    bootstrapped_points[i] = sample
    sampled_x = x[sample]
    sampled_y = y[sample]
    
    smoothed_values[i] = sm.nonparametric.lowess(
        exog=sampled_x, endog=sampled_y, xvals=eval_x, 
        frac=fr
    )


# Plot results of individual bootstraps
nc = int(np.floor(N**.5))
nr = int(np.ceil(N/nc))

fig, ax = plt.subplots(nr, nc, figsize=(5*cm*nc, 5*cm*nr))
ax = ax.flatten()
for i in range(len(smoothed_values)):
    ax[i].scatter(10**y[bootstrapped_points[i]], x[bootstrapped_points[i]], 
               marker='o', facecolor='none', edgecolor='tab:orange',
               alpha=.5,
               zorder=2)
    ax[i].plot(10**smoothed_values[i, :], eval_x, 
               color=pal_tracers[a], alpha=.8, 
               zorder=1)
    
    ax[i].set_xscale('log')
    ax[i].set(xlim=[.5, 50],
           ylim=[0, 1500],
           yticks=range(0, 1800, 300))
    ax[i].invert_yaxis()


fpath = ('figures/litrev/loess_test_bootstrap_indian_sf6_aour.pdf')
fig.savefig(fpath, format='pdf', bbox_inches='tight', transparent=False)


