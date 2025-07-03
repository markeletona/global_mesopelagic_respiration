# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:40:26 2023

TESTING INTERPOLATION METHODS

@author: Markel Gómez-Letona
"""

#%% IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton

#%% EXAMPLE FROM DOCUMENTATION

#### Test the different interpolation methods available in RBFInterpolator:

# Create random sequence of coordinates in space:
rng = np.random.default_rng(1789)
xobs = 2*Halton(2, seed=rng).random(100) - 1
yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

xgrid = np.mgrid[-1:1:50j, -1:1:50j]
xflat = xgrid.reshape(2, -1).T

krn = ['linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 
       'inverse_multiquadric', 'inverse_quadratic', 'gaussian']
ygrids = {}
cm = 1/2.54
fig_krn, ax_krn = plt.subplots(4, 2, figsize=(13*cm, 15*cm))
for ki, k in enumerate(krn):
    
    # Perform interpolation:
    yflat = RBFInterpolator(y=xobs, d=yobs,
                            neighbors=15, kernel=k, epsilon=2)(xflat)
    ygrid = yflat.reshape(50, 50)

    # Plot results:
    ax_krn[ki//2, ki%2].pcolormesh(*xgrid, ygrid,
                                   vmin=-0.25, vmax=0.25,
                                   shading='gouraud')
    p = ax_krn[ki//2, ki%2].scatter(*xobs.T, c=yobs, s=10, ec='k',
                                    linewidth=.5, vmin=-0.25, vmax=0.25)
    ax_krn[ki//2, ki%2].set_title(k, fontsize=6, y=.97)
    ax_krn[ki//2, ki%2].tick_params(axis='both', which='major', 
                                    direction='out', right=True, length=2, 
                                    labelsize=6)
    
    # Save results:
    ygrids[k] = ygrid
    
cb_krn = fig_krn.colorbar(p, ax=ax_krn.ravel().tolist(),
                          cax=fig_krn.add_axes([0.8, 0.31, 0.025, 0.38]))
cb_krn.ax.tick_params(labelsize=7)
fig_krn.subplots_adjust(hspace=.45, wspace=.35, right=.75)

fpath = "figures/interpolation_testing/interpolator_testing_kernel.pdf"
fig_krn.savefig(fpath, format='pdf', bbox_inches='tight', facecolor=None)


#### Compare differences between methods (use gaussian as reference):
fig_krn2, ax_krn2 = plt.subplots(4, 2, figsize=(13*cm, 15*cm))
for ki, k in enumerate(krn):

    # Plot results:
    pc = ax_krn2[ki//2, ki%2].pcolormesh(*xgrid,
                                         (ygrids[k]-ygrids['gaussian']),
                                         vmin=-0.07, vmax=0.07,
                                         cmap='RdBu',
                                         shading='gouraud')
    # Add contours of interpolation as reference:
    ax_krn2[ki//2, ki%2].contour(*xgrid, ygrids[k], 
                                 levels = [-.15, -.05, -.01, .01, .05, .15],
                                 linewidths=.4,
                                 colors='#aaa')
    ax_krn2[ki//2, ki%2].set_title(k + ' – gaussian', fontsize=6, y=.97)
    ax_krn2[ki//2, ki%2].tick_params(axis='both', which='major', 
                                    direction='out', right=True, length=2, 
                                    labelsize=6)
    
cb_krn2 = fig_krn2.colorbar(pc, ax=ax_krn2.ravel().tolist(),
                            cax=fig_krn2.add_axes([0.8, 0.31, 0.025, 0.38]))
cb_krn2.ax.tick_params(labelsize=7)
fig_krn2.subplots_adjust(hspace=.45, wspace=.35, right=.75)

fpath = "figures/interpolation_testing/interpolator_testing_kernel_diff.pdf"
fig_krn2.savefig(fpath, format='pdf', bbox_inches='tight', facecolor=None)


#### Test various neighbor numbers: 
ygrids2 = {}
fig_krn3, ax_krn3 = plt.subplots(3, 2, figsize=(13*cm, 12*cm))
neis = [3, 5, 8, 10, 15, 20]
for ni, n in enumerate(neis):
    
    # Perform interpolation:
    yflat = RBFInterpolator(y=xobs, d=yobs,
                            neighbors=n, kernel='gaussian', epsilon=2)(xflat)
    ygrid = yflat.reshape(50, 50)

    # Plot results:
    ax_krn3[ni//2, ni%2].pcolormesh(*xgrid, ygrid,
                                    vmin=-0.25, vmax=0.25,
                                    shading='gouraud')
    p = ax_krn3[ni//2, ni%2].scatter(*xobs.T, c=yobs, s=10, ec='k',
                                    linewidth=.5, vmin=-0.25, vmax=0.25)
    ax_krn3[ni//2, ni%2].set_title('neighbors = ' + str(n), fontsize=6, y=.97)
    ax_krn3[ni//2, ni%2].tick_params(axis='both', which='major', 
                                    direction='out', right=True, length=2, 
                                    labelsize=6)
    
    # Save results:
    ygrids2[str(n)] = ygrid
    
cb_krn3 = fig_krn3.colorbar(p, ax=ax_krn3.ravel().tolist(),
                            cax=fig_krn3.add_axes([0.8, 0.31, 0.025, 0.38]))
cb_krn3.ax.tick_params(labelsize=7)
fig_krn3.subplots_adjust(hspace=.45, wspace=.35, right=.75)

fpath = "figures/interpolation_testing/interpolator_testing_neighbors.pdf"
fig_krn3.savefig(fpath, format='pdf', bbox_inches='tight', facecolor=None)


#### Compare differences between neighbor numbers (use 20 as reference):
fig_krn4, ax_krn4 = plt.subplots(3, 2, figsize=(13*cm, 12*cm))
for ni, n in enumerate(neis):

    # Plot results:
    pc = ax_krn4[ni//2, ni%2].pcolormesh(*xgrid,
                                         (ygrids2[str(n)]-ygrids2['20']),
                                         vmin=-0.12, vmax=0.12,
                                         cmap='RdBu',
                                         shading='gouraud')
    # Add contours of interpolation as reference:
    ax_krn4[ni//2, ni%2].contour(*xgrid, ygrids2[str(n)], 
                                 levels = [-.15, -.05, -.01, .01, .05, .15],
                                 linewidths=.4,
                                 colors='#aaa')
    ax_krn4[ni//2, ni%2].set_title('neighbors(' + str(n) + ') – neighbors(20)',
                                   fontsize=6, y=.97)
    ax_krn4[ni//2, ni%2].tick_params(axis='both', which='major', 
                                    direction='out', right=True, length=2, 
                                    labelsize=6)
    
cb_krn4 = fig_krn4.colorbar(pc, ax=ax_krn4.ravel().tolist(),
                            cax=fig_krn4.add_axes([0.8, 0.31, 0.025, 0.38]))
cb_krn4.ax.tick_params(labelsize=7)
fig_krn4.subplots_adjust(hspace=.45, wspace=.35, right=.75)

fpath = "figures/interpolation_testing/interpolator_testing_neighbors_diff.pdf"
fig_krn4.savefig(fpath, format='pdf', bbox_inches='tight', facecolor=None)



#%% TEST WITH REAL (WOCE) DATA

# Filtered Hansell dataset:
fpath = "deriveddata/dom_hansell/Hansell_2022_o2_doc_cfc11_cfc12_sf6_with_ages.csv"
df = pd.read_csv(fpath, sep=",", header=0, dtype={'BOTTLE': str})

# Assign missing/invalid values as NAN:
df.replace(-999, np.nan, inplace=True)

# Subset A16N (2013) cruise:
df = df.loc[df['CRUISE']=="A16N (2013)",:]

# When trying to perform the interpolation for the entire section the function
# returns the error "LinAlgError: Singular matrix.". After searching for what
# might cause this error, I've read that duplicated data point coordinates
# [the 'y' in RBFInterpolator(y,...)] can be the reason. It turns out we *do*
# have a duplicated coord:
# df['id'] = df[['LATITUDE', 'CTD_PRESSURE']].astype(str).apply('_'.join, axis=1)
df2 = df.loc[df.duplicated(subset=['LATITUDE', 'CTD_PRESSURE']),:]

# Remove duplicate (values are basically the same):
df = df.loc[~df.duplicated(subset=['LATITUDE', 'CTD_PRESSURE']),:]
    

#### Create interpolation grid:
    
ix = df['LATITUDE']
iy = df['CTD_PRESSURE']
xmin = min(ix)
xmax = max(ix)
ymin = min(iy)
ymax = max(iy)

# Create the interpolation grid fitting to the requirements of RBFInterpolator:
# (I have slightly modified the grid creation process to better fit my needs)
ndim = 250
rx, ry = np.meshgrid(np.linspace(xmin, xmax, ndim),
                     np.linspace(ymin, ymax, ndim))
xflat = np.squeeze(np.array([rx.reshape(1, -1).T, ry.reshape(1, -1).T]).T)
# xgrid = np.mgrid[xmin:xmax:250j, ymin:ymax:250j]
# xflat = xgrid.reshape(2, -1).T    

# Assemble the data point coordinates and values as required:
coords = np.array([ix,iy]).T
vals = df['PT']

# Perform interpolation:
yflat = RBFInterpolator(y=coords, d=vals,
                        neighbors=20, kernel='linear', smoothing=0)(xflat)

# Reshape output into matrix:
ygrid = yflat.reshape(ndim, ndim)

# Plot results:
fig, ax = plt.subplots()
pc = ax.pcolormesh(rx, ry, ygrid, shading='gouraud')
# p = ax.scatter(*coords.T, c=vals, s=10, ec='k', linewidth=.5)
ax.invert_yaxis()
fig.colorbar(pc)
plt.show()

# Note that we get a pattern with horizontal bands because there is very little
# vertical influence of values due to the differences in magnitude of the 
# scales of X (DEGREES, 10^1) and Y (DBAR, 10^3). But, we can equalise this
# influence by scaling the two axes to the same scale by dividing each by their
# standard deviation. The scaling could be tuned by then multiplying by a 
# factor to give one of them more or less importance.
# 
# NOTE: apparently stations 70, 71 have nearly the same coordinates, with some
# days apart. I guess the cruise stopped some days for rest/supplies because 
# it was long and then restarted? Anyway, this results in very very close data
# points which seems to break the interpolation (unreal high/low values,
# except if the linear kernel is used). So remove one of the profiles.
# This seems to fix most of the issues but the gaussian kernel seems to still
# be very sensitive and creates important noise?
ix = df.loc[~(df['STATION']==71),'LATITUDE']
iy = df.loc[~(df['STATION']==71), 'CTD_PRESSURE']    
xsd = np.std(ix)
ysd = np.std(iy)

rx2, ry2 = np.meshgrid(np.linspace(xmin, xmax, ndim)/xsd,
                       np.linspace(ymin, ymax, ndim)/ysd)
xflat2 = np.squeeze(np.array([rx2.reshape(1, -1).T, ry2.reshape(1, -1).T]).T)
coords2 = np.array([ix/xsd,iy/ysd]).T
vals = df.loc[~(df['STATION']==71), 'PT']

yflat2 = RBFInterpolator(y=coords2, d=vals,
                         neighbors=20, kernel='linear', smoothing=0)(xflat2)
ygrid2 = yflat2.reshape(ndim, ndim)


#### Compare outputs (unscaled vs scaled):

fig, ax = plt.subplots(1, 2, figsize=(15*cm, 5*cm))
pc1 = ax[0].pcolormesh(rx, ry, ygrid, shading='gouraud',
                       vmin=np.min(vals), vmax=np.max(vals))
pc2 = ax[1].pcolormesh(rx, ry, ygrid2, shading='gouraud',
                       vmin=np.min(vals), vmax=np.max(vals))


ax[0].set(xlim=[xmin, xmax], xticks=range(0, 70, 10),
          ylim=[0, 1200], yticks=range(0, 1400, 200))
ax[0].set_xlabel("Latitude [$\degree$N]", fontsize=8, labelpad=2)
ax[0].set_ylabel("Depth [dbar]", fontsize=8)
ax[0].tick_params(axis='both', which='major', length=2.5, labelsize=7)
ax[0].set_title("Unscaled axes\nRBF = 'linear'", fontsize=8)
ax[1].set(xlim=[xmin, xmax], xticks=range(0, 70, 10),
          ylim=[0, 1200], yticks=range(0, 1400, 200))
ax[1].set_xlabel("Latitude [$\degree$N]", fontsize=8, labelpad=2)
ax[1].set_ylabel(None)
ax[1].tick_params(axis='both', which='major', length=2.5, labelsize=7)
ax[1].set_title("Scaled axes (/sd)\nRBF = 'linear'", fontsize=8)
ax[1].set_yticklabels([])
ax[0].invert_yaxis()
ax[1].invert_yaxis()
cbar = fig.colorbar(pc2, ax=ax.ravel().tolist(),
                    ticks=np.linspace(5, 25, 5))
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_ylabel("$\\theta$ [$\degree$C]", fontsize=8)
fig.subplots_adjust(wspace=.07, right=.76)


fpath = "figures/interpolation_testing/interpolator_testing_woce_scaling.pdf"
fig.savefig(fpath, format='pdf', bbox_inches='tight', facecolor=None)
