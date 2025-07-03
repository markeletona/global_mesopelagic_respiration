# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:31:16 2024

@author: Markel
"""

#%% IMPORTS

import numpy as np
import pandas as pd
import pathlib
from shapely import geometry, from_geojson
import matplotlib.pyplot as plt


#%% LOAD DATA

# Filtered, merged Hansell+GLODAP dataset:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
tbl = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                   'CRUISE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})



# Atmospheric tracers histories:
    
# In late July 2023 the Bullister (2015) dataset was updated. It differs 
# slightly in formatting from previous version.

fpath = 'rawdata/tracers_atmosphere/0164584/2.2/data/0-data/CFC_ATM_Hist_2022/Tracer_atmospheric_histories_revised_2023_Table1.csv'

## Note that in the data table:
#  - First row is the header
#  - Second row is hemisphere designation
#  - Third row are the corresponding units of each variable
#  - Rest are rows with actual data

# Get header (replace YEAR with Year to match the 2015 header):
atm_hdr = pd.read_csv(fpath, sep=',', header=None, nrows=1).replace("YEAR", "Year").values.flatten().tolist()

# Get hemispheres:
atm_hem = pd.read_csv(fpath, sep=',', header=None, nrows=1, skiprows=1).replace(np.nan, "").values.flatten().tolist()

# Create new header by concatenating header and hemispheres:
atm_hdr2 = [i+j for i, j in zip(atm_hdr, atm_hem)]
    
# Get units:
atm_units = pd.read_csv(fpath, sep=',', header=None, nrows=1, skiprows=2).values.flatten().tolist()

# Get the data:
atm = pd.read_csv(fpath, sep=',', header=None, names=atm_hdr2, skiprows=3)   

# Subset desired year range and create datetime column:
catm = atm.loc[atm['Year']>1940,:].copy()


#%% DATA HANDLING

# Assign missing/invalid values as NAN:
tbl.replace(-9999, np.nan, inplace=True)
# tbl.replace('-999', np.nan, inplace=True)

# Subset Atlantic cruises (Hotmix is split, ignore it for the time being)
# start doing only the A16 section (both N & S)
idx = (tbl['OCEAN']=="Atlantic")
a = tbl.loc[idx,:].copy()

# The last and first stations of the 2013 A16N and A16S, respectively, overlap. 
# Both stations share sampling depths. Average results.
idx = ((a['CRUISE']=="A16N (2013)") & (a['STATION']==145) |
       (a['CRUISE']=="A16S (2013)") & (a['STATION']==1))
ol = a.loc[idx,:].copy()
ol_av = []
for c, cv in ol.items():
    
    # Skip iteration if c is a flag (they will be treated along their variable)
    if "_FLAG_W" in c:
        continue
    
    # Average values if numeric; if string, take just the first one
    if (ol[c].dtype=='O') | (c in ['STATION', 'CAST']):
        newc = ol.groupby('BOTTLE')[c].first()
        ol_av.append(newc)
        
    else:
        # If the variable has a flag, consider it.
        cflag = c + '_FLAG_W'
        if cflag in ol.columns:
            ol_ss = ol[ol[cflag].isin([2, 6])] # discard anything that's not 2, 6)
            newc = ol_ss.groupby('BOTTLE')[c].mean() # ignores nans by default
            newcflag = ol_ss.groupby('BOTTLE')[cflag].mean()
            ol_av.append(newc)
            ol_av.append(newcflag)
        else:
            newc = ol.groupby('BOTTLE')[c].mean() 
            ol_av.append(newc)


# Convert into a dataframe
ol_melt = pd.DataFrame(ol_av).T
ol_melt = ol_melt.infer_objects()
ol_melt.BOTTLE = ol_melt.BOTTLE.astype('Int32').astype(str)
ol_melt.reset_index(drop=True, inplace=True)

# Re-introduce data to the section dataframe (deleting the overlap stations)
a_og = a # keep copy of original just in case
a = pd.concat([a.loc[~idx,:], ol_melt], ignore_index=True)

# Reorder according to cruise direction and depth, and reset index
# (This is to aid visualisation, has no other real effect)
a.sort_values(['EXPOCODE', 'STATION', 'CTD_PRESSURE'],
              ignore_index=True, inplace=True)


#%% ASSIGN WATER MASS

# Loads polygons created in:
# 
# scripts/ocean_wm_defs/ocean_water_mass_definitions.py 
# 
# and also uses SIGMA0 ranges gathered from the literature.


#%%% SET SIGMA0 RANGES

## Load SIGMA0 ranges
fpath = 'deriveddata/ocean_wm_defs/water_mass_sigma0_definitions.txt'
wm_sigma0 = pd.read_csv(fpath, sep='\t')
wm_sigma0 = {k:[v1, v2] for k, v1, v2 in zip(wm_sigma0.water_mass,
                                             wm_sigma0.sigma0_min,
                                             wm_sigma0.sigma0_max)}


#%%% LOAD POLYGONS

fpath = pathlib.Path('deriveddata/ocean_wm_defs/').glob('*')
dlist = [x for x in fpath if x.is_dir()]
wm_polys = {}
wm_depths = ['central', 'intermediate', 'deep']
for d in dlist:
    for z in wm_depths:
        
        # Get wm paths at depth z and ocean d
        flist = [*pathlib.Path(str(d) + "\\wms\\" + z).glob("*.geojson")]
        
        # Skip iteration if certain depth is absent (i.e. flist is empty)
        if not flist: continue
        
        for f in flist:
            
            # Get wm name (accounts for when the name itself has underscore, 
            # e.g. AAIW_P)            
            w = "_".join(str(f).split("\\")[-1].split("_")[0:-1])

            # Load polygon
            wm_polys[w] = from_geojson(f.read_text())
            


#%%% DEPTH

# Also, set the upper depth limit to exclude the mixed layer:
dlim = 150

# And the lower depth limit:
dlim2 = 1500


#%% ASSIGN WATER MASSES

# Add a new column to assign samples to water masses based on this criteria:
a['wm'] = 'none'

# Convert sample coordinates to geometry points to check against water mass 
# polygons
spoints = []
for i, r in a.iterrows():
    spoints.append(geometry.Point([r['LONGITUDE'], r['LATITUDE']]))
    
wm_boo = {}
wms = list(wm_sigma0.keys())
for iw, w in enumerate(wms):
    
    # Get the indices of the samples that fulfill the conditions of 
    # water mass w:
    idx = ((a['SIGMA0']>wm_sigma0[w][0]) &
           (a['SIGMA0']<=wm_sigma0[w][1]) &
           (wm_polys[w].contains(spoints)) &
           (a['CTD_PRESSURE']>=dlim) &
           (a['CTD_PRESSURE']<=dlim2))
    wm_boo[w] = idx
    
    # Assign water mass string to those samples:
    a.loc[idx, 'wm'] = w

# There is a small area+sigma range where MW and UNADW overlap due to how
# definitions have been kept as simple as reasonably possible. In those 
# overlaps, make sure to assign the samples as MW
mw_unadw_overlap = wm_boo['MW'] & wm_boo['UNADW']
sum(mw_unadw_overlap)
a.loc[mw_unadw_overlap, 'wm'] = 'MW'


#%% AGE vs AOU relationship

ss = a.loc[a.wm=='SPMW']
x = 'AGE_CFC11'
y = 'AOU'
cm = 1/2.54
fig, ax = plt.subplots(figsize=(14*cm, 8*cm))
sc = ax.scatter(ss[x], ss[y],
                linewidth=.5,
                s=10,
                vmin=1982,
                vmax=2021,
                c=ss.YEAR_SAMPLE)
cbar = fig.colorbar(sc)
cbar.ax.set_ylabel("YEAR_SAMPLE")
ax.set_xlabel(x)
ax.set_ylabel(y)


#%% MIXING TWO END-MEMBERS

# Water sample composed of an equal mixture of two end-members (50%-50% ratio)
# Each end-member dates from (for simplicity, matching dates from the
# atmospheric histories):
em1_r = .5
em2_r = 1 - em1_r
em1_y = 1958.5
em2_y = 1970.5
em1_c11 = catm.CFC11NH[catm.Year==em1_y].squeeze()
em2_c11 = catm.CFC11NH[catm.Year==em2_y].squeeze()

# The resulting mix would have:
mx_y = em1_r * em1_y + em2_r * em2_y
mx_c11 = em1_r * em1_c11 + em2_r * em2_c11

# Compute year associated with mx_c11
diff = mx_c11 - catm.CFC11NH
lowest_above = diff[diff<0].nlargest(1).index[0]
lowest_below = diff[diff>=0].nsmallest(1).index[0]
lowest2 = [lowest_below, lowest_above]
mx_ya = np.interp(x=mx_c11, xp=catm.CFC11NH[lowest2], fp=catm.Year[lowest2])


# Initialise plot
fig, ax = plt.subplots(figsize=(12*cm, 7*cm))
ax.plot(catm.Year, catm.CFC11NH, 
        c='#222',
        zorder=0)

## Add example of lower end

ax.plot([em1_y, em2_y], [em1_c11, em2_c11], 
        c='#666', 
        linewidth=1,
        zorder=1)
ax.plot([mx_y, mx_ya], [mx_c11, mx_c11],
        c='#aaa', 
        linewidth=.7,
        zorder=1)
ax.scatter([em1_y, em2_y], [em1_c11, em2_c11], 
           marker='o', facecolor='none', edgecolor='steelblue',
           s=30, linewidth=.7,
           zorder=2)
ax.scatter(mx_y, mx_c11, 
           marker='s', facecolor='none', edgecolor='goldenrod',
           s=20, linewidth=.7,
           zorder=2)
ax.scatter(mx_ya, mx_c11, 
           marker='d', facecolor='none', edgecolor='firebrick',
           s=20, linewidth=.7,
           zorder=2)
ax.text((em1_y - .5), (em1_c11 + 8), "EM$_1$ → " + str(em1_y), 
        fontsize=4, ha='right', va='center', color='#777')
ax.text((em2_y + 1.5), (em2_c11 - 1), "EM$_2$ → " + str(em2_y), 
        fontsize=4, ha='left', va='center', color='#777')
ax.text((mx_y - 1.5), (mx_c11 + 3), "Real → " + str(mx_y), 
        fontsize=6.5, ha='right', va='center', color='goldenrod')
ax.text((mx_ya + 1.2), (mx_c11 - 3), "Apparent → " + str(round(mx_ya, 1)), 
        fontsize=6.5, ha='left', va='center', color='firebrick')


## Add example of upper end

em1_r = .5
em2_r = 1 - em1_r
em1_y = 1982.5
em2_y = 2000.5
em1_c11 = catm.CFC11NH[catm.Year==em1_y].squeeze()
em2_c11 = catm.CFC11NH[catm.Year==em2_y].squeeze()
mx_y = em1_r * em1_y + em2_r * em2_y
mx_c11 = em1_r * em1_c11 + em2_r * em2_c11
catmss = catm[catm.Year<=mx_y]
diff = mx_c11 - catmss.CFC11NH
lowest_above = diff[diff<0].nlargest(1).index[0]
lowest_below = diff[diff>=0].nsmallest(1).index[0]
lowest2 = [lowest_below, lowest_above]
mx_ya = np.interp(x=mx_c11, xp=catm.CFC11NH[lowest2], fp=catm.Year[lowest2])

ax.plot([em1_y, em2_y], [em1_c11, em2_c11], 
        c='#666', 
        linewidth=1,
        zorder=1)
ax.plot([mx_y, mx_ya], [mx_c11, mx_c11],
        c='#aaa', 
        linewidth=.7,
        zorder=1)
ax.scatter([em1_y, em2_y], [em1_c11, em2_c11], 
           marker='o', facecolor='none', edgecolor='steelblue',
           s=30, linewidth=.7,
           zorder=2)
ax.scatter(mx_y, mx_c11, 
           marker='s', facecolor='none', edgecolor='goldenrod',
           s=20, linewidth=.7,
           zorder=2)
ax.scatter(mx_ya, mx_c11, 
           marker='d', facecolor='none', edgecolor='firebrick',
           s=20, linewidth=.7,
           zorder=2)
ax.text((em1_y - 1.5), (em1_c11 + 1), "EM$_1$ → " + str(em1_y), 
        fontsize=4, ha='right', va='center', color='#777')
ax.text((em2_y + 1.5), (em2_c11 + 5), "EM$_2$ → " + str(em2_y), 
        fontsize=4, ha='left', va='center', color='#777')
ax.text((mx_y + 1.8), (mx_c11 - 3), "Real → " + str(mx_y), 
        fontsize=6.5, ha='left', va='center', color='goldenrod')
ax.text((mx_ya - 1.5), (mx_c11 + 3), "Apparent → " + str(round(mx_ya, 1)), 
        fontsize=6.5, ha='right', va='center', color='firebrick')


# Set axis details
ax.set(ylim=[0, 300])
ax.tick_params(labelsize=8)
ax.set_ylabel("pCFC-11", fontsize=8)
ax.set_xlabel("Year", fontsize=8)

fpath = 'figures/test_.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')


#%% ARTIFACT: SAME AGE, DIFFERENT AOU


aour = 5 # umol/kg/y

em1_r = .5
em2_r = 1 - em1_r
em1_y = 1982.5
em2_y = 2015.5
em1_c11 = catm.CFC11NH[catm.Year==em1_y].squeeze()
em2_c11 = catm.CFC11NH[catm.Year==em2_y].squeeze()
mx_y = em1_r * em1_y + em2_r * em2_y
mx_c11 = em1_r * em1_c11 + em2_r * em2_c11
catmss = catm[catm.Year<=mx_y]
diff = mx_c11 - catmss.CFC11NH
lowest_above = diff[diff<0].nlargest(1).index[0]
lowest_below = diff[diff>=0].nsmallest(1).index[0]
lowest2 = [lowest_below, lowest_above]
mx_ya = np.interp(x=mx_c11, xp=catm.CFC11NH[lowest2], fp=catm.Year[lowest2])

sample_year = 2018
em1_aou = (sample_year - em1_y) * aour
em2_aou = (sample_year - em2_y) * aour
mx_aou = em1_r * em1_aou + em2_r * em2_aou


# Initialise plot
fig, ax = plt.subplots(figsize=(12*cm, 7*cm))
ax.plot(catm.Year, catm.CFC11NH, 
        c='#222',
        zorder=0)

## Add example of lower end

ax.plot([em1_y, em2_y], [em1_c11, em2_c11], 
        c='#666', 
        linewidth=1,
        zorder=1)
ax.plot([mx_y, mx_ya], [mx_c11, mx_c11],
        c='#aaa', 
        linewidth=.7,
        zorder=1)
ax.scatter([em1_y, em2_y], [em1_c11, em2_c11], 
            marker='o', facecolor='none', edgecolor='steelblue',
            s=30, linewidth=.7,
            zorder=2)
ax.scatter(mx_y, mx_c11, 
            marker='s', facecolor='none', edgecolor='goldenrod',
            s=20, linewidth=.7,
            zorder=2)
ax.scatter(mx_ya, mx_c11, 
            marker='d', facecolor='none', edgecolor='firebrick',
            s=20, linewidth=.7,
            zorder=2)
ax.axvline(sample_year, linestyle='--', linewidth=.7, color='#bbb')

ax.text((em1_y - 1.5), (em1_c11 + 0), "EM$_1$ → " + str(em1_y), 
        fontsize=4, ha='right', va='center', color='#777')
ax.text((em2_y + .5), (em2_c11 + 9), "EM$_2$ → " + str(em2_y), 
        fontsize=4, ha='left', va='center', color='#777')
ax.text((mx_y + 1.5), (mx_c11 - 9), "Real → " + str(mx_y), 
        fontsize=6.5, ha='left', va='center', color='goldenrod')
ax.text((mx_ya - 2), (mx_c11 - 1), "Apparent → " + str(round(mx_ya, 1)), 
        fontsize=6.5, ha='right', va='center', color='firebrick')


catmss = catm[catm.Year<=((mx_ya - em1_r * em1_y)/em2_r + 1)]
diff = em2_c11 - catmss.CFC11NH
lowest_above = diff[diff<0].nlargest(1).index[0]
lowest_below = diff[diff>=0].nsmallest(1).index[0]
lowest2 = [lowest_below, lowest_above]
em2_y = np.interp(x=em2_c11, xp=catm.CFC11NH[lowest2], fp=catm.Year[lowest2])
mx_y = em1_r * em1_y + em2_r * em2_y
mx_c11 = em1_r * em1_c11 + em2_r * em2_c11

em1_aou = (sample_year - em1_y) * aour
em2_aou = (sample_year - em2_y) * aour
mx_aou_b = em1_r * em1_aou + em2_r * em2_aou

ax.plot([em1_y, em2_y], [em1_c11, em2_c11], 
        c='#666', 
        linewidth=1,
        zorder=1)
ax.plot([mx_y, mx_ya], [mx_c11, mx_c11],
        c='#aaa', 
        linewidth=.7,
        zorder=1)
ax.scatter([em1_y, em2_y], [em1_c11, em2_c11], 
            marker='o', facecolor='none', edgecolor='steelblue',
            s=30, linewidth=.7,
            zorder=2)
ax.scatter(mx_y, mx_c11, 
            marker='s', facecolor='none', edgecolor='goldenrod',
            s=20, linewidth=.7,
            zorder=2)
ax.scatter(mx_ya, mx_c11, 
            marker='d', facecolor='none', edgecolor='firebrick',
            s=20, linewidth=.7,
            zorder=2)

ax.text((em2_y + 1.5), (em2_c11 - 1), "EM$_2$ → " + str(round(em2_y, 1)), 
        fontsize=4, ha='left', va='center', color='#777')
ax.text((mx_y - .5), (mx_c11 + 11), "Real → " + str(round(mx_y, 1)), 
        fontsize=6.5, ha='right', va='center', color='goldenrod')


ax.text(1976, 80, "At aOUR=5umol/kg/y, AOU = " + str(round(mx_aou, 0)) + " vs " + str(round(mx_aou_b, 0)), 
        fontsize=6, ha='left', va='center', color='#222')

ax.set(ylim=[0, 300])
ax.tick_params(labelsize=8)
ax.set_ylabel("pCFC-11", fontsize=8)
ax.set_xlabel("Year", fontsize=8)

fpath = 'figures/test_2.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
