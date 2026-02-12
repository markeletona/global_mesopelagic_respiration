# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:23:46 2024

@author: Markel

Merge the Hansell DOM and Glodap datasets.


"""

#%% IMPORTS

import pandas as pd
import numpy as np
import os


#%% READ DATA

# Filtered Hansell dataset:
fpath = "deriveddata/dom_hansell/Hansell_2022_o2_doc_cfc11_cfc12_sf6.csv"
hns = pd.read_csv(fpath, sep=",", header=0, dtype={'EXPOCODE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})

# Filtered GLODAPv2 dataset:
fpath = 'deriveddata/glodap/GLODAPv2.2023_o2_cfc11_cfc12_sf6.csv'
gv2 = pd.read_csv(fpath, sep=',', header=0, dtype={'G2expocode': str,
                                                   'G2bottle': str})


# Note: Joining datasets is mostly useful from the standpoint of maximasing 
# DOC measurements, as otherwise most cruises are already present in glodap.
# But, if comparing only the shared cruises, the Hansell dataset has 
# significantly more DOC values, 38500 vs 44000 (see these values at the end
# of the DATA HANDLING cell)


#%% DATA HANDLING

# Replace missing/bad values (-9999) with nan:
gv2 = gv2.replace(-9999, np.nan)
hns = hns.replace(-9999, np.nan)

# Convert POC3D from mg/m^3 to umol/L
gv2['G2poc3d'] = gv2['G2poc3d'] / 1000 / 1000 / 12.011 * 10**6
hns['POC3D'] = hns['POC3D'] / 1000 / 1000 / 12.011 * 10**6



# In Hansell's DOC, replace all flag values NOT 2|6 to 9 (and their values to
# NaN). (i.e., to make it easier to NOT consider unacceptable flags like 
# 1, 3, 4, 5)
# hns.DOC_FLAG_W.value_counts() # see
idx = ~(hns.DOC_FLAG_W.isin([2,6]))
hns.loc[idx, 'DOC_FLAG_W'] = 9
hns.loc[idx, 'DOC'] = np.nan


# Cruises need to be matched by EXPOCODE. However, there are a few 
# inconsistencies between datasets, with certain cruises not sharing the exact
# same expocode. E.g., The A16 (2003) cruise has two expocodes in the Hansell
# dataset, while it only has one in GLODAP:
# 33RO200306_01, 33RO200306_02 | 33RO20030604
#
# EXPOCODES from Hansell not appearing in GLODAP are:
g2expocode = list(gv2.G2expocode.unique())
expocode = list(hns.EXPOCODE.unique())
[i for i in expocode if i not in g2expocode]

# From those, the ones idenfitied as absent despite being present in both 
# (i.e., those with expocode issues) are:
# 
# |============================================|
# | Hansell ---------------------> GLODAP      |
# |============================================|
# |33RO200306_01, 33RO200306_02 -> 33RO20030604|
# |31DSCG94_1 -------------------> 31DS19940126|
# |31DSCG96_1, 31DSCG96_2 -------> 31DS19960105|
# |3175MB95_07 ------------------> 33MW19950922|
# |33RR20050106 -----------------> 33RR20050109|
# |3250020210420 ----------------> 325020210420|
# |============================================|
# 
# ss1 = hns.loc[hns['EXPOCODE']=="31DSCG94_1"].copy()
# ss2 = gv2.loc[gv2['G2expocode']=="31DS19940126", :].copy()
# ss1 = hns.loc[hns['EXPOCODE']=="31DSCG96_1"].copy()
# ss2 = gv2.loc[gv2['G2expocode']=="31DS19960105", :].copy()
# ss1 = hns.loc[hns['EXPOCODE']=="33RR20050106"].copy()
# ss2 = gv2.loc[gv2['G2expocode']=="33RR20050109", :].copy()
# ss1 = hns.loc[hns['EXPOCODE']=="3250020210420"].copy()
# ss2 = gv2.loc[gv2['G2expocode']=="325020210420", :].copy()

# Replace those EXPOCODES in Hansell with the matching ones in GLODAP
hns['EXPOCODE'] = hns['EXPOCODE'].replace({"33RO200306_01": "33RO20030604",
                                           "33RO200306_02": "33RO20030604",
                                           "31DSCG94_1": "31DS19940126",
                                           "31DSCG96_1": "31DS19960105",
                                           "31DSCG96_2": "31DS19960105",
                                           "33RR20050106": "33RR20050109",
                                           "3175MB95_07": "33MW19950922",
                                           "3250020210420": "325020210420"})

# Now we need to create an unique identifier for each sample.
# In some cruises from Hansell, bottle numbers are faulty (see e.g., 
# 31DS19940126 cruise, or 320620110219). In Glodap there are also issue, see
# 58JH19961030_73_11 (says same bottle 11 but samples are from different 
# depths)
# So matching will need to be done with pressures...
def round_to_closest_2(x): return 2 * round(x / 2)
hns['ID1'] = (hns['EXPOCODE'] + 
              "_" +
              hns['STATION'].astype(str) +
              "_" +
              hns['BOTTLE'])
hns['ID2'] = (hns['EXPOCODE'] + 
              "_" +
              hns['STATION'].astype(str) +
              "_" +
              round_to_closest_2(hns['CTD_PRESSURE']).astype(str))
gv2['ID1'] = (gv2['G2expocode'] + 
              "_" +
              gv2['G2station'].astype(int).astype(str) +
              "_" +
              gv2['G2bottle'])
gv2['ID2'] = (gv2['G2expocode'] + 
              "_" +
              gv2['G2station'].astype(int).astype(str) +
              "_" +
              round_to_closest_2(gv2['G2pressure']).astype(str))



# Find EXPOCODE matches to check the samples from the cruises that are shared 
# between both datasets but that do not match neither by bottle nor pressure 
bottle_match = gv2.ID1.isin(hns.ID1) 
pressu_match = gv2.ID2.isin(hns.ID2)
expoco_match = gv2.G2expocode.isin(hns.EXPOCODE)
ss2 = gv2.loc[(~bottle_match) & (~pressu_match) & expoco_match,
              ['G2expocode', 'G2station', 'G2bottle', 'G2pressure']].copy()
ss1 = hns.loc[hns['EXPOCODE'].isin(ss2['G2expocode']),
              ['EXPOCODE', 'STATION', 'BOTTLE', 'CTD_PRESSURE']].copy()

# Those from shared cruises with 'normal' bottle numbers that do not match are 
# because their depths are not present in Hansell.
# ss1['BOTTLE'] = ss1['BOTTLE'].astype(float)
# ss1 = ss1.loc[ss1['BOTTLE']<50,:]
# ss2 = ss2.loc[ss2.G2expocode.isin(ss1.EXPOCODE)]


# Important:
# A small fraction of samples (<1%) were taken at the same station and depth.
# 
# Average those values.
# 
# BEFORE THAT:
# Rename column names in gv2 to match those in hns.
# 
# First create date column for glodap (yyyymmdd)
date_list = []
for i, r in gv2.iterrows():
    dstr = (str(r['G2year']).zfill(4) +
            str(r['G2month']).zfill(2) + 
            str(r['G2day']).zfill(2))
    date_list.append(int(dstr))
gv2['DATE'] = date_list

# Modify name patterns
nms = gv2.columns.to_series().replace({'G2': '',
                                       'f$': '_FLAG_W',
                                       'watermass': 'WATER_MASS',
                                       'chla': 'CHLOROPHYLL_A',
                                       'pressure': 'CTD_PRESSURE',
                                       'theta': 'PT',
                                       'salinity': 'CTD_SALINITY',
                                       'cfc11': 'CFC_11',
                                       'cfc12': 'CFC_12',
                                       'silicate': 'SILICIC_ACID'},
                                      regex=True)
# Make all names uppercase
nms = [n.upper() for n in nms]

# Assign the new names
gv2.columns = nms

# Rename CTD_OXYGEN in hns as the oxygen in GLODAP comes from the bottles and 
# they do not exactly match* (otherwise it migh be misleading suggesting all
# comes from the CTD)
# * see: scripts/glodap/compare_ctd_bottle_oxy_a16n_2003.py
hns.columns = hns.columns.to_series().replace("CTD_OXYGEN", "OXYGEN",
                                              regex=True)

# Check columns left not matching
# [n for n in nms if n not in hns.columns]
# [n for n in hns.columns if n not in nms]

## Set shared columns and place them in the desired order
# 
# variables with no flags
vrs1 = ['EXPOCODE', 'CRUISE', 'STATION', 'BOTTLE',
        'DATE', 'LATITUDE', 'LONGITUDE',
        'OCEAN', 'WATER_MASS', 'LP',
        'CTD_PRESSURE', 'PT', 'CTD_SALINITY', 'SIGMA0', 'SIGMA1']
# variables with flags
vrs2 = ['OXYGEN',
        'CFC_11', 'CFC_12', 'SF6',
        'DOC', 'TDN',
        'NITRATE', 'NITRITE', 'PHOSPHATE', 'SILICIC_ACID',
        'CHLOROPHYLL_A']
vrs2 = [val for pair in zip(vrs2, [v + "_FLAG_W" for v in vrs2]) for val in pair]
# Other derived variables with no flags
vrs3 = ['AOU', 'NPP_EPPL', 'NPP_CBPM', 'B', 'POC3D']

# Combine all variable names
vrs = vrs1 + vrs2 + vrs3

# Subset tables
hns = hns.loc[:, vrs + ['ID2']]
gv2 = gv2.loc[:, vrs + ['ID2']]


# Now yes, aggregate to average values at same station~depth.
# 
# NOTE:
# Bad/missing values are a set as NaN, but make sure we do not take a 9 flag
# as representative when averaging values (the bad value does not get into
# the average but if the first flag value in the group is 9 it will be 
# confusing).

def first(x): return(x.iloc[0])
def first_flag(x):
    if (len(x)==1) | (all(x==9)):
        v = x.iloc[0]
    else:
        v = x[~(x==9)].iloc[0]
    return v

agg_dict = {'EXPOCODE': lambda x: first(x),
            'CRUISE': lambda x: first(x),
            'STATION': lambda x: first(x),
            'BOTTLE': lambda x: first(x),
            'DATE': lambda x: first(x),
            'LATITUDE': np.nanmean,
            'LONGITUDE': np.nanmean,
            'OCEAN': lambda x: first(x),
            'WATER_MASS': lambda x: first(x),
            'LP': lambda x: first(x),
            'CTD_PRESSURE': np.nanmean,
            'PT': np.nanmean,
            'CTD_SALINITY': np.nanmean,
            'SIGMA0': np.nanmean,
            'SIGMA1': np.nanmean,
            'OXYGEN': np.nanmean,
            'OXYGEN_FLAG_W': lambda x: first_flag(x),
            'CFC_11': np.nanmean,
            'CFC_11_FLAG_W': lambda x: first_flag(x),
            'CFC_12': np.nanmean,
            'CFC_12_FLAG_W': lambda x: first_flag(x),
            'SF6': np.nanmean,
            'SF6_FLAG_W': lambda x: first_flag(x),
            'DOC': np.nanmean,
            'DOC_FLAG_W': lambda x: first_flag(x),
            'TDN': np.nanmean,
            'TDN_FLAG_W': lambda x: first_flag(x),
            'NITRATE': np.nanmean,
            'NITRATE_FLAG_W': lambda x: first_flag(x),
            'NITRITE': np.nanmean,
            'NITRITE_FLAG_W': lambda x: first_flag(x),
            'PHOSPHATE': np.nanmean,
            'PHOSPHATE_FLAG_W': lambda x: first_flag(x),
            'SILICIC_ACID': np.nanmean,
            'SILICIC_ACID_FLAG_W': lambda x: first_flag(x),
            'CHLOROPHYLL_A': np.nanmean,
            'CHLOROPHYLL_A_FLAG_W': lambda x: first_flag(x),
            'AOU': np.nanmean,
            'NPP_EPPL': np.nanmean,
            'NPP_CBPM': np.nanmean,
            'B': np.nanmean,
            'POC3D': np.nanmean,
            'ID2': lambda x: first(x)}

gv2_agg = gv2.groupby('ID2').agg(agg_dict).reset_index(drop=True)
hns_agg = hns.groupby('ID2').agg(agg_dict).reset_index(drop=True)

any(gv2_agg.ID2.value_counts() > 1) # should be False
any(hns_agg.ID2.value_counts() > 1) # should be False


#%% COMBINE TABLES

# When merging:
# - For samples present in both datasets, take the DOC values from Hansell
#   and put them into Glodap.
# - There are a small number of samples not present in Glodap that are in 
#   Hansell. Also include them (with all the variables)

# First, save a copy to make be able to know how many DOC samples there were
# in the original gv2_agg prior to adding the Hansell values
gv2_agg_copy = gv2_agg.copy()

# Get the ID2 of valid DOC measurements in hns_agg. This will be put in gv2_agg
hns_agg_id_valid_DOC = hns_agg.ID2.loc[~(hns_agg.DOC_FLAG_W==9)]

# Retain only those that are actually in gv2_agg (most of them; the rest will
# be added as new rows with all variables, as said above)
hns_agg_valid_DOC_id_in_gv2_agg = hns_agg_id_valid_DOC[hns_agg_id_valid_DOC.isin(gv2_agg.ID2)]
gv2_agg = gv2_agg.set_index('ID2')
hns_agg = hns_agg.set_index('ID2')
gv2_agg.loc[hns_agg_valid_DOC_id_in_gv2_agg, 'DOC'] = hns_agg.loc[hns_agg_valid_DOC_id_in_gv2_agg, 'DOC']


# Add samples from hns_agg that are not present in gv2_agg
idx = ~(hns_agg.EXPOCODE.isin(gv2_agg.EXPOCODE))
# froga = hns_agg.loc[idx,:]
merged_tbl = pd.concat([gv2_agg, 
                        hns_agg.loc[idx, vrs]])


# Reorder rows to gather samples by cruises
merged_tbl = merged_tbl.sort_values(by=['EXPOCODE', 'STATION', 'CTD_PRESSURE'],
                                    ascending=[True, True, True])


#%%% SUMMARY

def summary_values(x):
    
    d = {}
    d['noxy'] = sum(~(np.isnan(x.OXYGEN)))
    d['ncfc11'] = sum(~(np.isnan(x.CFC_11)))
    d['ncfc12'] = sum(~(np.isnan(x.CFC_12)))
    d['nsf6'] = sum(~(np.isnan(x.SF6)))
    d['ndoc'] = sum(~(np.isnan(x.DOC)))
    d['poxy'] = round(100 * d['noxy'] / x.shape[0], 1)
    d['pcfc11'] = round(100 * d['ncfc11'] / x.shape[0], 1)
    d['pcfc12'] = round(100 * d['ncfc12'] / x.shape[0], 1)
    d['psf6'] = round(100 * d['nsf6'] / x.shape[0], 1)
    d['pdoc'] = round(100 * d['ndoc'] / x.shape[0], 1)
    
    vrs = ['oxy', 'cfc11', 'cfc12', 'sf6', 'doc']
    summary_string = [(v + " -> n = " + str(d['n' + v]) + 
                       " (" + str(d['p' + v]) + " %)\n") for v in vrs]
    summary_string = "".join(summary_string)
    print(summary_string)
    
    return d

#### Variable availability

merged_tbl_summary = summary_values(merged_tbl)

#### Compare DOC availability before and after adding Hansell values
ndoc_gv2 = sum(~(np.isnan(gv2_agg_copy.DOC)))
ndoc_gv2_shared_cruise = sum(~(np.isnan(gv2_agg_copy.loc[gv2_agg_copy.EXPOCODE.isin(hns_agg.EXPOCODE.unique()), "DOC"])))
merged_tbl_summary['ndoc'] - ndoc_gv2



#%% FILTER

#%%% Filter to keep only target depth range

drange = [150, 1500]
idx = ((merged_tbl.CTD_PRESSURE >= drange[0]) &
       (merged_tbl.CTD_PRESSURE <= drange[1]))
merged_tbl_f = merged_tbl.loc[idx, :]

#### Variable availability
merged_tbl_f_summary = summary_values(merged_tbl_f)

#### Compare DOC availability 
ndoc_gv2_f = sum(~np.isnan(gv2_agg_copy.loc[(gv2_agg_copy.CTD_PRESSURE >= drange[0]) &
                                            (gv2_agg_copy.CTD_PRESSURE <= drange[1]), 'DOC']))
merged_tbl_f_summary['ndoc'] - ndoc_gv2_f



#%%% Filter to keep only samples within a water mass

idx = ~(merged_tbl_f.WATER_MASS=="NO_WATER_MASS")
merged_tbl_f_wm = merged_tbl_f.loc[idx, :].copy()

merged_tbl_f_wm_summary = summary_values(merged_tbl_f_wm)


#%% ADD UNCERTAINTY

# Add columns with uncertainty of measurements / estimates

# GLODAP paper reports uncertainty of 0.005 for salinity, and 1% for oxygen.
# They don't provide error for temperature, presumably because it is very low.
# Alvarez et al. 2014 set 0.04 for temperature.
u_S = .005 # SALINITY uncert. as value
u_T = .04  # TEMPERATURE uncert. as value
u_O = .01  # OXYGEN uncert. as percentage

# AOU is harder, because the saturation state of waters when sinking during
# formation varies, and is usually not in equilibrium with the atmosphere.
# However, there are few studies that directly study this.
# - Wolf et al. (2018) found undersaturation of -6--7% during the formation of 
# the LSW for water sinking below 800 m.
# - Ito et al. (2004) modelling results that show values close to 100 % 
# saturation for most the of surface waters of the global ocean. For instance, 
# Stanley et al. (2012) also cite saturations of ~99-101% at BATS (subtropical 
# gyre). Undersaturation is present in water formation areas of the North 
# Atlantic and, specially, Southern Ocean, where undersaturation of 50-70 uM 
# can be reached (80-85%), e.g. in the Wedell Sea.
# This means that intermediate and deep water will be more affected by this,
# whereas the uncertainty for central water masses will be small. We will 
# parameterise this based on temperature (ranges).
def u_AOU(t):
    # Based on temperature, as overall in deep water formation areas uncert.
    # is higher, but for central waters not
    if (t >= 9):
        return .01 # ± -> 2%
    elif (t >= 5) & (t < 9):
        return .025 # ± -> 5%
    else:
        return .05 # ± -> 10%


# Add uncertainty columns
merged_tbl_f_wm['PT_U'] = u_T
merged_tbl_f_wm['CTD_SALINITY_U'] = u_S
merged_tbl_f_wm['OXYGEN_U'] = merged_tbl_f_wm['OXYGEN'] * u_O 
merged_tbl_f_wm['AOU_U'] = abs(merged_tbl_f_wm['AOU']) * [u_AOU(t) for t in merged_tbl_f_wm['PT']]


# Reorder columns so that uncertainties are next to their variables
cols = ['EXPOCODE', 'CRUISE', 'STATION', 'BOTTLE', 'DATE', 'LATITUDE',
        'LONGITUDE', 'OCEAN', 'WATER_MASS', 'LP', 'CTD_PRESSURE', 'PT', 'PT_U',
        'CTD_SALINITY', 'CTD_SALINITY_U', 'SIGMA0', 'SIGMA1',
        'OXYGEN', 'OXYGEN_U', 'OXYGEN_FLAG_W', 
        'CFC_11', 'CFC_11_FLAG_W', 'CFC_12', 'CFC_12_FLAG_W', 'SF6', 'SF6_FLAG_W', 
        'DOC', 'DOC_FLAG_W', 'TDN', 'TDN_FLAG_W', 'NITRATE', 'NITRATE_FLAG_W',
        'NITRITE', 'NITRITE_FLAG_W', 'PHOSPHATE', 'PHOSPHATE_FLAG_W',
        'SILICIC_ACID', 'SILICIC_ACID_FLAG_W', 'CHLOROPHYLL_A',
        'CHLOROPHYLL_A_FLAG_W', 'AOU', 'AOU_U', 
        'NPP_EPPL', 'NPP_CBPM', 'B', 'POC3D']
merged_tbl_f_wm = merged_tbl_f_wm.loc[:, cols]


#%% EXPORT

dpath = 'deriveddata/hansell_glodap/'
if not os.path.exists(dpath):
    os.makedirs(dpath)

fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6.csv'
merged_tbl_f_wm.to_csv(fpath, sep=',', header=True, index=False, na_rep='-9999')

# Save a copy of the unfiltered version too
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_unfiltered.csv'
merged_tbl.to_csv(fpath, sep=',', header=True, index=False, na_rep='-9999')

