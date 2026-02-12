# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:24:19 2023

@author: Markel Gómez Letona

Estimate tracer ages of samples based on the measured CFC-11, CFC-12 and SF6 
concentrations.

Note that measured concentrations from the Hansell and GLODAP datasets are 
expressed in pmol/kg (CFC-11, CFC-12) and fmol/kg (SF6).

Fine (2011)* describes that to estimate trace ages, the tracer partial pressure 
(pCFC) needs to be calculated as follows:
    
    pCFC = C/F(T,S)
    
where C is the measured seawater concentration of the tracer, and F its 
solubility function (dependent on temperature and salinity).

The solubility function is expressed as follows:
    
    ln F = a1 + a2*(100/T) + a3*ln(T/100) + a4*(T/100)^2 +
           S*[b1 + b2*(T/100) + b3*(T/100)^2]

where T is absolute temperature and S salinity (in parts per thousand). Valid 
in moist air and a total pressure of 1 atm. a1, a2, a3, a4, b1, b2, and b3 are 
constants determined from the least squares fit of the solubility measurements. 
They can be estimated both in gravimetric (mol·kg-1·atm-1) and volumetric 
(mol·L-1·atm-1) termns (we will use gravimetric because our tracer 
concentrations are in pmol/kg | fmol/kg), and are different for each tracer:
    
    - From Warner & Weiss (1985)**:
        For CFC-11 (in mol·kg-1·atm-1):
            a1 = -232.0411
            a2 = 322.5546
            a3 = 120.4956
            a4 = -1.39165
            b1 = -0.146531
            b2 = 0.093621
            b3 = -0.0160693
        For CFC-12 (in mol·kg-1·atm-1):
            a1 = -220.2120
            a2 = 301.8695
            a3 = 114.8533
            a4 = -1.39165
            b1 = -0.147718
            b2 = 0.093175
            b3 = -0.0157340
    - From Bullister et al. (2002)***:
        For SF6 (mol·kg-1·atm-1):
            a1 = -82.1639
            a2 = 120.152
            a3 = 30.6372
            b1 = 0.0293201
            b2 = -0.0351974
            b3 = 0.00740056
        Note that for SF6 the a4 term of the equation is absent.


The computed partial pressure is then compared to the atmospheric history 
of the tracer to determine a corresponding date were the value is the same.
Note that following the ideal gas law, at 1 atm, partial pressure of a gas 'i' 
is equal to its mole fraction (x_i, aka mixing ratio), since:
    
    p_i/p_tot = (n_iRT/V)/(n_totRT/V) = n_i/n_tot = x_i
    x_i = n_i/n_tot = p_i/p_tot    -> (p are pressures, n are moles)
    if p_tot = 1 atm, then p_i = x_i
    
This is noteworthy if atmospheric values are given in terms of mixing ratios.
    
The obtained date is then subtracted from the date the seawater sample was 
collected, the diffence between dates gives the apparent average age for the 
water parcel.

Nonetheless, not all tracers are equally accurate to determine ages across all
time periods. As Tanhua et al. (2004)‡ show (see their Fig. 5):
    
    «the CFC-11 [and CFC-12] is not a useful transient tracer for water 
    equilibrated after 1990 due to ambiguity in the atmospheric source 
    function, but is useful for about 40 years before that. Sulphur 
    hexafluoride, on the other hand, is useful in recently ventilated waters, 
    but does not have a very long atmospheric history and the analytical 
    uncertainties become significant before 1980.»

These uncertainties will need to be considered when addressing the results,
discarding estimates based on specific tracers for periods when the uncertainty 
is too high.

*   Fine, R (2011). Observations of CFCs and SF6 as Ocean Tracers. 
    Annual Review of Marine Science, 3: 173-195. 
    doi: 10.1146/annurev.marine.010908.163933
**  Warner, MJ; Weiss, RF (1985). Solubilities of chlorofluorocarbons 11 and 12
    in water and seawater. Deep Sea Research Part A, 32 (12): 1485-1497.
    doi: 10.1016/0198-0149(85)90099-8
*** Bullister, JL; Wisegarver, DP; Menzia, FA (2002). The solubility of sulfur 
    hexafluoride in water and seawater. Deep-Sea Research I, 49: 175-187.
    doi: 10.1016/S0967-0637(01)00051-6
‡   Tanhua, T; Olsson, KA; Fogelqvist, E (2004). A first study of SF6 as a 
    transient tracer in the Southern Ocean. Deep-Sea Research II, 51: 
    2683–2699. doi: 10.1016/j.dsr2.2001.02.001

"""

#%% IMPORTS

import numpy as np
import pandas as pd
import datetime as dt
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import statsmodels.formula.api as smf


#%% LOAD DATA

# Filtered Glodap-Hansell dataset:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6.csv'
df = pd.read_csv(fpath, sep=',', header=0, dtype={'EXPOCODE': str,
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


#%% DATA HANDLING


# Replace missing/bad values (-9999) with nan:
df = df.replace(-9999, np.nan)

# Create a column where we convert the DATE column to decimal year:
df['YEAR_SAMPLE'] = np.nan
df['DATETIME'] = dt.datetime(year=1, month=1, day=1)
for i, r in df.iterrows():
    
    # Separate sample date string into year, month and day:
    s = str(r['DATE'])
    year = pd.to_numeric(s[0:4])
    month = pd.to_numeric(s[4:6])
    day = pd.to_numeric(s[6:8])
    sdate = dt.datetime(year=year, month=month, day=day)
    
    # Get values for start and end of year (to account for leap year, do not 
    # assume a year has 365 days)
    start_of_this_year = dt.datetime(year=year, month=1, day=1)
    start_of_next_year = dt.datetime(year=year+1, month=1, day=1)
    
    # Estimate the fraction of the year that corresponds to the month, day:
    days_elapsed = sdate - start_of_this_year
    year_duration = start_of_next_year - start_of_this_year
    elapsed_fraction = days_elapsed/year_duration
    
    # Add fraction to year value and store it:
    df.at[i, 'YEAR_SAMPLE'] = round(year + elapsed_fraction, 3) # round to 3 digits because my resolution goes down only down to day, so more digits may mislead the precesion
    df.at[i, 'DATETIME'] = sdate # also store the date in datetime format


## Estimate atmospheric tracer ratios:
atm['CFC11CFC12NH'] = round(atm['CFC11NH']/atm['CFC12NH'], 3)
atm['SF6CFC11NH'] = round(atm['SF6NH']/atm['CFC11NH'], 4)
atm['SF6CFC12NH'] = round(atm['SF6NH']/atm['CFC12NH'], 4)
atm['CFC11CFC12SH'] = round(atm['CFC11SH']/atm['CFC12SH'], 3)
atm['SF6CFC11SH'] = round(atm['SF6SH']/atm['CFC11SH'], 4)
atm['SF6CFC12SH'] = round(atm['SF6SH']/atm['CFC12SH'], 4)

# Estimate global average values:
atm['CFC11GL'] = np.round(np.nanmean(atm.loc[:,['CFC11NH', 'CFC11SH']], axis=1), 2)
atm['CFC12GL'] = np.round(np.nanmean(atm.loc[:,['CFC12NH', 'CFC12SH']], axis=1), 2)
atm['SF6GL'] = np.round(np.nanmean(atm.loc[:,['SF6NH', 'SF6SH']], axis=1), 2)

# Estimate global ratios:
atm['CFC11CFC12GL'] = round(atm['CFC11GL']/atm['CFC12GL'], 3)
atm['SF6CFC11GL'] = round(atm['SF6GL']/atm['CFC11GL'], 4)
atm['SF6CFC12GL'] = round(atm['SF6GL']/atm['CFC12GL'], 4)


#%% PLOT ATMOSPHERIC HISTORIES

# Subset desired year range and create datetime column:
catm = atm.loc[atm['Year']>1940,:].copy()
catm['DATE'] = [dt.datetime(year=int(y), month=7, day=1) for y in np.floor(catm['Year'])] 

# Transform into long format:
catm2 = pd.melt(catm, id_vars = ['Year', 'DATE'], var_name='TRACER', value_name='MX_RATIO')


#%%% ABSOLUTE VALUES

# Create palette for tracers:
pal_tracers = {'CFC11': '#DDAA33',
               'CFC12': '#BB5566',
               'SF6': '#004488'}

cm = 1/2.54
fig_atm, ax_atm = plt.subplots(figsize=(15*cm, 10*cm))
ax_atm2 = ax_atm.twinx()
l_cfc11 = ax_atm.plot('DATE',
                      'MX_RATIO',
                      color=pal_tracers['CFC11'],
                      data=catm2.loc[catm2['TRACER']=='CFC11NH'])
l_cfc12 = ax_atm.plot('DATE',
                      'MX_RATIO',
                      color=pal_tracers['CFC12'],
                      data=catm2.loc[catm2['TRACER']=='CFC12NH'])
l_sf6 = ax_atm2.plot('DATE',
                     'MX_RATIO',
                     color=pal_tracers['SF6'],
                     data=catm2.loc[catm2['TRACER']=='SF6NH'])
ax_atm.set(ylabel = "CFC-11, CFC-12 [ppt]", ylim=[0, 600], 
           xlabel = "Year", xlim=[dt.datetime(1940,1,1), dt.datetime(2025,1,1)])
ax_atm.xaxis.set_minor_locator(mdates.YearLocator())
ax_atm2.set(ylabel = "SF$_6$ [ppt]", ylim=[0, 12])
ax_atm2.yaxis.label.set_color(pal_tracers['SF6'])
ax_atm2.spines['right'].set_color(pal_tracers['SF6'])
ax_atm2.tick_params(axis='y', colors=pal_tracers['SF6'])
fig_atm.legend(loc='upper left', bbox_to_anchor = [.13, .875],
               labels = ["CFC-11", "CFC-12", "SF$_6$"])


dpath = 'figures/tracers_atmosphere/'
if not os.path.exists(dpath):
    os.makedirs(dpath)

fpath = 'figures/tracers_atmosphere/atmospheric_history_nh_cfc11_cfc12_sf6.svg'
fig_atm.savefig(fpath, format='svg', bbox_inches='tight', facecolor=None)



#%%% RATIOS

fig_atmr, ax_atmr = plt.subplots(figsize=(15*cm, 10*cm))
ax_atmr2 = ax_atmr.twinx()
l_cfc11cfc12 = ax_atmr.plot('DATE',
                            'MX_RATIO',
                            color=pal_tracers['CFC11'],
                            data=catm2.loc[(catm2['TRACER']=='CFC11CFC12NH') & (catm2['Year']>1948),])
l_sf6cfc11 = ax_atmr2.plot('DATE',
                           'MX_RATIO',
                           color=pal_tracers['SF6'],
                           data=catm2.loc[(catm2['TRACER']=='SF6CFC11NH') & (catm2['Year']>1975),])
l_sf6cfc12 = ax_atmr2.plot('DATE',
                           'MX_RATIO',
                           color=pal_tracers['CFC12'],
                           data=catm2.loc[(catm2['TRACER']=='SF6CFC12NH') & (catm2['Year']>1975),])
ax_atmr.set(ylabel = "CFC-11:CFC-12", ylim=[0, .6], 
            xlabel = "Year", xlim=[dt.datetime(1940,1,1), dt.datetime(2025,1,1)])
ax_atmr.xaxis.set_minor_locator(mdates.YearLocator())
ax_atmr2.set(ylabel = "SF$_6$:CFC-11, SF$_6$:CFC-12", ylim=[0, .06])
ax_atmr.yaxis.label.set_color(pal_tracers['CFC11'])
ax_atmr2.spines['left'].set_color(pal_tracers['CFC11'])
ax_atmr.tick_params(axis='y', colors=pal_tracers['CFC11'])
fig_atmr.legend(loc='upper left', bbox_to_anchor = [.13, .875],
                labels = ["CFC-11:CFC-12", "SF$_6$:CFC-11", "SF$_6$:CFC-12"])    

fpath = 'figures/tracers_atmosphere/atmospheric_history_nh_cfc11_cfc12_sf6_ratios.svg'
fig_atmr.savefig(fpath, format='svg', bbox_inches='tight', facecolor=None)


#%% COMPUTE PARTIAL PRESSURES

# [takes ~ 1 min]

#### Before starting, we can check that we apply the solubility functions 
# correctly by doing the example provided by Bullister et al. (2002) for SF6. 
# For T=277.15ºK and S=34, F_gravimetric = 0.34670·10^-3 mol kg-1 atm-1. 
# numpy.log() is natural logarithm (ln) 
# Check:
a1 = -82.1639
a2 = 120.152
a3 = 30.6372
b1 = 0.0293201
b2 = -0.0351974
b3 = 0.00740056
T = 277.15
S = 34
F_test = np.exp(a1 + a2*(100/T) + a3*np.log(T/100) + S*(b1 + b2*(T/100) + b3*(T/100)**2))
print(F_test) # 0.00034669980002764933 ~= 0.34670·10^-3, great


# Now compute the partial pressures:

## CFC-11:

a1 = -232.0411
a2 = 322.5546
a3 = 120.4956
a4 = -1.39165
b1 = -0.146531
b2 = 0.093621
b3 = -0.0160693

df['pCFC11'] = np.nan # pre-allocate space
for index, row in df.iterrows():
    # Get potential temperature and salinity:
    T = row['PT'] + 273.15
    S = row['CTD_SALINITY']
    if S == -999: S = np.nan # few instances have no salinity
    # Estimate solubility:
    F = np.exp(a1 + a2*(100/T) + a3*np.log(T/100) + a4*(T/100)**2 + S*(b1 + b2*(T/100) + b3*(T/100)**2))
    # Estimate and store partial pressure:
    df.at[index, 'pCFC11'] = round(((row['CFC_11']/10**12)/F)*10**12, 2) # /10**12: to convert pmol/kg -> mol/kg | *10**12: to convert result to parts per trillion, the usual units
    
# Preserve values only if the measurement had a valid flag (set NaN otherwise)
# (the dataset was filtered to keep samples where either one of the three tracer
#  gases had a valid value (flags 2, 6), so specific samples might still carry
#  dubious or erroneous measurements (3, 4) that are not necessarily NaN, 
#  because one of the other gases in the sample had a valid measurement)
#  (0 is included for Glodap flags -> interpolated or calculated; although
#   apparently there are none for tracer gases)
idx = ~df['CFC_11_FLAG_W'].isin([0, 2, 6]) # samples with NO acceptable flag
df.loc[idx, 'pCFC11'] = np.nan # revert to NaN any value that has not valid flag

    
## CFC-12:
    
a1 = -220.2120
a2 = 301.8695
a3 = 114.8533
a4 = -1.39165
b1 = -0.147718
b2 = 0.093175
b3 = -0.0157340

df['pCFC12'] = np.nan
for index, row in df.iterrows():
    T = row['PT'] + 273.15
    S = row['CTD_SALINITY']
    if S == -999: S = np.nan
    F = np.exp(a1 + a2*(100/T) + a3*np.log(T/100) + a4*(T/100)**2 + S*(b1 + b2*(T/100) + b3*(T/100)**2))
    df.at[index, 'pCFC12'] = round(((row['CFC_12']/10**12)/F)*10**12, 2)

idx = ~df['CFC_12_FLAG_W'].isin([0, 2, 6])
df.loc[idx, 'pCFC12'] = np.nan


## SF6:

a1 = -82.1639
a2 = 120.152
a3 = 30.6372
b1 = 0.0293201
b2 = -0.0351974
b3 = 0.00740056

df['pSF6'] = np.nan
for index, row in df.iterrows():
    T = row['PT'] + 273.15
    S = row['CTD_SALINITY']
    if S == -999: S = np.nan
    F = np.exp(a1 + a2*(100/T) + a3*np.log(T/100) + S*(b1 + b2*(T/100) + b3*(T/100)**2)) # note that a4 term is absent
    df.at[index, 'pSF6'] = round(((row['SF6']/10**15)/F)*10**12, 2) # /10**15: to convert fmol/kg -> mol/kg

idx = ~df['SF6_FLAG_W'].isin([0, 2, 6]) 
df.loc[idx, 'pSF6'] = np.nan


# Numbers make sense, they are in the range of what is observed (see e.g. Fine 
# 2011, figure 1).



#%% ESTIMATE APPARENT AGES

start_time = dt.datetime.now()


# [takes ~ 10 min]

#### Atmospheric values are given yearly. We will need to interpolate between 
# the two closest values.

# For CFC-11 and CFC-12, decreases started following the Montreal Protocol,
# so for some atmospheric levels there are two dates. This means certain
# values cannot be used to compute ages, at least if the sample was collected 
# after that specific value was reached twice in the atmosphere:
#                 _______
#                /   |   \___
# --------------/------------\___--> (e.g. CFC-11 = 240 ppt in 1987 & 2010)
#              /|    |        |
#             / |    |        |
#            /  |    |        |
#           /   |    |        |
#        __/    |    |        |
# ______/       |    |        |
#               1987 1995     2012
# 
# For instance, if we were to have a sample with CFC-11 = 240 ppt collected in
# 1995 we could use the value to estimate the age, because until that point 240
# ppt was reached only once (~ 1987). However, if the sample with CFC-11 = 240 
# ppt was collect in 2012 we could not estimate the age, because we already 
# have 240 ppt at two points in the atmospheric history (~ 1987 & 2010): we 
# could not know for sure if that value would correspond to 1987 or 2010.

# Use Northern or Southern hemisphere values? Average? 
#   - For cruises in the NH -> NH.
#   - For cruises in the SH -> SH. 
#   - If a cruise spans across both hemispheres -> global average (to avoid 
#     "abrupt" change when crossing the equator). (if a cruise is mostly in one
#     hemisphere but has stations just a few degrees across the other (say,
#     5-6 deg), take the dominant hemisphere for all)
#
# Find out the geographical extent of each cruise:
uCRUISE = set(df['CRUISE'])
df['CRUISE_HEMISPHERE'] = ''
for cr in uCRUISE:
    lats = df.loc[df['CRUISE']==cr,'LATITUDE']
    if ((sum(lats > 0) / len(lats)) > (3/4)): # if majority of stations in NH
        df.loc[df['CRUISE']==cr,'CRUISE_HEMISPHERE'] = 'NH'
    elif ((sum(lats < 0) / len(lats)) > (3/4)):
        df.loc[df['CRUISE']==cr,'CRUISE_HEMISPHERE'] = 'SH'
    else:
        df.loc[df['CRUISE']==cr,'CRUISE_HEMISPHERE'] = 'GL'


# Check results by mapping
st = df[~df.duplicated(subset=['CRUISE', 'STATION'], keep='first')].copy()
ch_pal = {'NH': 'darkseagreen', 'SH': 'seagreen', 'GL': 'mediumorchid'}
# st['ch_color'] = 'k'
# for ir, r in st.iterrows():
#     st.loc[ir, 'ch_color'] = ch_pal[r['CRUISE_HEMISPHERE']]

# Map:
cm = 1/2.54
fig_n, ax_n = plt.subplots(figsize=(15*cm, 8*cm), 
                           subplot_kw={'projection': ccrs.Mollweide(
                               central_longitude=-160)})
ax_n.add_feature(cfeature.LAND, facecolor='#444444')
ax_n.add_feature(cfeature.OCEAN, facecolor='w')
for h in ch_pal:
    ss = st.loc[st.CRUISE_HEMISPHERE==h, :]
    s = ax_n.scatter(x=ss['LONGITUDE'],
                     y=ss['LATITUDE'],
                     facecolor=ch_pal[h],
                     label=h,
                     s=4,
                     transform=ccrs.PlateCarree())
ax_n.set_global()
leg = fig_n.legend(loc='upper center', bbox_to_anchor=[.5, .13],
                   labels=["Northern hemisphere", "Southern hemisphere",
                           "Global average"],
                   ncol=3,
                   frameon=False,
                   handletextpad=0)
for hndl in leg.legend_handles:
    hndl.set_sizes([30])


dpath = 'figures/hansell_glodap/tracer_ages/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    
fpath = 'figures/hansell_glodap/tracer_ages/map_filter_o2_cfc11_cfc12_sf6_hemisphere.svg'
fig_n.savefig(fpath, format='svg', bbox_inches='tight', transparent=True)



## Set tracer names to use in loop:
ptracers = ['pCFC11', 'pCFC12', 'pSF6']
ptracers_atm = ['CFC11', 'CFC12', 'SF6'] # NH/SH/GL suffix to be added within loop
tr_year_names = ['YEAR_CFC11', 'YEAR_CFC12', 'YEAR_SF6']

# Preallocate space:
for i in tr_year_names:
    df[i] = np.nan

# Estimate the ventilation year of the water samples (last time in contact with
# the atmosphere).
# We assume that the sample was in equilibrium with the atmosphere, i.e. 100% 
# saturation, but this is not always exactly true. Consider this uncertainty 
# when interpreting results.
cntr = 0
for idx, ptr in enumerate(ptracers):
    for idx_df, row in df.iterrows():
        
        # Subset the atmospheric history up to the year the sample was collected:
        YS = row['YEAR_SAMPLE']
        atm_ss = atm[(atm['Year']<=(YS + .5))] # add 0.5 because the atm tracers are reported at mid-year
        
        # Also, subset the atmospheric history to include the last 0 value 
        # prior to emissions starting. Otherwise, if the lower bound to find 
        # our year is 0, it could take any 0 from the start until that point.
        # For ratios, preemptively crop to discard early years of high 
        # uncertainty.
        # Consider the proper hemisphere as determined above for each cruise:
        ptracers_atm_h = ptracers_atm[idx] + row['CRUISE_HEMISPHERE']
        over0 = np.where(atm_ss[ptracers_atm_h]>0)[0] # where returns a tupple, [0] converts it to ndarray
        atm_ss = atm_ss.iloc[np.concatenate(((over0[0]-1), over0), axis=None),:]
        
        # Get the first and last value in that time range:
        atm_first = atm_ss[ptracers_atm_h].iat[0] # first is needed because we crop the ratio histories, so our sample value might be below the first, and thus we have to set that case as nan
        atm_last = atm_ss[ptracers_atm_h].iat[-1]
        
        if (np.isnan(row[ptr])) | (row[ptr]>=atm_last) | (row[ptr]<=atm_first) | (row[ptr]<=0):
            
            # There are a few instances in which ages cannot be computed:
            #
            # For CFC-11|CFC-12 after reaching the atm peak, if the sample 
            # value is equal/higher than the last value, it means that we have 
            # reached that value twice and thus we cannot estimate the age. 
            # For CFC-11|CFC-12 before reaching the atm peak, or for SF6,
            # if this happens it means that either there was an error due to it
            # being too close to the most recent atm value and the measured 
            # value resulted just above, or for SF6 the sample is too recent to
            # be compared to our current atmospheric dataset.
            #
            # In any case, both instances should be assigned as nan.
            # 
            # Also, if the sample value is zero (0), we cannot estimate the 
            # age (it could be anything prior to initiation of atmospheric
            # emissions).
            # 
            # Also, if the partial pressure is nan (because salinity was nan),
            # we cannot estimate of course.
    
            tracer_year = np.nan
            
        else:
            
            # Otherwise, estimate age by interpolation.
            # Subtract our measurment to atmospheric values to find the closest
            # two atmospheric values (above and below).
            # First:
            # Subset atm values again until max value of the tracer:
            # (as we have previously checked that it does not exceed/equal the last atm value before sampling, now this is safe)
            midx = atm_ss[ptracers_atm_h].idxmax()
            atm_ss = atm_ss.loc[:midx,:]
            
            # Get the atm values above and below our sample partial pressure:
            diff = row[ptr] - atm_ss[ptracers_atm_h]
            lowest_above = diff[diff<0].nlargest(1).index[0] # i was having issues when subsetting with lowest2 within the interp call (it could not find the indices??) so I forced the indices to be plain int by adding the [0] at the end
            lowest_below = diff[diff>=0].nsmallest(1).index[0]
            lowest2 = [lowest_below, lowest_above]
            

            # Interpolate to get the date of our exact tracer value:
            tracer_year = np.interp(x=row[ptr],
                                    xp=atm_ss.loc[lowest2, ptracers_atm_h], 
                                    fp=atm_ss.loc[lowest2, 'Year'],
                                    left=-777, # -999,
                                    right=-999)
            # the left|right out of bound cases should be dealt with in the 
            # previous conditions, but keep track just in case.
            
            #------------------------------------------------------------------
            # For CFCs measured after peaking, check if tracer partial pressure 
            # was not within the GLODAP uncertainty (5%) relative to the 
            # atmospheric values when sampling (that is, that the value in the 
            # sample is not within uncertainty of being repeated twice in the 
            # atmospheric history, and thus unable to be uniquely assigned)
            # 
            # Get peak year of tracer
            ptracers_atm_h_py = atm['Year'].iat[atm[ptracers_atm_h].idxmax()]
            # If CFC and sample year after peak, check value
            if (ptr in ['pCFC11', 'pCFC12']) & (YS > ptracers_atm_h_py):
                # get upper bound of uncertainty for sample, and compare it
                # to the atmospheric value of that year
                u_max = row[ptr] + row[ptr] * .05 
                if u_max >= atm_last:
                    tracer_year = np.nan
                    cntr += 1
            #------------------------------------------------------------------
                
        ## Store results:
        df.at[idx_df, tr_year_names[idx]] = round(tracer_year, 1)

end_time = dt.datetime.now()
print('Duration: {}'.format(end_time - start_time))
   
# Year estimates seem to differ considerably specially between CFC-11/-12 vs 
# SF6... Check results by plotting an example over the atmospheric history.
# Note that this actually agrees with the report of Tanhua et al. (2004): SF6
# tends to give younger ages.


# Based on Tanhua et al. (2004) set the time periods for which the ventilation
# year estimates based on each tracer a considered valid (i.e., reasonably low
# uncertainty):
# 
# CFC-11 --------> 1954 - 1990
# CFC-12 --------> 1948 - 1990
# CFC-11/CFC-12 -> 1954 - 1975
# SF6 -----------> 1980 - PRESENT
# 
df.loc[(df['YEAR_CFC11']<1954) | (df['YEAR_CFC11']>1990), 'YEAR_CFC11'] = np.nan
df.loc[(df['YEAR_CFC12']<1948) | (df['YEAR_CFC12']>1990), 'YEAR_CFC12'] = np.nan
df.loc[(df['YEAR_SF6']<1980), 'YEAR_SF6'] = np.nan

## Estimate ages by subtracting the sampling date to the tracer estimate:
df['AGE_CFC11'] = round(df['YEAR_SAMPLE'], 1) - df['YEAR_CFC11']
df['AGE_CFC12'] = round(df['YEAR_SAMPLE'], 1) - df['YEAR_CFC12']
df['AGE_SF6'] = round(df['YEAR_SAMPLE'], 1) - df['YEAR_SF6']

# Add age uncertainties based on Tanhua et al. (2004), Fig. 5.
def u_cfc11_age(y):
    if (y >= 1954) & (y <= 1961.5):
        return 1.5
    elif (y > 1961.5) & (y <= 1977.5):
        return 0.9
    elif (y > 1977.5) & (y <= 1990):
        return 2.3
    else:
        return np.nan

def u_cfc12_age(y):
    if (y >= 1948) & (y <= 1961.5):
        return 1.8
    elif (y > 1961.5) & (y <= 1977.5):
        return 1.1
    elif (y > 1977.5) & (y <= 1990):
        return 2.3
    else:
        return np.nan
    
def u_sf6_age(y):
    if (y >= 1980):
        return 1.5
    else:
        return np.nan

df['AGE_CFC11_U'] = [u_cfc11_age(y) for y in df['YEAR_CFC11']]
df['AGE_CFC12_U'] = [u_cfc12_age(y) for y in df['YEAR_CFC12']]
df['AGE_SF6_U'] = [u_sf6_age(y) for y in df['YEAR_SF6']]


# Make sure that all valid ages have a valid uncertainty, or that if they are nan
# the uncertainty is also nan
all((np.isnan(df['AGE_CFC11']) & np.isnan(df['AGE_CFC11_U'])) | 
    (~np.isnan(df['AGE_CFC11']) & ~np.isnan(df['AGE_CFC11_U'])))
all((np.isnan(df['AGE_CFC12']) & np.isnan(df['AGE_CFC12_U'])) | 
    (~np.isnan(df['AGE_CFC12']) & ~np.isnan(df['AGE_CFC12_U'])))
all((np.isnan(df['AGE_SF6']) & np.isnan(df['AGE_SF6_U'])) | 
    (~np.isnan(df['AGE_SF6']) & ~np.isnan(df['AGE_SF6_U'])))

# Set to NaN ages smaller than their uncertainty
df.loc[df['AGE_CFC11'] < df['AGE_CFC11_U'], 'AGE_CFC11'] = np.nan
df.loc[df['AGE_CFC12'] < df['AGE_CFC12_U'], 'AGE_CFC12'] = np.nan
df.loc[df['AGE_SF6'] < df['AGE_SF6_U'], 'AGE_SF6'] = np.nan



#%% EXAMPLE ESTIMATE PLOT

# Plot example to check that I have correctly selected the year correspoding to
# the atmospheric tracer concentration that matches the tracer concentration
# in our samples...

idx = (df['EXPOCODE']=='33RO20110926') & (df['STATION']==5) & (df['CTD_PRESSURE']==499.8)
exdata = df.loc[idx,:].copy().squeeze()
# idx = (df['CRUISE']=='A16N (2003)') & (df['STATION']==39) & (df['CTD_PRESSURE']==126.2)
# exdata = df.loc[idx,:].copy().squeeze()

# this does not consider leap years when getting the day but should be ok for the scale of this plot:
y = exdata['YEAR_CFC11']
exdata['DATE_CFC11'] = dt.datetime(year=int(round(y,0)), month=1, day=1) + dt.timedelta((y-round(y,0))*365-1)
y = exdata['YEAR_CFC12']
exdata['DATE_CFC12'] = dt.datetime(year=int(round(y,0)), month=1, day=1) + dt.timedelta((y-round(y,0))*365-1)
y = exdata['YEAR_SF6']
exdata['DATE_SF6'] = dt.datetime(year=int(round(y,0)), month=1, day=1) + dt.timedelta((y-round(y,0))*365-1)


ch = exdata['CRUISE_HEMISPHERE']
fig_exage, ax_exage = plt.subplots(figsize=(15*cm, 10*cm))
ax_exage2 = ax_exage.twinx()
for t in pal_tracers:
    
    ax = ax_exage2 if t=='SF6' else ax_exage
    p_t = ax.plot('DATE', 'MX_RATIO',
                  color=pal_tracers[t],
                  data=catm2.loc[catm2['TRACER']==(t + ch)])
    
# Customise axes
ax_exage.set(ylabel="CFC-11, CFC-12 [ppt]",
             ylim=[0, 600], 
             xlabel="Year", 
             xlim=[dt.datetime(1940,1,1), dt.datetime(2025,1,1)])

ax_exage2.set(ylabel="SF$_6$ [ppt]", ylim=[0, 12])
ax_exage.xaxis.set_minor_locator(mdates.YearLocator())
ax_exage2.yaxis.label.set_color(pal_tracers['SF6'])
ax_exage2.spines['right'].set_color(pal_tracers['SF6'])
ax_exage2.tick_params(axis='y', colors=pal_tracers['SF6'])
fig_exage.legend(loc='upper left', bbox_to_anchor=[.13, .875],
                 labels=["CFC-11", "CFC-12", "SF$_6$"])

# Add reference lines for the estimated values (sample value and corresponding 
# estimated year)
for t in pal_tracers:
    
    ax = ax_exage2 if t=='SF6' else ax_exage
    ax.axhline(y=exdata['p' + t],
               color=pal_tracers[t], alpha=.5, ls=':')
    ax.axvline(x=exdata['DATE_' + t], color=pal_tracers[t], alpha=.5, ls=':')

y = exdata['YEAR_SAMPLE']
exdata['DATE_SAMPLE'] = dt.datetime(year=int(round(y,0)), month=1, day=1) + dt.timedelta((y-round(y,0))*365-1)
ax_exage.axvline(x=exdata['DATE_SAMPLE'], color='grey', alpha=.5, linestyle='--') # sampling date
ax_exage.text(x=exdata['DATE_SAMPLE'],
              y=1.05,
              s=round(exdata['YEAR_SAMPLE'], 1),
              color='grey',
              ha='center', va='top',
              transform=ax_exage.get_xaxis_transform())

# Add labels:
for it, t in enumerate(pal_tracers):

    ax = ax_exage2 if t=='SF6' else ax_exage
    transx = ax.get_xaxis_transform() # to transforme between data and axis (0-1) coordinates
    
    deltay = 0 if t=='SF6' else .05
    ax.text(x=exdata['DATE_' + t],
            y=1.05 + it * deltay,
            s=round(exdata['YEAR_' + t], 1),
            color=pal_tracers[t],
            ha='center', va='top',
            transform=transx)

    transy = ax_exage2.get_yaxis_transform() if t=='SF6' else ax_exage.get_yaxis_transform()
    deltay = -.6 if t=='SF6' else 10
    ax_exage.text(x=.2, 
                  y=exdata['p' + t] + deltay,
                  s=str(round(exdata['p' + t], 1)) + " ppt",
                  color=pal_tracers[t],
                  ha='center', va='baseline',
                  transform=transy)

# Add title
ptitle = (exdata['EXPOCODE'] + "\n" + 
          str(exdata['CTD_PRESSURE']) + " dbar, " + 
          str(round(exdata['LATITUDE'],2)) + "$\degree$N, " + 
          str(round(exdata['LONGITUDE'],2)) + "$\degree$E")
fig_exage.suptitle(ptitle, x=.5, y = 1.07, weight='bold', color='#222')


# Well, my computations seem correct, the estimates align well in the plot with
# their respective atmospheric histories, and yeah, estimates differ depending
# on the tracer...

fpath = 'figures/hansell_glodap/tracer_ages/tracer_year_estimate_example.svg'
fig_exage.savefig(fpath, format='svg', bbox_inches='tight', facecolor=None)



#%% PLOT VENTILATION YEAR ESTIMATES

# Plot estimates for the northen hemisphere

nr = len(pal_tracers)
nc = 1

ylims = {k:v for k, v in zip(pal_tracers.keys(), [[0, 300],
                                                  [0, 600],
                                                  [0, 12]])}

fig_vy, ax_vy = plt.subplots(nrows=nr, ncols=nc,
                             figsize=(12*cm * nc, 8*cm * nr))
for it, t in enumerate(pal_tracers):

    idx = (df.CRUISE_HEMISPHERE=='NH') & ~np.isnan(df['YEAR_' + t])
    x = df.loc[idx, 'YEAR_' + t]
    x2 = df.loc[idx, 'YEAR_SAMPLE']
    y = df.loc[idx, 'p' + t]
    
    ax_vy[it].plot(atm['Year'], atm[t + 'NH'], 
                   linewidth=1, color='k',
                   zorder=2)
    ax_vy[it].scatter(x, y, 
                      marker='o', 
                      facecolor='none', edgecolor=pal_tracers[t],
                      s=15, linewidth=.3, 
                      zorder=1)
    ax_vy[it].scatter(x2, y, 
                      marker='|', 
                      facecolor='#777', linewidth=.5, s=3,
                      zorder=1)
    ax_vy[it].hlines(y, x, x2, color='#ddd', linewidth=.3,
                     zorder=0)
    ax_vy[it].set_ylabel("p" + t.replace("CFC", "CFC-"))
    ax_vy[it].set_xlabel("YEAR")
    ax_vy[it].set(ylim=ylims[t],
                  xlim=[1940, 2025])

fpath = 'figures/hansell_glodap/tracer_ages/tracer_year_estimates_NH.png'
fig_vy.savefig(fpath, format='png', bbox_inches='tight', transparent=True, 
               dpi=300)


#%% COMPARE ESTIMATES BY DIFFERENT TRACERS

## Compare with Tanhua et al. (2004), similar results!

# CFC-11 vs CFC-12:
fig_sca, ax_sca = plt.subplots(figsize=(10*cm, 10*cm))
ax_sca.scatter(x=df['AGE_CFC11'],
               y=df['AGE_CFC12'],
               facecolors='none', edgecolors='k',
               s=10, linewidth=.5)
ax_sca.axline((0, 0), slope=1, color='#aaa', linestyle='--', linewidth=.8)
ax_sca.set(xlim=[0, 80], ylim=[0, 80],
           xlabel="Age (CFC-11, years)",
           ylabel="Age (CFC-12, years)")
ax_sca.set_aspect('equal')
ax_sca.set_xticks(range(0,90,10))
ax_sca.set_yticks(range(0,90,10))
ax_sca.tick_params(direction='in', top=True, right=True, length=3)
# Add regression line to plot (only in our data range):
md = smf.ols('AGE_CFC12 ~ AGE_CFC11', data=df, missing='drop').fit()
x0 = np.nanmin(df['AGE_CFC11'])
y0 = md.params['Intercept'] + md.params['AGE_CFC11']*x0
x1 = np.nanmax(df['AGE_CFC11'])
y1 = md.params['Intercept'] + md.params['AGE_CFC11']*x1
ax_sca.plot([x0, x1], [y0, y1], c='#79c', linewidth=2, alpha=.9)
slp = str(round(md.params['AGE_CFC11'], 3))
if md.pvalues['AGE_CFC11']<.001:
    pval = '$\it{p}$ < 0.001'
elif md.pvalues['AGE_CFC11']<.05:
    pval = '$\it{p}$ = ' + str(round(md.pvalues['AGE_CFC11'], 3))
else:
    pval = '$\it{ns}$'
r2 = str(round(md.rsquared_adj, 2))
units = 'year·year$^{-1}$'
lb1 = "Slope = " + slp + ' ' + units + '\n' + pval + '; R$^2$ = ' + r2
ax_sca.text(.05, .82, lb1, size=8, transform=ax_sca.transAxes)     

fpath = 'figures/hansell_glodap/tracer_ages/tracer_age_comparison_cfc11_cfc12.png'
fig_sca.savefig(fpath, format='png', bbox_inches='tight', facecolor=None, dpi=300)


# CFC-11 vs SF6:
fig_sca2, ax_sca2 = plt.subplots(figsize=(10*cm, 10*cm))
ax_sca2.scatter(x=df['AGE_CFC11'],
                y=df['AGE_SF6'],
                facecolors='none', edgecolors='k',
                s=10, linewidth=.5)
ax_sca2.axline((0, 0), slope=1, color='#aaa', linestyle='--', linewidth=.8)
ax_sca2.set(xlim=[0, 80], ylim=[0, 80],
            xlabel="Age (CFC-11, years)",
            ylabel="Age (SF$_6$, years)")
ax_sca2.set_aspect('equal')
ax_sca2.set_xticks(range(0,90,10))
ax_sca2.set_yticks(range(0,90,10))
ax_sca2.tick_params(direction='in', top=True, right=True, length=3)
# Add regression line to plot (only in our data range):
md2 = smf.ols('AGE_SF6 ~ AGE_CFC11', data=df, missing='drop').fit()
x0 = np.nanmin(df['AGE_CFC11'])
y0 = md2.params['Intercept'] + md2.params['AGE_CFC11']*x0
x1 = np.nanmax(df['AGE_CFC11'])
y1 = md2.params['Intercept'] + md2.params['AGE_CFC11']*x1
ax_sca2.plot([x0, x1], [y0, y1], c='#79c', linewidth=2, alpha=.9)
slp = str(round(md2.params['AGE_CFC11'], 3))
if md2.pvalues['AGE_CFC11']<.001:
    pval = '$\it{p}$ < 0.001'
elif md2.pvalues['AGE_CFC11']<.05:
    pval = '$\it{p}$ = ' + str(round(md2.pvalues['AGE_CFC11'], 3))
else:
    pval = '$\it{ns}$'
r2 = str(round(md2.rsquared_adj, 2))
units = 'year·year$^{-1}$'
lb1 = "Slope = " + slp + ' ' + units + '\n' + pval + '; R$^2$ = ' + r2
ax_sca2.text(.05, .82, lb1, size=8, transform=ax_sca2.transAxes)

fpath = 'figures/hansell_glodap/tracer_ages/tracer_age_comparison_cfc11_sf6.png'
fig_sca2.savefig(fpath, format='png', bbox_inches='tight', facecolor=None, dpi=300)


# CFC-12 vs SF6:
fig_sca3, ax_sca3 = plt.subplots(figsize=(10*cm, 10*cm))
ax_sca3.scatter(x=df['AGE_CFC12'],
                y=df['AGE_SF6'],
                facecolors='none', edgecolors='k',
                s=10, linewidth=.5)
ax_sca3.axline((0, 0), slope=1, color='#aaa', linestyle='--', linewidth=.8)
ax_sca3.set(xlim=[0, 80], ylim=[0, 80],
            xlabel="Age (CFC-12, years)", 
            label="Age (SF$_6$, years)")
ax_sca3.set_aspect('equal')
ax_sca3.set_xticks(range(0,90,10))
ax_sca3.set_yticks(range(0,90,10))
ax_sca3.tick_params(direction='in', top=True, right=True, length=3)
# Add regression line to plot (only in our data range):
md3 = smf.ols('AGE_SF6 ~ AGE_CFC12', data=df, missing='drop').fit()
x0 = np.nanmin(df['AGE_CFC12'])
y0 = md3.params['Intercept'] + md3.params['AGE_CFC12']*x0
x1 = np.nanmax(df['AGE_CFC12'])
y1 = md3.params['Intercept'] + md3.params['AGE_CFC12']*x1
ax_sca3.plot([x0, x1], [y0, y1], c='#79c', linewidth=2, alpha=.9)
slp = str(round(md3.params['AGE_CFC12'], 3))
if md3.pvalues['AGE_CFC12']<.001:
    pval = '$\it{p}$ < 0.001'
elif md3.pvalues['AGE_CFC12']<.05:
    pval = '$\it{p}$ = ' + str(round(md3.pvalues['AGE_CFC12'], 3))
else:
    pval = '$\it{ns}$'
r2 = str(round(md3.rsquared_adj, 2))
units = 'year·year$^{-1}$'
lb1 = "Slope = " + slp + ' ' + units + '\n' + pval + '; R$^2$ = ' + r2
ax_sca3.text(.05, .82, lb1, size=8, transform=ax_sca3.transAxes)

fpath = 'figures/hansell_glodap/tracer_ages/tracer_age_comparison_cfc12_sf6.png'
fig_sca3.savefig(fpath, format='png', bbox_inches='tight', facecolor=None, dpi=300)



#%% EXPORT RESULTS

dpath = 'deriveddata/hansell_glodap/'
if not os.path.exists(dpath):
    os.makedirs(dpath)
    
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
df.to_csv(fpath, sep=',', header=True, index=False, na_rep='-9999')

