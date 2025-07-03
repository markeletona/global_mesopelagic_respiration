# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:57:50 2024

@author: Markel

Compare approaches to water mass mixing correction


"""

#%% IMPORTS

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import string



#%% LOAD DATA

# Filtered, merged Hansell+GLODAP dataset:
fpath = 'deriveddata/hansell_glodap/hansell_glodap_merged_o2_cfc11_cfc12_sf6_with_ages.csv'
tbl = pd.read_csv(fpath, sep=',', header=0, dtype={'EXPOCODE': str,
                                                   'CRUISE': str,
                                                   'BOTTLE': str,
                                                   'DATE': int})


#%% DATA HANDLING

# Assign missing/invalid values as NAN:
tbl.replace(-9999, np.nan, inplace=True)
# tbl.replace('-999', np.nan, inplace=True)



#%%% ESTIMATE COMPLEMENTARY VARIABLES

# Estimate complentary variables to use in water mass mixing corrections

# Potential temperature-squared (PT^2)
tbl['PT2'] = tbl['PT'] ** 2

# Salinity-squared (S^2)
tbl['S2'] = tbl['CTD_SALINITY'] ** 2

# Potential temperature times salinity (PT*S)
tbl['PTS'] = tbl['PT'] * tbl['CTD_SALINITY']



#%% ESTIMATE RESIDUALS (WATER-MASS MIXING CORRECTION)

# Perform linear regressions with pot. temperature and salinity as independent
# variables to account for the effect of water mass mixing on other variables.
# Linear regressions need to be done within each water mass.
# Extract residuals to use as "mixing-corrected" values.
# 
# Regressions can be done simply with potential temperature and salinity, or
# following De la Fuente et al. 2014 squared terms of PT and S, and PT*S i
# interaction terms can be introduced in the multiple linear regression 
# equations. This would make possible to account for the mixing of more than
# two end members. Test different approaches.
# 
# Given than the error/uncertainty of SALINITY and PT are much smaller 
# than those of biogeochemical variables -> fine to do a model I regression.

# Set variables for which residuals need to be estimated in the workflow
vrs = ['OXYGEN', 'AOU', 'DOC',
       'AGE_CFC11', 'AGE_CFC12', 'AGE_SF6',
       'NITRATE', 'PHOSPHATE', 'SILICIC_ACID']
vrs_flags = {'OXYGEN': 'OXYGEN_FLAG_W',
             'AOU': 'OXYGEN_FLAG_W',
             'DOC': 'DOC_FLAG_W',
             'AGE_CFC11': 'CFC_11_FLAG_W',
             'AGE_CFC12': 'CFC_12_FLAG_W',
             'AGE_SF6': 'SF6_FLAG_W',
             'NITRATE': 'NITRATE_FLAG_W',
             'PHOSPHATE': 'PHOSPHATE_FLAG_W',
             'SILICIC_ACID': 'SILICIC_ACID_FLAG_W'}


# Create function to compute the three versions of the mixing correction:
#   - var = PT + S + var_res
#   - var = PT + S + PT^2 + S^2 + var_res
#   - var = PT + S + PT^2 + S^2 + PT*S + var_res

def mixcorrs(v=None, w=None):
            
    # Subset samples of water mass w, with valid flags
    ss = tbl.loc[(tbl['WATER_MASS']==w) & (~np.isnan(tbl[v]))].copy()
    if v in vrs_flags.keys(): ss = ss.loc[ss[vrs_flags[v]].isin([2, 6]), :].copy()
    
    # Perform regressions
    md0 = smf.ols(v + ' ~ PT + CTD_SALINITY', data=ss).fit()
    md1 = smf.ols(v + ' ~ PT + CTD_SALINITY + PT2 + S2', data=ss).fit()
    md2 = smf.ols(v + ' ~ PT + CTD_SALINITY + PT2 + S2 + PTS', data=ss).fit()
    
    # Create new column for residual variable
    vres0 = v + '_RES0'
    vres1 = v + '_RES1'
    vres2 = v + '_RES2'
    ss[vres0] = np.nan
    ss[vres1] = np.nan
    ss[vres2] = np.nan
    
    # Introduce residual values in table using the index
    ss.loc[md0.resid.index, vres0] = md0.resid.astype('float64')
    ss.loc[md1.resid.index, vres1] = md1.resid.astype('float64')
    ss.loc[md2.resid.index, vres2] = md2.resid.astype('float64')
    
    return ((md0, md1, md2), ss)


# Compute the three correction for example
w = 'SPMW'
v = 'SIGMA0'
mdsSIGMA0, mdsSIGMA0_data = mixcorrs(v=v, w=w)

v = 'AOU'
mdsAOU, mdsAOU_data = mixcorrs(v=v, w=w)

v = 'AGE_CFC11'
mdsAGE11, mdsAGE11_data = mixcorrs(v=v, w=w)

v = 'AGE_SF6'
mdsAGESF6, mdsAGESF6_data = mixcorrs(v=v, w=w)


## Repeat it, but doing the regressions separately for each cruise
def mixcorrs_cruise(v=None, w=None):
            
    # Subset samples of water mass w, with valid flags
    ss = tbl.loc[(tbl['WATER_MASS']==w) & (~np.isnan(tbl[v]))].copy()
    if v in vrs_flags.keys(): ss = ss.loc[ss[vrs_flags[v]].isin([2, 6]), :].copy()
    
    # Create new column for residual variable
    vres0 = v + '_RES0'
    vres1 = v + '_RES1'
    vres2 = v + '_RES2'
    ss[vres0] = np.nan
    ss[vres1] = np.nan
    ss[vres2] = np.nan
    
    # Iterate through cruises
    uc = ss.CRUISE.unique()
    mds0 = []
    mds1 = []
    mds2 = []
    for c in uc:
        
        # Subset sample for cruise c
        ssc = ss[ss.CRUISE==c]
    
        # Perform regressions
        md0 = smf.ols(v + ' ~ PT + CTD_SALINITY', data=ssc).fit()
        md1 = smf.ols(v + ' ~ PT + CTD_SALINITY + PT2 + S2', data=ssc).fit()
        md2 = smf.ols(v + ' ~ PT + CTD_SALINITY + PT2 + S2 + PTS', data=ssc).fit()
    
        # Introduce residual values in table using the index
        ss.loc[md0.resid.index, vres0] = md0.resid.astype('float64')
        ss.loc[md1.resid.index, vres1] = md1.resid.astype('float64')
        ss.loc[md2.resid.index, vres2] = md2.resid.astype('float64')
        
        # Save models
        mds0.append(md0)
        mds1.append(md1)
        mds2.append(md2)
    
    return ((mds0, mds1, mds2), ss)


# Compute the three correction for example
w = 'SPMW'
v = 'SIGMA0'
mdsSIGMA0_c, mdsSIGMA0_data_c = mixcorrs_cruise(v=v, w=w)

v = 'AOU'
mdsAOU_c, mdsAOU_data_c = mixcorrs_cruise(v=v, w=w)

v = 'AGE_CFC11'
mdsAGE11_c, mdsAGE11_data_c = mixcorrs_cruise(v=v, w=w)

v = 'AGE_SF6'
mdsAGESF6_c, mdsAGESF6_data_c = mixcorrs_cruise(v=v, w=w)


#%% PLOT RESULTS

# Visualise the different fits. As they are multiple linear regressions this
# can't be done with an scatter plot, so plot the residuals to assess 
# differences in the wellness of the fits.

# Prepare equation text to add as tags on plots
def sar(x, n): return str(abs(round(x, n)))
def signchar(x): return "-" if x<0 else "+"
def rmse(x, xpredicted):
    # Root Mean Square Error
    return np.sqrt(np.mean((x - xpredicted)**2))

def pval_str(x):
    # Formatted pvalues
    if x>=.05: xs = "p = " + str(round(x, 2))
    elif x>=.001: xs = "p = " + str(round(x, 3)) 
    else: xs = "$\it{p}$ < 0.001"
    return xs


#%%% SIGMA0

v = 'SIGMA0'

# Parameters eq. 0
i_0 = sar(mdsSIGMA0[0].params['Intercept'], 2)
i_0_s = signchar(mdsSIGMA0[0].params['Intercept'])
p_pt_0 = sar(mdsSIGMA0[0].params['PT'], 2)
p_pt_0_s = signchar(mdsSIGMA0[0].params['PT'])
p_s_0 = sar(mdsSIGMA0[0].params['CTD_SALINITY'], 2)
p_s_0_s = signchar(mdsSIGMA0[0].params['CTD_SALINITY'])

# Parameters eq. 1
i_1 = sar(mdsSIGMA0[1].params['Intercept'], 2)
i_1_s = signchar(mdsSIGMA0[1].params['Intercept'])
p_pt_1 = sar(mdsSIGMA0[1].params['PT'], 2)
p_pt_1_s = signchar(mdsSIGMA0[1].params['PT'])
p_s_1 = sar(mdsSIGMA0[1].params['CTD_SALINITY'], 2)
p_s_1_s = signchar(mdsSIGMA0[1].params['CTD_SALINITY'])
p_pt2_1 = sar(mdsSIGMA0[1].params['PT2'], 4)
p_pt2_1_s = signchar(mdsSIGMA0[1].params['PT2'])
p_s2_1 = sar(mdsSIGMA0[1].params['S2'], 4)
p_s2_1_s = signchar(mdsSIGMA0[1].params['S2'])

# Parameters eq. 2
i_2 = sar(mdsSIGMA0[2].params['Intercept'], 2)
i_2_s = signchar(mdsSIGMA0[2].params['Intercept'])
p_pt_2 = sar(mdsSIGMA0[2].params['PT'], 2)
p_pt_2_s = signchar(mdsSIGMA0[2].params['PT'])
p_s_2 = sar(mdsSIGMA0[2].params['CTD_SALINITY'], 2)
p_s_2_s = signchar(mdsSIGMA0[2].params['CTD_SALINITY'])
p_pt2_2 = sar(mdsSIGMA0[2].params['PT2'], 4)
p_pt2_2_s = signchar(mdsSIGMA0[2].params['PT2'])
p_s2_2 = sar(mdsSIGMA0[2].params['S2'], 4)
p_s2_2_s = signchar(mdsSIGMA0[2].params['S2'])
p_pts_2 = sar(mdsSIGMA0[2].params['PTS'], 4)
p_pts_2_s = signchar(mdsSIGMA0[2].params['PTS'])


# Coefficients
n_ = str(len(mdsSIGMA0_data))
r2_0 = str(round(mdsSIGMA0[0].rsquared_adj, 3))
r2_1 = str(round(mdsSIGMA0[1].rsquared_adj, 3))
r2_2 = str(round(mdsSIGMA0[2].rsquared_adj, 3))
pv_0 = pval_str(mdsSIGMA0[0].f_pvalue)
pv_1 = pval_str(mdsSIGMA0[1].f_pvalue)
pv_2 = pval_str(mdsSIGMA0[2].f_pvalue)
rmse_0 = str(round(rmse(mdsSIGMA0_data[v], mdsSIGMA0[0].fittedvalues), 4))
rmse_1 = str(round(rmse(mdsSIGMA0_data[v], mdsSIGMA0[1].fittedvalues), 4))
rmse_2 = str(round(rmse(mdsSIGMA0_data[v], mdsSIGMA0[2].fittedvalues), 4))

eqs_sigma0 = ["$\sigma_{\\theta}$ = " + i_0_s + i_0 + " " + p_pt_0_s + " " + p_pt_0 + "$\mathbfit{\\theta}$ " + p_s_0_s + " " + p_s_0 + "$\mathbfit{S}$ + $\sigma_{\\theta\Delta}$",
              "$\sigma_{\\theta}$ = " + i_1_s + i_1 + " " + p_pt_1_s + " " + p_pt_1 + "$\mathbfit{\\theta}$ " + p_s_1_s + " " + p_s_1 + "$\mathbfit{S}$ " + p_pt2_1_s + " " + p_pt2_1 + "$\mathbfit{\\theta^2}$ " + p_s2_1_s + " " + p_s2_1 + "$\mathbfit{S^2}$ + $\sigma_{\\theta\Delta}$",
              "$\sigma_{\\theta}$ = " + i_2_s + i_2 + " " + p_pt_2_s + " " + p_pt_2 + "$\mathbfit{\\theta}$ " + p_s_2_s + " " + p_s_2 + "$\mathbfit{S}$ " + p_pt2_2_s + " " + p_pt2_2 + "$\mathbfit{\\theta^2}$ " + p_s2_2_s + " " + p_s2_2 + "$\mathbfit{S^2}$ " + p_pts_2_s + " " + p_pts_2 + "$\mathbfit{\\theta S}$ + $\sigma_{\\theta\Delta}$",
              "$\sigma_{\\theta}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $\sigma_{\\theta\Delta}$",
              "$\sigma_{\\theta}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + $\sigma_{\\theta\Delta}$",
              "$\sigma_{\\theta}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + $a_6\mathbfit{\\theta S}$ + $\sigma_{\\theta\Delta}$"]
cfs_sigma0 = ["R$^2_{adj}$ = " + r2_0 + "; " + pv_0 + "; RMSE = " + rmse_0 + "; n = " + n_,
              "R$^2_{adj}$ = " + r2_1 + "; " + pv_1 + "; RMSE = " + rmse_1 + "; n = " + n_,
              "R$^2_{adj}$ = " + r2_2 + "; " + pv_2 + "; RMSE = " + rmse_2 + "; n = " + n_]


# Do the plotting

cm = 1/2.54
fig, ax = plt.subplots(nrows=6, ncols=2,
                       width_ratios=[2, 1],
                       figsize=(8*cm * 2, 5*cm * 6))
for j in range(2):
    for i in range(3):
        if j==0:
            
            # Plot scatter of residuals against latitude
            ax[i, j].scatter(mdsSIGMA0_data['LATITUDE'],
                             mdsSIGMA0_data[v + '_RES' + str(i)],
                             marker='o',
                             facecolor='none',
                             edgecolor='steelblue',
                             linewidth=.5,
                             s=3)
            
            ax[i+3, j].scatter(mdsSIGMA0_data_c['LATITUDE'],
                               mdsSIGMA0_data_c[v + '_RES' + str(i)],
                               marker='o',
                               facecolor='none',
                               edgecolor='steelblue',
                               linewidth=.5,
                               s=3)
            
            # Customise limits and ticks
            yl = [-.35, .1]
            yt = np.arange(-.3, .2, .1)
            xl = [28, 72]
            xt = range(30, 80, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add equations and coefficients
            ax[i, j].text(.03, .15, eqs_sigma0[i],
                          size=5, transform=ax[i, j].transAxes)
            ax[i, j].text(.03, .05, cfs_sigma0[i],
                          size=5, transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .05, eqs_sigma0[i+3],
                            size=5, transform=ax[i+3, j].transAxes)
            
            # Add water mass label
            ax[i, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                          size=6, ha='right', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                            size=6, ha='right', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add tag informing about regression procedure
            ax[i, j].text(.03, .96, "$\mathit{All\ cruises\ regressed\ together}$",
                          size=4, ha='left', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .96, "$\mathit{Cruises\ regressed\ separately}$",
                            size=4, ha='left', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add plot labels
            ax[i, j].set_ylabel("$\sigma_{\\theta\Delta}$", fontsize=8)
            ax[i+3, j].set_ylabel("$\sigma_{\\theta\Delta}$", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Latitude", fontsize=8)
            
            # Add plot tags
            ax[i, j].text(-.2, .97, string.ascii_lowercase[i],
                          weight='bold',
                          fontsize=9,
                          ha='left',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(-.2, .97, string.ascii_lowercase[i+3],
                            weight='bold',
                            fontsize=9,
                            ha='left',
                            transform=ax[i+3, j].transAxes)


        if j==1:
            # Plot histogram of residual values
            ax[i, j].hist(mdsSIGMA0_data[v + '_RES' + str(i)],
                          bins=np.arange(-.35, .1, .02),
                          histtype='bar')
            ax[i+3, j].hist(mdsSIGMA0_data_c[v + '_RES' + str(i)],
                            bins=np.arange(-.35, .1, .02),
                            histtype='bar')
            
            # Customise limits and ticks
            yl = [0, 5000]
            yt = range(0, 6000, 1000)
            xl = [-.35, .1]
            xt = np.arange(-.3, .2, .1)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add plot labels
            ax[i, j].set_ylabel("No. of observations", fontsize=8)
            ax[i+3, j].set_ylabel("No. of observations", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("$\sigma_{\\theta\Delta}$", fontsize=8)


# Adjust subplot spacing and export
fig.subplots_adjust(wspace=.35, hspace=.2)

dpath = 'figures/mixcor/'
if not os.path.exists(dpath): os.makedirs(dpath)

fpath = 'figures/mixcor/test_mixing_corrections_' + v + '_' + w + '.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%%% AGE_CFC11

v = 'AGE_CFC11'

# Parameters eq. 0
i_0 = sar(mdsAGE11[0].params['Intercept'], 2)
i_0_s = signchar(mdsAGE11[0].params['Intercept'])
p_pt_0 = sar(mdsAGE11[0].params['PT'], 2)
p_pt_0_s = signchar(mdsAGE11[0].params['PT'])
p_s_0 = sar(mdsAGE11[0].params['CTD_SALINITY'], 2)
p_s_0_s = signchar(mdsAGE11[0].params['CTD_SALINITY'])

# Parameters eq. 1
i_1 = sar(mdsAGE11[1].params['Intercept'], 2)
i_1_s = signchar(mdsAGE11[1].params['Intercept'])
p_pt_1 = sar(mdsAGE11[1].params['PT'], 2)
p_pt_1_s = signchar(mdsAGE11[1].params['PT'])
p_s_1 = sar(mdsAGE11[1].params['CTD_SALINITY'], 1)
p_s_1_s = signchar(mdsAGE11[1].params['CTD_SALINITY'])
p_pt2_1 = sar(mdsAGE11[1].params['PT2'], 2)
p_pt2_1_s = signchar(mdsAGE11[1].params['PT2'])
p_s2_1 = sar(mdsAGE11[1].params['S2'], 2)
p_s2_1_s = signchar(mdsAGE11[1].params['S2'])

# Parameters eq. 2
i_2 = sar(mdsAGE11[2].params['Intercept'], 2)
i_2_s = signchar(mdsAGE11[2].params['Intercept'])
p_pt_2 = sar(mdsAGE11[2].params['PT'], 1)
p_pt_2_s = signchar(mdsAGE11[2].params['PT'])
p_s_2 = sar(mdsAGE11[2].params['CTD_SALINITY'], 1)
p_s_2_s = signchar(mdsAGE11[2].params['CTD_SALINITY'])
p_pt2_2 = sar(mdsAGE11[2].params['PT2'], 2)
p_pt2_2_s = signchar(mdsAGE11[2].params['PT2'])
p_s2_2 = sar(mdsAGE11[2].params['S2'], 1)
p_s2_2_s = signchar(mdsAGE11[2].params['S2'])
p_pts_2 = sar(mdsAGE11[2].params['PTS'], 1)
p_pts_2_s = signchar(mdsAGE11[2].params['PTS'])


# Coefficients
n_ = str(len(mdsAGE11_data))
r2_0 = str(round(mdsAGE11[0].rsquared_adj, 3))
r2_1 = str(round(mdsAGE11[1].rsquared_adj, 3))
r2_2 = str(round(mdsAGE11[2].rsquared_adj, 3))
pv_0 = pval_str(mdsAGE11[0].f_pvalue)
pv_1 = pval_str(mdsAGE11[1].f_pvalue)
pv_2 = pval_str(mdsAGE11[2].f_pvalue)
rmse_0 = str(round(rmse(mdsAGE11_data[v], mdsAGE11[0].fittedvalues), 2))
rmse_1 = str(round(rmse(mdsAGE11_data[v], mdsAGE11[1].fittedvalues), 2))
rmse_2 = str(round(rmse(mdsAGE11_data[v], mdsAGE11[2].fittedvalues), 2))

eqs_age11 = ["Age$_{CFC\u201011}$ = " + i_0_s + i_0 + " " + p_pt_0_s + " " + p_pt_0 + "$\mathbfit{\\theta}$ " + p_s_0_s + " " + p_s_0 + "$\mathbfit{S}$ + Age$_{CFC\u201011\Delta}$",
             "Age$_{CFC\u201011}$ = " + i_1_s + i_1 + " " + p_pt_1_s + " " + p_pt_1 + "$\mathbfit{\\theta}$ " + p_s_1_s + " " + p_s_1 + "$\mathbfit{S}$ " + p_pt2_1_s + " " + p_pt2_1 + "$\mathbfit{\\theta^2}$ " + p_s2_1_s + " " + p_s2_1 + "$\mathbfit{S^2}$ + Age$_{CFC\u201011\Delta}$",
             "Age$_{CFC\u201011}$ = " + i_2_s + i_2 + " " + p_pt_2_s + " " + p_pt_2 + "$\mathbfit{\\theta}$ " + p_s_2_s + " " + p_s_2 + "$\mathbfit{S}$ " + p_pt2_2_s + " " + p_pt2_2 + "$\mathbfit{\\theta^2}$ " + p_s2_2_s + " " + p_s2_2 + "$\mathbfit{S^2}$ " + p_pts_2_s + " " + p_pts_2 + "$\mathbfit{\\theta S}$ + Age$_{CFC\u201011\Delta}$",
             "Age$_{CFC\u201011}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + Age$_{CFC\u201011\Delta}$",
             "Age$_{CFC\u201011}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + Age$_{CFC\u201011\Delta}$",
             "Age$_{CFC\u201011}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + $a_6\mathbfit{\\theta S}$ + Age$_{CFC\u201011\Delta}$"]
cfs_age11 = ["R$^2_{adj}$ = " + r2_0 + "; " + pv_0 + "; RMSE = " + rmse_0 + "; n = " + n_,
             "R$^2_{adj}$ = " + r2_1 + "; " + pv_1 + "; RMSE = " + rmse_1 + "; n = " + n_,
             "R$^2_{adj}$ = " + r2_2 + "; " + pv_2 + "; RMSE = " + rmse_2 + "; n = " + n_]


# Do the plotting

fig, ax = plt.subplots(nrows=6, ncols=2,
                       width_ratios=[2, 1],
                       figsize=(8*cm * 2, 5*cm * 6))
for j in range(2):
    for i in range(3):
        if j==0:
            
            # Plot scatter of residuals against latitude
            ax[i, j].scatter(mdsAGE11_data['LATITUDE'],
                             mdsAGE11_data[v + '_RES' + str(i)],
                             marker='o',
                             facecolor='none',
                             edgecolor='steelblue',
                             linewidth=.5,
                             s=3)
            
            ax[i+3, j].scatter(mdsAGE11_data_c['LATITUDE'],
                               mdsAGE11_data_c[v + '_RES' + str(i)],
                               marker='o',
                               facecolor='none',
                               edgecolor='steelblue',
                               linewidth=.5,
                               s=3)
            
            # Customise limits and ticks
            yl = [-40, 35]
            yt = np.arange(-30, 45, 15)
            xl = [28, 72]
            xt = range(30, 80, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add equations and coefficients
            ax[i, j].text(.03, .12, eqs_age11[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i, j].text(.03, .05, cfs_age11[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .05, eqs_age11[i+3],
                            size=5, transform=ax[i+3, j].transAxes)
            
            # Add water mass label
            ax[i, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                          size=6, ha='right', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                            size=6, ha='right', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add tag informing about regression procedure
            ax[i, j].text(.03, .96, "$\mathit{All\ cruises\ regressed\ together}$",
                          size=4, ha='left', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .96, "$\mathit{Cruises\ regressed\ separately}$",
                            size=4, ha='left', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add plot labels
            ax[i, j].set_ylabel("Age$_{CFC\u201011\Delta}$", fontsize=8)
            ax[i+3, j].set_ylabel("Age$_{CFC\u201011\Delta}$", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Latitude", fontsize=8)
            
            # Add plot tags
            ax[i, j].text(-.2, .97, string.ascii_lowercase[i],
                          weight='bold',
                          fontsize=9,
                          ha='left',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(-.2, .97, string.ascii_lowercase[i+3],
                            weight='bold',
                            fontsize=9,
                            ha='left',
                            transform=ax[i+3, j].transAxes)

        if j==1:
            # Plot histogram of residual values
            ax[i, j].hist(mdsAGE11_data[v + '_RES' + str(i)],
                          bins=np.arange(-30, 30, 5),
                          histtype='bar')
            ax[i+3, j].hist(mdsAGE11_data_c[v + '_RES' + str(i)],
                            bins=np.arange(-30, 30, 5),
                            histtype='bar')
            
            # Customise limits and ticks
            yl=[0, 2400]
            yt=range(0, 2800, 400)
            xl=[-30, 30]
            xt=np.arange(-30, 30, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add plot labels
            ax[i, j].set_ylabel("No. of observations", fontsize=8)
            ax[i+3, j].set_ylabel("No. of observations", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Age$_{CFC\u201011\Delta}$", fontsize=8)


# Adjust subplot spacing and export
fig.subplots_adjust(wspace=.35, hspace=.2)
fpath = 'figures/mixcor/test_mixing_corrections_' + v + '_' + w + '.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)


#%%% SF6

v = 'AGE_SF6'

# Parameters eq. 0
i_0 = sar(mdsAGESF6[0].params['Intercept'], 2)
i_0_s = signchar(mdsAGESF6[0].params['Intercept'])
p_pt_0 = sar(mdsAGESF6[0].params['PT'], 2)
p_pt_0_s = signchar(mdsAGESF6[0].params['PT'])
p_s_0 = sar(mdsAGESF6[0].params['CTD_SALINITY'], 2)
p_s_0_s = signchar(mdsAGESF6[0].params['CTD_SALINITY'])

# Parameters eq. 1
i_1 = sar(mdsAGESF6[1].params['Intercept'], 2)
i_1_s = signchar(mdsAGESF6[1].params['Intercept'])
p_pt_1 = sar(mdsAGESF6[1].params['PT'], 2)
p_pt_1_s = signchar(mdsAGESF6[1].params['PT'])
p_s_1 = sar(mdsAGESF6[1].params['CTD_SALINITY'], 1)
p_s_1_s = signchar(mdsAGESF6[1].params['CTD_SALINITY'])
p_pt2_1 = sar(mdsAGESF6[1].params['PT2'], 2)
p_pt2_1_s = signchar(mdsAGESF6[1].params['PT2'])
p_s2_1 = sar(mdsAGESF6[1].params['S2'], 2)
p_s2_1_s = signchar(mdsAGESF6[1].params['S2'])

# Parameters eq. 2
i_2 = sar(mdsAGESF6[2].params['Intercept'], 2)
i_2_s = signchar(mdsAGESF6[2].params['Intercept'])
p_pt_2 = sar(mdsAGESF6[2].params['PT'], 1)
p_pt_2_s = signchar(mdsAGESF6[2].params['PT'])
p_s_2 = sar(mdsAGESF6[2].params['CTD_SALINITY'], 1)
p_s_2_s = signchar(mdsAGESF6[2].params['CTD_SALINITY'])
p_pt2_2 = sar(mdsAGESF6[2].params['PT2'], 2)
p_pt2_2_s = signchar(mdsAGESF6[2].params['PT2'])
p_s2_2 = sar(mdsAGESF6[2].params['S2'], 1)
p_s2_2_s = signchar(mdsAGESF6[2].params['S2'])
p_pts_2 = sar(mdsAGESF6[2].params['PTS'], 1)
p_pts_2_s = signchar(mdsAGESF6[2].params['PTS'])


# Coefficients
n_ = str(len(mdsAGESF6_data))
r2_0 = str(round(mdsAGESF6[0].rsquared_adj, 3))
r2_1 = str(round(mdsAGESF6[1].rsquared_adj, 3))
r2_2 = str(round(mdsAGESF6[2].rsquared_adj, 3))
pv_0 = pval_str(mdsAGESF6[0].f_pvalue)
pv_1 = pval_str(mdsAGESF6[1].f_pvalue)
pv_2 = pval_str(mdsAGESF6[2].f_pvalue)
rmse_0 = str(round(rmse(mdsAGESF6_data[v], mdsAGESF6[0].fittedvalues), 2))
rmse_1 = str(round(rmse(mdsAGESF6_data[v], mdsAGESF6[1].fittedvalues), 2))
rmse_2 = str(round(rmse(mdsAGESF6_data[v], mdsAGESF6[2].fittedvalues), 2))

eqs_agesf6 = ["Age$_{SF_6}$ = " + i_0_s + i_0 + " " + p_pt_0_s + " " + p_pt_0 + "$\mathbfit{\\theta}$ " + p_s_0_s + " " + p_s_0 + "$\mathbfit{S}$ + Age$_{SF_6\Delta}$",
              "Age$_{SF_6}$ = " + i_1_s + i_1 + " " + p_pt_1_s + " " + p_pt_1 + "$\mathbfit{\\theta}$ " + p_s_1_s + " " + p_s_1 + "$\mathbfit{S}$ " + p_pt2_1_s + " " + p_pt2_1 + "$\mathbfit{\\theta^2}$ " + p_s2_1_s + " " + p_s2_1 + "$\mathbfit{S^2}$ + Age$_{SF_6\Delta}$",
              "Age$_{SF_6}$ = " + i_2_s + i_2 + " " + p_pt_2_s + " " + p_pt_2 + "$\mathbfit{\\theta}$ " + p_s_2_s + " " + p_s_2 + "$\mathbfit{S}$ " + p_pt2_2_s + " " + p_pt2_2 + "$\mathbfit{\\theta^2}$ " + p_s2_2_s + " " + p_s2_2 + "$\mathbfit{S^2}$ " + p_pts_2_s + " " + p_pts_2 + "$\mathbfit{\\theta S}$ + Age$_{SF_6\Delta}$",
              "Age$_{SF_6}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + Age$_{SF_6\Delta}$",
              "Age$_{SF_6}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + Age$_{SF_6\Delta}$",
              "Age$_{SF_6}$ = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + $a_6\mathbfit{\\theta S}$ + Age$_{SF_6\Delta}$"]
cfs_agesf6 = ["R$^2_{adj}$ = " + r2_0 + "; " + pv_0 + "; RMSE = " + rmse_0 + "; n = " + n_,
              "R$^2_{adj}$ = " + r2_1 + "; " + pv_1 + "; RMSE = " + rmse_1 + "; n = " + n_,
              "R$^2_{adj}$ = " + r2_2 + "; " + pv_2 + "; RMSE = " + rmse_2 + "; n = " + n_]


# Do the plotting

fig, ax = plt.subplots(nrows=6, ncols=2,
                       width_ratios=[2, 1],
                       figsize=(8*cm * 2, 5*cm * 6))
for j in range(2):
    for i in range(3):
        if j==0:
            
            # Plot scatter of residuals against latitude
            ax[i, j].scatter(mdsAGESF6_data['LATITUDE'],
                             mdsAGESF6_data[v + '_RES' + str(i)],
                             marker='o',
                             facecolor='none',
                             edgecolor='steelblue',
                             linewidth=.5,
                             s=3)
            
            ax[i+3, j].scatter(mdsAGESF6_data_c['LATITUDE'],
                               mdsAGESF6_data_c[v + '_RES' + str(i)],
                               marker='o',
                               facecolor='none',
                               edgecolor='steelblue',
                               linewidth=.5,
                               s=3)
            
            # Customise limits and ticks
            yl = [-30, 25]
            yt = np.arange(-30, 30, 10)
            xl = [28, 72]
            xt = range(30, 80, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add equations and coefficients
            ax[i, j].text(.03, .12, eqs_agesf6[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i, j].text(.03, .05, cfs_agesf6[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .05, eqs_agesf6[i+3],
                            size=5, transform=ax[i+3, j].transAxes)
            
            # Add water mass label
            ax[i, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                          size=6, ha='right', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                            size=6, ha='right', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add tag informing about regression procedure
            ax[i, j].text(.03, .96, "$\mathit{All\ cruises\ regressed\ together}$",
                          size=4, ha='left', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .96, "$\mathit{Cruises\ regressed\ separately}$",
                            size=4, ha='left', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add plot labels
            ax[i, j].set_ylabel("Age$_{SF_6\Delta}$", fontsize=8)
            ax[i+3, j].set_ylabel("Age$_{SF_6\Delta}$", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Latitude", fontsize=8)
            
            # Add plot tags
            ax[i, j].text(-.2, .97, string.ascii_lowercase[i],
                          weight='bold',
                          fontsize=9,
                          ha='left',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(-.2, .97, string.ascii_lowercase[i+3],
                            weight='bold',
                            fontsize=9,
                            ha='left',
                            transform=ax[i+3, j].transAxes)

        if j==1:
            # Plot histogram of residual values
            ax[i, j].hist(mdsAGESF6_data[v + '_RES' + str(i)],
                          bins=np.arange(-30, 30, 5),
                          histtype='bar')
            ax[i+3, j].hist(mdsAGESF6_data_c[v + '_RES' + str(i)],
                            bins=np.arange(-30, 30, 5),
                            histtype='bar')
            
            # Customise limits and ticks
            yl = [0, 450]
            yt = range(0, 500, 100)
            xl = [-25, 25]
            xt = np.arange(-20, 30, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add plot labels
            ax[i, j].set_ylabel("No. of observations", fontsize=8)
            ax[i+3, j].set_ylabel("No. of observations", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Age$_{SF_6\Delta}$", fontsize=8)


# Adjust subplot spacing and export
fig.subplots_adjust(wspace=.35, hspace=.2)
fpath = 'figures/mixcor/test_mixing_corrections_' + v + '_' + w + '.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%%% AOU

v = 'AOU'

# Parameters eq. 0
i_0 = sar(mdsAOU[0].params['Intercept'], 2)
i_0_s = signchar(mdsAOU[0].params['Intercept'])
p_pt_0 = sar(mdsAOU[0].params['PT'], 2)
p_pt_0_s = signchar(mdsAOU[0].params['PT'])
p_s_0 = sar(mdsAOU[0].params['CTD_SALINITY'], 2)
p_s_0_s = signchar(mdsAOU[0].params['CTD_SALINITY'])

# Parameters eq. 1
i_1 = sar(mdsAOU[1].params['Intercept'], 2)
i_1_s = signchar(mdsAOU[1].params['Intercept'])
p_pt_1 = sar(mdsAOU[1].params['PT'], 2)
p_pt_1_s = signchar(mdsAOU[1].params['PT'])
p_s_1 = sar(mdsAOU[1].params['CTD_SALINITY'], 1)
p_s_1_s = signchar(mdsAOU[1].params['CTD_SALINITY'])
p_pt2_1 = sar(mdsAOU[1].params['PT2'], 2)
p_pt2_1_s = signchar(mdsAOU[1].params['PT2'])
p_s2_1 = sar(mdsAOU[1].params['S2'], 2)
p_s2_1_s = signchar(mdsAOU[1].params['S2'])

# Parameters eq. 2
i_2 = sar(mdsAOU[2].params['Intercept'], 2)
i_2_s = signchar(mdsAOU[2].params['Intercept'])
p_pt_2 = sar(mdsAOU[2].params['PT'], 1)
p_pt_2_s = signchar(mdsAOU[2].params['PT'])
p_s_2 = sar(mdsAOU[2].params['CTD_SALINITY'], 1)
p_s_2_s = signchar(mdsAOU[2].params['CTD_SALINITY'])
p_pt2_2 = sar(mdsAOU[2].params['PT2'], 2)
p_pt2_2_s = signchar(mdsAOU[2].params['PT2'])
p_s2_2 = sar(mdsAOU[2].params['S2'], 1)
p_s2_2_s = signchar(mdsAOU[2].params['S2'])
p_pts_2 = sar(mdsAOU[2].params['PTS'], 1)
p_pts_2_s = signchar(mdsAOU[2].params['PTS'])


# Coefficients
n_ = str(len(mdsAOU_data))
r2_0 = str(round(mdsAOU[0].rsquared_adj, 3))
r2_1 = str(round(mdsAOU[1].rsquared_adj, 3))
r2_2 = str(round(mdsAOU[2].rsquared_adj, 3))
pv_0 = pval_str(mdsAOU[0].f_pvalue)
pv_1 = pval_str(mdsAOU[1].f_pvalue)
pv_2 = pval_str(mdsAOU[2].f_pvalue)
rmse_0 = str(round(rmse(mdsAOU_data[v], mdsAOU[0].fittedvalues), 2))
rmse_1 = str(round(rmse(mdsAOU_data[v], mdsAOU[1].fittedvalues), 2))
rmse_2 = str(round(rmse(mdsAOU_data[v], mdsAOU[2].fittedvalues), 2))

eqs_aou = ["AOU = " + i_0_s + i_0 + " " + p_pt_0_s + " " + p_pt_0 + "$\mathbfit{\\theta}$ " + p_s_0_s + " " + p_s_0 + "$\mathbfit{S}$ + AOU$_{\Delta}$",
           "AOU = " + i_1_s + i_1 + " " + p_pt_1_s + " " + p_pt_1 + "$\mathbfit{\\theta}$ " + p_s_1_s + " " + p_s_1 + "$\mathbfit{S}$ " + p_pt2_1_s + " " + p_pt2_1 + "$\mathbfit{\\theta^2}$ " + p_s2_1_s + " " + p_s2_1 + "$\mathbfit{S^2}$ + AOU$_{\Delta}$",
           "AOU = " + i_2_s + i_2 + " " + p_pt_2_s + " " + p_pt_2 + "$\mathbfit{\\theta}$ " + p_s_2_s + " " + p_s_2 + "$\mathbfit{S}$ " + p_pt2_2_s + " " + p_pt2_2 + "$\mathbfit{\\theta^2}$ " + p_s2_2_s + " " + p_s2_2 + "$\mathbfit{S^2}$ " + p_pts_2_s + " " + p_pts_2 + "$\mathbfit{\\theta S}$ + AOU$_{\Delta}$",
           "AOU = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + AOU$_{\Delta}$",
           "AOU = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + AOU$_{\Delta}$",
           "AOU = $a_1$ + $a_2\mathbfit{\\theta}$ + $a_3\mathbfit{S}$ + $a_4\mathbfit{\\theta^2}$ + $a_5\mathbfit{S^2}$ + $a_6\mathbfit{\\theta S}$ + AOU$_{\Delta}$"]
cfs_aou = ["R$^2_{adj}$ = " + r2_0 + "; " + pv_0 + "; RMSE = " + rmse_0 + "; n = " + n_,
           "R$^2_{adj}$ = " + r2_1 + "; " + pv_1 + "; RMSE = " + rmse_1 + "; n = " + n_,
           "R$^2_{adj}$ = " + r2_2 + "; " + pv_2 + "; RMSE = " + rmse_2 + "; n = " + n_]


# Do the plotting

fig, ax = plt.subplots(nrows=6, ncols=2,
                       width_ratios=[2, 1],
                       figsize=(8*cm * 2, 5*cm * 6))
for j in range(2):
    for i in range(3):
        if j==0:
            
            # Plot scatter of residuals against latitude
            ax[i, j].scatter(mdsAOU_data['LATITUDE'],
                             mdsAOU_data[v + '_RES' + str(i)],
                             marker='o',
                             facecolor='none',
                             edgecolor='steelblue',
                             linewidth=.5,
                             s=3)
            
            ax[i+3, j].scatter(mdsAOU_data_c['LATITUDE'],
                               mdsAOU_data_c[v + '_RES' + str(i)],
                               marker='o',
                               facecolor='none',
                               edgecolor='steelblue',
                               linewidth=.5,
                               s=3)
            
            # Customise limits and ticks
            yl = [-120, 120]
            yt = np.arange(-120, 180, 60)
            xl = [28, 72]
            xt = range(30, 80, 10)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add equations and coefficients
            ax[i, j].text(.03, .12, eqs_aou[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i, j].text(.03, .05, cfs_aou[i],
                          size=4, transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .05, eqs_aou[i+3],
                            size=5, transform=ax[i+3, j].transAxes)
            
            # Add water mass label
            ax[i, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                          size=6, ha='right', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.985, .94, "$\mathbf{ " + w + "}$",
                            size=6, ha='right', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add tag informing about regression procedure
            ax[i, j].text(.03, .96, "$\mathit{All\ cruises\ regressed\ together}$",
                          size=4, ha='left', va='top',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(.03, .96, "$\mathit{Cruises\ regressed\ separately}$",
                            size=4, ha='left', va='top',
                            transform=ax[i+3, j].transAxes)
            
            # Add plot labels
            ax[i, j].set_ylabel("AOU$_{\Delta}$", fontsize=8)
            ax[i+3, j].set_ylabel("AOU$_{\Delta}$", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("Latitude", fontsize=8)
            
            # Add plot tags
            ax[i, j].text(-.2, .97, string.ascii_lowercase[i],
                          weight='bold',
                          fontsize=9,
                          ha='left',
                          transform=ax[i, j].transAxes)
            ax[i+3, j].text(-.2, .97, string.ascii_lowercase[i+3],
                            weight='bold',
                            fontsize=9,
                            ha='left',
                            transform=ax[i+3, j].transAxes)

        if j==1:
            # Plot histogram of residual values
            ax[i, j].hist(mdsAOU_data[v + '_RES' + str(i)],
                          bins=np.arange(-120, 120, 20),
                          histtype='bar')
            ax[i+3, j].hist(mdsAOU_data_c[v + '_RES' + str(i)],
                          bins=np.arange(-120, 120, 20),
                            histtype='bar')
            
            # Customise limits and ticks
            yl = [0, 2400]
            yt = range(0, 2800, 400)
            xl = [-120, 120]
            xt = np.arange(-120, 120, 40)
            ax[i, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i+3, j].set(ylim=yl, yticks=yt, xlim=xl, xticks=xt)
            ax[i, j].tick_params(axis='both', which='major', direction='in',
                                 top=True, right=True,
                                 length=2, labelsize=7)
            ax[i+3, j].tick_params(axis='both', which='major', direction='in',
                                   top=True, right=True,
                                   length=2, labelsize=7)
            
            # Add plot labels
            ax[i, j].set_ylabel("No. of observations", fontsize=8)
            ax[i+3, j].set_ylabel("No. of observations", fontsize=8)
            if i==2: ax[i+3, j].set_xlabel("AOU$_{\Delta}$", fontsize=8)
            

# Adjust subplot spacing and export
fig.subplots_adjust(wspace=.35, hspace=.2)
fpath = 'figures/mixcor/test_mixing_corrections_' + v + '_' + w + '.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%%% SCATTER AOU vs AGE11

idx = [x for x in mdsAOU_data.index if x in mdsAGE11_data.index]
x = mdsAGE11_data.AGE_CFC11_RES1.loc[idx]
y = mdsAOU_data.AOU_RES1.loc[idx]
c = mdsAGE11_data.YEAR_SAMPLE.loc[idx]

# Initialise figure with constrained layout to properly align subplots with and
# without colorbars
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14*cm, 6*cm * 2),
                       layout='constrained')
ax = ax.flatten()

# Plot the residuals of the mixing corrections done for the entire water mass
# at once
sc = ax[0].scatter(x,
                   y,
                   marker='o',
                   c=c,
                   vmin=1982,
                   vmax=2021,
                   alpha=1,
                   edgecolor='w',
                   linewidth=.1,
                   s=10)

# Repeat plot but highlighting specific cruises
# Plot separately to control which points appear on top
select_cruise = ['Other', '31', '42', '342', '1041']
curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGE11_data.CRUISE.loc[idx]]
pal = {'Other': '#ccc', '31': 'goldenrod', '42': 'green',
       '342': 'firebrick', '1041': 'steelblue'}
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[2].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)
ax[2].legend(loc='upper left',
             fontsize=5,
             frameon=False,
             handletextpad=0,
             scatterpoints=1, markerscale=1.5,
             reverse=True)


# Plot residuals of the corrections done separately per cruise
idx = [x for x in mdsAOU_data_c.index if x in mdsAGE11_data_c.index]
x = mdsAGE11_data_c.AGE_CFC11_RES1.loc[idx]
y = mdsAOU_data_c.AOU_RES1.loc[idx]
c = mdsAGE11_data_c.YEAR_SAMPLE.loc[idx]

# Reorder to plot most recent cruises on top, just to aid visualisation
x = x.loc[c.sort_values().index]
y = y.loc[c.sort_values().index]
c = c.loc[c.sort_values().index]

scRES = ax[1].scatter(x,
                      y,
                      marker='o',
                      c=c,
                      vmin=1982,
                      vmax=2021,
                      alpha=1,
                      edgecolor='w',
                      linewidth=.1,
                      s=10)

# Shared colorbar for the two top subplots
cbarRES = fig.colorbar(scRES, ax=ax[[0, 1]])
cbarRES.set_label("Sampling year", fontsize=7)
cbarRES.ax.tick_params(axis='y', which='major', direction='out',
                      length=2.5, labelsize=7)

# Repeat plot highlighting specific cruises
curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGE11_data_c.CRUISE.loc[idx]]
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[3].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)

# Set axis limits etc
for i in range(4):
    ax[i].set(ylim=[-80, 130],
              yticks=range(-80, 160, 40),
              xlim=[-25, 35],
              xticks=range(-20, 40, 10))
    ax[i].tick_params(axis='both', which='major', direction='in',
                      top=True, right=True,
                      length=3, labelsize=8)
    ax[i].set_ylabel("AOU$_{\Delta}$", fontsize=8, labelpad=0)
    ax[i].set_xlabel("Age$\mathregular{_{CFC\u201011\Delta}}$", fontsize=8, labelpad=2)

ax[0].text(.05, .95, w,
           fontsize=8,
           ha='left', va='top', 
           weight='bold',
           transform=ax[0].transAxes)

# Export
fpath = 'figures/mixcor/test_corrected_aou_vs_age11.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%%% SCATTER AOU vs AGESF6

idx = [x for x in mdsAOU_data.index if x in mdsAGESF6_data.index]
x = mdsAGESF6_data.AGE_SF6_RES1.loc[idx]
y = mdsAOU_data.AOU_RES1.loc[idx]
c = mdsAGESF6_data.YEAR_SAMPLE.loc[idx]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14*cm, 6*cm * 2),
                       layout='constrained')
ax = ax.flatten()

sc = ax[0].scatter(x,
                   y,
                   marker='o',
                   c=c,
                   vmin=1982,
                   vmax=2021,
                   alpha=1,
                   edgecolor='w',
                   linewidth=.1,
                   s=10)

select_cruise = ['Other', '2011', '329', '342', '1041']
curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGESF6_data.CRUISE.loc[idx]]
pal = {'Other': '#ccc', '2011': 'goldenrod', '329': 'green',
       '342': 'firebrick', '1041': 'steelblue'}
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[2].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)
ax[2].legend(loc='upper left', 
             fontsize=5,
             frameon=False,
             handletextpad=0,
             scatterpoints=1, markerscale=1.5,
             reverse=True)


# Plot residuals of the corrections done separately per cruise
idx = [x for x in mdsAOU_data_c.index if x in mdsAGESF6_data_c.index]
x = mdsAGESF6_data_c.AGE_SF6_RES1.loc[idx]
y = mdsAOU_data_c.AOU_RES1.loc[idx]
c = mdsAGESF6_data_c.YEAR_SAMPLE.loc[idx]
x = x.loc[c.sort_values().index]
y = y.loc[c.sort_values().index]
c = c.loc[c.sort_values().index]

scRES = ax[1].scatter(x,
                      y,
                      marker='o',
                      c=c,
                      vmin=1982,
                      vmax=2021,
                      alpha=1,
                      edgecolor='w',
                      linewidth=.1,
                      s=10)
cbarRES = fig.colorbar(scRES, ax=ax[[0, 1]])
cbarRES.set_label("Sampling year", fontsize=7)
cbarRES.ax.tick_params(axis='y', which='major', direction='out',
                      length=2.5, labelsize=7)

curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGESF6_data_c.CRUISE.loc[idx]]
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[3].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)

for i in range(4):
    ax[i].set(ylim=[-80, 130],
           yticks=range(-80, 160, 40),
           xlim=[-20, 30],
           xticks=range(-20, 40, 10))
    ax[i].tick_params(axis='both', which='major', direction='in',
                         top=True, right=True,
                         length=3, labelsize=8)
    ax[i].set_ylabel("AOU$_{\Delta}$", fontsize=8, labelpad=0)
    ax[i].set_xlabel("Age$\mathregular{_{SF_{6\Delta}}}$", fontsize=8, labelpad=2)

ax[0].text(.05, .95, w,
           fontsize=8,
           ha='left', va='top', 
           weight='bold',
           transform=ax[0].transAxes)


# Export
fpath = 'figures/mixcor/test_corrected_aou_vs_agesf6.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)



#%%% SCATTER AOU vs AGE11 (ONLY WITH SAMPLES THAT HAVE AGESF6 TOO)

# Repeat the AOU vs AGE11 but only with samples that also have SF6 ages

idx = [x for x in mdsAGE11_data.index if (x in mdsAGESF6_data.index) & (x in mdsAOU_data.index)]
x = mdsAGE11_data.AGE_CFC11_RES1.loc[idx]
y = mdsAOU_data.AOU_RES1.loc[idx]
c = mdsAGE11_data.YEAR_SAMPLE.loc[idx]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14*cm, 6*cm * 2),
                       layout='constrained')
ax = ax.flatten()
sc = ax[0].scatter(x,
                   y,
                   marker='o',
                   c=c,
                   vmin=1982,
                   vmax=2021,
                   alpha=1,
                   edgecolor='w',
                   linewidth=.1,
                   s=10)

select_cruise = ['Other', '31', '42', '342', '1041']
curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGE11_data.CRUISE.loc[idx]]
pal = {'Other': '#ccc', '31': 'goldenrod', '42': 'green',
       '342': 'firebrick', '1041': 'steelblue'}
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[2].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)
ax[2].legend(loc='upper left', 
             fontsize=5,
             frameon=False,
             handletextpad=0,
             scatterpoints=1, markerscale=1.5,
             reverse=True)


idx = [x for x in mdsAGE11_data_c.index if (x in mdsAGESF6_data_c.index) & (x in mdsAOU_data_c.index)]
x = mdsAGE11_data_c.AGE_CFC11_RES1.loc[idx]
y = mdsAOU_data_c.AOU_RES1.loc[idx]
c = mdsAGE11_data_c.YEAR_SAMPLE.loc[idx]
x = x.loc[c.sort_values().index]
y = y.loc[c.sort_values().index]
c = c.loc[c.sort_values().index]

scRES = ax[1].scatter(x,
                      y,
                      marker='o',
                      c=c,
                      vmin=1982,
                      vmax=2021,
                      alpha=1,
                      edgecolor='w',
                      linewidth=.1,
                      s=10)
cbarRES = fig.colorbar(scRES, ax=ax[[0, 1]])
cbarRES.set_label("Sampling year", fontsize=7)
cbarRES.ax.tick_params(axis='y', which='major', direction='out',
                      length=2.5, labelsize=7)

curated_cruise = [x if x in select_cruise else 'Other' for x in  mdsAGE11_data_c.CRUISE.loc[idx]]
for iv, v in enumerate(select_cruise):
    idx2 = [j==v for j in curated_cruise]
    sc = ax[3].scatter(x[idx2],
                       y[idx2],
                       marker='o',
                       c=pal[v],
                       label=v,
                       edgecolor='w',
                       linewidth=.1,
                       s=10,
                       zorder=iv)

for i in range(4):
    ax[i].set(ylim=[-80, 130],
              yticks=range(-80, 160, 40),
              xlim=[-25, 35],
              xticks=range(-20, 40, 10))
    ax[i].tick_params(axis='both', which='major', direction='in',
                      top=True, right=True,
                      length=3, labelsize=8)
    ax[i].set_ylabel("AOU$_{\Delta}$", fontsize=8, labelpad=0)
    ax[i].set_xlabel("Age$\mathregular{_{CFC\u201011\Delta}}$", fontsize=8, labelpad=2)

ax[0].text(.05, .95, w,
           fontsize=8,
           ha='left', va='top', 
           weight='bold',
           transform=ax[0].transAxes)


# Export
fpath = 'figures/mixcor/test_corrected_aou_vs_age11_only_matching_sf6.pdf'
fig.savefig(fpath, format='pdf', bbox_inches='tight')
fpath = fpath.replace("pdf", "png")
fig.savefig(fpath, format='png', bbox_inches='tight', dpi=600)


