# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:26:44 2024

@author: Markel

Assess relationships between rates/ratios estimates and environmental 
parameters for potential parametrisations.

"""

#%% IMPORTS

import pandas as pd
import numpy as np
import scripts.modules.RegressConsensus as rc

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
# from matplotlib.lines import Line2D

#%% LOAD DATA

fpath = 'deriveddata/hansell_glodap/global/o2_aou_doc_age_regressions.csv'
reg1 = pd.read_csv(fpath, sep=',')

fpath = 'deriveddata/hansell_glodap/global/stoichiometry_regressions.csv'
reg2 = pd.read_csv(fpath, sep=',')


#%% DATA HANDLING

# Replace bad values with nan
reg1 = reg1.replace(-9999, np.nan)
reg2 = reg2.replace(-9999, np.nan)


# Add individual columns for ocean, depth layer and water mass
reg1['water_mass_key'] = reg1['water_mass'].copy()
reg1['ocean'] = 'none'
reg1['depth'] = 'none'
reg1['water_mass'] = 'none'
for ir, r in reg1.iterrows():
    
    reg1.loc[ir, 'ocean'] = r.water_mass_key.split(";")[0] # ocean
    reg1.loc[ir, 'depth'] = r.water_mass_key.split(";")[1] # depth layer
    reg1.loc[ir, 'water_mass'] = r.water_mass_key.split(";")[2] # water mass code


reg2['water_mass_key'] = reg2['water_mass'].copy()
reg2['ocean'] = 'none'
reg2['depth'] = 'none'
reg2['water_mass'] = 'none'
for ir, r in reg2.iterrows():
    
    reg2.loc[ir, 'ocean'] = r.water_mass_key.split(";")[0]
    reg2.loc[ir, 'depth'] = r.water_mass_key.split(";")[1]
    reg2.loc[ir, 'water_mass'] = r.water_mass_key.split(";")[2]



#### Set limits to accept regressions.
pval_limit = .01 # acceptable if lower than
r2_limit = .15   # acceptable if greater than



#%% THERMODYNAMICS

# Represent the natural logarithm of OUR (and DOCUR) vs: 
#    - the inverse of pt (in Kelvin) -> 1/T
#    - 1000/RT (where R is the ideal gas constant = 8.314 J K−1 mol−1)

# To get the logarithm, values need to be positive. To keep track of the 
# meaning of the rate (e.g., DOC consumption or production), create a new
# column recording the sign of the rate.
reg1_copy = reg1.copy() # keep copy of original version
reg1['abs_rate'] = abs(reg1['slope'])
reg1['rate_sign'] = (reg1['slope'] > 0).astype(int) - (reg1['slope'] < 0).astype(int)
reg1['ln_rate'] = np.log(reg1['abs_rate'])
reg1['inv_PT'] = 1 / (reg1['PT'] + 273.15)
reg1['1000_RPT'] = 1000 / 8.314 * reg1['inv_PT'] # 1000/RT
new_vrs = ['inv_PT', '1000_RPT']


# Custom function to transform rate_sign into an actual character
def signchar(x): d = {-1: "−", 0: "", 1: "+"}; return d[x]


#%%% ARRHENIUS PLOTS: ALL ESTIMATES TOGETHER

#%%%% REGRESSION

# Do the ln_rate vs 1000_RPT regressions considering all estimated rates,
# with no geographical distinctions.

vrs = ['AOU_RES', 'DOC_RES']
ages = np.sort(reg1['x_tracer'].unique())
rate_reg = []
for v in vrs:
    for a in ages:
        for nv in new_vrs:
            for s in [-1, 1]:
                
                # subset values for variable v, age a, and sign s
                idx = ((reg1['y_var']==v) &
                       (reg1['x_tracer']==a) &
                       (reg1['rate_sign']==s))
                df = reg1.loc[idx, :].copy()
                
                # filter only significant rates
                idx = ((df['spvalue'] < pval_limit) &
                       (df['r2_adj'] > r2_limit))
                df = df.loc[idx, :]
                    
                # Do regression
                min_obs = 5 # set minimum obs to perform regression
                n_obs = sum(~np.isnan(df['ln_rate']))
                if n_obs < min_obs: 
                    msg = ("not enough observations (" + str(n_obs) + ")")
                    tmp = [v,
                           a,
                           nv,
                           s,
                           *[np.nan] * 15, # (15 = len of regresscons output)
                           msg             # add comment of too few obs
                           ]
                else:
                    r = rc.RegressConsensusW(df[nv],
                                             df['ln_rate'],
                                             Wx=.5)
                    tmp = [v,           # y variable (rate of the regression)
                           a,           # tracer age from which v was derived
                           nv,          # x variable of the regression
                           s,           # sign of rate
                           *r.values(), # all regression results
                           ''           # no comment
                           ]
                
                # append to previous results
                rate_reg.append(tmp)
                
                
                msg = ("Done: " + nv + " vs Ln(" + v + " rate)" +  
                       " [" + signchar(s) + ", " + a + "]\n"
                       "----------------------------------------------------")
                print(msg)
                
                
nms = ["y_var", "tracer_age", "x_var", "rate_sign", *r.keys(), "comment"]
rate_reg = pd.DataFrame(rate_reg, columns=nms)


#%%%% PLOT 

# custom colours and marker for each rate sign
pal_sign = {-1: '#4393c3', 1: '#d6604d'}
pal_sign_2 = {-1: '#053061', 1: '#67001f'}
mar_sign = {-1: '^', 1: 'o'}
sym_sign = {-1: '▲', 1: '○'}

# labels for each variable
lab_age = {'AGE_CFC11_RES': "$\mathregular{CFC\u201011}$",
           'AGE_CFC12_RES': "$\mathregular{CFC\u201012}$",
           'AGE_SF6_RES': "$\mathregular{SF_6}$"}
lab_rate = {'AOU_RES': "aOUR", 
            'DOC_RES': "aDOCUR"}

cm = 1/2.54
for iv, v in enumerate(vrs):
    
    fig_ar, ax_ar = plt.subplots(nrows=1, ncols=len(ages),
                                 figsize=(7*cm * len(ages), 6*cm))
    
    for ia, a in enumerate(ages):     
        for i, s in enumerate([-1, 1]):
        
            # Do not plot rates that represent increase of O2 with age
            # (negative rates with AOU or positive rates with OXY)
            if (((v=='AOU_RES') & (s==-1)) | ((v=='OXYGEN_RES') & (s==1))):
                continue
            
            # Subset rate data
            idx = ((reg1['y_var']==v) &
                   (reg1['x_tracer']==a) &
                   (reg1['rate_sign']==s))
            df = reg1.loc[idx, :].copy()
            
            # Filter only significant rates
            idx = ((df['spvalue'] < pval_limit) &
                   (df['r2_adj'] > r2_limit))
            df = df.loc[idx, :]
            
            # Get Arrhenius regression data
            idx = ((rate_reg['y_var']==v) &
                   (rate_reg['tracer_age']==a) &
                   (rate_reg['rate_sign']==s) &
                   (rate_reg['x_var']=='1000_RPT'))
            rg = rate_reg.loc[idx, :].copy()
            
            # Plot points
            ax_ar[ia].scatter(df['1000_RPT'], df['ln_rate'],
                              marker=mar_sign[s],
                              s=15,
                              c='#333',
                              edgecolor='w',
                              linewidth=.2)
            
            # Plot regression line if relationship is significant
            if ((rg['spvalue'] < pval_limit) & (rg['r2_adj'] > r2_limit)).squeeze():
                
                # Get the line coords within the data range and plot it
                x0 = np.nanmin(df['1000_RPT'])
                x1 = np.nanmax(df['1000_RPT'])
                slp = rg['slope'].squeeze()
                intr = rg['intercept'].squeeze()
                y0 = x0 * slp + intr
                y1 = x1 * slp + intr
                ax_ar[ia].plot([x0, x1], [y0, y1],
                               color='#222',
                               linestyle='--', linewidth=1)
                
                # Create text string to add equation to plot
                r2_adj = rg['r2_adj'].squeeze()
                p = rg['spvalue'].squeeze()
                if p < .001:
                    ptxt = "< 0.001"
                elif p < .01:
                    ptxt = "< 0.01"
                else:
                    ptxt = ("= " + str(round(p, 2)))
                txt = ("Ln(" + lab_rate[v] + "$_" + 
                       lab_age[a].replace("$", "") + "$ " + signchar(s) + ")"
                       " = " + str(round(slp, 1)) + " $\\times$ 1000/RT + " +
                       str(round(intr, 1)) + "\nR$^{2}$ = " + 
                       str(round(rg['r2_adj'].squeeze(), 2)) + "; $\it{p}$ " + 
                       ptxt)
                
                # Add text to plot
                txt_ypos = .97 - i * .08
                ax_ar[ia].text(x=.97, y=txt_ypos, s=txt,
                               size=4, c='#222',
                               linespacing=1,
                               ha='right',
                               transform=ax_ar[ia].transAxes)
            else:
                # If not significant, add text indicating that
                txt = ("Ln(" + lab_rate[v] + "$_" + 
                       lab_age[a].replace("$", "") + "$ " + signchar(s) + ")"
                       " $\u2192$ ns")
                txt_ypos = .05 + i * .05
                ax_ar[ia].text(x=.05, y=txt_ypos, s=txt,
                               size=4, c='#222',
                               linespacing=1,
                               transform=ax_ar[ia].transAxes)
        
        ## Customise axes
        # set limits dynamically
        # re-subset data to consider limits of both + & - (when valid), 
        # considering estimates across all tracer ages
        if v=='AOU_RES':
            valid_signs = [1]
        elif v=='OXYGEN_RES':
            valid_signs = [-1]
        else:
            valid_signs = [-1, 1]
            
        idx = ((reg1['y_var']==v) &
               (reg1['rate_sign'].isin(valid_signs)) &
               (reg1['spvalue'] < pval_limit) &
               (reg1['r2_adj'] > r2_limit))
        dfxy = reg1.loc[idx, :]
        xmin = np.floor(np.nanmin(dfxy['1000_RPT']) / .01) * .01
        xmax = np.ceil(np.nanmax(dfxy['1000_RPT']) / .005) * .005
        ymin = np.floor(np.nanmin(dfxy['ln_rate']) / .5) * .5
        ymax = np.ceil(np.nanmax(dfxy['ln_rate']) / .5) * .5
        ax_ar[ia].set(xlim=[xmin, xmax],
                      xticks=np.arange(xmin, xmax, .01),
                      ylim=[ymin, ymax],
                      yticks=np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1))
        ax_ar[ia].xaxis.set_minor_locator(mticker.MultipleLocator(.005))
        ax_ar[ia].yaxis.set_minor_locator(mticker.MultipleLocator(.5))
        ax_ar[ia].tick_params(axis='both', which='both',
                              direction='in',
                              top=True, right=True,
                              labelsize=7)
        ax_ar[ia].set_xlabel("1000/RT [mol·kJ$^{-1}$]", fontsize=7)
        ylab = ("Ln(" + lab_rate[v] + "$_" + lab_age[a].replace("$", "") + "$)"
                " [$\mathregular{\mu}$mol·kg$^{-1}$·yr$^{-1}$]")
        ax_ar[ia].set_ylabel(ylab, fontsize=7)

    # Adjust spacing between subplots
    fig_ar.subplots_adjust(wspace=.3)

    # Save figure
    fpath = ('figures/hansell_glodap/global/parameterisations/' +
             'hansell_glodap_' + v + '_rate_arrhenius_plots_ALL.pdf')
    fig_ar.savefig(fpath, format='pdf', bbox_inches='tight')
    

#%%% ARRHENIUS PLOTS: PER OCEAN

#%%%% REGRESSION

oceans = reg1.ocean.unique()
depths = reg1.depth.unique()
wms = reg1.water_mass.unique()

rate_reg_ocean = []
for v in vrs:
    for a in ages:
        for nv in new_vrs:
            for o in oceans:
                for s in [-1, 1]:
                    
                    # Subset values for variable v, age a, and sign s
                    idx = ((reg1['y_var']==v) &
                           (reg1['x_tracer']==a) &
                           (reg1['ocean']==o) &
                           (reg1['rate_sign']==s))
                    df = reg1.loc[idx, :].copy()
                    
                    # Filter only significant rates
                    idx = ((df['spvalue'] < pval_limit) &
                           (df['r2_adj'] > r2_limit))
                    df = df.loc[idx, :]
                        
                    # Do regression
                    min_obs = 5 # set minimum obs to perform regression
                    n_obs = sum(~np.isnan(df['ln_rate']))
                    if n_obs < min_obs: 
                        msg = ("not enough observations (" + str(n_obs) + ")")
                        tmp = [v,
                               a,
                               nv,
                               o,
                               s,
                               *[np.nan] * 15,
                               msg
                               ]
                    else:
                        r = rc.RegressConsensusW(df[nv],
                                                 df['ln_rate'],
                                                 Wx=.5)
                        tmp = [v,
                               a,
                               nv,
                               o,
                               s,
                               *r.values(),
                               ''
                               ]
                    
                    # Append to previous results
                    rate_reg_ocean.append(tmp)
                    
                    
                    msg = ("Done: " + nv + " vs Ln(" + v + " rate)" +  
                           " [" + signchar(s) + ", " + a + ", " + o + "]\n"
                           "-------------------------------------------------------------")
                    print(msg)


nms = ['y_var', 'tracer_age', 'x_var', 'ocean', 'rate_sign', *r.keys(), 
       'comment']
rate_reg_ocean = pd.DataFrame(rate_reg_ocean, columns=nms)


#%%%% PLOT


nr = len(oceans)
nc = len(ages)
pal_ocean = {k:v for k, v in zip(oceans, ['#44AA99', '#88CCEE', '#EE8866'])}
pal_ages = {k:v for k, v in zip(ages, ['#DDAA33','#BB5566', '#004488'])}

for iv, v in enumerate(vrs):
    
    fig_aro, ax_aro = plt.subplots(nrows=nr, ncols=nc, 
                                 figsize=(7*cm*nc, 6*cm*nr))
    
    for ia, a in enumerate(ages):
        for io, o in enumerate(oceans):
            for i, s in enumerate([-1, 1]):
        
                # Do not plot rates that represent increase of O2 with age
                # (negative rates with AOU or positive rates with OXY)
                if (((v=='AOU_RES') & (s==-1)) | ((v=='OXYGEN_RES') & (s==1))):
                    continue
                
                # subset data
                idx = ((reg1['y_var']==v) &
                       (reg1['x_tracer']==a) &
                       (reg1['ocean']==o) &
                       (reg1['rate_sign']==s))
                df = reg1.loc[idx, :].copy()
                
                # Filter only significant rates
                idx = ((df['spvalue'] < pval_limit) &
                       (df['r2_adj'] > r2_limit))
                df = df.loc[idx, :]
                
                # Get regression data
                idx = ((rate_reg_ocean['y_var']==v) &
                       (rate_reg_ocean['tracer_age']==a) &
                       (rate_reg_ocean['ocean']==o) &
                       (rate_reg_ocean['rate_sign']==s) &
                       (rate_reg_ocean['x_var']=='1000_RPT'))
                rg = rate_reg_ocean.loc[idx, :].copy()
                
                # Plot points
                ax_aro[io, ia].scatter(df['1000_RPT'], df['ln_rate'],
                                       marker=mar_sign[s],
                                       s=15,
                                       c='w',
                                       edgecolor='#222',
                                       linewidth=.8)
                
                # Plot regression line if relationship is significant
                txt_ypos = .96 - i * .08 if v=='DOC_RES' else .96
                if ((rg['spvalue'] < pval_limit) & (rg['r2_adj'] > r2_limit)).squeeze():
                    
                    # Get the line coords within the data range and plot it
                    x0 = np.nanmin(df['1000_RPT'])
                    x1 = np.nanmax(df['1000_RPT'])
                    slp = rg['slope'].squeeze()
                    intr = rg['intercept'].squeeze()
                    y0 = x0 * slp + intr
                    y1 = x1 * slp + intr
                    ax_aro[io, ia].plot([x0, x1], [y0, y1],
                                        color='#222',
                                        linestyle='--', linewidth=1)
                    
                    # Create text string to add equation to plot
                    r2_adj = rg['r2_adj'].squeeze()
                    p = rg['spvalue'].squeeze()
                    if p < .001:
                        ptxt = "< 0.001"
                    elif p < .01:
                        ptxt = "< 0.01"
                    else:
                        ptxt = ("= " + str(round(p, 2)))
                    txt = ("Ln(" + lab_rate[v] + "$_" + 
                           lab_age[a].replace("$", "") + 
                           "$ " + signchar(s) + ")" +
                           " = " + str(round(slp, 1)) + 
                           " $\\times$ 1000/RT + " +
                           str(round(intr, 1)) + "\nR$^{2}$ = " + 
                           "{:.2f}".format(round(rg['r2_adj'].squeeze(), 2)) + 
                           "; $\it{p}$ " + 
                           ptxt)
                    
                    # Add text to plot
                    ax_aro[io, ia].text(x=.97, y=txt_ypos, s=txt,
                                        size=4, c='#222',
                                        linespacing=1,
                                        ha='right', va='top',
                                        transform=ax_aro[io, ia].transAxes)
                else:
                    # if not significant, add text indicating that
                    txt = ("Ln(" + lab_rate[v] + "$_" + 
                           lab_age[a].replace("$", "") + "$ " + signchar(s) + ")"
                           " $\u2192$ ns")
                    # txt_ypos = .05 + i * .05
                    ax_aro[io, ia].text(x=.97, y=txt_ypos, s=txt,
                                        size=4, c='#222',
                                        linespacing=1,
                                        ha='right', va='top',
                                        transform=ax_aro[io, ia].transAxes)
        
            ## customise axes
            # set limits dynamically
            # re-subset data to consider limits of both + & - (when valid), 
            # and ages, together
            if v=='AOU_RES':
                valid_signs = [1]
            elif v=='OXYGEN_RES':
                valid_signs = [-1]
            else:
                valid_signs = [-1, 1]
                
            idx = ((reg1['y_var']==v) &
                   (reg1['rate_sign'].isin(valid_signs)) &
                   (reg1['spvalue'] < pval_limit) &
                   (reg1['r2_adj'] > r2_limit))
            dfxy = reg1.loc[idx, :]
            xmin = np.floor(np.nanmin(dfxy['1000_RPT']) / .01) * .01
            xmax = np.ceil(np.nanmax(dfxy['1000_RPT']) / .01) * .01
            ymin = np.floor(np.nanmin(dfxy['ln_rate']) / .5) * .5
            ymax = np.ceil(np.nanmax(dfxy['ln_rate']) / .5) * .5
            ax_aro[io, ia].set(xlim=[xmin, xmax],
                               xticks=np.arange(xmin, xmax + .01, .01),
                               ylim=[ymin, ymax],
                               yticks=np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1))
            ax_aro[io, ia].xaxis.set_minor_locator(mticker.MultipleLocator(.005))
            ax_aro[io, ia].yaxis.set_minor_locator(mticker.MultipleLocator(.5))
            ax_aro[io, ia].tick_params(axis='both', which='both',
                                       direction='in',
                                       top=True, right=True,
                                       labelsize=7)
            if io==(nr-1):
                ax_aro[io, ia].set_xlabel("1000/RT [mol·kJ$^{-1}$]",
                                          fontsize=7)
            if ia==0:
                ylab = ("Ln(" + lab_rate[v] + ")" +
                        " [$\mathregular{\mu}$mol·kg$^{-1}$·yr$^{-1}$]")
                ax_aro[io, ia].set_ylabel(ylab, fontsize=7)
                
            if io==0:
                ax_aro[io, ia].text(.5, 1.05, lab_age[a],
                                    fontsize=8, fontweight='bold',
                                    color=pal_ages[a],
                                    path_effects=[pe.withStroke(linewidth=.6, 
                                                                foreground='k')],
                                    va='center', ha='center',
                                    transform=ax_aro[io, ia].transAxes)
            if ia==(nc-1):
                ax_aro[io, ia].text(1.05, .5, o.capitalize(),
                                    fontsize=8, fontweight='bold',
                                    color=pal_ocean[o],
                                    path_effects=[pe.withStroke(linewidth=.6, 
                                                                foreground='k')],
                                    va='center', ha='center',
                                    rotation=270,
                                    transform=ax_aro[io, ia].transAxes)

    # adjust spacing between subplots
    fig_aro.subplots_adjust(wspace=.2, hspace=.15)

    # save figure
    fpath = ('figures/hansell_glodap/global/parameterisations/' +
             'hansell_glodap_' + v + '_rate_arrhenius_plots_per_ocean.pdf')
    fig_aro.savefig(fpath, format='pdf', bbox_inches='tight')

    


    
#%%% Ea & Q10

# Calculate:
#     - Activation energy: slope of the 1000/RT vs Ln(rate) regression. Can 
#           also be estimated as Ea=−slope×R, where slope is the slope of the 
#           1/T vs Ln(rate) regression
#     - Q10 as: exp((-10*slope)/(T1*T2), were slope is the slope of the 1/T vs 
#           Ln(rate) regression. Use 10ºC as TempRef

# set T1 and T2
tr = 273.15 + 10
t1 = tr - 5
t2 = tr + 5

for v in vrs:
    for a in ages:
        for s in [-1, 1]:
            
            # Get the inv_pt vs ln_our regression for variable v, tracer a and
            # sign s
            idx = ((rate_reg['x_var']=='inv_PT') &
                   (rate_reg['y_var']==v) &
                   (rate_reg['tracer_age']==a) &
                   (rate_reg['rate_sign']==s))
            slp = rate_reg.loc[idx, 'slope'].squeeze()
            
            # Estimate Ea and Q10
            ea = -slp * 8.314
            q10 = np.exp((ea / 8.314) * (10 / (t1 * t2)))
            
            # give Ea in kJ/mol -> /1000
            print("------------------------------------------------")
            print("Ea (" + v + ", " + a + ", " + signchar(s) + ") = " + 
                  str(round(ea/1000, 2)) + " kJ/mol")
            print("Q10 (" + v + ", " + a + ", " + signchar(s) + ") = " + 
                  str(round(q10, 2)))


#%% OUR, DOCUR VS VARIABLES

#%%% regressions

## Regressions of OUR, DOCURs against environmental variables

ev = ['DOC', 'AOU', 'NITRATE', 'CTD_PRESSURE', 'SIGMA0', 'AGE']
reg_ev = []
for v in vrs:
    for e in ev:
        
        track_e = e # keep record of the current e
                    # (important for e='AGE', see below)
        
        for ia, a in enumerate(ages):
            
            # only regress tracer ages against the OUR/DOCOUR estimated based
            # on that tracer age.
            if track_e=='AGE':
                e = a.replace("_RES", "")
                
            for s in [-1, 1]:
                
                # Subset data of rates for v, tracer a and sign s
                idx = ((reg1['y_var']==v) &
                       (reg1['x_tracer']==a) &
                       (reg1['rate_sign']==s))
                ss = reg1.loc[idx, :].copy()
                
                # Filter only significant rates
                ss = ss.loc[ss['spvalue']<.05, :]
                
                # Do regression
                min_obs = 4 # set minimum obs to perform regression
                n_obs = sum(~np.isnan(ss['slope']))
                if n_obs < min_obs: 
                    msg = ("not enough observations (" + str(n_obs) + ")")
                    tmp = [v,
                           e,
                           a,
                           s,
                           *[np.nan] * 15, # (15 = len of regresscons output)
                           msg             # add comment of too few obs
                           ]
                else:
                    r = rc.RegressConsensusW(ss[e], ss['slope'], Wx=.5)
                    tmp = [v,           # y variable (rate of the regression)
                           e,           # tracer age from which v was derived
                           a,           # x variable of the regression
                           s,           # sign of rate
                           *r.values(), # all regression results
                           np.nan       # no comment
                           ]
                
                # Store results
                reg_ev.append(tmp)
                
                msg = ("Done: " + e + " vs " + v + " rate" +  
                       " [" + signchar(s) + ", " + a + "]\n"
                       "----------------------------------------------------")
                print(msg)

# convert to dataframe
nms = ['y_var', 'x_var', 'tracer_age', 'rate_sign', *r.keys(), "comment"]
reg_ev = pd.DataFrame(reg_ev, columns=nms)


#%%% plots

# Set axis parameters for each variable
yaxpar = {'AOU_RES': [[-30, 45],          # y limits
                      range(-30, 50, 10), # y ticks
                      5,                  # y minor tick multiple
                      "aOUR [$\mathregular{\mu}$mol·kg$^{-1}$·yr$^{-1}$]" # ylab
                      ],
          'DOC_RES': [[-3, 2.5],
                      np.arange(-3, 3, 1),
                      .5,
                      "aDOCUR [$\mathregular{\mu}$mol·kg$^{-1}$·yr$^{-1}$]"
                      ]
          }
xaxpar = {'DOC': [[38, 52],           # x limits
                  range(38, 54, 4),   # x ticks
                  2,                  # x minor tick multiple
                  "DOC [$\mathregular{\mu}$mol·kg$^{-1}$]" # x axis label
                            ],
          'AOU': [[10, 210],
                  range(10, 250, 40),
                  20,
                  "AOU [$\mathregular{\mu}$mol·kg$^{-1}$]"],
          'NITRATE': [[4, 36],
                      range(4, 44, 8),
                      4,
                      "NO$_3^{-}$ [$\mathregular{\mu}$mol·kg$^{-1}$]"],
          'CTD_PRESSURE': [[100, 1400],
                           range(100, 1500, 200),
                           100,
                           "Depth [dbar]"],
          'SIGMA0': [[26.2, 27.8],
                     np.arange(26.2, 27.8, .4),
                     .2,
                     "$\sigma_\\theta$ [g·kg$^{-1}$]"
                     ],
          'AGE_CFC11': [[0, 70],
                        range(0, 80, 20),
                        10,
                        "Age$_{\mathregular{CFC\u201011}}$ [y]"],
          'AGE_CFC12': [[0, 70],
                        range(0, 80, 20),
                        10,
                        "Age$_{\mathregular{CFC\u201012}}$ [y]"],
          'AGE_SF6': [[0, 70],
                      range(0, 80, 20),
                      10,
                      "Age$_{\mathregular{SF_6}}$ [y]"]
          }


# subscripts to dynamically modify y axis label (add tracer age tag)
sscript = {'AGE_CFC11_RES': "$_{\mathregular{CFC\u201011}}$",
           'AGE_CFC12_RES': "$_{\mathregular{CFC\u201012}}$",
           'AGE_SF6_RES': "$_{\mathregular{SF_6}}$"}

txt_nline = 1
for v in vrs:
    for e in ev:

        fig_ev, ax_ev = plt.subplots(nrows=1, ncols=len(ages),
                                     figsize=(7*cm * len(ages), 6*cm))
        track_e = e
    
        for ia, a in enumerate(ages):
            
            if track_e=='AGE':
                e = a.replace("_RES", "")
                
            xl, xtck, xtck_mi, xlab = xaxpar[e]
            yl, ytck, ytck_mi, ylab = yaxpar[v]
            
            # modify ylab for each tracer age
            ylab = ylab.replace("UR ", ("UR" + sscript[a] + " "))
                
            for s in [-1, 1]:
                
                # Subset data of rates for v, tracer a and sign s
                idx = ((reg1['y_var']==v) &
                       (reg1['x_tracer']==a) &
                       (reg1['rate_sign']==s))
                ss = reg1.loc[idx, :].copy()
                
                # Filter only significant rates
                ss = ss.loc[ss['spvalue']<.05, :]
                
                # Get regression data
                idx = ((reg_ev['y_var']==v) &
                       (reg_ev['x_var']==e) &
                       (reg_ev['tracer_age']==a) &
                       (reg_ev['rate_sign']==s))
                rev = reg_ev.loc[idx, :]
                
                # plot points
                ax_ev[ia].scatter(ss[e],
                                  ss['slope'],
                                  marker=mar_sign[s],
                                  s=15,
                                  c=pal_sign[s],
                                  edgecolor="#222",
                                  linewidth=.5)
                
                # plot regression line if significant
                if rev['spvalue'].squeeze()<.05:
                    
                    # compute line for range of data
                    x0 = np.nanmin(ss[e])
                    x1 = np.nanmax(ss[e])
                    slp = rev['slope'].squeeze()
                    intr = rev['intercept'].squeeze()
                    y0 = x0 * slp + intr
                    y1 = x1 * slp + intr
                    
                    # plot line
                    ax_ev[ia].plot([x0, x1], [y0, y1],
                                   color=pal_sign_2[s],
                                   linestyle='--', linewidth=1)
                    
                    # create text string to add equation to plot
                    r2_adj = rev['r2_adj'].squeeze()
                    p = rev['spvalue'].squeeze()
                    ptxt = "< .01" if p<.01 else ("= " + str(round(p, 2)))
                    # round slp and intr adequantly according to magnitude
                    slp_mag = np.floor(np.log10(abs(slp)))
                    if slp_mag<3:
                        round_decimals = int(2 - slp_mag)
                    else:
                        round_decimals = 0 # if large, limit rounding
                    rslp = round(slp, round_decimals)
                    rintr = round(intr, round_decimals)
                    txt = (lab_rate[v] + "$_" + 
                           lab_age[a].replace("$", "") + "$ [" + 
                           signchar(s) + "]"
                           " = " + str(rslp) + " $\\times$ " + e + 
                           " " + signchar(np.sign(rintr)) + " " +
                           str(abs(rintr)) + 
                           "\nR$^{2}$ = " + 
                           str(round(rev['r2_adj'].squeeze(), 2)) + 
                           "; $\it{p}$ " + ptxt)
                else:
                    # if not significant, add text indicating that
                    txt = (lab_rate[v] + "$_" +
                           lab_age[a].replace("$", "") + "$ [" +
                           signchar(s) + "] $\u2192$ ns")
                
                # add equation to plot
                txt_ypos = .1 * (s + 1)/2 + .04 #+ txt_nline
                ax_ev[ia].text(x=.05, y=txt_ypos, s=txt,
                               size=4, c=pal_sign_2[s],
                               linespacing=1,
                               transform=ax_ev[ia].transAxes)
                txt_nline = txt.count("\n") + 1 # count no. of lines of txt
                
                # set axes
                ax_ev[ia].set(xlim=xl,
                              xticks=xtck,
                              ylim=yl,
                              yticks=ytck)
                ax_ev[ia].xaxis.set_minor_locator(mticker.MultipleLocator(xtck_mi))
                ax_ev[ia].yaxis.set_minor_locator(mticker.MultipleLocator(ytck_mi))
                ax_ev[ia].set_xlabel(xlab, fontsize=7)
                ax_ev[ia].set_ylabel(ylab, fontsize=7)
                ax_ev[ia].tick_params(axis='both', which='both',
                                      direction='in',
                                      labelsize=7,
                                      top=True, right=True)
            
            # adjust spacing between subplots
            fig_ev.subplots_adjust(wspace=.35)
                        
        
        fpath = ("figures/dom_hansell/A16/parameterisations/" +
                 "hansell2022_a16_2013_" + v + "_vs_" + track_e + ".pdf")
        fig_ev.savefig(fpath, format='pdf', bbox_inches='tight')



