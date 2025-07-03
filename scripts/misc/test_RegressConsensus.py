# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:44:19 2023

@author: Markel Gómez Letona

Test the RegressConsensus module.

"""

#%% IMPORTS

import scripts.modules.RegressConsensus as rc
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#%% DATA

# Create dummy data for test
s = 3
i = 2.5
np.random.seed(1905)
x = np.linspace(0, 100, 100) + np.random.normal(20, 10, 100)
y = i + s * np.linspace(0, 100, 100) + np.random.normal(80, 40, 100)

# Check it
def plot_base_scatter(x=x, y=y, xl=10, yl=10):
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(xl*cm, yl*cm))
    ax.plot(x, y, marker='o', markersize=5, 
             linestyle='none', markeredgewidth=1,
             c='none', mec='k')
    ax.set(xlim=[-5, 145], xticks=range(0, 150, 25),
            ylim=[-20, 480], yticks=range(0, 500, 100))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(20))
    ax.tick_params(which='both', axis='both', direction='in', top=True, right=True)
    ax.set_xlabel("x variable [units]")
    ax.set_ylabel("y variable [units]")
    return(fig, ax)

plot_base_scatter()


#%% PERFORM ORDINARY SLR (MODEL I) & COMPARE W/ STATSMODELS


# With RegressConsensus we can perform a model I regression by giving a weight
# of zero to the X axis.
rc_mdi = rc.RegressConsensusW(x, y, Wx=0)
sm_mdi = smf.ols('y ~ x', data={'x': x, 'y': y}).fit()

src = '{0:.6f}'.format(round(rc_mdi['slope'], 6))
sprc = '{0:.6f}'.format(round(rc_mdi['spvalue'], 6))
irc = '{0:.6f}'.format(round(rc_mdi['intercept'], 6))
iprc = '{0:.6f}'.format(round(rc_mdi['ipvalue'], 6))
ssm = '{0:.6f}'.format(round(sm_mdi.params['x'], 6))
ism = '{0:.6f}'.format(round(sm_mdi.params['Intercept'], 6))
spsm = '{0:.6f}'.format(round(sm_mdi.pvalues['x'], 6))
ipsm = '{0:.6f}'.format(round(sm_mdi.pvalues['Intercept'], 6))

print("\n####### Comparison of regression routines #######\n"
      "                 ________________________________\n" +
      "                |                   |            |\n" +
      "                | RegressConsensusW | statsmodel |\n" +
      " _______________|___________________|____________|\n" + 
      "|               |                   |            |\n" +
      "| slope         | " + src  + "          | " + ssm + "   |\n" +
      "| p (slope)     | " + sprc + "          | " + spsm + "   |\n"
      "| intercept     | " + irc + "         | " + ism + "  |\n" +
      "| p (intercept) | " + sprc + "          | " + spsm + "   |\n" + 
      "|_______________|___________________|____________|\n\n")


fig1, ax1 = plot_base_scatter()
x0 = min(x)
x1 = max(x)
y0 = rc_mdi['intercept'] + x0 * rc_mdi['slope']
y1 = rc_mdi['intercept'] + x1 * rc_mdi['slope']
ax1.plot([x0, x1], [y0, y1],
         c='#b0c4de', 
         linewidth=2, dashes=[1, 0])
rc_txt = ("$\mathbf{RegressConsensusW~(W_x = 0):}$\n" + 
          "y = " + irc + " + " + src + " · x\n" +
          "R$^2$ = " + str(round(rc_mdi['r2'], 3)) +
          "; $\mathit{p}$ = " + sprc) 
ax1.text(5, 385, rc_txt,
         size=7, color='#b0c4de')
ax1.plot([x0, x1], [y0, y1],
         c='#eea2ad', 
         linewidth=2, dashes=[1, 1])
sm_txt = ("$\mathbf{statsmodels:}$\n" + 
          "y = " + ism + " + " + ssm + " · x\n" +
          "R$^2$ = " + str(round(sm_mdi.rsquared, 3)) +
          "; $\mathit{p}$ = " + spsm)
ax1.text(65, 0, sm_txt,
         size=7, color='#eea2ad')

fpath = "figures/misc/regressconsensusw_vs_statsmodels.pdf"
fig1.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)


#%% TEST RANGE OF WX

wx = [0, .5, 1]
wx_c = {0: '#d32a2d', .5: '#006666', 1: '#ffa01d'}
rcs = {}
pp = {}
txty = {0: 435, .5: 410, 1: 385}
fig2, ax2 = plot_base_scatter()
for w in wx:
    rcw = rc.RegressConsensusW(x, y, Wx=w)
    x0 = min(x)
    x1 = max(x)
    y0 = rcw['intercept'] + x0 * rcw['slope']
    y1 = rcw['intercept'] + x1 * rcw['slope']
    ax2.plot([x0, x1], [y0, y1],
             c=wx_c[w], 
             linewidth=2, dashes=[4, 2],
             zorder=1)
    txt = ("W$_x$ = " + '{0:.1f}'.format(w) + "  \u2192  y = " + 
           str(round(rcw['intercept'], 1)) + " + " +
           str(round(rcw['slope'], 2)) + " · x")
    ax2.text(5, txty[w], txt,
             size=7,
             color=wx_c[w])
    # Save to add range shade polygon
    rcs[w] = rcw
    pp[str(w) + '_y0'] = y0
    pp[str(w) + '_y1'] = y1

# don't really need to estimate intersection of lines
# pp['cross'] = (rcs[1]['intercept']-rcs[0]['intercept'])/(rcs[0]['slope']-rcs[1]['slope']) 
ypoly = [pp['0_y0'], pp['0_y1'], pp['1_y1'], pp['1_y0']]
xpoly = [min(x), max(x), max(x), min(x)]
ax2.fill(xpoly, ypoly,
         c='#aaa', alpha=.2,
         zorder=0)

fpath = "figures/misc/regressconsensusw_range.pdf"
fig2.savefig(fpath, format='pdf', bbox_inches='tight', transparent=True)
