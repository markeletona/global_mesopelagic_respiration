# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:31:26 2024

Compute sigma0 ranges for certain waters masses based on temperature and
salinity data from the literature.

@author: Markel
"""

#%% IMPORTS

import gsw


#%% SIGMA0

# WATER_MASS = [[T_MIN, T_MAX], [S_MIN, S_MAX]]

# As CT and SA need coordinates and pressure, and the given T,S values are
# general ones, use them directly and consider results as approximate
def sigma0(ts):
    sgm = [round(gsw.sigma0(ts[1][1], ts[0][1]), 1),
           round(gsw.sigma0(ts[1][0], ts[0][0]), 1)]
    return sgm


#%% PACIFIC

#--------///

# Table 1 from Emery (2001)

# Give T-S diagrams, T-S pairs tend to be 
# colder ~ less-saline and warmer ~ more-saline

WNPCW = [[10.0, 22.0], [34.2, 35.2]]
WNPCW_SIGMA0 = sigma0(WNPCW)

ENPCW = [[12.0, 20.0], [34.2, 35.0]]
ENPCW_SIGMA0 = sigma0(ENPCW)

ENPTW = [[11.0, 20.0], [33.8, 34.3]]
ENPTW_SIGMA0 = sigma0(ENPTW)

PEW = [[7.0, 23.0], [34.5, 36.0]]
PEW_SIGMA0 = sigma0(PEW)

WSPCW = [[6.0, 22.0], [34.5, 35.8]]
WSPCW_SIGMA0 = sigma0(WSPCW)

ESPCW = [[8.0, 24.0], [34.4, 36.4]]
ESPCW_SIGMA0 = sigma0(ESPCW)

ESPTW = [[14.0, 20.0], [34.6, 35.2]]
ESPTW_SIGMA0 = sigma0(ESPTW)

# For PSUW tends to be different:
# colder ~ more-saline and warmer ~ less-saline
PSUW = [[3.0, 15.0], [33.6, 32.6]]
PSUW_SIGMA0 = sigma0(PSUW)


#%% ATLANTIC 

#--------///

# Table 1 from Hinrichsen & Tomczak (1993)

WNACW = [[7.0, 19.0], [35.0, 36.65]]
WNACW_SIGMA0 = sigma0(WNACW)


#--------///

# Table 1 from Emery (2001)
WNACW = [[7.0, 20.0], [35.0, 36.7]]
WNACW_SIGMA0 = sigma0(WNACW)
ENACW = [[8.0, 18.0], [35.2, 36.7]]
ENACW_SIGMA0 = sigma0(ENACW)


#--------///

# Table 1 from Poole & Tomczak (1999)
WSACW = [[6.6, 16.3], [34.4, 35.7]]
WSACW_SIGMA0 = sigma0(WSACW)
ESACW = [[6.0, 14.4], [34.4, 35.3]]
ESACW_SIGMA0 = sigma0(ESACW)


#--------///

# Bashmachnikov et al. (2015)
WNACW = [[7.0, 19.0], [35.1, 36.4]]
WNACW_SIGMA0 = sigma0(WNACW)
ENACW = [[8.5, 19.0], [35.3, 36.5]]
ENACW_SIGMA0 = sigma0(ENACW)

# (Table 2)
SAIW = [[5.6, 5.6], [34.70, 34.70]]
SAIW_SIGMA0 = sigma0(SAIW)

#--------///

# Arhan (1990)
SAIW = [[4.0, 7.0], [34.9, 34.9]]
SAIW_SIGMA0 = sigma0(SAIW)


#--------///

# Table 1 from Garcia-Ibanez et al. (2018)
SPMW8 = [[8, 8], [35.23, 35.23]]
SPMW8_SIGMA0 = sigma0(SPMW8)
SPMW7 = [[7.1, 7.1], [35.16, 35.16]]
SPMW7_SIGMA0 = sigma0(SPMW7)
IrSPMW = [[5, 5], [35.01, 35.01]]
IrSPMW_SIGMA0 = sigma0(IrSPMW)
ENACW = [[12.3, 16], [35.66, 36.2]]
ENACW_SIGMA0 = sigma0(ENACW)


#--------///

# Table 2 from Alvarez et al (2014):
AAIW_A = [[3.02, 5.08], [34.12, 34.14]]
AAIW_A_SIGMA0 = sigma0(AAIW_A)
CDW = [[1.60, 1.60], [34.72, 34.72]]
CDW_SIGMA0 = sigma0(CDW)

#%% INDIAN

#--------///

# Table 1 from Emery (2001)
AAIW = [[2, 10], [33.8, 34.8]]
AAIW_SIGMA0 = sigma0(AAIW)

IIW = [[3.5, 5.5], [34.6, 34.7]]
IIW_SIGMA0 = sigma0(IIW)

#--------///

# From Fig. 12.5 from Tomczak & Godfrey (1994)
# values extract with plotdigitizer
ICW = [[5.0, 17.0], [34.45, 35.65]]
ICW_SIGMA0 = sigma0(ICW)


#%% MEDITERRANEAN

AW = [[14.8, 14.8], [37, 37]]
AW_SIGMA0 = sigma0(AW)

MAW = [[14.740, 14.740], [37.150, 37.150]]
MAW_SIGMA0 = sigma0(MAW)

LSW = [[17.800, 17.800], [39.180, 39.180]]
LSW_SIGMA0 = sigma0(LSW)

EIW = [[14.220, 14.220], [38.840, 38.840]]
EIW_SIGMA0 = sigma0(EIW)

WIW = [[12.934, 12.934], [38.259, 38.259]]
WIW_SIGMA0 = sigma0(WIW)

LIW = [[16.910, 16.910], [39.270, 39.270]]
LIW_SIGMA0 = sigma0(LIW)


WMDW = [[12.840, 12.840], [38.460, 38.460]]
WMDW_SIGMA0 = sigma0(WMDW)

EMDW = [[13.670, 13.570], [38.744, 38.776]]
EMDW_SIGMA0 = sigma0(EMDW)


