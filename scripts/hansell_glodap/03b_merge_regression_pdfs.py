# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:27:48 2025

Merge individual pds files with regression results for each pixel into a 
single file.

https://stackoverflow.com/questions/3444645/merge-pdf-files


@author: Markel
"""

#%% IMPORTS

import os
from pypdf import PdfWriter


#%% MERGING

# Get list of files
dpath = 'figures/hansell_glodap/global/regressions'
flist = os.listdir(dpath)

# Only include AOU rate regressions
rate_str = ['_AOU_RES_AGE']
pdfs_to_merge = [any([s in x for s in rate_str]) for x in flist]
pdfs_to_merge = [x for x, y in zip(flist, pdfs_to_merge) if y]
pdfs_to_merge = [dpath + '/' + x for x in pdfs_to_merge]

merger = PdfWriter()
for pdf in pdfs_to_merge:
    merger.append(pdf)

out = dpath + '/Supplementary_file_all_pixel_regressions.pdf' 
merger.write(out)
merger.close()

