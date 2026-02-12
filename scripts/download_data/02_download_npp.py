# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:08:42 2025

Download NPP and associated SST data from:
http://orca.science.oregonstate.edu/npp_products.php

Eppley-VGPM and CbPM models, monthly data, 2160 x 4320, MODIS R2022 data.

Data from 2003-2021 year, i.e., the years that are complete, with no month
missing, so as to not bias the climatology. 2022 and 2023 have data but are
missing a month.


@author: Markel
"""

#%% IMPORTS

from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import ssl
import sys


#%% SETUP

# When downloading, check whether file already exists to avoid unnecessarily 
# downloading  files repeatedly.         
# 
# But, also give option to force download and overwrite in case this 
# is desired
#
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

force_download_and_overwrite = False

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

# Set desired timeout of connections when downloading
timeout_seconds = 100


# Server and local parents
server_parent = "http://orca.science.oregonstate.edu/data/2x4/monthly/"
local_parent = Path("rawdata") / "npp"

# Set models to download, and required codes to fill paths etc.:
mods = {'cbpm': {'url_subdir': 'cbpm2.modis.r2022',
                 'url_fname_code': 'cbpm.m.',
                 'local_dir': 'cbpm/'},
        'eppley': {'url_subdir': 'eppley.r2022.m.chl.m.sst',
                   'url_fname_code': 'eppley.m.',
                   'local_dir': 'eppley-vgpm/'}}

# Set years to download data from
years = range(2003, 2022) # this creates 2003...2021

# Create necessary local directories
for m in mods.values():
    (local_parent / m["local_dir"]).mkdir(parents=True, exist_ok=True)


# It seems that Oregon State University's webpage currently has outdated
# certificates. This might be fixed in the future. As a workaround, temporarily 
# disable SSL verification if needed.

def detect_ssl_issue_and_get_context(test_url: str = "https://orca.science.oregonstate.edu/npp_products.php",
                                     timeout: int = 20):
    """
    Try to open test_url. If a URLError occurs that appears to be due to SSL
    certificate problems, return an unverified SSL context; otherwise return None.
    """
    try:
        # Try with default context first
        urllib.request.urlopen(test_url, timeout=timeout)
        return None
    except urllib.error.URLError:
        # If the error appears SSL-related (certificate verification), return unverified context        
        try:
            return ssl._create_unverified_context()
        except Exception:
            return None
    except Exception:
        return None


def download_file(url: str, dest: Path, force: bool = False, timeout: int = 60, context=None):
    """
    Download `url` to `dest`.
    - Streams in chunks to avoid loading whole file into memory.
    - Uses a temporary .part file and atomically replaces the final file.
    - Handles common exceptions (HTTPError, URLError, timeout).
    - context: optional ssl.SSLContext to pass to urlopen (for unverified SSL).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Skipping existing file: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    #print(f"Starting download: {url} -> {dest}")
    try:
        # Use context parameter only if not None (urlopen accepts context=...)
        if context is not None:
            resp = urllib.request.urlopen(url, timeout=timeout, context=context)
        else:
            resp = urllib.request.urlopen(url, timeout=timeout)

        with resp:
            # Optionally, check HTTP status code
            code = getattr(resp, "getcode", lambda: None)()
            if code is not None and code >= 400:
                raise urllib.error.HTTPError(url, code, "HTTP error", hdrs=None, fp=None)

            with tmp.open("wb") as out_f:
                # shutil.copyfileobj will stream in chunks
                shutil.copyfileobj(resp, out_f)

        tmp.replace(dest)
        print(f"Downloaded: {dest}")
        
    except urllib.error.HTTPError:
        # Clean up partial file
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    except urllib.error.URLError:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    except socket.timeout:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise


#%% SSL handling

# Get ssl context
ssl_context = detect_ssl_issue_and_get_context()


#%% DOWNLOAD DATA

for mname, m in mods.items():
    subdir = m["url_subdir"]
    fname_code = m["url_fname_code"]
    local_dir = local_parent / m["local_dir"]

    for y in years:
        fname = f"{fname_code}{y}.tar"
        fpath_local = local_dir / fname
        url = f"{server_parent}{subdir}/hdf/{fname}"

        try:
            download_file(url, fpath_local, force=force_download_and_overwrite,
                          timeout=timeout_seconds, context=ssl_context)
        except Exception as exc:
            # Don't stop the whole loop on single failure; report and continue
            print(f"Failed to download {url}: {exc}", file=sys.stderr)
            continue
