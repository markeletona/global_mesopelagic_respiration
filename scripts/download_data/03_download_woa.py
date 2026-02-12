# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:32:16 2025

Download World Ocean Atlas temperature climatological fields.

https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23.pl

It will be used to estimate b values following Marsay et al. (2015), Fig. 2A

@author: Markel
"""

#%% IMPORTS

from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import sys


#%% SETUP

local_dir = Path("rawdata") / "woa"
local_dir.mkdir(parents=True, exist_ok=True)

# When downloading, check whether file already exists to avoid unnecessarily 
# downloading files repeatedly.         
# 
# But, also give option to force download and overwrite in case this 
# is desired
#
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

force_download_and_overwrite = False

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

# Timeout (seconds) for socket operations (connect and read). This is a per-socket-operation
# timeout, not a total-download timeout. Increase if the server is slow.
timeout_seconds = 100



def download_file(url: str, dest: Path, force: bool = False, timeout: int = 30):
    """
    Download `url` to `dest` using urllib (stdlib only).
    - Streams in chunks to avoid loading whole file into memory.
    - Writes to a temporary .part file and atomically replaces the final file.
    - Cleans up partial files on error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Skipping existing file: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Starting download: {url} -> {dest}")
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        with resp:
            code = getattr(resp, "getcode", lambda: None)()
            if code is not None and code >= 400:
                raise urllib.error.HTTPError(url, code, "HTTP error", hdrs=None, fp=None)

            with tmp.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f)

        tmp.replace(dest)
        print(f"Downloaded: {dest}")
    except urllib.error.HTTPError:
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


#%% DOWNLOAD

# Grid ---> 1º
# Field --> statistical mean
# Period -> averaged decades

# Set paths in server and local
url = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/csv/decav/1.00/woa23_decav_t00mn01.csv.gz"
fpath_local = local_dir / "woa23_decav_t00mn01.csv.gz"

try:
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)
except Exception as e:
    print(f"Failed to download {url}: {e}", file=sys.stderr)
    raise
