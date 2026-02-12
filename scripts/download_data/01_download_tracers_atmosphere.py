# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:35:23 2025ç

Download atmospheric histories of tracer gases.

https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0164584


@author: Markel
"""

#%% IMPORTS

from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import tarfile
import os
import sys

#%% SETUP

local_dir = Path("rawdata") / "tracers_atmosphere"
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

# Timeout (seconds) for socket operations (connect and read). This is per-socket-operation,
# not a total-download timeout. Increase if the server is slow.
timeout_seconds = 100




def download_file(url: str, dest: Path, force: bool = False, timeout: int = 30):
    """
    Download `url` to `dest` using urllib (stdlib only).
    - Streams in chunks to avoid loading whole file into memory.
    - Uses a temporary .part file and atomically replaces the final file.
    - Handles common exceptions (HTTPError, URLError, timeout).
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


#%% DOWNLOAD & EXTRACT

url = "https://www.ncei.noaa.gov/archive/archive-management-system/OAS/bin/prd/jquery/download/164584.2.2.tar.gz"
fpath_local = local_dir / "164584.2.2.tar.gz"

try:
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)
except Exception as e:
    print(f"Failed to download {url}: {e}", file=sys.stderr)
    raise

# Extract contents (no path-traversal checks as requested)
try:
    with tarfile.open(fpath_local, "r:gz") as tar:
        tar.extractall(path=str(local_dir))
    print(f"Extracted tarball into {local_dir}")
except Exception as e:
    print(f"Failed to extract {fpath_local}: {e}", file=sys.stderr)
    raise