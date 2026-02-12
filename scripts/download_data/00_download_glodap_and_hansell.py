# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:23:08 2025

Download GLODAP dataset from:
https://glodap.info/index.php/merged-and-adjusted-data-product-v2-2023/

Download GLODAP dataset from:
https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0227166/

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

# Create necessary directories
glodap_dir = Path("rawdata") / "glodap"
hansell_dir = Path("rawdata") / "dom_hansell"
glodap_dir.mkdir(parents=True, exist_ok=True)
hansell_dir.mkdir(parents=True, exist_ok=True)


# When downloading, check whether file already exists to avoid unnecessarily 
# downloading files repeatedly.         
# 
# But, also give option to force download and overwrite in case this 
# is desired
#
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

force_download_and_overwrite = False

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

# Set desired timeout of connections when downloading
timeout_seconds = 100


# Create function to download files
def download_file(url: str, dest: Path, force: bool = False, timeout: int = 60):
    """
    Download `url` to `dest`.
    - Streams in chunks to avoid loading whole file into memory.
    - Uses a temporary .part file and atomically replaces the final file.
    - Handles common exceptions (HTTPError, URLError, timeout).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Skipping existing file: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            # Optional: check HTTP status via resp.getcode()
            code = getattr(resp, "getcode", lambda: None)()
            if code is not None and code >= 400:
                raise urllib.error.HTTPError(url, code, "HTTP error", hdrs=None, fp=None)

            with tmp.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f)  # streams efficiently
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
        # URLError wraps socket.timeout and other connection errors
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


#%% DOWNLOAD GLODAP


try:
    # Set paths in server and local
    url = "https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_Merged_Master_File.csv.zip"
    fpath_local = glodap_dir / "GLODAPv2.2023_Merged_Master_File.csv.zip"
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)

    # Ancillary files
    url = "https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_EXPOCODES.txt"
    fpath_local = glodap_dir / "GLODAPv2.2023_EXPOCODES.txt"
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)

    url = "https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_DOIs.csv"
    fpath_local = glodap_dir / "GLODAPv2.2023_DOIs.csv"
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)
except Exception as e:
    print(f"Error downloading GLODAP files: {e}", file=sys.stderr)
    




#%% DOWNLOAD HANSELL DOM

try:
    url = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0227166/All_Basins_Data_Merged_Hansell_2022.xlsx"
    fpath_local = hansell_dir / "All_Basins_Data_Merged_Hansell_2022.xlsx"
    download_file(url, fpath_local, force=force_download_and_overwrite, timeout=timeout_seconds)
except Exception as e:
    print(f"Error downloading Hansell file: {e}", file=sys.stderr)