#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:25:34 2025
generate daily MIRA-files with cloudtop height + selected variables
@author: clerx
"""

#%% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import os, glob, np, xr
from src.utils import generate_date_range
from src.radar_processing import preprocess, preprocess_MIRA
from scipy.ndimage import label

def cloudtop_height(mask, n_consec=6, dim='range'):
    """
    Find the highest 'range' coordinate below which there are at least
    n_consecutive True values along `dim`.
    Works for 1D or 2D DataArrays.
    """
    mask_np = mask.values  # convert to numpy array
    coords_range = mask[dim].values

    if mask_np.ndim == 1:
        labeled, num_features = label(mask_np)
        max_idx = -1
        for i in range(1, num_features + 1):
            inds = np.where(labeled == i)[0]
            if len(inds) >= n_consec:
                max_idx = max(max_idx, inds[-1])
        return coords_range[max_idx] if max_idx >= 0 else np.nan

    elif mask_np.ndim == 2:
        results = []
        for row in mask_np:
            labeled, num_features = label(row)
            max_idx = -1
            for i in range(1, num_features + 1):
                inds = np.where(labeled == i)[0]
                if len(inds) >= n_consec:
                    max_idx = max(max_idx, inds[-1])
            results.append(coords_range[max_idx] if max_idx >= 0 else np.nan)
        return xr.DataArray(results, dims=[mask.dims[0]], coords={mask.dims[0]: mask.coords[mask.dims[0]]})

    else:
        raise NotImplementedError("Only 1D or 2D arrays supported")

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024'
MIRA_dir = os.path.join(base_dir, 'MIRA/L1')
ERA_dir = os.path.join(base_dir, 'ERA5/netcdf')
peaktree_dir = os.path.join(base_dir, 'peaktree')


#%% dates
startdate = '2024-10-18'
enddate = '2025-01-24'

dates = generate_date_range(startdate, enddate)

output_campaign = f"{MIRA_dir}/full_campaign.nc"
vars_to_keep = ['Z', 'VELg', 'SNRg', 'LDRg', 'RHO', 'SKWg', 'RR', 'NPKg', 'RMSg']
vars_mmclx = ['nfft', 'prf', 'NyquistVelocity', 'nave', 'ovl',  'zrg', 'rg0', 'drg', 
              'lambda', 'microsec', 'tpow', 'npw1', 'npw2', 'cpw1', 'cpw2', 'grst',
              'azi', 'elv', 'aziv', 'northangle', 'elvv', 'LO_Frequency', 'DetuneFine',
              'SNR', 'VEL', 'RMS', 'LDR', 'NPK', 'SNRg', 'VELg', 'RMSg', 'LDRg', 'NPKg',
              'SNRplank', 'VELplank', 'RMSplank', 'LDRplank', 'NPKplank', 'SNRrain',
              'VELrain', 'RMSrain', 'LDRrain', 'NPKrain', 'SNRcl', 'VELcl', 'RMScl', 
              'LDRcl', 'NPKcl', 'SNRice', 'VELice', 'RMSice', 'LDRice', 'NPKice', 'RHO',
              'RHOwav', 'DPS', 'DPSwav', 'HSDco', 'HSDcx', 'Ze', 'Zg', 'Z', 'RR', 'LWC',
              'TEMP', 'MeltHei', 'MeltHeiDet', 'MeltHeiDB', 'ISDRco', 'ISDRcx', 'MRMco',
              'MRMcx', 'RadarConst', 'SNRCorFaCo', 'SNRCorFaCx']


for i, date in enumerate(dates):
    year, month, day = date.year, date.month, date.day
    outname = f"{MIRA_dir}/{year}{month:02}{day:02}_small.nc"
    if not os.path.exists(outname):
        MIRA_filedir = os.path.join(MIRA_dir, f"{year}/{month:02}/{day:02}")
        MIRA_files = sorted(glob.glob(f"{MIRA_filedir}/*_merged.nc"))
        if MIRA_files:
            mira = preprocess_MIRA(xr.open_mfdataset(MIRA_files, combine='by_coords', 
                                                     chunks={'time': 720, 'range': 50, 'doppler': 64},
                                                     preprocess=preprocess, parallel=True))
            print(f"opened files for {year}-{month:02}-{day:02}")
            cloudtop = cloudtop_height(mira['Zg'] > -35, n_consec=4)
            mira_small = mira[vars_to_keep]
            mira_small = mira_small.assign(cloud_top_height = cloudtop)
            encoding = {}
            for var in mira_small.data_vars:
                dims = mira_small[var].dims
                if len(dims) == 1:
                    encoding[var] = {'zlib': True, 'complevel': 1, 'chunksizes': (720,)}
                elif len(dims) == 2:
                    encoding[var] = {'zlib': True, 'complevel': 1, 'chunksizes': (720, 50)}
                elif len(dims) == 3:
                    encoding[var] = {'zlib': True, 'complevel': 1, 'chunksizes': (720, 50, 64)}
            mira_small.to_netcdf(outname, encoding=encoding)
            mira.close()
            print(f"processed file for {year}-{month:02}-{day:02}")
    else:
        print(f"daily MIRA file for {year}-{month:02}-{day:02} already exists, skipping.")

# add temperature to _small .nc files
for date in dates[1:]:
    year, month, day = date.year, date.month, date.day
    fname = f"{MIRA_dir}/{year}{month:02}{day:02}_small.nc"
    mfiles = sorted(glob.glob(f"{MIRA_dir}/{year}/{month:02}/{day:02}/*_mmclx.nc"))
    print(f"Processing {year:02}-{month:02}-{day:02} with {len(mfiles)} mmclx files...")
    if os.path.exists(fname) and (len(mfiles) > 0):
        ds = xr.open_dataset(fname).load()
        ds.close()
        mmclx = xr.open_mfdataset(mfiles, drop_variables=[v for v in vars_mmclx if v != 'TEMP'])
        ds['TEMP'] = mmclx['TEMP']
        ds.to_netcdf(fname, mode='w')
    else:
        print(f"File {fname} not found, skipping.")
    