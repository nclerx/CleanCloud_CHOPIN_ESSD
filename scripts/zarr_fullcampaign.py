#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:25:31 2025

script to create campaign-long zarr files for all radars

@author: clerx
"""
#%% imports & constants
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
from src import os, pd, xr, zarr, datetime
from src.constants_input import common_range as range_bins
from src.constants_input import base_dir, dirs, zarr_variables, att_fn, DFR_fn, BASTAmode, start_date, end_date, radar_ranges, time_bins_radars, BASTAmodes, common_range
from src.utils import get_DFRdata
from src.zarr_utils import generate_zarr_encodings, create_empty_zarr, campaign_to_zarr

# DFRs = get_DFRdata(DFR_files) # outdated - now using the merged file in which MIRA reflectivities are corrected
cloudtopDFRs = pd.read_csv(DFR_fn, index_col=0)

# t_res = '30s'
t_res = '5s'
# t_res = '15s' # for BASTA sensitivity
# t_res = '3s' # for BASTA zarrs
time_bins = pd.date_range(start_date, end_date, freq=t_res) # time bins for the campaign
range_bins = common_range

# radars = ['MIRA', 'BASTA', 'MXPol', 'peaktree']
# radars = ['MIRA', 'MXPol', 'BASTA']
radars = ['MXPol', 'MIRA']
#%% make & fill zarr-files

# zarrdir = f"{dirs['zarr']}/30s_25m"
# zarrdir = f"{dirs['zarr']}/no_resampling"
# zarrdir = f"{dirs['zarr']}/5s_25m"
# zarrdir = f"{dirs['zarr']}/BASTAmodes_5s"
# zarrdir = f"{dirs['zarr']}/BASTAmodes_25m"
zarrdir = f"{dirs['zarr']}/no_resampling_wspectral"
os.makedirs(zarrdir, exist_ok=True)

drop_spectral = False
resample_time = False
resample_range = False
method='linear'

encodings = {}
for radar in radars: 
    encodings[radar] = generate_zarr_encodings(zarr_variables[radar])

precip = pd.read_excel(f"{base_dir}/precip_temp_stats.xlsx", index_col=0)
precip_times = pd.Series(False, index=time_bins)
for _, row in precip.iterrows():
    precip_times.loc[row['start']:row['end']] = True

time_precip = time_bins[precip_times]

# create empty zarr files for each radar and mode (if not already existing)
for radar in radars:
    if not resample_time: 
        time_bins = time_bins_radars[radar]
    if not resample_range:
        range_bins = radar_ranges[radar] if not radar == 'BASTA' else radar_ranges[f"{radar}_{BASTAmode}"]

    # zarr_path = f"{zarrdir}/{radar}.zarr"
    if radar == 'BASTA':
        for mode in BASTAmodes:
            zarr_path = f"{zarrdir}/{radar}_{mode}.zarr"
            if not resample_range:
                range_bins = radar_ranges[f"{radar}_{mode}"]
                if not os.path.exists(zarr_path):
                    print(f"Creating empty zarr for {radar} ({mode} mode)...")
                    ds = create_empty_zarr(zarr_path, radar, time_bins, range_bins, encodings[radar], drop_spectral=drop_spectral)
            else:
                if not os.path.exists(zarr_path):
                    print(f"Creating empty zarr for {radar} ({mode} mode)...")
                    ds = create_empty_zarr(zarr_path, radar, time_bins, range_bins, encodings[radar], drop_spectral=drop_spectral)        
    else:
        zarr_path = f"{zarrdir}/{radar}.zarr"
        if not os.path.exists(zarr_path):
            print(f"Creating empty zarr for {radar}...")
            ds = create_empty_zarr(zarr_path, radar, time_bins, range_bins, encodings[radar], drop_spectral=drop_spectral)        

campaign_to_zarr(pd.date_range(start_date, end_date), precip_times=time_precip, dirs=dirs, DFRs=cloudtopDFRs, att_fn=att_fn, 
                 zarrdir=zarrdir, encodings=encodings, BASTAmode=BASTAmode, radars=radars, interp_method=method,
                 resample_time=resample_time, resample_time_interval=t_res, resample_range=resample_range, drop_spectral=drop_spectral)





