#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:25:34 2025

script that calculates the cloudtop DFR for BASTA-MIRA 
and saves output to daily .csv-files

@author: clerx
"""

#%% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import os, glob, np, pd, xr, datetime, timedelta
from src.constants_input import calibrationvalues, frequencies
from src.utils import generate_date_range, safe_reindex, read_ERA_data, get_DFRdata
from src.radar_processing import preprocess, preprocess_BASTA, preprocess_MIRA, preprocess_MXPol, safe_openmfdataset, correct_gas_attenuation, DFR_offset
from src.plotting import plot_spectrogram_MIRA

#%% dates
startdate = '2024-10-18'
enddate = '2025-01-24'

#%% input data directories & other constants
BASTAmode = '25m'

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024'
MIRA_dir = os.path.join(base_dir, 'MIRA/L1/')
BASTA_dir = os.path.join(base_dir, f'BASTA/L1/{BASTAmode}')
ERA_dir = os.path.join(base_dir, 'ERA5/netcdf')

att_fn = os.path.join(base_dir, "ERA5", "ERA_attenuation_PAMTRA.nc")

#%% DFR at cloud top
check_numvalpt = False
check_corr = False
check_var = False

#%% daily quicklooks
dates = generate_date_range(startdate, enddate)

for date in dates:
    # delete any (memory-consuming) pre-existing variables that are defined in this cell
    for var in ['MIRA_data', 'BASTA_data', 'DFR_correction']:
        if var in locals():
            del globals()[var]

    year, month, day = date.year, date.month, date.day
    print(f"Calculating DFR correction for {year}-{month:02}-{day:02}")

    MIRA_filedir = os.path.join(MIRA_dir, f"{year}/{month:02}/{day:02}")
    MIRA_files = sorted(glob.glob(f"{MIRA_filedir}/*_merged.nc"))
    if MIRA_files:
        MIRA_data = preprocess_MIRA(xr.open_mfdataset(MIRA_files, 
                                                      combine='by_coords', 
                                                      chunks={'time':3600}, 
                                                      preprocess=preprocess, 
                                                      parallel=True))
    else:
        MIRA_data = None
    BASTA_files = sorted(glob.glob(os.path.join(BASTA_dir, f"BASTA_L1_{BASTAmode}_{year}{month:02}{day:02}*.nc")))
    if BASTA_files:
        BASTA_data = preprocess_BASTA(xr.open_mfdataset(BASTA_files, combine='by_coords'))
    else:
        BASTA_data = None
    
    if not MIRA_data or not BASTA_data:
        if BASTA_data and not MIRA_data:
            print(f'No MIRA-data for {year}-{month:02}-{day:02}')
        if MIRA_data and not BASTA_data:
            print(f'No BASTA-data for {year}-{month:02}-{day:02}')
        if not MIRA_data and not BASTA_data:
            print(f'No MIRA- or BASTA-data for {year}-{month:02}-{day:02}')
        continue

    dataset_times = [MIRA_data['time'].values, BASTA_data['time'].values]
    tmin = max(np.min(times) for times in dataset_times)
    tmax = min(np.max(times) for times in dataset_times)

    full_time = pd.date_range(tmin, tmax, freq='5s') # start to end of available data resampled to every 5 seconds
    common_range = np.arange(400, 10e3, 25) # 0.4-10 km elevation window at a 25 m resolution

    MIRA = safe_reindex(MIRA_data, dim='time', full_dim=full_time, tolerance='2.5s', name='MIRA')
    MIRA = MIRA.reindex(range=common_range, method='nearest')
    BASTA = safe_reindex(BASTA_data, dim='time', full_dim=full_time, tolerance='1.5s', name='BASTA')
    BASTA = BASTA.reindex(range=common_range, method='nearest')

    # correct BASTA for atmospheric gas attenuation (without MIRA-BASTA calibration constant)
    BASTA['reflectivity_corrected'] = xr.DataArray(correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequencies['BASTA'], tmin, tmax).values, dims=BASTA['reflectivity'].dims, 
                                                   coords={dim: BASTA[dim] for dim in BASTA['reflectivity'].dims})

    # calculate or import DFR at top cloud (MIRA-BASTA) to flag extreme BASTA-attenuation
    DFR_outname = f"{base_dir}/radar_calib/cloudtopDFR/{year}{month:02}{day:02}_MIRA-BASTA_DFR.csv"
    if not os.path.exists(DFR_outname):
        DFR_correction = DFR_offset(BASTA, MIRA, (check_numvalpt, check_corr, check_var))
        DFR_correction.to_csv(DFR_outname, index=False)


#%% adjust DFR correction for wrong MIRA reflectivities (following MBP mail 28.05.2026) - make 1 new file that is used later on
# this was run only once to manually correct the cloudtop DFR values
files = glob.glob(f"{base_dir}/radar_calib/cloudtopDFR/*_MIRA-BASTA_DFR.csv")

dfrs = get_DFRdata(files)
dfrs['mean_DFR'] = dfrs['mean_DFR'] + 0.89
dfrs['daily_mean_DFR'] = dfrs['daily_mean_DFR'] + 0.89

dfrs.to_csv(f"{base_dir}/radar_calib/cloudtopDFR/MIRA-BASTA_DFRs.csv")