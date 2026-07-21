#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 10:47:03 2025
Script to merge and chunk MIRA .znc data into hourly .nc-files.

@author: clerx
"""
import glob
import xarray as xr
import os
import pandas as pd
import numpy as np
import dask
import sys
sys.path.append('/home/clerx/cleancloud/scripts/github')

from MIRA_auxiliary import *
from MIRA_plotting import *

from datetime import datetime, timedelta
from dask import delayed


@delayed
def process_file(f):
    """Processes a single file to extract its valid time range if it contains zenith measurements."""
    ds = xr.open_dataset(f, chunks='auto', decode_times=False)  # Lazy loading
    ds = ds.where(ds.elv == 90, drop=True)  # Keep only zenith scans
    
    if 'time' in ds.dims and ds.sizes['time'] > 0:
        file_time = pd.to_datetime(ds.coords['time'].values, unit='s')
        return (f, (file_time.min(), file_time.max()))
    return None


def mask(files, time_ranges, hour, date_output_dir):
    """
    Creates one dataset per hour, masking the periods in between the various files.
    Skips processing if the output file already exists.
    """
    # Define the output filename and path
    year, month, day = [int(i) for i in date_output_dir.split('/')[-3:]]
    output_filename = f"{year}{month:02}{day:02}_{hour:02d}00_merged.nc"
    output_path = os.path.join(date_output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping.")
        return #Skip processing for this hour

    if not files:
        print(f"No files to process for hour {hour}.")
        return

    # Open and process the datasets
    parts = date_output_dir.strip('/').split('/')
    print(f"Opening files for {hour:02}:00 to {hour+1:02}:00 on {parts[-3]}-{parts[-2]}-{parts[-1]}, {len(files)} files.")
    with dask.config.set(scheduler='processes'):
        hr = xr.open_mfdataset(files, combine='by_coords', chunks='auto', preprocess=preprocess, parallel=True)
        msk = ((hr.elv == 90) | (hr.elv == -1000)).compute()
        hr = hr.where(msk, drop=True)
        msk.close()
        
        if 'time' in hr.dims and hr.sizes['time'] > 0:
            hr = hr.assign_coords(time=pd.to_datetime(hr.coords['time'].values, unit='s'))
            first_ts = pd.to_datetime(hr.time.values[0])
            year, month, day = first_ts.year, first_ts.month, first_ts.day
            full_time = pd.date_range(start=datetime(year, month, day, hour), end=datetime(year, month, day, hour)+timedelta(minutes=59,seconds=59), freq='5s')
            hr = hr.reindex(time=full_time, method='nearest')
            
            msk = xr.DataArray(np.full(len(hr.coords['time']), True), coords={'time':hr.coords['time']}, dims=['time'])
            
            if len(files) > 1:  
                for i, (start_time, endtime) in enumerate(time_ranges):
                    if i + 1 < len(time_ranges):
                        starttime = time_ranges[i+1][0]
                        cond = (msk.coords['time'] > endtime) & (msk.coords['time'] < starttime)
                        msk = msk.where(~cond, other=False)
                        hr = hr.where(msk)
                        msk.close()
            msk2 = xr.DataArray(np.full(len(hr.coords['time']), True), coords={'time': hr.coords['time']}, dims=['time'])
            cond2 = (hr.coords['time'] < time_ranges[0][0]) | (hr.coords['time'] > time_ranges[-1][1])
            msk2 = msk2.where(~cond2, other=False)  # False for out-of-range times
            hr = hr.where(msk2)
        
            chunked = hr.chunk({'time': 600, 'range': 240, 'doppler': 128})
            
            hr.close()
        
            # Encoding and handling chunks
            encoding = {}
            for var in chunked.data_vars:
                if chunked[var].chunks is None:
                    # Skip chunking for variables with undefined chunks
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 5,
                    }
                else:
                    # Apply chunking for variables with defined chunks
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 5,
                        'chunksizes': [
                            size[0] if isinstance(size, tuple) else size for size in chunked[var].chunks
                        ],
                    }
                    
            with dask.config.set(scheduler='threads'):
                chunked.to_netcdf(output_path, encoding=encoding, compute=True)
            print(f"Saved merged data for hour {hour} to {output_path}")        
            chunked.close()            

        else:
            print(f"No zenith-data available for {hour:02}:00 to {hour+1:02}:00 on {parts[-3]}-{parts[-2]}-{parts[-1]}.")


def mask_mmclx(files, time_ranges, hour, date_output_dir):
    """
    Creates one dataset per hour, masking the periods in between the various files.
    Skips processing if the output file already exists.
    """
    # Define the output filename and path
    year, month, day = [int(i) for i in date_output_dir.split('/')[-3:]]
    output_filename = f"{year}{month:02}{day:02}_mmclx.nc"
    output_path = os.path.join(date_output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping.")
        return #Skip processing for this hour

    if not files:
        print(f"No files to process for hour {hour}.")
        return

    # Open and process the datasets
    parts = date_output_dir.strip('/').split('/')
    print(f"Opening files for {hour:02}:00 to {hour+1:02}:00 on {parts[-3]}-{parts[-2]}-{parts[-1]}, {len(files)} files.")
    with dask.config.set(scheduler='processes'):
        try:
            hr = xr.open_mfdataset(files, combine='by_coords', chunks='auto', preprocess=preprocess_mmclx, parallel=True)
            msk = ((hr.elv == 90) | (hr.elv == -1000.)).compute()
            hr = hr.where(msk, drop=True)
            msk.close()
            
            if 'time' in hr.dims and hr.sizes['time'] > 0:
                hr = hr.assign_coords(time=pd.to_datetime(hr.coords['time'].values, unit='s'))
                first_ts = pd.to_datetime(hr.time.values[0])
                year, month, day = first_ts.year, first_ts.month, first_ts.day
                full_time = pd.date_range(start=datetime(year, month, day, hour), end=datetime(year, month, day, hour)+timedelta(minutes=59,seconds=59), freq='5s')
                hr = hr.reindex(time=full_time, method='nearest')
                
                msk = xr.DataArray(np.full(len(hr.coords['time']), True), coords={'time':hr.coords['time']}, dims=['time'])
                
                if len(files) > 1:  
                    for i, (start_time, endtime) in enumerate(time_ranges):
                        if i + 1 < len(time_ranges):
                            starttime = time_ranges[i+1][0]
                            cond = (msk.coords['time'] > endtime) & (msk.coords['time'] < starttime)
                            msk = msk.where(~cond, other=False)
                            hr = hr.where(msk)
                            msk.close()
                msk2 = xr.DataArray(np.full(len(hr.coords['time']), True), coords={'time': hr.coords['time']}, dims=['time'])
                cond2 = (hr.coords['time'] < time_ranges[0][0]) | (hr.coords['time'] > time_ranges[-1][1])
                msk2 = msk2.where(~cond2, other=False)  # False for out-of-range times
                hr = hr.where(msk2)
            
                chunked = hr.chunk({'time': 600, 'range': 240})
                
                hr.close()
            
                # Encoding and handling chunks
                encoding = {}
                for var in chunked.data_vars:
                    if chunked[var].chunks is None:
                        # Skip chunking for variables with undefined chunks
                        encoding[var] = {
                            'zlib': True,
                            'complevel': 5,
                        }
                    else:
                        # Apply chunking for variables with defined chunks
                        encoding[var] = {
                            'zlib': True,
                            'complevel': 5,
                            'chunksizes': [
                                size[0] if isinstance(size, tuple) else size for size in chunked[var].chunks
                            ],
                        }
                        
                with dask.config.set(scheduler='threads'):
                    chunked.to_netcdf(output_path, encoding=encoding, compute=True)
                print(f"Saved merged data for hour {hour} to {output_path}")        
                chunked.close()            
    
            else:
                print(f"No zenith-data available for {hour:02}:00 to {hour+1:02}:00 on {parts[-3]}-{parts[-2]}-{parts[-1]}.")
                
        except Exception as e:
            print(f"Error while processing files for hour {hour}: {e}.")


def process_and_plot_dates(start_date, end_date, base_dir, output_dir):
    """
    Process the data for each day between start_date and end_date, create a daily file, and generate a plot.
    """
    dates = generate_date_range(start_date, end_date)
    for date in dates:
        year, month, day = date.year, date.month, date.day
        file_dir = os.path.join(base_dir, f"{year}/{month:02d}/{day:02d}")
        files_ranges = get_files_ranges(day, month, year, file_dir)
        date_output_dir = os.path.join(output_dir, f"{year}/{month:02d}/{day:02d}")
        os.makedirs(date_output_dir, exist_ok=True)

        for hour in range(24):
            if files_ranges[hour]['files']:
                mask(files_ranges[hour]['files'], files_ranges[hour]['timeranges'], hour, date_output_dir)

        files = sorted(glob.glob(os.path.join(date_output_dir, '*_merged.nc')))
        if files:
            plot_hourly_data(files, date, date_output_dir)
        else:
            if not os.listdir(date_output_dir):
                print(f"No data to plot for {date}, removed L1-folder {date_output_dir}.")
                os.rmdir(date_output_dir)
            

def process_mmclx(start_date, end_date, base_dir, output_dir):
    dates = generate_date_range(start_date, end_date)
    for date in dates:
        year, month, day = date.year, date.month, date.day
        file_dir = os.path.join(base_dir, f"{year}/{month:02}/{day:02}")
        files_ranges = get_files_ranges_mmclx(day, month, year, file_dir)
        date_output_dir = os.path.join(output_dir, f"{year}/{month:02d}/{day:02d}")
        os.makedirs(date_output_dir, exist_ok=True)
        
        for hour in range(24):
            if files_ranges[hour]['files']:
                mask_mmclx(files_ranges[hour]['files'], files_ranges[hour]['timeranges'], hour, date_output_dir)
       
    
# base_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/mom"
# output_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/L1"

# date = datetime(2025, 1, 14)
# year, month, day = date.year, date.month, date.day
# file_dir = os.path.join(base_dir, f"{year}/{month:02d}/{day:02d}")
# files_ranges = get_files_ranges(day, month, year, file_dir)
# files = [file for hour in range(24) for file in files_ranges[hour]['files']]             
# hr = xr.open_mfdataset(files, combine='by_coords', chunks='auto', preprocess=preprocess)

if __name__ == "__main__":
    base_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/mom"
    output_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/L1"
    start_date = "2024-10-16"
    end_date = "2025-01-24"
    # process_and_plot_dates(start_date, end_date, base_dir, output_dir)
    # process_mmclx(start_date, end_date, base_dir, output_dir)
    # plot_dates(start_date, end_date, base_dir, output_dir)


