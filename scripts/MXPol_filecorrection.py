#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 2 11:51:32 2026

Correct MXPol files

@author: clerx
"""
#%% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import glob, os, pd, gc, plt, pyart, time, datetime
from src.constants_input import dirs, base_dir, start_date, end_date
from src.utils import generate_date_range, check_time
from src.radar_processing import correct_MXPol_file
from MXPol_hydroclassif import classify_hm, plot_classification


#%%
if __name__ == '__main__':
    dates = generate_date_range(start_date, end_date)
    for date in dates:
        t_date = time.time()
        year, month, day = date.year, date.month, date.day
        files = sorted(glob.iglob(os.path.join(base_dir, dirs['MXPol'], f"{year:04d}/{month:02d}/{day:02d}/XPOL-{year:04d}{month:02d}{day:02d}*.nc")))

        if not files:
            continue

        RHIs = [f for f in files if 'RHI' in f]
        t_RHI = time.time()
        print(f"Correcting {len(RHIs)} RHI files for {year}-{month:02}-{day:02}, starting {pd.to_datetime(t_RHI, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
        for fname in RHIs:
            name = fname.split('/')[-1].split('.')[0]
            ftime = datetime.strptime(name.split('_')[0][5:], '%Y%m%d-%H%M%S')
            if check_time(ftime):
                print(f"{name} in time interval of lost frequency tracking, skipping.")
                continue

            scantype = 'RHI'
            outname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}/{name}.nc")
            if os.path.exists(outname):
                print(f"Processed file {name} already exists, skipping.")
                continue
            try:
                figname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/plots/{year:04d}/{month:02d}/{day:02d}/{name}_hydroclassif.png")
                radar = classify_hm(fname, scantype=scantype, rain_flag=None)
                if radar is None:
                    continue
                plot_classification(radar, scantype=scantype, savepath=figname)
                pyart.io.write_cfradial(outname, radar)
                del radar
            except Exception as e:
                print(e)
            gc.collect()

        t_end = time.time()
        print(f"Saved {len(RHIs)} files for {year}-{month:02}-{day:02}, time taken: {t_end-t_RHI:.2f} seconds\n")
        

        sectors = [f for f in files if 'sector' in f]
        t_sector = time.time()
        print(f"Correcting {len(sectors)} sector files for {year}-{month:02}-{day:02}, starting {pd.to_datetime(t_RHI, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
        for fname in sectors:
            name = fname.split('/')[-1].split('.')[0]
            ftime = datetime.strptime(name.split('_')[0][5:], '%Y%m%d-%H%M%S')
            if check_time(ftime):
                print(f"{name} in time interval of lost frequency tracking, skipping.")
                continue

            scantype = 'sector'
            outname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}/{name}.nc")
            if os.path.exists(outname):
                print(f"Processed file {name} already exists, skipping.")
                continue
            try:
                figname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/plots/{year:04d}/{month:02d}/{day:02d}/{name}_hydroclassif.png")
                radar = classify_hm(fname, scantype=scantype, rain_flag=None)
                if radar is None:
                    continue
                plot_classification(radar, scantype=scantype, savepath=figname)
                pyart.io.write_cfradial(outname, radar)
                del radar
            except Exception as e:
                print(e)
            gc.collect()

        t_end = time.time()
        print(f"Saved {len(sectors)} files for {year}-{month:02}-{day:02}, time taken: {t_end-t_sector:.2f} seconds\n")
        
        Zdrs = [f for f in files if 'Zdr' in f]
        t_Zdr = time.time()
        print(f"Correcting {len(Zdrs)} Zdr files for {year}-{month:02}-{day:02}, starting {pd.to_datetime(t_Zdr, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")

        os.makedirs(os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}"), exist_ok=True )
        for fname in Zdrs:
            scantype = 'Zdr'
            name = fname.split('/')[-1].split('.')[0]
            ftime = datetime.strptime(name.split('_')[0][5:], '%Y%m%d-%H%M%S')
            if check_time(ftime):
                print(f"{name} in time interval of lost frequency tracking, skipping.")
                continue
            outname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}/{name}.nc")
            if os.path.exists(outname):
                print(f"Processed file {name} already exists, skipping.")
                continue
            try:
                radar = correct_MXPol_file(fname, scantype=scantype)
                pyart.io.write_cfradial(outname, radar)                
                del radar
            except Exception as e:
                print(e)
            gc.collect()
        t_end = time.time()
        print(f"Saved Zdr files for {year}-{month:02}-{day:02}, time taken: {t_end-t_Zdr:.2f} seconds\n")

