#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 09:09:02 2025

Convert ERA5 .grib-files to .nc-files, inspired by https://github.com/ecmwf-projects/dss-notebooks/blob/main/documentation/grib_to_netcdf.ipynb.
Possible to convert ERA5 reanalysis data at pressure & model level.

@author: clerx
"""
#%%
from src import os, np, xr, glob, Path

#%% constants & references

p0 = 100000.0 # reference pressure [Pa]
g = 9.80665 # gravitational constant [m/s^2]
r_earth = 6378e3 # earth radius [m]
Rd = 287.05 # air gas constant of

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/ERA5'
variables_ml = ['t', 'q', 'u', 'v', 'z', 'lnsp']

calculate_height_ml = False

#%% convert .grib-files to netcdf

files = sorted(glob.glob(f"{base_dir}/grib/*_large.grib"))

print(f"Found {len(files)} large grib files")
for f in files: 
    outname = f"{base_dir}/netcdf/{Path(f).stem}.nc"
    if os.path.exists(outname):
        print(f"{os.path.basename(outname)} already exists, skipping...")
        continue
    if not 'modellevels' in f:
        try:
            ds = xr.open_dataset(f, engine='cfgrib', backend_kwargs={'indexpath':'', 'decode_timedelta':True})
            ds = xr.decode_cf(ds)
            if 'time' in ds.coords and 'step' in ds.coords:
                ds = ds.assign_coords(time=ds.time + ds.step)
                ds = ds.drop_vars('step')
        except Exception as e:
            print(f"Error while converting {os.path.basename(f)}: {e}")
        ds.to_netcdf(outname)
        ds.close()
    else:
        datasets = []
        for var in variables_ml:
            try:
                filter_keys = {'shortName': var}

                if var == 'z':
                    filter_keys['typeOfLevel'] = 'hybrid'  # or 'surface' if you want surface geopotential

                ds = xr.open_dataset(
                    f,
                    engine="cfgrib",
                    backend_kwargs={
                        'filter_by_keys': filter_keys,
                        'indexpath': ''  # prevent cfgrib from using index files
                    }
                )
                datasets.append(ds)
                print(f"Loaded variable: {var}")
            except Exception as e:
                print(f"Skipping variable: {var} due to error: {e}")
        combined = xr.merge(datasets)
        combined = xr.decode_cf(combined)   # ensure CF decoding before time arithmetic
        if 'time' in combined.coords and 'step' in combined.coords:
            combined = combined.assign_coords(time=combined['time'] + combined['step'])
            combined = combined.drop_vars('step')
        combined.to_netcdf(outname, mode='w')
        combined.close()
    print(f"Saved {os.path.basename(outname)} to {os.path.dirname(outname)}")
    # else:
    #     print(f"{os.path.basename(outname)} already exists")
    #     continue


#%% calculate geopotential height and altitude for model levels ########## not working - try with updated .grib file
if calculate_height_ml: 
    ds = xr.open_dataset(outname)
    a = ds['hyam']  # shape: (model_levels,)
    b = ds['hybm']  # shape: (model_levels,)
    T = ds['t']     # shape: (time, level, lat, lon)
    lnsp = ds['lnsp']  # shape: (time, lat, lon)

    ps = np.exp(lnsp)
    p = a * p0 + b * ps[:, None, :, :]

    Phi = xr.zeros_like(T)

    levels = T.level.values
    nlev = len(levels)

    Phi_vals = np.zeros_like(T.values)

    for k in reversed(range(nlev -1)):
        dlnp = np.log(p[:, k+1, :, :] / p[:, k, :, :])
        Phi_vals[:, k, :, :] = Phi_vals[:, k+1, :, :] + Rd * T.values[:, k, :, :] * dlnp

    z = Phi_vals / g

    ds['z'] = (('time', 'level', 'latitude', 'longitude'), z)
    ds['z'].attrs['units'] = 'm'
    ds['z'].attrs['long_name'] = 'Geopotential height'

