#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 5 15:04:34 2025

@author: clerx
"""

#%% imports

import glob 
import os 
import re 
import dask
import sys
import itertools

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import xarray as xr

from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from matplotlib.dates import date2num
from matplotlib.patches import Patch, Rectangle

project_path = '/home/clerx/scripts/'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from MIRA_auxiliary import preprocess_mmclx

#%% helper functions & constants
def dtm(s): return datetime.strptime(s, '%Y.%m.%d %H:%M')

def parse_ranges(ranges):
    return [(datetime.strptime(start, '%Y.%m.%d %H:%M'), (datetime.strptime(end, '%Y.%m.%d %H:%M')))
             for start, end in ranges]

plt.rcParams.update({    
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})

def isnear(time, target, tolerance_minutes=3):
    delta = abs((dt.datetime.combine(dt.date.min, time) -
                 dt.datetime.combine(dt.date.min, target)))
    return delta <= dt.timedelta(minutes=tolerance_minutes)

def merge_precip_intervals(df_precip, gap_threshold_minutes=10, min_precip_duration_minutes=5):
    """
    Merge precipitation intervals that are less than gap_threshold_minutes apart.
    
    Parameters:
    df_precip: DataFrame with columns ['start', 'end', 'duration_min', temp/meltheight stats]
    gap_threshold_minutes: Maximum gap between intervals to merge (default 10 minutes)
    
    Returns:
    DataFrame with merged intervals
    """
    if df_precip.empty:
        return df_precip
    
    # Sort by start time to ensure proper order
    df_sorted = df_precip.sort_values('start').reset_index(drop=True)
    
    merged_periods = []
    current_period = df_sorted.iloc[0].to_dict()
    
    for i in range(1, len(df_sorted)):
        next_period = df_sorted.iloc[i].to_dict()
        
        # Calculate gap between current period end and next period start
        gap = next_period['start'] - current_period['end']
        gap_minutes = gap.total_seconds() / 60
        
        if gap_minutes <= gap_threshold_minutes:
            # Merge intervals
            # Update end time and duration
            current_period['end'] = next_period['end']
            current_period['duration_min'] = (current_period['end'] - current_period['start']).total_seconds() / 60
            
            # Merge temperature statistics (taking min/max across both periods)
            for stat in ['temp_min', 'temp_max', 'meltheight_min', 'meltheight_max']:
                if current_period[stat] is not None and next_period[stat] is not None:
                    if 'min' in stat:
                        current_period[stat] = min(current_period[stat], next_period[stat])
                    else:  # max
                        current_period[stat] = max(current_period[stat], next_period[stat])
                elif next_period[stat] is not None:
                    current_period[stat] = next_period[stat]
            
            # Merge mean statistics (weighted average by duration)
            for stat in ['temp_mean', 'meltheight_mean']:
                if current_period[stat] is not None and next_period[stat] is not None:
                    # Weight by original durations
                    curr_weight = df_sorted.iloc[i-1]['duration_min'] if i > 0 else current_period['duration_min']
                    next_weight = next_period['duration_min']
                    total_weight = curr_weight + next_weight
                    
                    current_period[stat] = (
                        current_period[stat] * curr_weight + 
                        next_period[stat] * next_weight
                    ) / total_weight
                elif next_period[stat] is not None:
                    current_period[stat] = next_period[stat]
        else:
            # Gap is too large, save current period and start new one
            merged_periods.append(current_period)
            current_period = next_period.copy()
    
    # Don't forget the last period
    merged_periods.append(current_period)

    df_merged = pd.DataFrame(merged_periods)
    df_filtered = df_merged[df_merged['duration_min'] >= min_precip_duration_minutes]
    
    return df_filtered

def add_precip_type(df_precip, temp_threshold=1.5, meltheight_threshold=100,
                   temp_col='temp_mean', meltheight_col='meltheight_mean'):
    """
    Add a 'precip_type' column to precipitation dataframe based on temperature and melting height criteria.
    
    Notes:
    ------
    Precipitation types:
    - 'solid': temp_mean < temp_threshold AND meltheight_mean < meltheight_threshold
    - 'liquid': temp_mean >= temp_threshold OR meltheight_mean >= meltheight_threshold
    - 'unknown': when both temperature and melting height data are missing
    """
    
    # Make a copy to avoid modifying original dataframe
    df_result = df_precip.copy()
    
    # Initialize precip_type column as 'liquid' (default)
    df_result['precip_type'] = 'liquid'
    
    # Create masks for conditions
    temp_available = df_result[temp_col].notna()
    meltheight_available = df_result[meltheight_col].notna()
    
    # SOLID: BOTH conditions must be true (temp < threshold AND meltheight < threshold)
    temp_cold = df_result[temp_col] < temp_threshold
    meltheight_low = df_result[meltheight_col] < meltheight_threshold
    
    # Only classify as solid if BOTH conditions are met AND both data are available
    solid_mask = temp_available & meltheight_available & temp_cold & meltheight_low
    df_result.loc[solid_mask, 'precip_type'] = 'solid'
    
    # Handle cases where only one parameter is available
    # If only temp available: use temp threshold
    only_temp = temp_available & ~meltheight_available
    df_result.loc[only_temp & temp_cold, 'precip_type'] = 'solid'
    df_result.loc[only_temp & ~temp_cold, 'precip_type'] = 'liquid'
    
    # If only meltheight available: use meltheight threshold  
    only_meltheight = meltheight_available & ~temp_available
    df_result.loc[only_meltheight & meltheight_low, 'precip_type'] = 'solid'
    df_result.loc[only_meltheight & ~meltheight_low, 'precip_type'] = 'liquid'
    
    # Mark as unknown if both temperature and melting height are missing
    both_missing = ~temp_available & ~meltheight_available
    df_result.loc[both_missing, 'precip_type'] = 'unknown'
    
    return df_result

def create_precipitation_chart(df_precip_merged, start_date, end_date, ax, y_position=1):
    """
    Create a bar chart showing precipitation periods and solid/liquid classification.
    
    Parameters:
    df_precip_merged: DataFrame with merged precipitation intervals
    start_date, end_date: datetime objects defining the plot range
    ax: matplotlib axis object
    y_position: vertical position for the precipitation bars
    
    Returns:
    matplotlib axis object
    """
    import numpy as np
    
    # Create full time range for the chart
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Initialize precipitation array (0 = no precip, 1 = liquid, 2 = solid)
    precip_array = np.zeros(len(date_range))
    
    # Fill in precipitation periods
    for _, period in df_precip_merged.iterrows():
        # Find indices corresponding to this precipitation period
        start_idx = np.searchsorted(date_range, period['start'])
        end_idx = np.searchsorted(date_range, period['end'])
        
        # Classify as solid or liquid based on temperature
        if period['temp_mean'] is not None and period['temp_mean'] < 1.5 and period['meltheight_mean'] <= 100:
            precip_type = 2  # Solid
        else:
            precip_type = 1  # Liquid
            
        # Fill the array for this period
        precip_array[start_idx:end_idx+1] = precip_type
    
    # Convert to datetime for plotting
    times = mdates.date2num(date_range)
    
    # Plot bars using the integrated color scheme
    for i in range(len(times)-1):
        width = times[i+1] - times[i]
        precip_type = int(precip_array[i])
        
        if precip_type == 1:  # Liquid precipitation
            ax.barh(y_position, width, left=times[i], height=0.8,
                   color=colors['precip_liquid'], alpha=0.7,
                   edgecolor='black', linewidth=0.3,
                   label='Liquid precipitation (T ≥ 1.5°C)' if i == 0 else "")
        elif precip_type == 2:  # Solid precipitation
            ax.barh(y_position, width, left=times[i], height=0.8,
                   color=colors['precip_solid'], alpha=0.7,
                   edgecolor='black', linewidth=0.3,
                   label='Potentially solid precipitation (T < 1.5°C)' if i == 0 else "")
    
    return ax

def add_precipitation_to_plot(ax, df_precip_merged, start_date, end_date, height=0.5, y_position=1):
    """
    Add precipitation bars to existing axis
    """
    if df_precip_merged is None or len(df_precip_merged) == 0:
        return ax
        
    # Create full time range for the chart
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Initialize precipitation array (0 = no precip, 1 = liquid, 2 = solid)
    precip_array = np.zeros(len(date_range))
    
    # Fill in precipitation periods
    for _, period in df_precip_merged.iterrows():
        # Find indices corresponding to this precipitation period
        start_idx = np.searchsorted(date_range, period['start'])
        end_idx = np.searchsorted(date_range, period['end'])
        
        # Classify as solid or liquid based on temperature
        if period['temp_mean'] is not None and period['temp_mean'] < 1.5 and period['meltheight_mean'] <= 100:
            precip_type = 2  # Solid
        else:
            precip_type = 1  # Liquid
            
        # Fill the array for this period
        precip_array[start_idx:end_idx+1] = precip_type
    
    # Convert to datetime for plotting
    times = mdates.date2num(date_range)
    
    # Track if we've added labels
    liquid_labeled = False
    solid_labeled = False
    
    # Plot bars
    for i in range(len(times)-1):
        width = times[i+1] - times[i]
        precip_type = int(precip_array[i])
        
        if precip_type == 1:  # Liquid precipitation
            label = 'Liquid precipitation (T ≥ 1.5°C)' if not liquid_labeled else ""
            ax.barh(y_position, width, left=times[i], height=height,
                   color=colors['precip_liquid'], label=label)
            liquid_labeled = True
        elif precip_type == 2:  # Solid precipitation
            label = 'Potentially solid precipitation (T < 1.5°C)' if not solid_labeled else ""
            ax.barh(y_position, width, left=times[i], height=height,
                   color=colors['precip_solid'], label=label)
            solid_labeled = True
    
    return ax

def calc_alt(h):
    """calculate geometric height from geopotential height"""
    G = 9.80665 # gravitational constant [m/s^2]
    R_earth = 6378e3 # earth radius [m]
    alt = (R_earth * (h/G)) / (R_earth - (h/G))
    
    return alt

def read_ERA_data(date, ERA_dir):
    latlon_Helmos = (38.007, 22.196)
    ERA_file = glob.glob(f"{ERA_dir}/*{date.strftime('%Y%m%d')}*.nc")
    if not ERA_file:
        print(f"No ERA5 file for {date.strftime('%Y-%m-%d')}")
        return None
    else:
        ds = xr.open_dataset(ERA_file[0], decode_timedelta=True)
        altitude = calc_alt(ds['z'].mean(dim=['time', 'latitude', 'longitude']))
        ds = ds.assign_coords(altitude=altitude)
        ds_Helmos = ds.sel(latitude=latlon_Helmos[0], longitude=latlon_Helmos[1], method='nearest')
        ds_Helmos['t'] = ds_Helmos['t'] - 273.15
        return ds_Helmos

def find_temp_levels(ds, target_temps, temp_var='t', alt_var='altitude', interp_kind='linear', extrapolate=False):
    """
    find altitude (wrt msl) of target_temperature (base input dataset = ERA output data)
    returns xarray.DataArray of altitude values
    """
    temp_celsius = ds[temp_var] - 273.15 if ds[temp_var].mean() > 200 else ds[temp_var]

    results = {}

    for target_temp in target_temps:
        target_altitudes = []

        for time_idx in range(len(ds.time)):
            temp = temp_celsius.isel(time=time_idx).values.ravel()
            alt  = ds[alt_var].isel(time=time_idx).values.ravel()

            valid_mask = ~(np.isnan(temp) | np.isnan(alt))
            if valid_mask.sum() < 2:
                target_altitudes.append(np.nan)
                continue
            
            temp_clean = temp[valid_mask]
            alt_clean = alt[valid_mask]

            sort_idx = np.argsort(temp_clean)
            temp_sorted = temp_clean[sort_idx]
            alt_sorted = alt_clean[sort_idx]
            temp_min, temp_max = temp_sorted.min(), temp_sorted.max()

            if (temp_min <= target_temp <= temp_max) or extrapolate:
                try:
                    fill_value = np.nan if not extrapolate else 'extrapolate'
                    f = interp1d(temp_sorted, alt_sorted, kind=interp_kind, 
                            bounds_error=False, fill_value=fill_value)
                    
                    target_alt = f(target_temp)
                    target_altitudes.append(float(target_alt) - radar_altitude)
                except Exception as e:
                    print(f"Interpolation failed at time {time_idx}: {e}")
                    target_altitudes.append(np.nan)
            else:
                target_altitudes.append(np.nan)

        var_name = f'temp_{int(target_temp) if target_temp == int(target_temp) else target_temp}C_altitude'
        # var_name = var_name.replace('-', 'minus_').replace('.', 'p')
        
        results[var_name] = xr.DataArray(
            target_altitudes,
            coords={'time': ds.time},
            dims=['time'],
            attrs={
                'units': 'm',
                'long_name': f'{target_temp}°C isotherm height above radar',
                'target_temperature': target_temp
            }
        )
        out = xr.Dataset(results)
        out.attrs['description'] = f"isotherm heights relative to radar altitude {int(radar_altitude)} m a.s.l."
    
    return out

def safe_reindex(data, dim, full_dim, tolerance, name=''):
    if dim == 'time':
        reindexed = data.reindex(time=full_dim, method='nearest', tolerance=pd.Timedelta(tolerance))
        if reindexed.time.isnull().any():
            print(f"Warning: Some time points in {name} were not reindexed due to missing data within tolerance.")
    if dim == 'range':
        reindexed = data.reindex(range=full_dim, method='nearest', tolerance=tolerance)
        if reindexed.range.isnull().any():
            print(f"Warning: Some range points in {name} were not reindexed due to missing data within tolerance.")
    return reindexed

#%% data availability per radar
radar_altitude = 1690 # m a.s.l.

basta_ranges = parse_ranges([
    ("2024.10.18 00:00", "2024.11.30 01:00"),
    ("2024.12.02 12:00", "2024.12.07 12:00"),
    ("2024.12.18 13:15", "2025.01.20 23:50"),
    ("2025.01.21 11:45", "2025.01.22 08:45"),
    ("2025.01.23 09:30", "2025.01.24 00:00"),
])

basta_errors = [
    ("2024.12.20 08:00", "2024.12.20 13:00"),
    ("2024.12.21 00:00", "2024.12.27 07:30"),
    ("2025.01.15 11:30", "2025.01.15 20:00"),
    ("2025.01.21 22:30", "2025.01.22 08:45")
] # approximate timing of attenuation due to snow cover

mira_ranges = parse_ranges([
    ("2024.10.15 00:00", "2024.12.26 04:00"),
    ("2024.12.26 15:15", "2025.01.03 00:00"),  # zenith
    ("2025.01.03 00:00", "2025.01.13 17:00"),
    ("2025.01.14 04:00", "2025.01.14 14:00"),  # zenith
    ("2025.01.14 14:50", "2025.01.18 01:20"),  # zenith
    ("2025.01.19 21:00", "2025.01.20 17:55"),  # zenith
    ("2025.01.20 17:55", "2025.01.23 23:59"),
])
mira_zenith_flags = [False, True, False, True, True, True, False]
mira_errors = [
    ("2024.12.26 15:00", "2024.12.30 00:00"),
    ("2025.01.13 03:40", "2025.01.13 17:00"),
    ("2025.01.14 04:00", "2025.01.14 14:00")
] # approximate timing based on LDR variation

mxpol_scan_full = parse_ranges([
    ("2024.11.29 00:00", "2024.11.29 15:32"),
    ("2024.11.30 06:02", "2024.12.01 20:18"),
    ("2024.12.02 06:05", "2024.12.02 15:04"),
    ("2024.12.03 05:41", "2024.12.03 17:57"),
    ("2024.12.04 10:48", "2024.12.04 23:31"),
    ("2024.12.05 00:00", "2024.12.15 15:25"),
    ("2024.12.06 06:23", "2024.12.07 06:37"),
    ("2024.12.07 02:00", "2024.12.07 07:00"),
    ("2024.12.09 06:19", "2024.12.25 21:00"),
])

mxpol_zenith_full = parse_ranges([
    ("2024.12.25 13:35", "2025.01.06 06:44"),
    ("2025.01.07 00:00", "2025.01.08 00:29"),
    ("2025.01.10 00:21", "2025.01.12 07:30"),
    ("2025.01.13 20:27", "2025.01.13 22:14"),
    ("2025.01.14 01:56", "2025.01.14 18:50"),
    ("2025.01.15 00:00", "2025.01.15 21:55"),
    ("2025.01.16 01:16", "2025.01.16 23:44"),
    ("2025.01.17 06:01", "2025.01.17 07:06"),
    ("2025.01.18 01:46", "2025.01.18 17:54"),
    ("2025.01.19 20:18", "2025.01.19 20:23"),
    ("2025.01.20 07:16", "2025.01.20 12:39"),
    ("2025.01.21 04:01", "2025.01.21 15:42"),
])
mxpol_lost_frequency = [
     ('20241129-093758', '20241129-111859'),
     ('20241209-061913', '20241209-062252'),
     ('20241210-045727', '20241210-065155'),
     ('20241211-005731', '20241211-131323'),
     ('20241212-065545', '20241212-065937'),
     ('20241213-090121', '20241214-220406'),
     ('20241215-205044', '20241216-084431'),
     ('20241217-014915', '20241220-094708'),
     ('20241222-081854', '20241222-083141'),
     ('20241224-123639', '20241224-123914'), #check if elevation correction goes OK
     ('20250103-072540', '20250103-074247'),
     ('20250104-021318', '20250106-062100'), # from here onwards maybe some files already OK  (or bad) before the mentioned files
     ('20250107-010046', '20250107-062514'),
     ('20250110-002155', '20250112-073000'),
     ('20250113-202700', '20250113-221400'),
     ('20250114-015600', '20250114-063819'),
    #  ('20250111-013936', '20250114-063819'), 
     ('20250115-194048', '20250115215500'),
     ('20250116-011600', '20250116-234400'),
     ('20250117-060100', '20250117-070600'),
     ('20250118-014600', '20250118-144705'), # plenty of files not recorded properly (0 bytes)
     ('20250120-071620', '20250120-123900'),
     ('20250121-040120', '20250121-120002')
 ]

start_date = pd.Timestamp('2024-10-14')
end_date = pd.Timestamp('2025-01-25')

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024'
ERA_dir = os.path.join(base_dir, 'ERA5/netcdf')

all_dates = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D').date

fn_precip_stats = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/precip_temp_stats.csv"

# %% webcam images availability

img_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/webcam/jpg'
image_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))

image_fns = [f for f in sorted(os.listdir(f"{img_dir}")) if f.endswith(".jpg")]

pattern = re.compile(r"(\d{8}-\d{6})")
timestamps_list = []

for filename in image_fns:
    match = pattern.search(filename)
    timestamps_list.append(match.group(1))
timestamps = pd.to_datetime(timestamps_list).sort_values()

timestamps = timestamps[(timestamps >= start_date) & (timestamps <= end_date)]

df = pd.DataFrame({'timestamp': timestamps})
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time

all_gaps = []
for date in all_dates:
    day_start = pd.Timestamp(f"{date} 05:00")
    day_end = pd.Timestamp(f"{date} 17:00")

    group = df[df['date'] == date]
    times = group['timestamp']
    times = times[(times >= day_start) & (times <= day_end)]

    if times.empty:
        all_gaps.append({'start': day_start, 
                         'end': day_end, 
                         'gap': pd.Timedelta(days=1), 
                         'note': 'full day missing'})
        continue

    if times.iloc[0] > day_start + pd.Timedelta(minutes=6):
        all_gaps.append({'start': day_start,
                         'end': times.iloc[0],
                         'gap': times.iloc[0] - day_start,
                         'note': 'late start'
                         })
    
    if times.iloc[-1] < day_end - pd.Timedelta(minutes=6):
        all_gaps.append({'start': times.iloc[-1],
                         'end': day_end,
                         'gap': day_end - times.iloc[-1],
                         'note': 'early end'})
        
    deltas = times.diff().dropna()
    for i, delta in enumerate(deltas):
        if delta > pd.Timedelta(minutes=6):
            all_gaps.append({
                'start': times.iloc[i],
                'end': times.iloc[i+1],
                'gap': delta,
                'note': 'mid-day gap'
            })

gap_df = pd.DataFrame(all_gaps)

webcam = []
for date in all_dates:
    day_start = pd.Timestamp(f"{date} 05:00")
    day_end = pd.Timestamp(f"{date} 17:00")

    day_gaps = gap_df[
        (gap_df['start'] < day_end) & (gap_df['end'] > day_start) & (gap_df['note'] == 'fulll day missing')
    ]

    if not day_gaps.empty:
        continue

    day_gaps = gap_df[
        (gap_df['start'] < day_end) & (gap_df['end'] > day_start)
    ].sort_values('start')

    last_covered = day_start

    for _, gap in day_gaps.iterrows():
        gap_start = max(gap['start'], day_start)
        gap_end = min(gap['end'], day_end)

        if last_covered < gap_start:
            webcam.append((last_covered, gap_start))
        last_covered = max(last_covered, gap_end)

    if last_covered < day_end:
        webcam.append((last_covered, day_end))

# apparently recording error every day between 07:55 and 10:00 UTC (unclear why)
recgap_start = dt.time(7, 55)
recgap_end = dt.time(10, 0)
tolerance = 3

mask = ~(
    gap_df['start'].dt.time.apply(lambda t: isnear(t, recgap_start, tolerance)) & 
    gap_df['end'].dt.time.apply(lambda t: isnear(t, recgap_end, tolerance))
)

filtered_gaps = gap_df[mask]

# with pd.option_context("display.max_rows", 150, "display.max_columns", 5):
#     display(gap_df)


#%% 

def compute_precip_periods(all_dates, mira_dir, ERA_dir, threshold_VEL, min_precip_gates, min_duration, max_allowed_gap):
    precip_periods = []
    for date in all_dates:
        # print(f'{date} | threshold={threshold_VEL}, gates={min_precip_gates}, duration={min_duration}, gap={max_allowed_gap}')
        year, month, day = date.year, date.month, date.day
        flist = sorted(glob.glob(f"{mira_dir}/{year}/{month:02}/{day:02}/*_mmclx.nc"))
        if not flist:
            continue

        ERA1 = read_ERA_data(date, ERA_dir)
        ERA2 = read_ERA_data(date + pd.Timedelta(days=1), ERA_dir)
        if ERA1 and ERA2:
            ERA_data = xr.concat([ERA1, ERA2], dim='time').sortby('time')
        else:
            continue

        ds = xr.open_mfdataset(
            flist, combine='by_coords', preprocess=preprocess_mmclx,
            drop_variables=[var for var in variables_mira_all if var not in variables_mira]
        )

        altitudes = find_temp_levels(ERA_data, target_temps=[1.5, 0, -1])
        altitudes = safe_reindex(altitudes, 'time', ds.time, tolerance='30m', name='altitudes')

        low_gates = ds.isel(range=slice(0, 12))
        el_limit = ds.range[14].values

        precip_flag = (low_gates.VELg < threshold_VEL).sum(dim='range') >= min_precip_gates
        data_available = low_gates.VELg.notnull().any(dim='range')

        era_mask = (ERA_data['altitude'] >= radar_altitude) & (ERA_data['altitude'] <= el_limit + radar_altitude).broadcast_like(ERA_data['t'])
        mean_temps = ERA_data['t'].where(era_mask).mean(dim='isobaricInhPa', skipna=True)
        meltheights = altitudes['temp_0C_altitude']

        precip_flag, data_available, mean_temps, meltheights = dask.compute(
            precip_flag, data_available, mean_temps, meltheights
        )

        precip_flag = precip_flag.to_series().astype(bool)
        data_available = data_available.to_series().astype(bool)
        mean_temps = mean_temps.to_series().resample('5s').interpolate('linear')
        meltheights = meltheights.to_series()

        group_ids = precip_flag.ne(precip_flag.shift()).cumsum()
        grouped = precip_flag.groupby(group_ids)

        for _, group in grouped:
            if not group.iloc[0]:
                continue
            start, end = group.index[0], group.index[-1]
            duration = end - start

            if duration >= min_duration:
                precip_periods.append({
                    'start': start,
                    'end': end,
                    'duration_min': duration.total_seconds() / 60,
                })

    # Convert to DataFrame and summarize
    if precip_periods:
        df = pd.DataFrame(precip_periods)
        return {
            'num_events': len(df),
            'mean_duration_min': df['duration_min'].mean(),
            'total_precip_time_min': df['duration_min'].sum(),
        }
    else:
        return {'num_events': 0, 'mean_duration_min': 0, 'total_precip_time_min': 0}


def analyze_one_date(date, mira_dir, ERA_dir, param_grid):
    """
    Load all data for a given date once, then compute precipitation periods
    for all parameter combinations.
    """
    print(f"\n=== Processing {date} ===")
    year, month, day = date.year, date.month, date.day
    flist = sorted(glob.glob(f"{mira_dir}/{year}/{month:02}/{day:02}/*_mmclx.nc"))
    if not flist:
        print(f"No MIRA data for {date}")
        return []

    # Load ERA data
    ERA1 = read_ERA_data(date, ERA_dir)
    ERA2 = read_ERA_data(date + pd.Timedelta(days=1), ERA_dir)
    if not (ERA1 and ERA2):
        print(f"No ERA data for {date}")
        return []

    ERA_data = xr.concat([ERA1, ERA2], dim='time').sortby('time')

    # Load MIRA data
    ds = xr.open_mfdataset(
        flist,
        combine='by_coords',
        preprocess=preprocess_mmclx,
        drop_variables=[var for var in variables_mira_all if var not in variables_mira]
    )

    # Prepare common data
    altitudes = find_temp_levels(ERA_data, target_temps=[1.5, 0, -1])
    altitudes = safe_reindex(altitudes, 'time', ds.time, tolerance='30m', name='altitudes')

    low_gates = ds.isel(range=slice(0, 12))
    el_limit = ds.range[14].values

    era_mask = (
        (ERA_data['altitude'] >= radar_altitude)
        & (ERA_data['altitude'] <= el_limit + radar_altitude)
    ).broadcast_like(ERA_data['t'])
    mean_temps = ERA_data['t'].where(era_mask).mean(dim='isobaricInhPa', skipna=True)
    meltheights = altitudes['temp_0C_altitude']

    # Precompute (common to all parameter combos)
    mean_temps = mean_temps.to_series().resample('5s').interpolate('linear')
    meltheights = meltheights.to_series()

    results = []

    # --- Now loop through parameter combinations ---
    for threshold_VEL, min_precip_gates, min_duration, max_allowed_gap in param_grid:
        precip_flag = (low_gates.VELg < threshold_VEL).sum(dim='range') >= min_precip_gates
        precip_flag = precip_flag.to_series().astype(bool)

        group_ids = precip_flag.ne(precip_flag.shift()).cumsum()
        grouped = precip_flag.groupby(group_ids)

        precip_periods = []
        for _, group in grouped:
            if not group.iloc[0]:
                continue
            start, end = group.index[0], group.index[-1]
            duration = end - start
            if duration >= min_duration:
                precip_periods.append(duration.total_seconds() / 60)

        if precip_periods:
            results.append({
                'date': date,
                'threshold_VEL': threshold_VEL,
                'min_precip_gates': min_precip_gates,
                'min_duration_min': min_duration.total_seconds() / 60,
                'max_allowed_gap_min': max_allowed_gap.total_seconds() / 60,
                'num_events': len(precip_periods),
                'mean_duration_min': np.mean(precip_periods),
                'total_precip_time_min': np.sum(precip_periods)
            })
        else:
            results.append({
                'date': date,
                'threshold_VEL': threshold_VEL,
                'min_precip_gates': min_precip_gates,
                'min_duration_min': min_duration.total_seconds() / 60,
                'max_allowed_gap_min': max_allowed_gap.total_seconds() / 60,
                'num_events': 0,
                'mean_duration_min': 0,
                'total_precip_time_min': 0
            })

    ds.close()
    return results


vels = [-0.1, -0.2, -0.3]
gates = [5, 6, 7, 8]
min_durations = [timedelta(minutes=1), timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=5), timedelta(minutes=10)]
max_gaps = [timedelta(minutes=1), timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=5), timedelta(minutes=10)]
mira_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/L1"
param_grid = list(itertools.product(vels, gates, min_durations, max_gaps))

all_results = []
for date in all_dates:
    daily_results = analyze_one_date(date, mira_dir, ERA_dir, param_grid)
    all_results.extend(daily_results)

sensitivity_df = pd.DataFrame(all_results)
sensitivity_df.to_csv(f"{base_dir}/precip_sensitivity.csv", index=False)
print(sensitivity_df)

# results = []
# for threshold, gates, mindur, maxgap in itertools.product(vels, gates, min_durations, max_gaps):
#     res = compute_precip_periods(all_dates, mira_dir, ERA_dir, threshold, gates, mindur, maxgap)
#     res.update({
#         'threshold_VEL': threshold,
#         'min_precip_gates': gates,
#         'min_duration_min': mindur.total_seconds() / 60,
#         'max_allowed_gap_min': maxgap.total_seconds() / 60,
#     })
#     results.append(res)
# sensitivity_df = pd.DataFrame(results)

# %% calculate precipitation periods (save to file)
mira_dir = "/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MIRA/L1"
variables_mira_all = ['nfft', 'prf', 'NyquistVelocity', 'nave', 'ovl', 'zrg', 'rg0', 'drg', 
                     'lambda', 'microsec', 'tpow', 'npw1', 'npw2', 'cpw1', 'cpw2', 
                     'grst', 'azi', 'elv', 'aziv', 'northangle', 'elvv', 'LO_Frequency', 
                     'DetuneFine', 'SNR', 'VEL', 'RMS', 'LDR', 'NPK', 'SNRg', 'VELg', 'RMSg', 
                     'LDRg', 'NPKg', 'SNRplank', 'VELplank', 'RMSplank', 'LDRplank', 'NPKplank', 
                     'SNRrain', 'VELrain', 'RMSrain', 'LDRrain', 'NPKrain', 'SNRcl', 'VELcl', 
                     'RMScl', 'LDRcl', 'NPKcl', 'SNRice', 'VELice', 'RMSice', 'LDRice', 'NPKice', 
                     'RHO', 'RHOwav', 'DPS', 'DPSwav', 'HSDco', 'HSDcx', 'Ze', 'Zg', 'Z', 'RR', 
                      'LWC', 'TEMP', 'MeltHei', 'MeltHeiDet', 'MeltHeiDB', 'ISDRco', 'ISDRcx', 
                      'MRMco', 'MRMcx', 'RadarConst', 'SNRCorFaCo', 'SNRCorFaCx']
variables_mira = ['elv', 'Z', 'VELg', 'TEMP', 'MeltHei']

vels = [-0.05, -0.1, -0.2, -0.3]
gates = [5, 6, 7, 8]
durations = [1, 2, 5, 10] # minutes
gaps = [0.1, 0.25, 0.5, 0.75, 1] # fraction of duration

threshold_VEL = -0.2
min_precip_gates = 6 # half of the gates in the lower 600 m above the radar

min_duration = timedelta(minutes=1)
max_allowed_gap = timedelta(minutes=10)
min_steps = int(min_duration / timedelta(seconds=5))

precip_periods = []
for date in all_dates:
    print(f'{date}')
    year, month, day = date.year, date.month, date.day
    flist = sorted(glob.glob(f"{mira_dir}/{year}/{month:02}/{day:02}/*_mmclx.nc"))
    if not flist:
        continue
    ERA1 = read_ERA_data(date, ERA_dir)
    ERA2 = read_ERA_data(date + pd.Timedelta(days=1), ERA_dir)
    if ERA1 and ERA2:
        ERA_data = xr.concat([ERA1, ERA2], dim='time').sortby('time')
    else:
        continue
    ds = xr.open_mfdataset(flist, combine='by_coords', preprocess=preprocess_mmclx, drop_variables=[var for var in variables_mira_all if var not in variables_mira])
    altitudes = find_temp_levels(ERA_data, target_temps=[1.5, 0, -1])
    altitudes = safe_reindex(altitudes, 'time', ds.time, tolerance='30m', name='altitudes')

    low_gates = ds.isel(range=slice(0, 12))
    el_limit = ds.range[14].values

    precip_flag = (low_gates.VELg < threshold_VEL).sum(dim='range') >= min_precip_gates
    data_available = low_gates.VELg.notnull().any(dim='range')

    era_mask = (ERA_data['altitude'] >= radar_altitude) & (ERA_data['altitude'] <= el_limit + radar_altitude).broadcast_like(ERA_data['t'])
    mean_temps = ERA_data['t'].where(era_mask).mean(dim='isobaricInhPa', skipna=True)
    meltheights = altitudes['temp_0C_altitude']
    # mean_temps = low_gates.TEMP.mean(dim='range')
    # meltheights = low_gates.MeltHei

    precip_flag, data_available, mean_temps, meltheights = dask.compute(
        precip_flag, data_available, mean_temps, meltheights
    )

    def ensure_series_bool(obj):
        if isinstance(obj, xr.DataArray):
            return obj.to_series().astype(bool)
        return obj.astype(bool)
    
    precip_flag = ensure_series_bool(precip_flag)
    data_available = ensure_series_bool(data_available)
    mean_temps = mean_temps.to_series()
    mean_temps_fine = mean_temps.resample('5s').interpolate('linear')
    meltheights = meltheights.to_series()

    group_ids = precip_flag.ne(precip_flag.shift()).cumsum()
    grouped = precip_flag.groupby(group_ids)

    last_end = None

    for grid, group in grouped:
        if not group.iloc[0]:
            continue 

        start = group.index[0]
        end = group.index[-1]
        actual_duration = end - start

        if actual_duration >= min_duration:
            temp = mean_temps_fine[start:end].dropna()
            meltheight = meltheights[start:end].dropna()
            precip_periods.append({
                'start': start,
                'end': end,
                'duration_min': actual_duration.total_seconds() / 60,
                'temp_min': temp.min() if not temp.empty else None,
                'temp_max': temp.max() if not temp.empty else None,
                'temp_mean': temp.mean() if not temp.empty else None,
                'meltheight_min': meltheight.min() if not meltheight.empty else None,
                'meltheight_max': meltheight.max() if not meltheight.empty else None,
                'meltheight_mean': meltheight.mean() if not meltheight.empty else None,
            })
            last_end = end

df_precip_all = pd.DataFrame(precip_periods)

min_precip_duration_minutes = 2
gap_threshold_minutes = 5
df_precip = merge_precip_intervals(df_precip_all, gap_threshold_minutes=10, min_precip_duration_minutes=min_precip_duration_minutes)
# df_precip.to_csv(fn_precip_stats)
fn_precip = f"{base_dir}/precip_intervals_merged_gap{gap_threshold_minutes}min_mindur{min_precip_duration_minutes}min.xlsx"
df_precip.to_excel(fn_precip, index=False)

#%%

df_precip = pd.read_csv(fn_precip_stats)

#%% plotting
savefig = True
plot_webcam = True

start = pd.Timestamp('2024-10-1')
end = pd.Timestamp('2025-01-25')
# days = [1, 10, 20]
days = [1, 15]

major_ticks = pd.to_datetime([
    pd.Timestamp(year=dt.year, month=dt.month, day=d)
    for dt in pd.date_range(start+pd.Timedelta(days=16), end, freq='MS')  # MS = Month Start
    for d in days
    if pd.Timestamp(year=dt.year, month=dt.month, day=1).replace(day=d) <= end
])

add_dates = pd.to_datetime(['2025-01-25', '2024-10-14'])
major_ticks = major_ticks.union(add_dates).sort_values()

colors = {
    'mxpol_scan': '#2E8B57',      # Sea green (darker, more saturated)
    'mxpol_zenith': '#90EE90',    # Light green (better contrast)
    'mira_scan': '#DC143C',     # Crimson (true red)
    'mira_zenith': '#FF6B6B',     # Light coral (lighter red)
    'basta': "#FF8C00",           # Dark orange
    'webcam': '#8A2BE2',          # Blue violet
    'precip_liquid': '#000080',   # Dark blue for rain (T >= 1.5°C)
    'precip_solid': '#87CEEB'     # Light blue for snow (T < 1.5°C)
}

legend_handles = [
    Patch(facecolor=colors['mxpol_scan'], label='MXPol: scans + zenith'),
    Patch(facecolor=colors['mxpol_zenith'], label='MXPol: zenith only'),
    Patch(facecolor=colors['mira_scan'], label='MIRA: scanning'),
    Patch(facecolor=colors['mira_zenith'], label='MIRA: zenith only'),
    Patch(facecolor=colors['basta'], label='BASTA: zenith only'),
    Patch(facecolor='none', hatch='///', label='data quality issues'),
    Patch(facecolor=colors['precip_liquid'], label='Liquid precipitation (T ≥ 1.5°C)'),
    Patch(facecolor=colors['precip_solid'], label='Potentially solid precipitation (T < 1.5°C)'),
]

height = 0.7

if plot_webcam:
    yticks = [5, 4, 3, 2, 1]
    yticklabels = ['MXPol (9.4 GHz)', 'MIRA (35 GHz)', 'BASTA (95 GHz)', 'webcam images', 'precipitation']
    ypos = {'MXPol': 5,
            'MIRA': 4,
            'BASTA': 3,
            'webcam': 2,
            'precipitation': 1}
    fig, ax = plt.subplots(figsize=(len(yticks)*3+2, len(yticks)))
else:
    yticks = [4, 3, 2, 1]
    yticklabels = ['MXPol (9.4 GHz)', 'MIRA (35 GHz)', 'BASTA (95 GHz)', 'precipitation']
    ypos = {'MXPol': 4,
            'MIRA': 3,
            'BASTA': 2,
            'precipitation': 1}
    fig, ax = plt.subplots(figsize=(len(yticks)*4+2, len(yticks)+2))

# MXPol
for start, end in mxpol_scan_full:
    left = date2num(start)
    width = date2num(end) - left
    ax.barh(ypos['MXPol'], width, left=left, height=height, color=colors['mxpol_scan'], label='scans + zenith')
for start, end in mxpol_zenith_full:
    left = date2num(start)
    width = date2num(end) - left
    ax.barh(ypos['MXPol'], width, left=left, height=height, color=colors['mxpol_zenith'], alpha=0.5, label='zenith only')

for start, end in mxpol_lost_frequency:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    left = date2num(start_dt)
    width = date2num(end_dt) - left
    ax.barh(ypos['MXPol'], width, left=left, height=height, facecolor='none', hatch='///')

# MIRA
for (start, end), zenith in zip(mira_ranges, mira_zenith_flags):
    color = colors['mira_zenith'] if zenith else colors['mira_scan']
    left = date2num(start)
    width = date2num(end) - left
    ax.barh(ypos['MIRA'], width, left=left, height=height, color=color)

for start, end in mira_errors:
    start_dt = dtm(start)
    end_dt = dtm(end)
    left = date2num(start_dt)
    width = date2num(end_dt) - left
    ax.barh(ypos['MIRA'], width, left=left, height=height, facecolor='none', hatch='///')

# BASTA
for start, end in basta_ranges:
    left = date2num(start)
    width = date2num(end) - left
    ax.barh(ypos['BASTA'], width, left=left, height=height, color=colors['basta'])

for start, end in basta_errors:
    start_dt = dtm(start)
    end_dt = dtm(end)
    left = mdates.date2num(start_dt)
    width = mdates.date2num(end_dt) - left
    ax.barh(ypos['BASTA'], width, left=left, height=height, facecolor='none', hatch='///')

# webcam
if plot_webcam:
    for start, end in webcam:
        left = date2num(start)
        width = date2num(end) - left
        ax.barh(ypos['webcam'], width, left=left, height=0.5, color=colors['webcam'])

add_precipitation_to_plot(ax, df_precip, start_date, end_date, y_position=ypos['precipitation'])

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
# ax.set_xlabel("Date")
# ax.xaxis.set_major_locator(CustomMajorLocator())
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 10, 20]))
ax.set_xticks(major_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(True, which='major', axis='x', linestyle='--', linewidth=1.)
ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.3)
ax.set_xlim(datetime(2024, 10, 14), datetime(2025, 1, 25))
plt.xticks(rotation=45)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')
plt.title("Radar data availability during the CHOPIN campaign (Oct 2024 – Jan 2025)", fontweight='bold')

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
# ax.legend(unique.values(), unique.keys(), loc='upper left')
ax.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=4)

plt.tight_layout()
if plot_webcam:
    plt.subplots_adjust(bottom=0.1)
else:
    plt.subplots_adjust(bottom=0.2)
if savefig:
    plt.savefig("/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/radardata_availability_overview.png", dpi=300, bbox_inches='tight', facecolor='w')
plt.show()
# %%

df_full = add_precip_type(df_precip)
fn_precip_stats_xls = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/precip_temp_stats.xlsx'

# df_full.to_excel(fn_precip_stats_xls)
df_full['start'] = pd.to_datetime(df_full['start'])
precip_dates = sorted({(i.strftime('%Y%m%d') if isinstance(i, pd.Timestamp) else i)for i in df_full['start']})
precip_days = len(precip_dates)

# print total hours of rain vs. snow
df_full.groupby('precip_type')['duration_min'].sum() / 60
df_full['date'] = df_full['start'].dt.date

prdays = []
for i in precip_dates:
    prdays.append(i[:10])
print(f"days with precipitation: {len(np.unique(prdays))}")

unique_solid_days = (
    df_full.loc[df_full['precip_type'] == 'solid', 'start']
    .dt.date
    .nunique()
)
print(f"days with solid precipitation: {unique_solid_days}")

unique_liquid_days = (
    df_full.loc[df_full['precip_type'] == 'liquid', 'start']
    .dt.date
    .nunique()
)
print(f"days with liquid precipitation: {unique_liquid_days}")


only_solid_days = (
    df_full.groupby('date')['precip_type']
    .nunique()
    .loc[lambda x: x == 1]
    .index
)
only_solid_days = [
    d for d in only_solid_days
    if (df_full.loc[df_full['date'] == d, 'precip_type'].iloc[0] == 'solid')
]
print(f"days with only solid precip: {len(only_solid_days)}")