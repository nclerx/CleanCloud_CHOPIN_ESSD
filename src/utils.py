#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src.constants_input import headers, G, R_earth, latlon_Helmos, radar_altitude, temps, MXPol_lost_frequency_times
from src import glob, os, np, pd, xr, da, re, datetime, timedelta, interp1d

import logging
from scipy import stats


def find_nearest(array, value):
    """find closest value in array to specified value and return index of that value
    array: array of values
    value: float"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_idx(lats, lons, target_coords):
    """
    lats: array of latitudes
    lons: array of longitudes
    target_coords: tuple of target coords (lat-lon)

    returns lat_idx and lon_idx, integer values of closest lat & lon id 
    """
    dist = np.sqrt((lats - target_coords[0])**2 + (lons - target_coords[1])**2)
    lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)

    return lat_idx, lon_idx


def safe_reindex(data, dim, full_dim, tolerance, method='nearest'):
    """reindex xarray object onto a target coordinate grid with specified tolerance and interpolation method
    data: xr.DataArray or xr.Dataset
    dim: str, dimension to reindex ('time' or 'range')
    full_dim: array-like, target coordinate values for the specified dimension
    tolerance: str, float or pd.Timedelta, maximum allowed gap for filling/matching
    method: {'nearest', 'linear'}, default 'nearest', reindexing strategy (nearest-neighbor or linear interpolation)"""
    if method == "linear":
        return data.interp({dim: full_dim}, method="linear")
    else:
        gap_tolerance = pd.Timedelta(tolerance) if dim == "time" else tolerance
        return data.reindex(
            {dim: full_dim},
            method=method,
            tolerance=gap_tolerance,
        )

def remove_nans(*arrays):
    """remove nan values from multiple arrays, keeping only the indices where all arrays have valid (non-nan) values
    *arrays: array-like"""
    assert all(array.shape ==
               arrays[0].shape for array in arrays), "All arrays must have the same shape"
    mask = np.ones_like(arrays[0], dtype=bool)
    for array in arrays:
        mask &= ~np.isnan(array)
    return tuple(array[mask] for array in arrays)


def create_file_with_headers(headers, file_path):
    """create file for writing stats output"""
    # check if the file already exists
    if not os.path.exists(file_path):
        # create a new file and write the headers
        with open(file_path, 'w') as file:
            file.write('\t'.join(headers) + '\n')


def sum_column(file_path, column_name):
    """sum all the values in the given column (by header name) in the file"""
    total_sum = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split('\t')
        # map headers to column indices
        header_index = {header[i]: i for i in range(len(header))}

        # iterate through the rest of the rows
        for line in lines[1:]:
            row = line.strip().split('\t')
            try:
                # add value from the specified column to the sum
                total_sum += float(row[header_index[column_name]])
            except ValueError:
                continue

    return total_sum


def append_data(file_path, data):
    """append data to specified file"""
    with open(file_path, 'a') as file:
        # convert each element to string, but leave floats as they are
        formatted_data = [f"{item:.6f}" if isinstance(item, float) or isinstance(
            item, np.float32) else str(item) for item in data]
        file.write('\t'.join(formatted_data) + '\n')


def write_calibrationfile(output_path, data):
    """write calibration statistics to file"""
    create_file_with_headers(headers, output_path)
    append_data(output_path, data)


def is_within_limits(value, limits):
    """check whether is within specified limits
    value: float
    limits: tuple"""
    return limits[0] <= value <= limits[1]


def remove_nans(data1, data2):
    """match two data arrays such that there are no data pairs in which one of
    the values is a nan"""
    mask = ~np.isnan(data1) & ~np.isnan(data2)

    data1_clean = data1[mask]
    data2_clean = data2[mask]

    return data1_clean, data2_clean


def fit_slope(x, y):
    """fit y = x + b" with fixed slope = 1 and return fit statistics"""
    intercept = np.mean(y - x)  # optimal intercept
    intercept_std = np.std(y - (x + intercept))
    r = np.corrcoef(x, y)[0, 1]  # pearson correlation coefficient

    return intercept, intercept_std, r


def ensure_time_range_order(*datasets):
    """check whether data set dimensions are in the correct or der 
    - to use before flatting specific arrays.
    (not sure if the transposing actually works)"""
    for dataset in datasets:
        if dataset.dims != ('time', 'range'):
            # if not, transpose the dimensions
            dataset = dataset.transpose('time', 'range')
    return datasets


def generate_date_range(start_date, end_date):
    """generates a list of dates between start_date and end_date (inclusive)
    start_date, end_date: string in format 'YYYY-MM-DD' or datetime objects"""
    start = datetime.strptime(
        start_date, "%Y-%m-%d") if type(start_date) is str else start_date
    end = datetime.strptime(
        end_date, "%Y-%m-%d") if type(end_date) is str else end_date
    delta = timedelta(days=1)

    dates = []
    while start <= end:
        dates.append(start)
        start += delta
    return dates


def calc_alt(h):
    """calculate geometric height from geopotential height"""
    alt = (R_earth * (h/G)) / (R_earth - (h/G))

    return alt


def clean_dataframe(df):
    """strip whitespaces from strings, convert trings to float/int where possible"""
    df = df.copy()

    # strip whitespace and replace empty strings with nans
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace(
            '', np.nan).infer_objects(copy=False)
    df = df.infer_objects(copy=False)

    return df


def read_ERA_data(date, ERA_dir, large_file=False, reduce_dims=True, latlon=None, ref_altitude=None):
    """load ERA5 reanalysis .nc for a specific date, optionally reduce dimensions to a specific lat-lon box and/or altitude range
    date: datetime-like, date for which to load ERA5 data used to match the ERA5 filename
    ERA_dir: str, directory containing ERA5 .nc files
    large_file: bool, default False, whether to look for large ERA5 files (with '_large' in filename)
    reduce_dims: bool, default True, whether to reduce dimensions to a specific lat-lon box and/or altitude range
    latlon: tuple (optional), default None, bounding latitude and longitude ranges as ((min_lat, max_lat), (min_lon, max_lon))
    ref_altitude: array-like float or None (optional), default None, reference altitude for the dataset
    returns ds: xr.Dataset containing the loaded ERA5 dataset with calculated 'altitude' (m)
    """
    # latlon = ([min_lat, max_lat], [min_lon, max_lon])
    if large_file:
        files = glob.glob(f"{ERA_dir}/*{date.strftime('%Y%m%d')}_large.nc")
        ERA_file = files[0] if files else None
    else:
        files = glob.glob(f"{ERA_dir}/*{date.strftime('%Y%m%d')}*.nc")
        ERA_file = files[0] if files else None

    if ERA_file is None:
        print(f"No ERA5 file for {date.strftime('%Y-%m-%d')}")
        return None
    
    ds = xr.open_dataset(ERA_file)
    if reduce_dims:
        if latlon is None:
            ds = ds.sel(latitude=latlon_Helmos[0], longitude=latlon_Helmos[1], method='nearest')
        else:
            lat_range, lon_range = latlon
            min_lat, max_lat = lat_range
            min_lon, max_lon = lon_range
            if ds.latitude.values[0] > ds.latitude.values[-1]:
                lat_slice = slice(max_lat, min_lat)
            else:
                lat_slice = slice(min_lat, max_lat)
            lon_slice = slice(min_lon, max_lon)
            ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
    
    if ref_altitude is None:
        altitude = calc_alt(ds['z'].mean(dim=[d for d in ds['z'].dims if d != 'isobaricInhPa']))
    else:
        altitude = ref_altitude
    
    ds = ds.assign_coords(altitude=altitude)
    ds['t'] = ds['t'] - 273.15
    
    return ds


def parse_cloudtop_range(x):
    """parse string representation of cloudtop ranges into a list of floats, handling empty or NaN values
    x: str or NaN (raw string values from the 'cloudtop_ranges' column in the DFRs dataframe)
    returns list or float (parsed numeric value or empty if 'x' is NaN or empty)"""
    if pd.isna(x) or not x.strip():
        return []
    x = x.strip().replace('[', '').replace(']', '')  # remove brackets
    return [float(val) for val in x.split() if val]


def get_DFRdata(DFR_fns):
    """load and process cloudtop DFR data from a list of CSV files
    DFR_fns: list of str, file paths to CSV files containing cloudtop DFR data
    returns DFRs: pd.DataFrame, cleaned and merged dataframe with additional daily statistics"""
    dataframes = []

    for f in DFR_fns:
        if os.path.getsize(f) > 0:
            try:
                df = pd.read_csv(f)
                dataframes.append(df)
            except pd.errors.EmptyDataError:
                continue

    DFRs = pd.concat(dataframes, ignore_index=True)
    DFRs['time'] = pd.to_datetime(DFRs['time'], format='mixed')
    DFRs['date'] = DFRs['time'].dt.date

    DFRs['cloudtop_ranges'] = DFRs['cloudtop_ranges'].apply(
        parse_cloudtop_range)
    DFRs['cloudtop_min'] = DFRs['cloudtop_ranges'].apply(
        lambda r: min(r) if r else np.nan)
    DFRs['cloudtop_max'] = DFRs['cloudtop_ranges'].apply(
        lambda r: max(r) if r else np.nan)
    DFRs['cloudtop_avg'] = DFRs['cloudtop_ranges'].apply(
        lambda r: np.mean(r) if r else np.nan)

    daily_avg = DFRs.groupby('date')[
        ['cloudtop_min', 'cloudtop_max', 'cloudtop_avg', 'mean_DFR']].mean().reset_index()
    daily_avg = daily_avg.rename(columns={
        'cloudtop_min': 'daily_cloudtop_min',
        'cloudtop_max': 'daily_cloudtop_max',
        'cloudtop_avg': 'daily_cloudtop_avg',
        'mean_DFR': 'daily_mean_DFR'
    })
    DFRs = DFRs.drop(columns=['DFR', 'cloudtop_times', 'cloudtop_ranges'])
    DFRs = DFRs.merge(daily_avg, on='date', how='left')
    DFRs = DFRs.sort_values(by='time')

    return DFRs


def format_list(items):
    """format a list of items into a human-readable string with commas and 'and' before the last item"""
    items = list(map(str, items))
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return f"{', '.join(items[:-1])} and {items[-1]}"


def merge_hourly_peaktree_files(outputdir, date):
    """merge all hourly peaktree files for a specific date into a single daily file
    outputdir: str, directory containing the hourly files
    date: datetime or str, date for which to merge the files (format 'YYYYMMDD')
    writes a daily merged file to <outputdir>/<YYYYMMDD>_daily_Hlm_peakTree_mbr5.nc4 
    and returns True if successful, False otherwise
    """
    hourly_dir = os.path.join(outputdir, 'hourly_files')
    if not os.path.exists(hourly_dir):
        print(f"Error: directory {hourly_dir} does not exist")
        return False

    if isinstance(date, datetime):
        date_obj = date
    elif isinstance(date, str):
        try:
            date_obj = datetime.strptime(date, '%Y%m%d')
        except ValueError:
            print(f"Error: date string '{date}' must be in format 'YYYYMMDD'")
            return False
    else:
        print(f"Error: date must be datetime object or string in format 'YYYYMMDD'")
        return False

    year, month, day = date_obj.year, date_obj.month, date_obj.day
    files = sorted(glob.glob(f"{outputdir}/hourly_files/{year}{month:02}{day:02}_*.nc4"))

    if not files:
        print(f"Error: no hourly files found for {year}{month:02}{day:02}")
        return False
    
    ds = xr.open_mfdataset(files)
    encoding = {var: {
        'dtype': ds[var].dtype,  # Keep original dtype for all variables
        'zlib': True,
        'complevel': 4
    } for var in ds.variables if ds[var].dtype.kind != 'M'}
    outname = f"{outputdir}/{year}{month:02}{day:02}_daily_Hlm_peakTree_mbr5.nc4"
    ds.to_netcdf(outname, format='NETCDF4', mode='w',
                 engine='h5netcdf', encoding=encoding)
    
    return True


def find_temp_levels(ds, target_temps=temps, temp_var='t', alt_var='altitude', interp_kind='linear', extrapolate=False):
    """
    find altitude (wrt msl) of target_temperature (base input dataset = ERA output data)
    returns xarray.DataArray of altitude values
    """
    temp_celsius = ds[temp_var] - \
        273.15 if ds[temp_var].mean() > 200 else ds[temp_var]

    results = {}

    for target_temp in target_temps:
        target_altitudes = []

        for time_idx in range(len(ds.time)):
            temp = temp_celsius.isel(time=time_idx).values
            alt = ds[alt_var].isel(time=time_idx).values

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

    return xr.Dataset(results)


def flatten_for_densityplot(da1, da2):
    """flatten + clean input data arrays for plotting density histogram"""
    da1_flat = da1.flatten()
    da2_flat = da2.flatten()
    valid_mask = ~(np.isnan(da1_flat) | np.isnan(da2_flat))
    da1_clean = da1_flat[valid_mask]
    da2_clean = da2_flat[valid_mask]

    return da1_clean, da2_clean


def find_temperature_altitude(ERA_data, altitudes, time=None, alt_range=(0, 10), temps=temps):
    """
    find altitudes corresponding to specified temperatures in an ERA5 temperature profile

    parameters:
        ERA_data : xr.Dataset (temperature variable named 't' with dimensions including 'time' and 'altitude')    
        altitudes : xr.DataArray or np.ndarray (array of altitude values corresponding to vertical temperature-dimension)
        time : datetime-like, optional (if None, the first time index is used)
        alt_range : tuple of float, minimum and maximum altitude range (same units as `altitudes`)
        temps : list or array-like of target temperature (in Kelvin) for which to find the corresponding altitudes

    returns:
        temp_altitudes : dict mapping each requested temperature to the corresponding ERA5 altitude (float) within 
        ±2 K tolerance (temperature key is omitted if no corresponding altitude)
    """
    alt_min, alt_max = alt_range
    valid_alt_mask = (altitudes >= alt_min) & (altitudes <= alt_max)
    valid_altitudes = altitudes[valid_alt_mask]

    if time is None:
        temp_profile = ERA_data['t'].isel(time=0).values
    else:
        temp_profile = ERA_data['t'].interp(time=time, method='linear').values

    temp_altitudes = {}

    for temp in temps:
        temp_diff = np.abs(temp_profile[valid_alt_mask] - temp)
        idx = np.argmin(temp_diff)
        if temp_diff[idx] < 2.0:
            temp_altitudes[temp] = valid_altitudes[idx]['altitude'].values

    return temp_altitudes


def find_temperature_altitude_xr(
    ERA_data: xr.Dataset,
    temps: list | np.ndarray = temps,
    alt_range: tuple = (0, 15000),
    radar_altitude: float = 0.0,) -> xr.Dataset:
    """
    Find altitudes corresponding to specified temperatures in an ERA5 temperature profile,
    optionally for all timesteps, to use directly as a zarr variable (e.g. 'zdi').

    Parameters
    ----------
    ERA_data : xr.Dataset
        ERA5 dataset containing variable 't' [K] with dimensions ('time', 'altitude')
    temps : list or np.ndarray, default [0]
        Target temperatures in °C or K (values <100 assumed to be °C, converted to K).
    alt_range : tuple, default (0, 15000)
        Minimum and maximum altitude range (in meters).
    radar_altitude : float, default 0.0
        Radar altitude above sea level [m]. Will be subtracted from ERA altitudes
        to return altitude *above radar level*.

    Returns
    -------
    xr.Dataset
        Dataset with one variable per target temperature (e.g. 'z0degC' for 0°C),
        each containing altitude [m] as a function of time.
    """
    alt_min, alt_max = alt_range

    altitude = ERA_data['altitude']  # (time, isobaricInhPa)
    temperature = ERA_data['t']   # (time, isobaricInhPa)
    times = ERA_data['time']

    results = {}

    # loop over target temps
    for target_T in temps:
        tvals = []
        for t in times:
            alt_profile = altitude.sel(time=t).values  # 1D
            temp_profile = temperature.sel(time=t).values  # 1D

            mask = (alt_profile >= alt_min) & (alt_profile <= alt_max)
            if not np.any(mask):
                tvals.append(np.nan)
                continue

            alt_valid = alt_profile[mask]
            temp_valid = temp_profile[mask]

            diff = np.abs(temp_valid - target_T)
            idx = np.argmin(diff)
            if diff[idx] < 2.0:
                tval = alt_valid[idx] - radar_altitude
                tvals.append(tval)
            else:
                tvals.append(np.nan)

        # create DataArray for this temperature
        label = f"z{int(target_T):+d}degC".replace("+0", "0")
        da = xr.DataArray(
            tvals,
            dims=["time"],
            coords={"time": times},
            attrs={
                "long_name": f"Altitude of {target_T:.0f} °C isotherm",
                "units": "m above radar",
                "tolerance": "±2 K",
                "source": "ERA5 temperature profile",
            },
        )
        results[label] = da

    return xr.Dataset(results)


def daily_monthly_stats(da):
    """
    Compute daily and monthly summary statistics (min, max, mean) from an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension. The time coordinate should be compatible
        with xarray's resampling functionality (i.e., datetime-like values).

    Returns
    -------
    daily_stats : xr.DataArray
        DataArray containing daily minimum, maximum, and mean values.
        Has an added 'statistic' dimension with labels ['min', 'max', 'mean'].
        Dimensions: ('time', 'statistic', ...)

    monthly_stats : xr.DataArray
        DataArray containing monthly minimum, maximum, and mean values.
        Has an added 'statistic' dimension with labels ['min', 'max', 'mean'].
        Dimensions: ('time', 'statistic', ...)

    Notes
    -----
    - Uses xarray's `resample` method with '1D' (daily) and '1M' (monthly) frequencies.
    - The output retains other non-time dimensions (if present) from the input DataArray.
    - The 'statistic' coordinate distinguishes between the computed metrics.
    """
    daily_min = da.resample(time='1D').min()
    daily_max = da.resample(time='1D').max()
    daily_mean = da.resample(time='1D').mean()
    daily_stats = xr.concat(
        [daily_min, daily_max, daily_mean], dim='statistic')
    daily_stats = daily_stats.assign_coords(statistic=['min', 'max', 'mean'])

    monthly_min = da.resample(time='1M').min()
    monthly_max = da.resample(time='1M').max()
    monthly_mean = da.resample(time='1M').mean()
    monthly_stats = xr.concat(
        [monthly_min, monthly_max, monthly_mean], dim='statistic')
    monthly_stats = monthly_stats.assign_coords(
        statistic=['min', 'max', 'mean'])

    return daily_stats, monthly_stats


def initialize_dask_cluster(minimum_memory=None):
    """initialize Dask Cluster"""
    import dask
    import psutil

    # Silence dask warnings
    # dask.config.set({"logging.distributed": "error"})
    # Import dask.distributed after setting the config
    from dask.distributed import Client, LocalCluster
    from dask.utils import parse_bytes

    # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Retrieve the number of processes to run
    available_workers = os.cpu_count() - 2  # if not set, all CPUs minus 2
    num_workers = dask.config.get("num_workers", available_workers)

    # If memory limit specified, ensure correct amount of workers
    if minimum_memory is not None:
        # Compute available memory (in bytes)
        total_memory = psutil.virtual_memory().total
        # Get minimum memory per worker (in bytes)
        minimum_memory = parse_bytes(minimum_memory)
        # Determine number of workers constrained by memory
        maximum_workers_allowed = max(1, total_memory // minimum_memory)
        # Respect both CPU and memory requirements
        num_workers = min(maximum_workers_allowed, num_workers)

    # Create dask.distributed local cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit=0,  # this avoid flexible dask memory management
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    return cluster, client


def check_resampling(ds, dimension, target_coords, tolerance, verbose=False):
    """check if resampling is needed along specified dimension (returns True if so)"""
    if dimension not in ds.dims:
        raise ValueError(
            f"Dimension '{dimension}' not found in dataset dimensions {ds.dims}")

    if len(ds[dimension]) < 2:
        return False

    coords = ds[dimension].values
    target_coords = np.asarray(target_coords)

    if len(coords) != len(target_coords):
        if verbose:
            print(
                f"  {dimension}: length mismatch {len(coords)} != {len(target_coords)}")
        return True

    if dimension == 'time':
        tolerance_val = pd.Timedelta(
            tolerance).total_seconds() * 1e9  # nanoseconds
        coords_ns = coords.astype('datetime64[ns]').astype(np.int64)
        target_ns = target_coords.astype('datetime64[ns]').astype(np.int64)
        values_match = np.allclose(coords_ns, target_ns, atol=tolerance_val)

        if verbose:
            print(f"  {dimension}: values_match={values_match}")
            print(f"            range: {coords[0]} to {coords[-1]}")
            print(
                f"            target: {target_coords[0]} to {target_coords[-1]}")

    elif dimension == 'range':
        tolerance_val = float(tolerance)
        values_match = np.allclose(coords, target_coords, atol=tolerance_val)

        if verbose:
            print(f"  {dimension}: values_match={values_match}")
            print(f"            range: {coords[0]:.1f} to {coords[-1]:.1f}")
            print(
                f"            target: {target_coords[0]:.1f} to {target_coords[-1]:.1f}")

    return not values_match


def format_elapsed(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """write timedelta as HH:MM:SS (progress checking)"""
    delta = end - start
    total_seconds = int(delta.total_seconds())

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def align_times_to_grid(daily_ds, target_time, tol_factor=0.51):
    """resample xr dataset to target time grid, using nearest neighbor interpolation with specified tolerance factor
    daily_ds: xr.Dataset, input dataset with 'time' dimension
    target_time: pd.DatetimeIndex (or similar), target time grid to align to
    tol_factor: float, default 0.51, factor to determine tolerance for nearest neighbor matching (as a fraction of the time step)
    returns: xr.Dataset, resampled dataset aligned to target_time"""
    if daily_ds.sizes.get("time", 0) == 0:
        # no source data for this window, return an all-NaN dataset on the target grid
        data_vars = {
            var: (da.dims, np.full((len(target_time),) + da.shape[1:], np.nan, dtype=da.dtype))
            for var, da in daily_ds.data_vars.items()
        }
        coords = {k: v for k, v in daily_ds.coords.items() if k != 'time'}
        coords['time'] = target_time
        return xr.Dataset(data_vars, coords=coords, attrs=daily_ds.attrs)
    
    src_times = daily_ds.time.values.astype("datetime64[ns]").astype(np.int64)
    tgt_times = target_time.values.astype("datetime64[ns]").astype(np.int64)

    idx = np.searchsorted(src_times, tgt_times)
    idx = np.clip(idx, 1, len(src_times) - 1)
    before = np.abs(src_times[idx - 1] - tgt_times)
    after = np.abs(src_times[idx] - tgt_times)
    nearest = np.where(before < after, idx - 1, idx)

    if target_time.freq is not None:
        tol_ns = int(
            tol_factor * pd.to_timedelta(target_time.freq).total_seconds() * 1e9)
    else:
        dt = np.median(np.diff(src_times))
        tol_ns = int(tol_factor * dt)
    tolerance = np.timedelta64(tol_ns, "ns")

    mask = np.abs(src_times[nearest] - tgt_times) <= tolerance

    data_vars = {
        var: (da.dims, np.full((len(target_time),) +
              da.shape[1:], np.nan, dtype=da.dtype))
        for var, da in daily_ds.data_vars.items()
    }

    for var, da in daily_ds.data_vars.items():
        data_vars[var][1][mask] = da.values[nearest[mask]]

    coords = {k: v for k, v in daily_ds.coords.items() if k != 'time'}
    coords['time'] = target_time

    new_ds = xr.Dataset(data_vars, coords=coords, attrs=daily_ds.attrs)

    return new_ds


def determine_nbins(data, max_bins=200):
    """Freedman-Diaconis rule-based bin quantity identification with sensible limits"""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:  # handle case where IQR is 0
        return 50

    bin_width = 2 * iqr / (len(data) ** (1/3))
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    n_bins = max(10, min(n_bins, max_bins))    # constrain to reasonable range

    return n_bins


def determine_nhexbins(x, y):
    """freedman-diaconis rule-based bin # identification for hexbin plot"""
    x = da.asarray(x)
    y = da.asarray(y)
    if isinstance(x, da.Array):
        # Use small random subset to avoid huge compute
        x = x[::1000].compute()
        y = y[::1000].compute()

    q75_x, q25_x = np.percentile(x, [75, 25])
    iqr_x = q75_x - q25_x
    bin_width_x = 2 * iqr_x / (len(x) ** (1/3)) if iqr_x > 0 else 1.0

    q75_y, q25_y = np.percentile(y, [75, 25])
    iqr_y = q75_y - q25_y
    bin_width_y = 2 * iqr_y / (len(y) ** (1/3)) if iqr_y > 0 else 1.0

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    n_bins_x = int(np.ceil((x_max - x_min) / bin_width_x))
    n_bins_y = int(np.ceil((y_max - y_min) / bin_width_y))
    gridsize = int(np.sqrt(n_bins_x * n_bins_y))

    return max(gridsize, 10)


def mask_to_intervals(mask, time_coord='time', min_gap=None):
    """return start/end timestamps of True-runs in a boolean mask (pandas groupby version)
    if min_gap is set, small False-gaps (<= min_gap points) are filled in first, merging nearby True-runs."""

    mask_series = pd.Series(
        mask.values, index=pd.to_datetime(mask[time_coord].values))

    if min_gap is not None:
        mask_filled = mask_series.copy()
        false_groups = (~mask_series).astype(int).groupby(
            mask_series.ne(mask_series.shift()).cumsum())
        for group_id, group in false_groups:
            if len(group) <= min_gap and group.iloc[0] == 1:  # small False gap
                mask_filled.loc[group.index] = True
        mask_series = mask_filled

    # find where mask changes
    change = mask_series.ne(mask_series.shift(fill_value=mask_series.iloc[0]))
    # assign a group id to each contiguous block
    groups = change.cumsum()

    intervals = []
    for _, group in mask_series.groupby(groups):
        if group.iloc[0]:  # only True intervals
            intervals.append((group.index[0], group.index[-1]))
    return intervals


def mask_to_intervals_fast(mask, time_coord='time', min_gap=None):
    """same as mask_to_intervals but vectorized with numpy diff/cumsum for speed on large arrays"""
    time = pd.to_datetime(mask[time_coord].values)
    values = mask.values.astype(bool)
    if min_gap is not None:
        is_false = ~values
        change = np.diff(is_false, prepend=is_false[0])
        run_id = np.cumsum(change != 0)

        run_lengths = np.bincount(run_id)
        run_is_false = is_false[np.r_[0, np.where(change != 0)[0]]]

        small_false_runs = (run_is_false & (run_lengths <= min_gap))
        fill_mask = small_false_runs[run_id]

        values = values | fill_mask

    change = np.diff(values, prepend=values[0])
    run_id = np.cumsum(change != 0)

    intervals = []
    for rid in np.unique(run_id[values]):
        idx = np.where(run_id == rid)[0]
        intervals.append((time[idx[0]], time[idx[-1]]))

    return intervals  


def mask_to_intervals_numpy(values, time, min_gap=None):
    """same as mask_to_intervals_fast but takes raw values/time arrays instead of an xarray object (no time_coord lookup)"""
    values = values.astype(bool)
    n = len(values)

    if min_gap is not None:
        is_false = ~values
        change = np.diff(is_false.astype(int), prepend=int(not is_false[0]))
        run_starts = np.where(change == 1)[0]
        run_ends = np.where(change == -1)[0]

        if len(run_starts) == 0:
            run_starts = np.array([0])
        if len(run_ends) == 0 or run_ends[-1] < run_starts[-1]:
            run_ends = np.append(run_ends, n)

        run_lengths = run_ends - run_starts
        small_false_runs = (run_lengths <= min_gap) & (~values[run_starts])
        for start, length in zip(run_starts[small_false_runs], run_lengths[small_false_runs]):
            values[start:start+length] = True

    change = np.diff(values.astype(int), prepend=int(not values[0]))
    interval_starts = np.where(change == 1)[0]
    interval_ends = np.where(change == -1)[0]

    if len(interval_starts) == 0:
        return []
    if len(interval_ends) == 0 or interval_ends[-1] < interval_starts[-1]:
        interval_ends = np.append(interval_ends, n-1)

    intervals = list(zip(time[interval_starts], time[interval_ends]))
    return intervals

    
def fast_intervals(mask, times):
    """return start/end intervals from boolean mask in a fully vectorized way."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    # edge cases
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    return list(zip(times[starts], times[ends]))


def calculate_edges(centers):
    """calculates the edges for pcolormesh from a 1D array of centers."""
    spacing = np.diff(centers)

    start_edge = centers[0] - spacing[0] / 2
    mid_edges = centers[:-1] + spacing / 2
    end_edge = centers[-1] + spacing[-1] / 2

    return np.concatenate([np.array([start_edge]), mid_edges, np.array([end_edge])])


def combine_intervals_to_patch(intervals, ymin=0, ymax=1):
    """Convert a list of (start, end) intervals into vertices for a Polygon."""
    verts = []
    for start, end in intervals:
        verts.extend([(start, ymin), (start, ymax), (end, ymax), (end, ymin)])
    return verts


def merge_close_intervals(intervals, max_gap_minutes=5):
    """merge consecutive (start, end) intervals if the gap between them is <= max_gap_minutes"""
    starts, ends = zip(*intervals)
    starts = np.array(starts)
    ends = np.array(ends)
    max_gap = np.timedelta64(max_gap_minutes, 'm')
    gaps = starts[1:] - ends[:-1]
    split_points = np.where(gaps > max_gap)[0] + 1
    merged = []
    prev = 0
    for point in split_points:
        merged.append((starts[prev], ends[point-1]))
        prev = point
    merged.append((starts[prev], ends[-1]))
    return merged


def intervals_to_mask(intervals, time_vals):
    """convert a list of (start, end) intervals back into a boolean mask over time_vals"""
    time_vals = pd.to_datetime(time_vals).values
    mask = np.zeros(len(time_vals), dtype=bool)
    starts, ends = zip(*intervals)
    starts = np.array(starts, dtype=time_vals.dtype)
    ends = np.array(ends, dtype=time_vals.dtype)
    starts_idx = np.searchsorted(time_vals, starts, side='left')
    ends_idx = np.searchsorted(time_vals, ends, side='right')
    for s, e in zip(starts_idx, ends_idx):
        mask[s:e] = True
    return mask


def check_time(time, intervals=MXPol_lost_frequency_times):
    """check if a given time falls within any of the specified 'YYYYMMDD-HHMMSS' string intervals"""
    for start_str, end_str in intervals:
        start_time = datetime.strptime(start_str, '%Y%m%d-%H%M%S')
        end_time = datetime.strptime(end_str, '%Y%m%d-%H%M%S')
        if start_time <= time <= end_time:
            return True
    return False


def format_elapsed(seconds):
    """format elapsed time in seconds into a human-readable string"""
    c = pd.Timedelta(seconds=seconds).components
    if c.days:
        return f"{c.days}d {c.hours}h {c.minutes}m {c.seconds}s"
    elif c.hours:
        return f"{c.hours}h {c.minutes}m {c.seconds}s"
    elif c.minutes:
        return f"{c.minutes}m {c.seconds}s"
    else:
        return f"{seconds:.2f}s"


def fix_encoding(ds, fill_value=-9999.0):
    """remove conflicting _FillValue/missing_value encoding and attrs, replace with 
    a single consistent fill value (used to correct MXPol L1 .nc-files)"""
    for var in list(ds.data_vars) + list(ds.coords):
        # determine the correct fill value for this dtype
        if np.issubdtype(ds[var].dtype, np.integer):
            fv = np.int8(-1) if ds[var].dtype == np.int8 else np.int32(-9999)
        elif np.issubdtype(ds[var].dtype, np.floating):
            fv = np.float32(fill_value) if ds[var].dtype == np.float32 else fill_value
        else:
            continue  # skip string/object vars

        # clear both from encoding and attrs, then set consistently
        ds[var].encoding.pop('_FillValue', None)
        ds[var].encoding.pop('missing_value', None)
        ds[var].attrs.pop('_FillValue', None)
        ds[var].attrs.pop('missing_value', None)

        # set in encoding only (xarray writes these as netCDF attributes)
        ds[var].encoding['_FillValue'] = fv
        ds[var].encoding['missing_value'] = fv
    return ds


def find_MXPol_files(files, pad_start, pad_end):
    """
    select MXPol files overlapping [pad_start, pad_end] based on the
    timestamp embedded in the filename (XPOL-YYYYMMDD-HHMMSS_Zdr.nc), 
    including the file immediately preceding pad_start too, since its
    data can extend past its filename timestamp into the window
    """
    parsed = []
    _MXPOL_FN_PATTERN = re.compile(r'XPOL-(\d{8}-\d{6})_Zdr')

    for f in files:
        m = _MXPOL_FN_PATTERN.search(f)
        if not m:
            continue
        ts = pd.Timestamp(datetime.strptime(m.group(1), '%Y%m%d-%H%M%S'))
        parsed.append((ts, f))
    parsed.sort(key=lambda x: x[0])

    if not parsed:
        return []

    selected = []
    prev_file = None
    for ts, f in parsed:
        if ts > pad_end:
            break
        if ts >= pad_start:
            selected.append(f)
        else:
            prev_file = f  # keeps getting overwritten -> ends up as the last file before pad_start
    if prev_file is not None:
        selected.insert(0, prev_file)
    return selected


def find_MIRA_files(files, pad_start, pad_end):
    """
    select MIRA files overlapping [pad_start, pad_end] based on the
    timestamp embedded in the filename (MIRA_YYYYMMDD_HHMMSS.nc),
    including the file immediately preceding pad_start too, since its
    data can extend past its filename timestamp into the window
    """
    parsed = []
    _MIRA_FN_PATTERN = re.compile(r'(\d{8})_(\d{4})_merged')

    for f in files:
        m = _MIRA_FN_PATTERN.search(f)
        if not m:
            continue
        ts_str = f"{m.group(1)}_{m.group(2)}"
        ts = pd.Timestamp(datetime.strptime(ts_str, '%Y%m%d_%H%M%S'))
        parsed.append((ts, f))
    parsed.sort(key=lambda x: x[0])

    if not parsed:
        return []

    selected = []
    prev_file = None
    for ts, f in parsed:
        if ts > pad_end:
            break
        if ts >= pad_start:
            selected.append(f)
        else:
            prev_file = f  # keeps getting overwritten -> ends up as the last file before pad_start
    if prev_file is not None:
        selected.insert(0, prev_file)
    return selected


def safe_reindex_dim(data, dim, full_dim, tolerance):
    """reindex xarray object onto a target coordinate grid using majority-vote (mode)
    over all native samples falling in each target cell, respecting a tolerance for gaps
    data: xr.DataArray or xr.Dataset (bool/int flag variables)
    dim: str, dimension to reindex ('time' or 'range')
    full_dim: array-like, target coordinate values for the specified dimension
    tolerance: str, float or pd.Timedelta, max allowed gap for a target point to be considered covered
    returns: xr.DataArray or xr.Dataset, majority-voted onto full_dim; -1 (int8) where no native data within tolerance"""
    data = data.chunk({dim: -1})  # ensure chunking along the dimension to reindex
    is_time = dim == 'time'
    gap_tolerance = pd.Timedelta(tolerance).to_numpy().astype('int64') if is_time else tolerance

    src = data[dim].values.astype('datetime64[ns]').astype(np.int64) if is_time \
        else np.asarray(data[dim].values, dtype=float)
    tgt = full_dim.values.astype('datetime64[ns]').astype(np.int64) if is_time \
        else np.asarray(full_dim, dtype=float)

    # bin edges = midpoints between consecutive target points (open-ended at both ends)
    mids = (tgt[:-1] + tgt[1:]) / 2
    edges = np.concatenate(([-np.inf], mids, [np.inf]))
    bin_idx = np.digitize(src, edges) - 1  # maps each native sample to a target bin

    # nearest-gap mask: distance from each target point to nearest native sample
    nearest_src_idx = np.clip(np.searchsorted(src, tgt), 0, len(src) - 1)
    nearest_src_idx_m1 = np.clip(nearest_src_idx - 1, 0, len(src) - 1)
    gap = np.minimum(np.abs(src[nearest_src_idx] - tgt), np.abs(src[nearest_src_idx_m1] - tgt))
    covered = gap <= gap_tolerance

    def _mode_along_bins(arr):
        arr = arr.astype(np.int8)
        n_bins = len(tgt)
        valid = bin_idx >= 0
        arr_valid = arr[valid]
        bins_valid = bin_idx[valid]

        out = np.full(n_bins, -1, dtype=np.int8)
        if arr_valid.size == 0:
            out[~covered] = -1
            return out

        n_values = int(arr_valid.max()) + 1
        flat_idx = bins_valid * n_values + arr_valid.astype(np.int64)
        counts = np.bincount(flat_idx, minlength=n_bins * n_values).reshape(n_bins, n_values)

        has_data = counts.sum(axis=1) > 0
        out[has_data] = counts[has_data].argmax(axis=1).astype(np.int8)
        out[~covered] = -1
        return out

    result = xr.apply_ufunc(
        _mode_along_bins, data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        exclude_dims={dim},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int8],
        dask_gufunc_kwargs={'output_sizes': {dim: len(tgt)}},
    )
    result = result.assign_coords({dim: full_dim})
    return result
