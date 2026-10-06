# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:18:21 2025

script to do campaign-wide statistics/plotting using zarr files

@author: clerx
"""
# %% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import os, glob, np, pd, xr, plt, sns, datetime, mcolors, LogNorm, mdates, time, scipy, timedelta
from src.constants_input import base_dir, dirs, zarr_variables, df_precip, pltConfig, plotdates_weekly, start_date, end_date, DFR_fn, date_ticks, radars, BASTAmodes, radar_ranges, calibrationvalues, frequencies, att_fn
from src.plotting import plot_density_histogram, plot_density, FuncFormatter, meters_to_km, meters_to_km_num, weekly_plot, plot_weekly_HALO, weekly_plot_paper, weekly_plot_paper_ext, paper_plots
from src.zarr_utils import generate_zarr_encodings, decode_and_combine_radars, decode_and_combine_modes, update_resampled_zarr, safe_reindex_dim
from src.radar_processing import DFRcorrection_array as add_DFRcorrection, add_DFRcorrection_xr
from src.radar_processing import cloudtop_height, find_continuous_cloudtop
from src.utils import determine_nbins, read_ERA_data, get_DFRdata, mask_to_intervals, format_elapsed
from matplotlib.ticker import MultipleLocator

if not 'peaktree' in radars:
    radars.append('peaktree')
# radar = 'MIRA'

freqs = ['X', 'Ka', 'W']
variables = ['Z', 'velocity', 'numpeaks', 'width', 'LDR', 'skewness']
masks = [None, 'orographic', 'frontal', 'precip', 'orographic_smoothed', 'frontal_smoothed', 'is_precip']

encodings = {}
for radar in radars:
    encodings[radar] = generate_zarr_encodings(zarr_variables[radar])

zarrdir = f"{dirs['zarr']}/30s_25m"
zarrdir_modes = f"{dirs['zarr']}/BASTAmodes_5s"
zarrdir_modes_25m = f"{dirs['zarr']}/BASTAmodes_25m"
zarrdir_native = f"{dirs['zarr']}/no_resampling"
zarrdir_spectral = f"{dirs['zarr']}/no_resampling_wspectral"

cloudtopDFRs = pd.read_csv(DFR_fn, index_col=0)

# what = 'HALO_weekly_plots'
# what = 'cloudtype_statistics'
# what = 'campaign_stats'
# what = 'analyze_campaign_stats'
# what = 'paper_plots'
# what = 'BASTA_resampling'
# what = 'BASTA_masking'
what = 'zarr_corrections'
# what = None

# %% functions

def define_temperature_zones(ds, lapse_rate=6.5):
    """
    Define temperature zones relative to 0°C isotherm.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with 'range' and 'zdi' (0°C isotherm height above radar)
    lapse_rate : float
        Temperature lapse rate in K/km (default: 6.5)

    Returns:
    --------

    ds : xarray.Dataset with added temperature and zone masks
    """
    # calculate height relative to 0°C
    # zdi and cloudtop are heights above radar
    ds['height_rel_to_0C'] = ds['range'] - ds['zdi']

    # approximate temperature (°C)
    ds['temperature_approx'] = -1 * \
        (ds['height_rel_to_0C'] / 1000) * lapse_rate

    # define SIP zones (height in meters above 0°C)
    ds['HM_zone'] = (ds['height_rel_to_0C'] >= 460) & (
        ds['height_rel_to_0C'] <= 1230)  # -3 to -8°C
    ds['droplet_shatter_zone'] = (ds['height_rel_to_0C'] > 1230) & (
        ds['height_rel_to_0C'] <= 2300)  # -8 to -15°C
    ds['ice_collision_zone'] = (ds['height_rel_to_0C'] > 2300) & (
        ds['height_rel_to_0C'] <= 3850)  # -15 to -25°C

    return ds


def classify_cloud_type(ds, orographic_threshold=3000):
    """
    Classify clouds as orographic vs. frontal/developed based on cloud top height.

    Parameters:
    -----------
    ds : xarray.Dataset
        Must contain 'cloudtop_height'
    orographic_threshold : float
        Height threshold in meters above radar (default: 3000m)

    Returns:
    --------
    ds : xarray.Dataset with 'cloud_type' variable
    """
    ds['cloud_type'] = xr.where(ds['cloudtop'] < orographic_threshold,
                                'orographic', 'frontal')
    return ds


def calculate_vertical_gradients(ds, radar='X'):
    """
    Calculate vertical gradients of reflectivity and spectral width.

    Parameters:
    -----------
    ds : xarray.Dataset
    radar : str
        Radar to use ('X' or 'Ka')

    Returns:
    --------
    ds : xarray.Dataset with added gradient variables
    """
    z_var = f'Z_{radar}'
    width_var = f'width_{radar}'

    # calculate gradients (centered difference)
    if z_var in ds:
        ds[f'dZ_dh_{radar}'] = ds[z_var].differentiate('range')

    if width_var in ds:
        ds[f'dW_dh_{radar}'] = ds[width_var].differentiate('range')

    return ds


def prepare_data_profiled(dsets, dataset_key, variable_key, maskname=None):
    t0 = time.time()
    z = dsets[dataset_key][variable_key].values
    original_shape = z.shape
    print(f"Loaded {variable_key}: {time.time()-t0:.2f}s")

    t0 = time.time()
    z_flat = z.ravel()
    print(f"Flatten: {time.time()-t0:.2f}s")

    t0 = time.time()
    msk_2d = ~np.isnan(z)

    if 'background_mask' in dsets[dataset_key].data_vars:
        bg = dsets[dataset_key]['background_mask'].values == 1
        msk_2d = msk_2d & bg

    if maskname is not None:
        if maskname in dsets[dataset_key].data_vars:
            msk_2d = msk_2d & (dsets[dataset_key][maskname].values == 0)
        else:
            print(f"Warning: '{maskname}' not found in dataset {dataset_key}, skipping mask.")
    print(f"Load & process mask: {time.time()-t0:.2f}s")

    t0 = time.time()
    msk = msk_2d.ravel()
    z = z_flat[msk]
    print(f"Apply mask: {time.time()-t0:.2f}s")

    t0 = time.time()
    rga_2d = np.broadcast_to(dsets[dataset_key]['range'].values, original_shape)
    rga = rga_2d.ravel()
    rg = rga[msk]
    # rga = np.tile(dsets[dataset_key]['range'].values,
    #               dsets[dataset_key][variable_key].shape[0])
    rg = rga[msk]
    print(f"Create & mask range: {time.time()-t0:.2f}s\n")

    return z, rg, msk


def prepare_data_hist(dset, variable):
    t0 = time.time()
    z = dset[variable].values
    print(f"Load {variable}: {time.time()-t0:.2f}s")

    t0 = time.time()
    z = z.flatten()
    print(f"Flatten: {time.time()-t0:.2f}s")

    mask = None
    if 'background_mask' in dset.data_vars:
        t0 = time.time()
        mask = dset['background_mask'].values.flatten() == 1
        print(f"Load & process mask: {time.time()-t0:.2f}s")

    t0 = time.time()
    if mask is not None:
        msk = mask & ~np.isnan(z)
    else:
        msk = ~np.isnan(z)
    print(f"Combine masks: {time.time()-t0:.2f}s")

    t0 = time.time()
    z = z[msk]
    print(f"Apply mask: {time.time()-t0:.2f}s")

    t0 = time.time()
    rga = np.tile(dset['range'].values,
                  dset[variable].shape[0])
    rg = rga[msk]
    print(f"Create & mask range: {time.time()-t0:.2f}s\n")

    return z, rg, msk


def prepare_data_chunked(dsets, dataset_key, variable_key):
    da = dsets[dataset_key][variable_key]
    mask_da = dsets[dataset_key].get('background_mask')
    rg_vals = dsets[dataset_key]['range'].values

    z_list, rg_list = [], []

    for i in range(0, da.sizes['time'], 1000):
        z_chunk = da.isel(time=slice(i, i+1000)).values
        if mask_da is not None:
            m = mask_da.isel(time=slice(i, i+1000)).values.flatten() == 1
        else:
            m = np.ones(z_chunk.size, dtype=bool)

        z_flat = z_chunk.flatten()
        msk = m & ~np.isnan(z_flat)
        z_list.append(z_flat[msk])
        rg_list.append(np.tile(rg_vals, z_chunk.shape[0])[msk])
        
    return np.concatenate(z_list), np.concatenate(rg_list)


def precompute_histograms(data_list, bins_list,
                          range_x=[-60, 45], range_y=[0, 12000]):
    """Pre-compute all histograms (do once, use many times)"""
    histograms = []

    for (x, y), bins in zip(data_list, bins_list):
        print(f"Computing histogram for {len(x):,} points...")
        H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                           range=[range_x, range_y])
        histograms.append((H, xedges, yedges))

    return histograms


def nested_dict_to_dataframe(results):
    """Convert nested results dict to a clean DataFrame"""
    rows = []
    for freq, variables in results.items():
        for var, stats in variables.items():
            row = {'frequency': freq, 'variable': var}
            row.update({k: v.item() if hasattr(v, 'item') else v
                       for k, v in stats.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns for readability
    cols = ['frequency', 'variable', 'count', 'min',
            'q25', 'median', 'mean', 'q75', 'max', 'std']
    # Only include columns that exist
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def plot_variable_by_frequency(df_frontal, df_orographic, variable='Z'):
    """
    Group by frequency, show all statistics with synchronized y-axis
    """
    df_f = df_frontal[df_frontal['variable'] == variable].copy()
    df_o = df_orographic[df_orographic['variable'] == variable].copy()

    frequencies = sorted(df_f['frequency'].unique())
    stats = ['min', 'q25', 'median', 'mean', 'q75', 'max', 'std']

    # Calculate global min/max across all frequencies for y-axis limits
    all_frontal_vals = df_f[stats].values.flatten()
    all_orographic_vals = df_o[stats].values.flatten()
    all_vals = np.concatenate([all_frontal_vals, all_orographic_vals])

    y_min = np.min(all_vals)
    y_max = np.max(all_vals)

    # Add some padding (10%)
    y_range = y_max - y_min
    y_min = y_min - 0.1 * y_range
    y_max = y_max + 0.1 * y_range

    fig, axes = plt.subplots(
        1, len(frequencies), figsize=(6*len(frequencies), 5))
    if len(frequencies) == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]

        frontal_vals = df_f[df_f['frequency'] == freq][stats].values[0]
        orographic_vals = df_o[df_o['frequency'] == freq][stats].values[0]

        x = np.arange(len(stats))
        width = 0.35

        bars1 = ax.bar(x - width/2, frontal_vals, width,
                       label='Frontal', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, orographic_vals, width,
                       label='Orographic', alpha=0.8, color='coral')

        ax.set_title(f'{freq}-band', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Set same y-limits for all subplots
        ax.set_ylim([y_min, y_max])

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if y_min <= height <= y_max:  # Only show if within range
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if y_min <= height <= y_max:  # Only show if within range
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        # Only show y-label on leftmost plot
        if idx == 0:
            ax.set_ylabel(pltConfig[variable]['label_short'],
                          fontsize=11, fontweight='bold')
            ax.legend()

    plt.suptitle(f"Frontal vs. orographic: {pltConfig[variable]['label_short']}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def dominant_flag(flag1, flag2, precip, window_minutes=60, dt_seconds=30, precip_thresh=0.75):
    """
    Apply rolling dominant-flag smoothing to precipitation flags.

    Parameters
    ----------
    oro_flag : xarray.DataArray
        Boolean orographic precipitation flag (short-term smoothed).
    fr_flag : xarray.DataArray
        Boolean frontal precipitation flag (short-term smoothed).
    precip : xarray.DataArray
        Boolean precipitation presence mask.
    window_minutes : float
        Rolling window size in minutes.
    dt_seconds : float
        Time resolution of the data in seconds (default 30 s).
    precip_thresh : float
        Minimum fraction of precipitation in window to apply dominance (default 0.8).

    Returns
    -------
    orographic_final : xarray.DataArray
        Orographic flag after rolling dominant smoothing.
    frontal_final : xarray.DataArray
        Frontal flag after rolling dominant smoothing.
    """

    # Compute rolling window in number of samples
    W = int(window_minutes*60 / dt_seconds)
    if W < 1:
        W = 1

    # Rolling precipitation fraction
    precip_frac = precip.rolling(time=W, center=True).mean()
    precip_extended = precip_frac > precip_thresh

    # Rolling flag occurrence fraction
    flag1_frac = flag1.rolling(time=W, center=True).mean()
    flag2_frac = flag2.rolling(time=W, center=True).mean()

    # Determine dominant flag in window
    dominant = xr.full_like(flag1_frac, 0, dtype=int)

    dominant = xr.where(
        precip_extended,
        xr.where(
            (flag1_frac <= 0) & (flag2_frac <= 0),
            0,  # No flag in window
            xr.where(flag1_frac > flag2_frac, 1, 2)  # 1=orographic, 2=frontal
        ),
        0  # No precipitation → no fill
    )

    # Construct final flags: keep original short-term smoothed flags, plus dominant-window fill
    flag1_final = flag1 | (dominant == 1)
    flag2_final = flag2 | (dominant == 2)

    return flag1_final, flag2_final, precip_extended


def precip_from_zarr(ds, radar='Ka', Zthresh=-25, Vthresh=-0.2, n_lower=10, frac_lower=0.5, min_valid=5, smoothing_min=10):
    var_Z = f"Z_{radar}"
    var_V = f"velocity_{radar}"

    Z_mask = ds[var_Z] >= Zthresh
    V_mask = ds[var_V] <= Vthresh
    crit_lower = (Z_mask & V_mask).isel(range=slice(0, n_lower))

    n_valid = crit_lower.notnull().sum(dim='range')
    frac_met = crit_lower.sum(dim='range') / \
        n_valid.where(n_valid >= min_valid)
    precip_flag = frac_met >= frac_lower

    return precip_flag.rolling(time=smoothing_min*2, center=True, min_periods=1).max().astype(bool)


def calc_campaign_stats(da, hours=3, mask=None):
    """
    da: DataArray(time, range), NaNs already masked
    uses approxiamte median (tdigest) for speed
    """
    if mask is not None:
        mask = mask.broadcast_like(da)
        da = da.where(mask)
    da = da.persist()

    da.chunk({'time': 'auto', 'range': -1})
    
    # ---- per-range (range dimension only)
    per_range = xr.Dataset(
        {
            "mean":   da.mean("time", skipna=True),
            "median": da.quantile(0.5, dim="time", skipna=True).squeeze(),
            "std":    da.std("time", skipna=True),
            "min":    da.min("time", skipna=True),
            "max":    da.max("time", skipna=True),
            "count":  da.count("time"),
        }
    )

    # ---- per-time (time dimension only)
    da_hourly = da.resample(time=f"{hours}H").mean(skipna=True)
    valid_hours = ~da_hourly.isnull().all(dim='range')
    da_hourly = da_hourly.sel(time=valid_hours)
    per_time_hourly = xr.Dataset(
        {
            "mean":   da_hourly.mean("range", skipna=True),
            "median": da_hourly.quantile(0.5, dim="range", skipna=True).squeeze(),
            "std":    da_hourly.std("range", skipna=True),
            "min":    da_hourly.min("range", skipna=True),
            "max":    da_hourly.max("range", skipna=True),
            "count":  da_hourly.count("range"),
        }
    )
    da_daily = da.resample(time='1D').mean(skipna=True)
    valid_days = ~da_daily.isnull().all(dim='range')
    da_daily = da_daily.sel(time=valid_days)
    
    per_time_daily = xr.Dataset(
        {
            "mean":   da_daily.mean("range", skipna=True),
            "median": da_daily.quantile(0.5, dim="range", skipna=True).squeeze(),
            "std":    da_daily.std("range", skipna=True),
            "min":    da_daily.min("range", skipna=True),
            "max":    da_daily.max("range", skipna=True),
            "count":  da_daily.count("range"),
        }
    )

    # ---- campaign-wide scalars (explicit scalar dimension)
    campaign = xr.Dataset(
        {
            "mean":   per_time_hourly["mean"].mean("time", skipna=True),
            "median": per_time_hourly["mean"].quantile(0.5, dim="time", skipna=True).squeeze(),
            "std":    per_time_hourly["mean"].std("time", skipna=True),
            "min":    per_time_hourly["mean"].min("time", skipna=True),
            "max":    per_time_hourly["mean"].max("time", skipna=True),
            "count":  per_time_hourly["mean"].count("time"),
        }
    ).expand_dims(stat=["campaign"])

    return per_range, per_time_hourly, per_time_daily, campaign


def campaign_stats_dataset(ds, varname, maskvar=None):
    if maskvar is not None:
        per_range, per_time_hourly, per_time_daily, campaign = calc_campaign_stats(
            ds[varname], mask=ds[maskvar]
        )
        suffix = f"_{maskvar}"
    else:
        per_range, per_time_hourly, per_time_daily, campaign = calc_campaign_stats(
            ds[varname]
        )
        suffix = ""

    ds_out = xr.Dataset()

    # ---- per-range stats
    for v in per_range.data_vars:
        ds_out[f"{varname}_{v}_per_range{suffix}"] = per_range[v]

    # ---- per-time hourly stats
    for v in per_time_hourly.data_vars:
        ds_out[f"{varname}_{v}_per_time_hourly{suffix}"] = per_time_hourly[v]

    # ---- per-time daily stats
    per_time_daily_renamed = per_time_daily.rename({"time": "time_daily"})
    for v in per_time_daily_renamed.data_vars:
        ds_out[f"{varname}_{v}_per_time_daily{suffix}"] = per_time_daily_renamed[v]

    # ---- campaign-wide scalars
    for v in campaign.data_vars:
        ds_out[f"{varname}_{v}_campaign{suffix}"] = campaign[v]

    return ds_out


def cfad(da, heights, z_bins, h_bins):
    """
    da: DataArray(time, range)
    heights: DataArray(time, range) or 1D range
    """

    z = da.values.ravel()
    h = heights.values.ravel()

    valid = np.isfinite(z) & np.isfinite(h)

    H, _, _ = np.histogram2d(
        h[valid], z[valid],
        bins=[h_bins, z_bins]
    )

    return xr.DataArray(
        H,
        dims=("height", "Z"),
        coords={
            "height": 0.5 * (h_bins[:-1] + h_bins[1:]),
            "Z": 0.5 * (z_bins[:-1] + z_bins[1:]),
        }
    )


def break_on_time_gaps(time, values, max_gap):
    """
    Insert NaNs where time gaps exceed max_gap so matplotlib breaks lines.
    """
    time = pd.to_datetime(time)
    values = np.asarray(values)

    dt = np.diff(time)
    gap_idx = np.where(dt > max_gap)[0]

    if len(gap_idx) == 0:
        return time, values

    time_out = []
    values_out = []

    for i in range(len(time)):
        time_out.append(time[i])
        values_out.append(values[i])

        if i in gap_idx:
            time_out.append(time[i] + max_gap / 2)
            values_out.append(np.nan)

    return np.array(time_out), np.array(values_out)


def plot_stats(ds, freq, var, maskvar=None):
    varname = f"{var}_{freq}"
    if maskvar is not None:
        title = f"Statistics of {freq}-band {pltConfig[var]['label_short']} ({maskvar})"
        suffix = f"_{maskvar}"
    else:
        title = f"Statistics of {freq}-band {pltConfig[var]['label_short']}"
        suffix = ''
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.15, width_ratios=[1, 2])

    ax_range = fig.add_subplot(gs[:, 0])
    ax_hourly = fig.add_subplot(gs[0, 1])
    ax_daily = fig.add_subplot(gs[1, 1])

    # range stats
    range_vals = ds['range'].values
    mean_r = ds[f"{varname}_mean_per_range{suffix}"].values
    std_r = ds[f"{varname}_std_per_range{suffix}"].values

    ax_range.plot(mean_r, range_vals, label="mean", linewidth=2, color='#2E86AB', zorder=3)
    ax_range.plot(
        ds[f"{varname}_median_per_range{suffix}"].values,
        range_vals,
        linestyle="--",
        linewidth=2,
        label="median",
        color='#A23B72',
        zorder=3
    )

    ax_range.fill_betweenx(range_vals, mean_r - std_r, mean_r + std_r, alpha=0.3, label="±1 std dev", color='#2E86AB', zorder=1)
    ax_range.plot(ds[f"{varname}_min_per_range{suffix}"].values,
        range_vals,
        linestyle=":",
        linewidth=1,
        label="min",
        color='#C73E1D',
        zorder=2
    )
    ax_range.plot(
        ds[f"{varname}_max_per_range{suffix}"].values,
        range_vals,
        linestyle=":",
        linewidth=1,
        label="max",
        color='#F18F01',
        zorder=2
    )

    ax_range.set_xlim(pltConfig[var]["vmin"], pltConfig[var]["vmax"])
    ax_range.set_ylim(0, 12000)
    ax_range.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    ax_range.set_xlabel(pltConfig[var]["label_short_unit"])
    ax_range.set_ylabel("range [km]")
    ax_range.set_title(f"{title} per range", fontweight="bold")
    ax_range.grid(True, linestyle="--")

    ax_range_count = ax_range.twiny()
    count_range = ds[f"{varname}_count_per_range{suffix}"].values
    ax_range_count.plot(count_range, range_vals, color='gray', linewidth=1, label='count')
    ax_range_count.set_xscale('log')
    ax_range_count.set_xlabel('sample count', fontsize=10, color='gray')
    ax_range_count.tick_params(axis='x', labelcolor='gray')

    # hourly stats
    time_h = ds["time"].values
    max_gap_h = pd.Timedelta(hours=3.5)

    mean_h = ds[f"{varname}_mean_per_time_hourly{suffix}"].values
    std_h = ds[f"{varname}_std_per_time_hourly{suffix}"].values
    median_h = ds[f"{varname}_median_per_time_hourly{suffix}"].values
    min_h = ds[f"{varname}_min_per_time_hourly{suffix}"].values
    max_h = ds[f"{varname}_max_per_time_hourly{suffix}"].values
    count_h = np.ma.masked_where(
        ds[f"{varname}_count_per_time_hourly{suffix}"].values == 0,
        ds[f"{varname}_count_per_time_hourly{suffix}"].values.astype(float),
    )

    t_plot, mean_plot = break_on_time_gaps(time_h, mean_h, max_gap_h)
    _, std_plot = break_on_time_gaps(time_h, std_h, max_gap_h)
    _, median_plot = break_on_time_gaps(time_h, median_h, max_gap_h)
    _, min_plot = break_on_time_gaps(time_h, min_h, max_gap_h)
    _, max_plot = break_on_time_gaps(time_h, max_h, max_gap_h)

    ax_hourly.plot(t_plot, mean_plot, linewidth=2, label="mean", color='#2E86AB', zorder=3)
    ax_hourly.plot(t_plot, median_plot, linewidth=2, linestyle="--", label="median", color='#A23B72', zorder=3)

    ax_hourly.fill_between(
        t_plot,
        mean_plot - std_plot,
        mean_plot + std_plot,
        alpha=0.3,
        label="±1 std dev",
        zorder=1,
        color='#2E86AB'
    )

    ax_hourly.plot(t_plot, min_plot, linestyle=":", linewidth=1, label="min", color='#C73E1D', zorder=2)
    ax_hourly.plot(t_plot, max_plot, linestyle=":", linewidth=1, label="max", color='#F18F01', zorder=2)

    ax_hourly.set_ylim(pltConfig[var]["vmin"], pltConfig[var]["vmax"])
    ax_hourly.set_ylabel(pltConfig[var]["label_short_unit"])
    ax_hourly.set_title(f"{title} per time (3-hourly)", fontweight="bold")
    ax_hourly.grid(True, linestyle="--")
    ax_hourly.set_xticks(date_ticks)
    ax_hourly.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_hourly.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax_hourly_count = ax_hourly.twinx()
    ax_hourly_count.plot(time_h, count_h, color='gray', linestyle='-', linewidth=1, label='count')
    ax_hourly_count.fill_between(time_h, 0.1, count_h,
                                  color='gray', alpha=0.1, zorder=0, label='count')
    ax_hourly_count.set_yscale('log')
    ax_hourly_count.set_ylabel('sample count', fontsize=10, color='gray')
    ax_hourly_count.tick_params(axis='y', labelcolor='gray')

    # daily stats
    time_d = ds[f"{varname}_mean_per_time_daily{suffix}"].time_daily.values
    max_gap_d = pd.Timedelta(days=1.5)

    mean_d = ds[f"{varname}_mean_per_time_daily{suffix}"].values
    std_d = ds[f"{varname}_std_per_time_daily{suffix}"].values
    median_d = ds[f"{varname}_median_per_time_daily{suffix}"].values
    min_d = ds[f"{varname}_min_per_time_daily{suffix}"].values
    max_d = ds[f"{varname}_max_per_time_daily{suffix}"].values
    count_d = np.ma.masked_where(
        ds[f"{varname}_count_per_time_daily{suffix}"].values == 0,
        ds[f"{varname}_count_per_time_daily{suffix}"].values.astype(float),
    )

    t_plot_d, mean_plot_d = break_on_time_gaps(time_d, mean_d, max_gap_d)
    _, std_plot_d = break_on_time_gaps(time_d, std_d, max_gap_d)
    _, median_plot_d = break_on_time_gaps(time_d, median_d, max_gap_d)
    _, min_plot_d = break_on_time_gaps(time_d, min_d, max_gap_d)
    _, max_plot_d = break_on_time_gaps(time_d, max_d, max_gap_d)

    ax_daily.plot(t_plot_d, mean_plot_d, linewidth=2, label="mean", color='#2E86AB', zorder=3)
    ax_daily.plot(t_plot_d, median_plot_d, linewidth=2, linestyle="-", label="median", color='#A23B72', zorder=3)

    ax_daily.fill_between(
        t_plot_d,
        mean_plot_d - std_plot_d,
        mean_plot_d + std_plot_d,
        alpha=0.3,
        label="±1 std dev",
        zorder=1,
        color='#2E86AB'
    )

    ax_daily.plot(t_plot_d, min_plot_d, linestyle=":", linewidth=1, label="min", color='#C73E1D', zorder=2)
    ax_daily.plot(t_plot_d, max_plot_d, linestyle=":", linewidth=1, label="max", color='#F18F01', zorder=2)
    ax_daily.set_ylim(pltConfig[var]["vmin"], pltConfig[var]["vmax"])
    ax_daily.set_ylabel(pltConfig[var]["label_short_unit"])
    ax_daily.grid(True, linestyle="--")
    ax_daily.set_xticks(date_ticks)
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax_daily_count = ax_daily.twinx()
    ax_daily_count.plot(time_d, count_d, color='gray', linestyle='-', linewidth=1, label='count')
    ax_daily_count.fill_between(time_d, 0.1, count_d,
                                color='gray', alpha=0.1, zorder=0, label='count')
    ax_daily_count.set_yscale('log')  # Log scale for count
    ax_daily_count.set_ylabel('sample count', fontsize=10, color='gray')
    ax_daily_count.tick_params(axis='y', labelcolor='gray')

    ax_daily.legend(loc="best", fontsize=10)

    plt.tight_layout()
    return fig


def adjust_range(ds, mode):
    """helper function to interpolate/adjust range coordinates for BASTA modes"""
    dbz_vars = ["reflectivity", "reflectivity_raw", "z_30s", "z_120s"]
    vel_vars = ["velocity", "vel_30s", "vel_120s"]
    max_vars = ["background_mask", "bg_30s", "bg_120s", "is_precip", "cloudtop", "zdi"]
    target_range = radar_ranges['BASTA_25m']

    out = {}
    for var in ds.data_vars:
        da = ds[var]
        if 'range' not in da.dims:
            out[var] = ds[var]
            continue

        if var in dbz_vars:
            lin = 10 ** (da / 10)

            if mode == '12m5':
                lin = lin.rolling(range=2, center=True, min_periods=1).mean()
            
            lin = lin.interp(range=target_range)
            out[var] = 10 * np.log10(lin)

        elif var in vel_vars:
            if mode == '12m5':
                da = da.rolling(range=2, center=True, min_periods=1).mean()
                
            out[var] = da.interp(range=target_range)

        elif var in max_vars:
            if mode == '12m5':
                out[var] = da.coarsen(range=2, boundary='trim').max().reindex(range=target_range, method='nearest', tolerance=13.)
            elif mode == '100m_18km':
                out[var] = da.reindex(range=target_range, method='nearest', tolerance=50.)
            else:
                out[var] = da.reindex(range=target_range, method='nearest', tolerance=13.)

        else:
            out[var] = da.interp(range=target_range)

    return xr.Dataset(out, coords={**{c: ds.coords[c] for c in ds.coords if c != 'range'}, 'range': target_range}, attrs=ds.attrs)



# %% loading data
print('loading data from .zarr files')
t0 = time.time()

ds_combined, dsets = decode_and_combine_radars(zarrdir, radars[:-1])
dsets_modes = decode_and_combine_modes(zarrdir_modes_25m, BASTAmodes)
dsets_native = decode_and_combine_modes(zarrdir_native, BASTAmodes)

mira_native = xr.open_zarr(f"{zarrdir_native}/MIRA.zarr", consolidated=True).unify_chunks()
basta_native = xr.open_zarr(f"{zarrdir_native}/BASTA_25m.zarr", consolidated=True).unify_chunks()
mxpol_native = xr.open_zarr(f"{zarrdir_native}/MXPol.zarr", consolidated=True).unify_chunks()

mira_resampled = xr.open_zarr(f"{zarrdir}/MIRA.zarr", consolidated=True).unify_chunks()
basta_resampled = xr.open_zarr(f"{zarrdir}/BASTA.zarr", consolidated=True).unify_chunks()
mxpol_resampled = xr.open_zarr(f"{zarrdir}/MXPol.zarr", consolidated=True).unify_chunks()

mira_spectral = xr.open_zarr(f"{zarrdir_spectral}/MIRA.zarr", consolidated=True).unify_chunks()
mxpol_spectral = xr.open_zarr(f"{zarrdir_spectral}/MXPol.zarr", consolidated=True).unify_chunks()

print(f"finished loading data ({time.time()-t0:.2f}s), starting calculations")
t0 = time.time()

dsets_tr = {}
for mode in BASTAmodes:
    dsets_tr[mode] = xr.open_zarr(f"{os.path.dirname(zarrdir)}/BASTAmodes_5s_25m/BASTA_{mode}.zarr")
    dsets_tr[mode] = dsets_tr[mode].chunk({'time': 3600, 'range': -1})

# load HALO data
HALO_data_all = xr.open_mfdataset([f"{dirs['HALO']}/ABLclassification.nc", f"{dirs['HALO']}/epsilon.nc"], combine='by_coords')

print(f"finished loading data from .zarr files ({time.time()-t0:.2f}s)\n")


# %% zarr file check
if what == 'test_plot':
    start, end = datetime(2024, 11, 11), datetime(2024, 11, 17)
    # start, end = datetime(2024, 12, 8), datetime(2024, 12, 15)
    # start, end = datetime(2024, 12, 1), datetime(2024, 12, 8)
    start, end = datetime(2025, 1, 12), datetime(2025, 1, 18)
    subset = ds_combined.sel(time=slice(start, end))

    fig, ax = plt.subplots(figsize=(20, 5))

    im = subset['Z_Ka'].plot.imshow(
        x='time',
        y='range',
        vmin=-60, vmax=35,
        cmap='Spectral_r',
        ax=ax,
        add_colorbar=True
    )
    timevals = subset.time.values

    # overlay cloudtop line (same time axis, y=cloudtop)
    ax.plot(subset['time'], subset['cloudtop'], color='black', lw=0.8)
    ax.set_ylim([0, 12e3])
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.legend()
    plt.tight_layout()
    plt.show()


#%% weekly plots with multiple flags
if what == 'weekly_plots_flags':
    for start, end in plotdates_weekly:
        print(f"Making plot for {start:%Y-%m-%d} to {end:%Y-%m-%d} at {pd.Timestamp.now()}")

        figpath = (f"{base_dir}/summaryQLs/weeklyplots/"
                   f"{start:%Y%m%d}-{end:%Y%m%d}_flags.png")

        # coarsen
        subset = (
            ds_combined
            .sel(time=slice(start, end))
            .coarsen(time=4, range=2, boundary="trim")
            .mean()
            .compute()
        )

        fig, ax = plt.subplots(figsize=(20, 5))

        im = ax.imshow(
            subset["Z_Ka"].T,
            origin="lower",
            aspect="auto",
            extent=[
                subset.time.values[0],
                subset.time.values[-1],
                subset.range.values[0],
                subset.range.values[-1],
            ],
            vmin=-60,
            vmax=35,
            cmap="Spectral_r",
        )
        plt.colorbar(im, ax=ax, label="Reflectivity [dBZ]")

        timevals = subset.time.values
        cloudtop = subset.cloudtop.values

        ax.fill_between(
            timevals, cloudtop, 12e3,
            where=subset.orographic.values,
            color="darkgreen", alpha=0.25
        )
        ax.fill_between(
            timevals, cloudtop, 12e3,
            where=subset.frontal.values,
            color="orange", alpha=0.25
        )
        ax.fill_between(
            timevals, cloudtop, 12e3,
            where=subset.orographic_smoothed.values,
            color="deepskyblue", alpha=0.15
        )
        ax.fill_between(
            timevals, cloudtop, 12e3,
            where=subset.frontal_smoothed.values,
            color="fuchsia", alpha=0.15
        )
        ax.fill_between(
            timevals, 50, 200,
            where=subset.precip.values,
            color="darkblue", alpha=0.4
        )

        ax.plot(timevals, cloudtop, color="black", lw=0.8)

        ax.set_ylim(0, 12e3)
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.tight_layout()
        plt.savefig(figpath, dpi=200, bbox_inches="tight", facecolor="w")
        plt.close()

        print(f"Saved {figpath}")




# %% density plots - radar sensitivity
if what == 'refl_densityplots':
    # zw, rgw, mskw = prepare_data_profiled(dsets, 'W', 'Z_W')
    zw, rgw, mskw = prepare_data_profiled(dsets_modes, '25m', 'reflectivity', maskname='mask_artefacts_loose') # with new artefact mask
    zk, rgk, mskk = prepare_data_profiled(dsets, 'Ka', 'Z_Ka')
    zx, rgx, mskx = prepare_data_profiled(dsets, 'X', 'Z_X')

    data_list = [(zx, rgx), (zk, rgk), (zw, rgw)]
    # data_list = [(zx, rgx), (zk, rgk), ((z25f-calibrationvalues['BASTA'][0]), rg25f)] # quick and dirty plot for filtered 25m BASTA data - only works after having run 
    # bins_list = ((105, int(12e3/30)),
    #              (105, int(12e3/31.2)),
    #              (105, int(12e3/25)))
    bins_list = ((105, int(12e3/25)), (105, int(12e3/25)), (105, int(12e3/25))) # this line for consistent binning across frequencies, for native range resolution use above variable
    # data_list = [(zx, rgx), (zxf, rgxf)]
    # bins_list =[(105, int(12e3/25)), (105, int(12e3/25))]
    histograms = precompute_histograms(data_list, bins_list)

    nonzero_vals = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms])
    vmin = np.nanmin(nonzero_vals)
    vmax = np.nanmax(nonzero_vals)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    norms = (LogNorm(vmin, 200),
             LogNorm(vmin, 1000),
             LogNorm(vmin, 15000))
    # these logs for plotting with consistent binning across frequencies and filtered 25m BASTA data
    # norms = (LogNorm(vmin, 250),
    #          LogNorm(vmin, 1000),
    #          LogNorm(vmin, 7000))
    vmaxs = [4000, 6000, 15e3]
    vmaxs = [250, 1000, 1e4]

    # plotting all in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms, norms, vmaxs,
            ['$Ze_{X}$ [dBZ]', '$Ze_{Ka}$ [dBZ]', '$Ze_{W}$ [dBZ]'],
            ['Sensitivity X', 'Sensitivity Ka', 'Sensitivity W'],
    )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                           cmap='inferno',
                           norm=norm
                           # vmin=vmin, vmax=vmax
                           )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (resampled to 5s and 25m)',
                 fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/summaryQLs/datapaper/densityplots_refl_vs_range.png", dpi=200, bbox_inches="tight", facecolor="w")

    # print statistics for corrected BASTA-data
    # upper limit
    count = (ds_combined['Z_W'] > 20).sum().compute().item()
    total = ds_combined['Z_W'].notnull().sum().compute().item()
    pct = count / total * 100
    print(f"{pct:.2f}% of data > 20 dBZ for DFR-corrected BASTA data ({count} of {total} valid points)")
    # lower limit
    refl = ds_combined['Z_W']
    rng = ds_combined['range']

    valid = refl.notnull()
    mask = (
        (refl < -20) &
        (rng > 3e3) & 
        (rng > (1e2 * refl + 5e3))
    )
    count = mask.sum().compute().item()
    total = valid.sum().compute().item()
    pct = count / total * 100

    print(
        f"{pct:.2f}% of data < -20 dBZ above 3 km "
        f"and above curve for DFR-corrected BASTA data "
        f"({count} of {total} valid points)"
    )



#%% density plots for different BASTA modes

if what == 'refl_densityplots_BASTA':
    zw, rgw, mskw = prepare_data_profiled(dsets_modes, '12m5', 'reflectivity')
    zk, rgk, mskk = prepare_data_profiled(dsets_modes, '25m', 'reflectivity')
    zx, rgx, mskx = prepare_data_profiled(dsets_modes, '100m_18km', 'reflectivity')

    data_list = [(zw, rgw), (zk, rgk), (zx, rgx)]
    # bins_list = ((105, int(12e3/30)),
    #              (105, int(12e3/31.2)),
    #              (105, int(12e3/25)))
    bins_list = ((105, 240), (105, 240), (105, 120))
    # data_list = [(zx, rgx), (zxf, rgxf)]
    # bins_list =[(105, int(12e3/25)), (105, int(12e3/25))]
    histograms = precompute_histograms(data_list, bins_list)

    nonzero_vals = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms])
    vmin = np.nanmin(nonzero_vals)
    vmax = np.nanmax(nonzero_vals)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    norms = (LogNorm(vmin, 15e3),
             LogNorm(vmin, 6000),
             LogNorm(vmin, 2500))
    vmaxs = [15e3, 6000, 2500]

    # plotting individual modes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms, norms, vmaxs,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5', 'Sensitivity 25m', 'Sensitivity 100m_18km'],
    )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='plasma',norm=norm,# vmin=vmin, vmax=vmax
                           )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        xline1 = np.linspace(-60, -20, 500)
        yline1 = 1e2 * xline1 + 5e3
        ax.plot(xline1, yline1, color='cyan', linewidth=2, linestyle='--', label='range = 100·dBZ + 5000')
        ax.vlines(x=-20, ymin=3e3, ymax=12e3, color='cyan', linewidth=2, linestyle='--', label='-20 dBZ')

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (resampled)',
                 fontweight='bold', fontsize=18)
    plt.tight_layout()


    # calculate statistics for all modes together (for comparison with individual modes)
    z_all = np.concatenate([zx, zk, zw])
    rg_all = np.concatenate([rgx, rgk, rgw])

    w_x = np.ones(len(zx)) / len(zx)
    w_k = np.ones(len(zk)) / len(zk)
    w_w = np.ones(len(zw)) / len(zw)
    weights = np.concatenate([w_x, w_k, w_w])

    # three modes in one figure
    Ha, xedgesa, yedgesa = np.histogram2d(z_all, rg_all, bins=(105, 120),
                                        range=[[-60, 45], [0, 12e3]],
                                        weights=weights)
    
    fig, ax = plt.subplots(figsize=(6, 6))

    H_masked_all = np.ma.masked_where(Ha == 0, Ha)
    im = ax.pcolormesh(xedgesa, yedgesa, H_masked_all.T,
                    cmap='plasma',
                    norm=LogNorm(vmin=1e-6, vmax=H_masked_all.max()))

    ax.set_xlabel('Ze [dBZ]')
    ax.set_ylabel('Range [km]')
    ax.set_xlim([-60, 45])
    ax.set_ylim([0, 12e3])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
    ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=ax, label='normalised count')
    plt.suptitle('Density plot of reflectivity vs. range (all modes combined)', fontweight='bold', fontsize=18)
    plt.tight_layout()

    # lower two modes
    z_two = np.concatenate([zk, zw])
    rg_two = np.concatenate([rgk, rgw])

    weights_two = np.concatenate([w_k, w_w])

    # three modes in one figure
    Ht, xedgest, yedgest = np.histogram2d(z_two, rg_two, bins=(105, 120),
                                        range=[[-60, 45], [0, 12e3]],
                                        weights=weights_two)
    
    fig, ax = plt.subplots(figsize=(6, 6))

    H_masked_two = np.ma.masked_where(Ht == 0, Ht)
    im = ax.pcolormesh(xedgest, yedgest, H_masked_two.T,
                    cmap='plasma',
                    norm=LogNorm(vmin=1e-6, vmax=H_masked_two.max()))

    ax.set_xlabel('Ze [dBZ]')
    ax.set_ylabel('Range [km]')
    ax.set_xlim([-60, 45])
    ax.set_ylim([0, 12e3])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
    ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=ax, label='normalised count')
    plt.suptitle('Density plot of reflectivity vs. range (12m5 and 25m modes)', fontweight='bold', fontsize=18)
    plt.tight_layout()

    # print statistics of percentage of points above 20 dBZ for each mode
    for mode in BASTAmodes:
        count = (dsets_modes[mode]['reflectivity'] > 20).sum().compute().item()

        # Percentage of all valid (non-NaN) points
        total = dsets_modes[mode]['reflectivity'].notnull().sum().compute().item()
        pct = count / total * 100

        print(f"{pct:.2f}% of data > 20 dBZ for {mode} ({count} of {total} valid points)")

    # print statistics of percentage of points below -above 20 dBZ above 4 km and above curve (range = 6e3 * dBZ) for each mode
    for mode in BASTAmodes:
        refl = dsets_modes[mode]['reflectivity']
        rng = dsets_modes[mode]['range']

        valid = refl.notnull()
        mask = (
            (refl < -20) &
            (rng > 3e3) & 
            (rng > (1e2 * refl + 5e3))
        )
        count = mask.sum().compute().item()
        total = valid.sum().compute().item()
        pct = count / total * 100

        print(
            f"{pct:.2f}% of data < -20 dBZ above 3 km "
            f"and above curve for {mode} "
            f"({count} of {total} valid points)"
        )


#%% campaign stats
if what == 'campaign_stats':
    all_stats = []
    startstats = pd.Timestamp.now()
    print(f"calculating campaign statistics, starting {startstats.strftime('%Y-%m-%d %H:%M:%S')}")
    for freq in freqs:
        startfreq = pd.Timestamp.now()
        print(f"processing frequency {freq}, starting {startfreq.strftime('%Y-%m-%d %H:%M:%S')}")
        for var in variables:
            for maskvar in masks:
                startvar = pd.Timestamp.now()
                varname = f"{var}_{freq}"
                if not varname in ds_combined:
                    continue
                print(f"calculating campaign stats for {varname} with mask '{mask}' starting {startvar.strftime('%Y-%m-%d %H:%M:%S')}")
                if mask is None:
                    ds_stats = campaign_stats_dataset(ds_combined, varname=varname).compute()
                else:
                    ds_stats = campaign_stats_dataset(ds_combined, varname=varname, maskvar=mask).compute()
                all_stats.append(ds_stats)

                if mask:
                    fn = f"{base_dir}/campaign_stats/campaign_stats_{varname}_{mask}.png"
                else:
                    fn = f"{base_dir}/campaign_stats/campaign_stats_{varname}.png"
                fig = plot_stats(ds_stats, freq, var, maskvar)
                plt.savefig(fn, dpi=300, bbox_inches='tight', facecolor='w')
                plt.close(fig)
                endvar = pd.Timestamp.now()
                print(f"done {varname}, time elapsed {(endvar - startvar).total_seconds():.2f} seconds")
        endfreq = pd.Timestamp.now()
        print(f"done frequency {freq}, time elapsed {(endfreq - startfreq).total_seconds():.2f} seconds")

    startmerge = pd.Timestamp.now()
    print(f"merging and saving all campaign stats, starting {startmerge.strftime('%Y-%m-%d %H:%M:%S')}")
    ds_all_stats = xr.merge(all_stats)
    endmerge = pd.Timestamp.now()
    print(f"merging done, time elapsed {endmerge - startmerge}\nsaving to netcdf, starting {endmerge.strftime('%Y-%m-%d %H:%M:%S')}")
    ds_all_stats.to_netcdf(f"{base_dir}/campaign_stats/campaign_stats_allvariables.nc")
    endsave = pd.Timestamp.now()
    print(f"saving done, time elapsed {(endsave - endmerge).total_seconds():.2f} seconds")

#%% open stats from netcdf
if what == 'analyze_campaign_stats':
    ds_all_stats = xr.open_dataset(f"{base_dir}/campaign_stats/campaign_stats_allvariables.nc")

# %% orographic vs. frontal cloud statistics
# orographic: cloudtop <= 2 km
# frontal: cloudtop >= 3 km

if what == 'cloudtype_statistics':
    from scipy.stats import skew

    def summarize(masked):
        # Load into memory once
        da = masked.compute()

        # da = da.where(ranges <= cloudtops)

        max_ = da.max('range', skipna=True)
        mean_ = da.mean('range', skipna=True)
        median_ = da.median('range', skipna=True)
        min_ = da.min('range', skipna=True)
        std_ = da.std('range', skipna=True)
        q25_ = da.quantile(0.25, dim='range', skipna=True)
        q75_ = da.quantile(0.75, dim='range', skipna=True)
        count_ = da.count('range')
        iqr_ = q75_ - q25_

        skew_ = xr.apply_ufunc(
            skew, da,
            input_core_dims=[['range']],
            vectorize=True,
            kwargs={'nan_policy': 'omit'},
            dask='parallelized',
            output_dtypes=[float]
        )
        return dict(
            max=float(max_.mean('time')),
            mean=float(mean_.mean('time')),
            median=float(median_.mean('time')),
            min=float(min_.mean('time')),
            std=float(std_.mean('time')),
            q25=float(q25_.mean('time')),
            q75=float(q75_.mean('time')),
            count=float(count_.sum('time')),
            iqr=float(iqr_.mean('time')),
            skew=float(skew_.mean('time'))
        )

    # print(f'calculating frontal statistics, starting {pd.Timestamp.now()}')
    # results_frontal = {}
    # for freq in freqs:
    #     results_frontal[freq] = {}

    #     for var in variables:
    #         varname = f"{var}_{freq}"
    #         if varname not in ds_combined:
    #             continue
    #         print(f"running {varname} at {pd.Timestamp.now()}")

    #         masked = ds_combined[varname].where(
    #             (ds_combined['precip']) &
    #             (ds_combined['frontal'])
    #         )
    #         cloudtops = ds_combined['cloudtop'].where(
    #             (ds_combined['precip']) &
    #             (ds_combined['frontal'])).broadcast_like(masked)
    #         ranges = ds_combined['range'].broadcast_like(masked)

    #         results_frontal[freq][var] = summarize(masked, cloudtops, ranges)

    # df_frontal = nested_dict_to_dataframe(results_frontal)
    # df_frontal.to_csv('/home/clerx/results_frontal_v10dec.csv',
    #                   index=False, float_format='%.4f')
    # df_frontal = pd.read_csv('/home/clerx/results_frontal_v10dec.csv')

    print(f'calculating orographic statistics, starting {pd.Timestamp.now()}')
    results_orographic = {}
    for freq in freqs:
        results_orographic[freq] = {}

        for var in variables:
            varname = f"{var}_{freq}"
            if varname not in ds_combined:
                continue
            print(f"running {varname} at {pd.Timestamp.now()}")

            masked = ds_combined[varname].where(
                (ds_combined['precip']) &
                (ds_combined['orographic_inst']) & 
                (ds_combined['range'] <= ds_combined['cloudtop'])
            )
            cloudtops = ds_combined['cloudtop'].where(
                (ds_combined['precip']) &
                (ds_combined['orographic_inst'])).broadcast_like(masked)
            ranges = ds_combined['range'].broadcast_like(masked)

            results_orographic[freq][var] = summarize(masked, cloudtops, ranges)

    df_orographic = nested_dict_to_dataframe(results_orographic)

    df_orographic.to_csv(
        '/home/clerx/results_orographic_10dec_min0dB.csv', index=False, float_format='%.4f')
    # df_orographic = pd.read_csv('/home/clerx/results_orographic_10dec_min0dB.csv')

    dt_hours = (ds_combined.time.diff("time").isel(time=0) / np.timedelta64(1, "h"))

    n_total = ds_combined.sizes['time']
    n_hours = n_total * dt_hours

    counts = ds_combined[['precip', 'orographic', 'frontal']].sum(dim='time').compute()
    hours = (counts * dt_hours)

    n_precip_hours = counts['precip'].item() * dt_hours
    n_orographic_hours = counts['orographic'].item() * dt_hours
    n_frontal_hours = counts['frontal'].item() * dt_hours

    n_nonprecip_hours = (n_total - counts['precip'].item()) * dt_hours


# %% weekly plots
if what == 'weekly_plots':
    # load all ERA-data once
    ERA = []
    for date in pd.date_range(start_date, end_date):
        ds = read_ERA_data(date, dirs['ERA'])
        ERA.append(ds)
    ERA_data = xr.concat(ERA, dim='time').sortby('time')
    del ERA
    # start, end = plotdates_weekly[6]
    for start, end in plotdates_weekly:
        print(
            f"Making plot for {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        figpath = f"{base_dir}/summaryQLs/weeklyplots/{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}.png"
        weekly_plot(ds_combined, ERA_data, start, end, figpath)


# %%
if what == 'HALO_weekly_plots':
    # load all ERA-data once
    ERA = []
    for date in pd.date_range(start_date, end_date):
        ds = read_ERA_data(date, dirs['ERA'])
        ERA.append(ds)
        ERA_data = xr.concat(ERA, dim='time').sortby('time')
    del ERA

    HALO_data = xr.open_mfdataset(
        [f"{dirs['HALO']}/ABLclassification.nc", f"{dirs['HALO']}/epsilon.nc"], combine='by_coords')

    # start, end = plotdates_weekly[6]
    for start, end in plotdates_weekly:
        print(
            f"Making weekly HALO plot for {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        figpath = f"{base_dir}/summaryQLs/weeklyplots/{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}_wHALO.png"
        plot_weekly_HALO(ds_combined, HALO_data, ERA_data, start, end, figpath)


# %% data paper plots
if what == 'paper_plots':
    plot_ranges = [
        (pd.to_datetime('2025-01-06 12:00'), pd.to_datetime('2025-01-08 18:00')),
        (pd.to_datetime('2025-01-15 00:00'), pd.to_datetime('2025-01-16 00:00'))
    ]
    for start, end in plot_ranges:
        print(f"\n{'='*60}")
        print(f"Processing {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        t0 = time.time()
        print("Subsetting combined dataset...")
        needed_vars = ['Z_Ka', 'Z_W', 'velocity_Ka', 'LDR_Ka', 'DFR_KaW', 'DFR_XKa', 'Z_X', 'time', 'range', 'DFRcorrection_W']
        ds_subset = ds_combined.sel(time=slice(start, end))[needed_vars]
        ds_mode = dsets_modes['25m'].sel(time=slice(start, end)).reindex({'time': ds_subset.time.values})
        print(f"Loaded subset ({time.time()-t0:.2f}s)")
        
        t0 = time.time()
        print("Loading ERA data...")
        ERA = []
        ref_altitude = None
        for date in pd.date_range(start.date(), end.date()):
            ds = read_ERA_data(date, dirs['ERA'])
            if ref_altitude is None:
                ref_altitude = ds['altitude']
            ds = ds.assign_coords(altitude=ref_altitude)
            ERA.append(ds)        
        ERA_data = xr.concat(ERA, dim='time').sortby('time')
        ERA_data = ERA_data.sel(time=slice(start, end)).load()  # Subset and load ERA too
        del ERA
        print(f"Loaded ERA data ({time.time()-t0:.2f}s)")

        y_range = (0, 7e3) if (start, end) == plot_ranges[1] else (0, 10e3)

        HALO_data = HALO_data_all.sel(time_3min=slice(start, end)).drop_dims(['time_30min', 'time_60min'])
        t0 = time.time()
        print("Creating plots...")
        figpath = f"{base_dir}/summaryQLs/datapaper/{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
        paper_plots(ds_subset, ds_mode, ERA_data, HALO_data, figpath, y_range)
        
        # figpath_small = f"{base_dir}/summaryQLs/datapaper/{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}.png"
        # figpath_large = f"{base_dir}/summaryQLs/datapaper/{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}_ext.png"
        # weekly_plot_paper(ds_subset, ERA_data, HALO_data, figpath_small, y_range)
        # weekly_plot_paper_ext(ds_subset, ERA_data, HALO_data, figpath_large, y_range)
        print(f"Finished plots for {date.strftime('%Y-%m-%d')} ({time.time()-t0:.2f}s)")        
        del ds_subset, ERA_data


#%% combine BASTA data into one dataset
if what =='BASTA_resampling':
    # code below was used once to create new .zarr files 
    dbz_vars = ["reflectivity", "reflectivity_raw", "z_30s", "z_120s"]
    vel_vars = ["velocity", "vel_30s", "vel_120s"]
    max_vars = ["background_mask", "bg_30s", "bg_120s", "is_precip", "cloudtop", "zdi"]
    t_res = '5s'

    dsets_r = {}
    for mode in BASTAmodes:
        if mode == '25m':
            continue
        t0 = time.time()
        print(f"resampling zarr for {mode}")
        ds = dsets_native[mode]
        ds = ds.chunk({'time': 3600, 'range': -1})

        ds_r = adjust_range(ds, mode) if mode != '25m' else ds
        dsets_r[mode] = ds_r
        print(f"  finished processing {mode} (time-resampling + range adjustment), at {pd.to_datetime(time.time(), unit='s').strftime('%Y-%m-%d %H:%M:%S')}, time elapsed {time.time()-t0:.2f}s")
        
    for mode in BASTAmodes:
        if mode == '25m':
            continue
        t0 = time.time()
        print(f"saving zarr for {mode}")
        encoding = {}
        for var in dsets_r[mode].data_vars:
            enc = dict(dsets_r[mode][var].encoding)
            enc.pop('chunks', None)  # remove existing chunking info to let to_zarr decide
            enc.pop('preferred_chunks', None)
            encoding[var] = enc
        dsets_r[mode].to_zarr(f"{os.path.dirname(zarrdir)}/BASTAmodes_25m/BASTA_{mode}.zarr", mode='w', consolidated=True, encoding=encoding)
        print(f"  saved range-resampled zarr for {mode} at {pd.to_datetime(time.time(), unit='s').strftime('%Y-%m-%d %H:%M:%S')}, time elapsed {time.time()-t0:.2f}s")


#%% BASTA composite
if what == 'BASTA_masking':
    # # below code was run once to add 'mask_artefacts' to the zarr files (with mask == 1 where data should be included, similar to background_mask)
    # import zarr
    # print(f"calculating masks...")
    # threshold_abs = 10
    # for mode in dsets_modes:
    #     dsets_modes[mode] = dsets_modes[mode].chunk({'time': 3600, 'range': -1})
        
    # diff_25m_12m5 = (dsets_modes['25m']['reflectivity'].fillna(999) - dsets_modes['12m5']['reflectivity'].fillna(999))
    # # diff_25m_12m5_vel = (dsets_modes['25m']['velocity'] - dsets_modes['12m5']['velocity'])
    # include25 = ((diff_25m_12m5.range < 0.4e3) | ((1e3 < diff_25m_12m5.range) & (diff_25m_12m5.range < 2.7e3)))
    # # mask_25m_strict = (np.abs(diff_25m_12m5) > threshold_abs)
    # mask_25m_strict = (np.abs(diff_25m_12m5) > threshold_abs)#.where(np.abs(diff_25m_12m5_vel) > 5).astype(bool)
    # mask_25m_loose = mask_25m_strict.where(include25, False).astype(bool) # only apply mask where range < 3 km, to avoid flagging artefacts at far ranges where no artefacts are present

    # diff_100m_25m_strict = (dsets_modes['100m_18km']['reflectivity'].fillna(999) - dsets_modes['25m']['reflectivity'].where(mask_25m_strict == False).fillna(999))
    # diff_100m_25m_loose = (dsets_modes['100m_18km']['reflectivity'].fillna(999) - dsets_modes['25m']['reflectivity'].where(mask_25m_loose == False).fillna(999))
    # include100 = (
    #     (diff_100m_25m_loose.range < 0.6e3) 
    #     | ((diff_100m_25m_loose.range > 4.3e3) & (diff_100m_25m_loose.range < 5.5e3))
    #     | ((diff_100m_25m_loose.range > 6.7e3) & (diff_100m_25m_loose.range < 7.3e3))
    #     | ((diff_100m_25m_loose.range > 8.9e3) & (diff_100m_25m_loose.range < 10e3))
    #     )
    # mask_100m_strict = (np.abs(diff_100m_25m_strict) > threshold_abs)
    # mask_100m_loose = (np.abs(diff_100m_25m_loose) > threshold_abs).where(include100, False) # only apply mask in intervals where artefacts are clearly visible in original density plots

    # dsets_modes['25m']['mask_artefacts_loose'] = ~mask_25m_loose
    # dsets_modes['25m']['mask_artefacts_strict'] = ~mask_25m_strict
    # dsets_modes['100m_18km']['mask_artefacts_loose'] = ~mask_100m_loose
    # dsets_modes['100m_18km']['mask_artefacts_strict'] = ~mask_100m_strict

    # masks = {
    #     '25m': {'loose': mask_25m_loose, 'strict': mask_25m_strict},
    #     '100m_18km': {'loose': mask_100m_loose, 'strict': mask_100m_strict},
    #     }

    # t = pd.Timestamp.now()
    # print(f"saving masks to zarrs, starting {t.strftime('%Y-%m-%d %H:%M:%S')}")
    # for mode, mode_masks in masks.items():
    #     zarr_path = f"{os.path.dirname(zarrdir)}/BASTAmodes_25m/BASTA_{mode}.zarr"
    #     z = zarr.open(zarr_path, mode='a')

    #     if 'mask_artefacts_loose_new' in z:
    #         del z['mask_artefacts_loose_new']
    #     if 'mask_artefacts_strict_new' in z:
    #         del z['mask_artefacts_strict_new']

    #     for mask_type, mask in mode_masks.items():
    #         var_name = f'mask_artefacts_{mask_type}'  # → mask_artefacts_loose / mask_artefacts_strict

    #         if var_name in z:
    #             del z[var_name]

    #         mask.name = var_name
    #         mask.to_zarr(zarr_path, mode='a', compute=True)
        
    #     zarr.consolidate_metadata(zarr_path)
    # print(f"finished saving masks to zarrs, finished {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # print(f"elapsed time: {format_elapsed((pd.Timestamp.now() - t).total_seconds())}\n")

    # # resampling back to 100m native range resolution of mask
    # t = pd.Timestamp.now()
    # print(f"resampling masks back to 100m native range resolution, starting {t.strftime('%Y-%m-%d %H:%M:%S')}")
    # # import zarr
    # range100 = dsets_native['100m_18km']['range'].values
    # edges = np.concatenate([[range100[0] - 50], (range100[:-1] + range100[1:]) / 2, [range100[-1] + 50]])
    # mask100_strict = (dsets_modes['100m_18km']['mask_artefacts_strict'].groupby_bins('range', bins=edges).mean() >= 0.5)
    # mask100_strict = xr.DataArray(mask100_strict.data, dims=('time', 'range'), coords={'time': mask100_strict.time, 'range': dsets_native['100m_18km']['range']}, name='mask_artefacts_strict')
    # mask100_loose = (dsets_modes['100m_18km']['mask_artefacts_loose'].groupby_bins('range', bins=edges).mean() >= 0.5)
    # mask100_loose = xr.DataArray(mask100_loose.data, dims=('time', 'range'), coords={'time': mask100_loose.time, 'range': dsets_native['100m_18km']['range']}, name='mask_artefacts_loose')
    # dsets_native['100m_18km']['mask_artefacts_strict'] = mask100_strict
    # dsets_native['100m_18km']['mask_artefacts_loose'] = mask100_loose
    
    # mask25_loose = dsets_modes['25m']['mask_artefacts_loose']
    # mask25_strict = dsets_modes['25m']['mask_artefacts_strict']
    # dsets_native['25m']['mask_artefacts_loose'] = mask25_loose
    # dsets_native['25m']['mask_artefacts_strict'] = mask25_strict

    # zarr_path100 = f"{os.path.dirname(zarrdir)}/no_resampling/BASTA_100m_18km.zarr"
    # z = zarr.open(zarr_path100, mode='a')
    # if 'mask_artefacts' in z:
    #     del z['mask_artefacts']
    # mask100_strict.chunk('auto').to_zarr(zarr_path100, mode='a', compute=True)
    # mask100_loose.chunk('auto').to_zarr(zarr_path100,  mode='a', compute=True)
    # zarr.consolidate_metadata(zarr_path100)

    # zarr_path25 = f"{os.path.dirname(zarrdir)}/no_resampling/BASTA_25m.zarr"
    # z = zarr.open(zarr_path25, mode='a')
    # existing_range = z['range'][:]  # 720-gate range from the store
    
    # if 'mask_artefacts' in z:
    #     del z['mask_artefacts']

    # mask25_loose_reindexed = mask25_loose.reindex(range=existing_range, fill_value=False)
    # mask25_strict_reindexed = mask25_strict.reindex(range=existing_range, fill_value=False)

    # mask25_loose_reindexed.to_zarr(zarr_path25, mode='a', compute=True, align_chunks=True)
    # mask25_strict_reindexed.to_zarr(zarr_path25, mode='a', compute=True, align_chunks=True)
    # zarr.consolidate_metadata(zarr_path25)
    # print(f"finished resampling masks back to 100m native range resolution, finished {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # print(f"elapsed time: {format_elapsed((pd.Timestamp.now() - t).total_seconds())}\n")

    # example plot
    date = datetime(2024, 11, 22)
    date = datetime(2025, 1, 12)
    mode = '25m'
    mode = '100m_18km'
    dates = [datetime(2024, 11, 22), datetime(2025, 1, 12)]
    modes = ['25m', '100m_18km']
    for date in dates:
        for mode in modes:
            print(f"making plot for {mode} on {date.strftime('%Y-%m-%d')    }")
            # data_subset = dsets_modes[mode].sel(time=slice(date, date+pd.Timedelta(days=1))).chunk({'time': 3600, 'range': -1})
            data_subset = dsets_native[mode].sel(time=slice(date, date+pd.Timedelta(days=1))).chunk({'time': 3600, 'range': -1})
            native = True
            maskflag = 0 if native else 1
            
            # quicklook showing different filters
            fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True, sharey=True)
            data_subset['reflectivity'].where(data_subset['background_mask']).plot.imshow(
                ax=axs[0], 
                x='time', 
                cmap='viridis',
                cbar_kwargs={'label': 'dBZ'},
                vmin=-60, vmax=30
            )
            axs[0].set_title("unfiltered", fontsize=14)
            axs[0].set_ylabel("Range [km]")
            axs[0].yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            axs[0].set_ylim([0, 12e3])

            filtered_refl = data_subset['reflectivity'].where(data_subset['mask_artefacts_loose'] == maskflag).where(data_subset['background_mask'])
            filtered_refl.plot.imshow(
                ax=axs[1], 
                x='time', 
                cmap='viridis', 
                cbar_kwargs={'label': 'dBZ'},
                vmin=-60, vmax=30
            )
            axs[1].set_title("filtered (loose artefacts-mask aplied)", fontsize=14)
            axs[1].set_ylabel("Range [km]")
            axs[1].set_xlabel("Time")
            axs[1].set_ylim([0, 12e3])
            axs[1].yaxis.set_major_formatter(FuncFormatter(meters_to_km))

            filtered_refl = data_subset['reflectivity'].where(data_subset['mask_artefacts_strict'] == maskflag).where(data_subset['background_mask'])
            filtered_refl.plot.imshow(
                ax=axs[2], 
                x='time', 
                cmap='viridis', 
                cbar_kwargs={'label': 'dBZ'},
                vmin=-60, vmax=30
            )
            axs[2].set_title("filtered (strict artefacts-mask aplied)", fontsize=14)
            axs[2].set_ylabel("Range [km]")
            axs[2].set_xlabel("Time")
            axs[2].set_ylim([0, 12e3])
            axs[2].yaxis.set_major_formatter(FuncFormatter(meters_to_km))

            plt.suptitle(f"Example of artefact filtering for {mode} reflectivity on {date.strftime('%Y-%m-%d')}", fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{base_dir}/BASTA/{date.year}{date.month:02}{date.day:02}_example_artefactmask_{mode}_(native).png")
            plt.show()


#%% zarr corrections (run only once)
"""below code was run once to correct the zarr files (fill DFRs, add/remove variables to individual zarrs, etc.)"""
if what == 'zarr_corrections':
    # add BASTA cloudtop DFR correction + gas attenuation
    # resample masks from native / 25m resolution BASTA to time-upscaled zarr
    # compute X-Ka, Ka-W and X-W DFRs
    # store output zarrs
    
    import zarr

    flag_vars = ['background_mask', 'bg_30s', 'bg_120s', 'mask_artefacts_loose', 'mask_artefacts_strict']
    exclude_vars = {'mask_artefacts_loose', 'mask_artefacts_strict', 'time_original', 'is_precip'}

    # remove / update boolean vars
    mira_flag_vars = [v for v in flag_vars if v in mira_native.data_vars and v not in exclude_vars]
    basta_flag_vars = [v for v in flag_vars if v in basta_native.data_vars and v not in exclude_vars]
    mxpol_flag_vars = [v for v in flag_vars if v in mxpol_native.data_vars and v not in exclude_vars]

    mira_linearvars = [i for i in mira_native.data_vars if i not in flag_vars and i not in exclude_vars]
    basta_linearvars = [i for i in basta_native.data_vars if i not in flag_vars and i not in exclude_vars]
    mxpol_linearvars = [i for i in mxpol_native.data_vars if i not in flag_vars and i not in exclude_vars]

    # confirm each radar's daily step count before setting chunk sizes
    for name, ds in [#('MIRA', mira_resampled), 
                     ('BASTA', basta_resampled), 
                     ('MXPol', mxpol_resampled)]:
        one_day = ds.time.sel(time=slice(str(ds.time.values[0])[:10], None)).sel(
            time=slice(str(ds.time.values[0])[:10], (pd.Timestamp(ds.time.values[0]) + pd.Timedelta('1D'))))
        print(f"{name}: steps/day = {len(one_day)}")

    # build rechunked template + run update, per radar, into SEPARATE paths (not overwriting originals yet)
    for name, native_ds, resampled_ds, linvars, flagvars, time_chunk, out_suffix in [
        #('MIRA', mira_native, mira_resampled, mira_linearvars, mira_flag_vars, 2880, '_rechunked'),
        ('BASTA', basta_native, basta_resampled, basta_linearvars, basta_flag_vars, 2880, '_rechunked'),
        ('MXPol', mxpol_native, mxpol_resampled, mxpol_linearvars, mxpol_flag_vars, 2880, '_rechunked'),
    ]:
        out_path = f"{zarrdir}/{name}{out_suffix}.zarr"
        template = resampled_ds.chunk({'time': time_chunk, 'range': -1})
        for var in template.variables:
            template[var].encoding.pop('chunks', None)
            template[var].encoding.pop('preferred_chunks', None)
        template.to_zarr(out_path, mode='w', compute=False)
        print(f"{name}: template written to {out_path}")

        update_resampled_zarr(
            name, native_ds, out_path, resampled_ds,
            time_tol='15s', range_tol=13,
            linear_vars=linvars, flag_vars=flagvars, freq='1D',
        )
        print(f"{name}: update complete, verify before swapping into {zarrdir}/{name}.zarr")

    # resampling 3s-resolution artefact masks to 30s_25m zarr file
    source = xr.open_zarr(f"{zarrdir_native}/BASTA_25m.zarr", consolidated=True).unify_chunks()
    mask_zarr_loose = safe_reindex_dim(source[['mask_artefacts_loose']], 'time', basta_resampled.time, '15s')
    mask_zarr_loose = safe_reindex_dim(mask_zarr_loose, 'range', basta_resampled.range, 13)
    mask_zarr_loose = (mask_zarr_loose == 1).compute()
    mask_zarr_loose.drop_vars('range', errors='ignore').chunk({'time': 2880, 'range': -1})
    mask_zarr_loose.to_zarr(f"{zarrdir}/BASTA.zarr", mode='a', compute=True)
    mask_zarr_strict = safe_reindex_dim(source[['mask_artefacts_strict']], 'time', basta_resampled.time, '15s')
    mask_zarr_strict = safe_reindex_dim(mask_zarr_strict, 'range', basta_resampled.range, 13)
    mask_zarr_strict = (mask_zarr_strict == 1).compute()
    mask_zarr_strict.drop_vars('range', errors='ignore').chunk({'time': 2880, 'range': -1})
    mask_zarr_strict.to_zarr(f"{zarrdir}/BASTA.zarr", mode='a', compute=True)
    zarr.consolidate_metadata(f"{zarrdir}/BASTA.zarr")

    # include cloudtop DFR values in BASTA zarr
    basta_native_cloudtopDFRs = add_DFRcorrection_xr(cloudtopDFRs, basta_native.time.values).rename('cloudtopDFRs')
    basta_native_cloudtopDFRs.chunk('auto').to_zarr(f"{zarrdir_native}/BASTA_25m.zarr", mode='a', compute=True)
    zarr.consolidate_metadata(f"{zarrdir_native}/BASTA_25m.zarr")

    basta_resampled_cloudtopDFRs = add_DFRcorrection_xr(cloudtopDFRs, basta_resampled.time.values).rename('cloudtopDFRs')
    basta_resampled_cloudtopDFRs.chunk({'time': 2880}).to_zarr(f"{zarrdir}/BASTA.zarr", mode='a', compute=True)
    zarr.consolidate_metadata(f"{zarrdir}/BASTA.zarr")

    # recalculate DFRs
    DFR_XKa = (mxpol_resampled['reflectivity'] - mira_resampled['reflectivity']).rename('DFR_XKa')
    DFR_KaW = (mira_resampled['reflectivity'] - basta_resampled['reflectivity']).rename('DFR_KaW')
    DFR_XW = (mxpol_resampled['reflectivity'] - basta_resampled['reflectivity']).rename('DFR_XW')

    for arr in (DFR_XKa, DFR_KaW, DFR_XW):
        arr.encoding.pop('chunks', None)
        arr.encoding.pop('preferred_chunks', None)
        for coord in arr.coords:
            arr[coord].encoding.pop('chunks', None)
            arr[coord].encoding.pop('preferred_chunks', None)

    mira_path = f"{zarrdir}/MIRA.zarr"
    mxpol_path = f"{zarrdir}/MXPol.zarr"
    basta_path = f"{zarrdir}/BASTA.zarr"

    writes = [(mira_path, [DFR_XKa, DFR_KaW]), (mxpol_path, [DFR_XKa, DFR_XW]), (basta_path, [DFR_XW, DFR_KaW]),]

    for path, vars_to_write in writes:
        z = zarr.open(path, mode='a')
        for var in vars_to_write:
            if var.name in z:
                del z[var.name]
        zarr.consolidate_metadata(path)

        for var in vars_to_write:
            var_out = var.drop_vars('range', errors='ignore').chunk({'time': 2880, 'range': -1})
            var_out.to_dataset().to_zarr(path, mode='a')
        zarr.consolidate_metadata(path)
        print(f"overwrote {[v.name for v in vars_to_write]} in {path}")


#%% CFADs for BASTA sensitivity
if what == 'CFAD_BASTA_resampled':
    # make filtered reflectivity density/sensitivity plots for resampled data
    z12f, rg12f, msk12f = prepare_data_profiled(dsets_modes, '12m5', 'reflectivity')
    z25f, rg25f, msk25f = prepare_data_profiled(dsets_modes, '25m', 'reflectivity')
    z25fl, rg25fl, msk25fl = prepare_data_profiled(dsets_modes, '25m', 'reflectivity', maskname='mask_artefacts_loose')
    z25fs, rg25fs, msk25fs = prepare_data_profiled(dsets_modes, '25m', 'reflectivity', maskname='mask_artefacts_strict')
    z100f, rg100f, msk100f = prepare_data_profiled(dsets_modes, '100m_18km', 'reflectivity')
    z100fl, rg100fl, msk100fl = prepare_data_profiled(dsets_modes, '100m_18km', 'reflectivity', maskname='mask_artefacts_loose')
    z100fs, rg100fs, msk100fs = prepare_data_profiled(dsets_modes, '100m_18km', 'reflectivity', maskname='mask_artefacts_strict')

    # no filter
    data_list_f = [(z12f, rg12f), (z25f, rg25f), (z100f, rg100f)]
    bins_list_f = ((105, int(12e3/25)), (105, int(12e3/25)), (105, int(12e3/25)))
    histograms_f = precompute_histograms(data_list_f, bins_list_f)
    nonzero_vals_f = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_f])
    vmin_f = np.nanmin(nonzero_vals_f)
    vmax_f = np.nanmax(nonzero_vals_f)
    norms_f = (LogNorm(vmin_f, 15e3), LogNorm(vmin_f, 15e3), LogNorm(vmin_f, 15e3))
    vmaxs_f = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_f, norms_f, vmaxs_f,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 ', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            # cmap=white_turbo,
                            norm=norm
                            # vmin=vmin_fl, vmax=vmax_fl
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (resampled in range to 25m, unfiltered)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual_25m.png")

    # combined plot
    H_combined_resampled = sum(H for H, _, _ in histograms_f)
    xedges = histograms_f[0][1]
    yedges = histograms_f[0][2]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(xedges, yedges, H_combined_resampled.T,
                    cmap='inferno',
                    norm=LogNorm(vmin_f, 15e3)
                    )
    ax.set_xlabel('$Ze_{modes}$ [dBZ]')
    ax.set_title('Combined sensitivity (resampled in range)', fontweight='bold')
    ax.set_xlim([-60, 45])
    ax.set_ylim([0, 12e3])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
    ax.set_ylabel('range [km]')
    ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=ax, label='count')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_combined_25m.png")


    # loose filter
    data_list_fl = [(z12f, rg12f), (z25fl, rg25fl), (z100fl, rg100fl)]
    bins_list_fl = ((105, int(12e3/25)), (105, int(12e3/25)), (105, int(12e3/25)))
    histograms_fl = precompute_histograms(data_list_fl, bins_list_fl)
    nonzero_vals_fl = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_fl])
    vmin_fl = np.nanmin(nonzero_vals_fl)
    vmax_fl = np.nanmax(nonzero_vals_fl)
    norms_fl = (LogNorm(vmin_fl, 15e3), LogNorm(vmin_fl, 15e3), LogNorm(vmin_fl, 15e3))
    vmaxs_fl = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_fl, norms_fl, vmaxs_fl,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 (not filtered)', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            # cmap=white_turbo,
                            norm=norm
                            # vmin=vmin_fl, vmax=vmax_fl
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (resampled in range to 25m)\nfiltered for coupling artefacts (loose)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual_25m_filtered_loose.png")

    # combined plot
    H_combined_resampled = sum(H for H, _, _ in histograms_fl)
    xedges = histograms_fl[0][1]
    yedges = histograms_fl[0][2]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(xedges, yedges, H_combined_resampled.T,
                    cmap='inferno',
                    norm=LogNorm(vmin_fl, 15e3)
                    )
    ax.set_xlabel('$Ze_{modes}$ [dBZ]')
    ax.set_title('Combined sensitivity (resampled in range)\nfiltered for coupling artefacts (loose)', fontweight='bold')
    ax.set_xlim([-60, 45])
    ax.set_ylim([0, 12e3])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
    ax.set_ylabel('range [km]')
    ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=ax, label='count')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_combined_25m_filtered_loose.png")


    # strict filter
    data_list_fs = [(z12f, rg12f), (z25fs, rg25fs), (z100fs, rg100fs)]
    bins_list_fs = ((105, int(12e3/25)), (105, int(12e3/25)), (105, int(12e3/25)))
    histograms_fs = precompute_histograms(data_list_fs, bins_list_fs)
    nonzero_vals_fs = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_fs])
    vmin_fs = np.nanmin(nonzero_vals_fs)
    vmax_fs = np.nanmax(nonzero_vals_fs)
    norms_fs = (LogNorm(vmin_fs, 15e3), LogNorm(vmin_fs, 15e3), LogNorm(vmin_fs, 15e3))
    vmaxs_fs = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_fs, norms_fs, vmaxs_fs,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 (not filtered)', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            norm=norm
                            # vmin=vmin, vmax=vmax
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (resampled in range to 25m)\nfiltered for coupling artefacts (strict)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual_25m_filtered_strict.png")

    # combined plot
    H_combined_resampled = sum(H for H, _, _ in histograms_fs)
    xedges = histograms_fs[0][1]
    yedges = histograms_fs[0][2]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(xedges, yedges, H_combined_resampled.T,
                    cmap='inferno',
                    norm=LogNorm(vmin_fs, 15e3)
                    )
    ax.set_xlabel('$Ze_{modes}$ [dBZ]')
    ax.set_title('Combined sensitivity (resampled in range)\nfiltered for coupling artefacts (strict)', fontweight='bold')
    ax.set_xlim([-60, 45])
    ax.set_ylim([0, 12e3])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
    ax.set_ylabel('range [km]')
    ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=ax, label='count')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_combined_25m_filtered_strict.png")

#%% make filtered reflectivity density/sensitivity plots for native data
if what == 'CFAD_BASTA_native':
    z12f, rg12f, msk12f = prepare_data_profiled(dsets_native, '12m5', 'reflectivity')
    z25f, rg25f, msk25f = prepare_data_profiled(dsets_native, '25m', 'reflectivity')
    z25fl, rg25fl, msk25fl = prepare_data_profiled(dsets_native, '25m', 'reflectivity', maskname='mask_artefacts_loose')
    z25fs, rg25fs, msk25fs = prepare_data_profiled(dsets_native, '25m', 'reflectivity', maskname='mask_artefacts_strict')
    z100f, rg100f, msk100f = prepare_data_profiled(dsets_native, '100m_18km', 'reflectivity')
    z100fl, rg100fl, msk100fl = prepare_data_profiled(dsets_native, '100m_18km', 'reflectivity', maskname='mask_artefacts_loose')
    z100fs, rg100fs, msk100fs = prepare_data_profiled(dsets_native, '100m_18km', 'reflectivity', maskname='mask_artefacts_strict')

    # no filter
    data_list_f = [(z12f, rg12f), (z25f, rg25f), (z100f, rg100f)]
    bins_list_f = ((105, int(12e3/12.5)), (105, int(12e3/25)), (105, int(12e3/100)))
    histograms_f = precompute_histograms(data_list_f, bins_list_f)
    nonzero_vals_f = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_f])
    vmin_f = np.nanmin(nonzero_vals_f)
    vmax_f = np.nanmax(nonzero_vals_f)
    norms_f = (LogNorm(vmin_f, 15e3), LogNorm(vmin_f, 15e3), LogNorm(vmin_f, 15e3))
    vmaxs_f = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_f, norms_f, vmaxs_f,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 ', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            # cmap=white_turbo,
                            norm=norm
                            # vmin=vmin_fl, vmax=vmax_fl
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range (unfiltered)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual.png")


    # loose filter
    data_list_fl = [(z12f, rg12f), (z25fl, rg25fl), (z100fl, rg100fl)]
    bins_list_fl = ((105, int(12e3/12.5)), (105, int(12e3/25)), (105, int(12e3/100)))
    histograms_fl = precompute_histograms(data_list_fl, bins_list_fl)
    nonzero_vals_fl = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_fl])
    vmin_fl = np.nanmin(nonzero_vals_fl)
    vmax_fl = np.nanmax(nonzero_vals_fl)
    norms_fl = (LogNorm(vmin_fl, 15e3), LogNorm(vmin_fl, 15e3), LogNorm(vmin_fl, 15e3))
    vmaxs_fl = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_fl, norms_fl, vmaxs_fl,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 (not filtered)', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            # cmap=white_turbo,
                            norm=norm
                            # vmin=vmin_fl, vmax=vmax_fl
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range\nfiltered for coupling artefacts (loose)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual_filtered_loose.png")


    # strict filter
    data_list_fs = [(z12f, rg12f), (z25fs, rg25fs), (z100fs, rg100fs)]
    bins_list_fs = ((105, int(12e3/12.5)), (105, int(12e3/25)), (105, int(12e3/100)))
    histograms_fs = precompute_histograms(data_list_fs, bins_list_fs)
    nonzero_vals_fs = np.concatenate([H[H > 0].ravel() for (H, *_) in histograms_fs])
    vmin_fs = np.nanmin(nonzero_vals_fs)
    vmax_fs = np.nanmax(nonzero_vals_fs)
    norms_fs = (LogNorm(vmin_fs, 15e3), LogNorm(vmin_fs, 15e3), LogNorm(vmin_fs, 15e3))
    vmaxs_fs = [1250, 1250, 1250]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, (H, xedges, yedges),  norm, vmax, xlabel, title) in enumerate(zip(
            axes, histograms_fs, norms_fs, vmaxs_fs,
            ['$Ze_{12m5}$ [dBZ]', '$Ze_{25m}$ [dBZ]', '$Ze_{100m}$ [dBZ]'],
            ['Sensitivity 12m5 (not filtered)', 'Sensitivity 25m', 'Sensitivity 100m'],
            )):

        H_masked = np.ma.masked_where(H == 0, H)
        im = ax.pcolormesh(xedges, yedges, H_masked.T,
                            cmap='inferno',
                            norm=norm
                            # vmin=vmin, vmax=vmax
                            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim([-60, 45])
        ax.set_ylim([0, 12e3])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(2000))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km_num))
        ax.grid(True, color='grey', alpha=0.5, linewidth=0.5)

        if i == 2:
            plt.colorbar(im, ax=ax, label='count')
        else:
            plt.colorbar(im, ax=ax)

    axes[0].set_ylabel('range [km]')
    plt.suptitle('Density plots of reflectivity vs. range\nfiltered for coupling artefacts (strict)', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/BASTA/sensitivity_allmodes_individual_filtered_strict.png")

