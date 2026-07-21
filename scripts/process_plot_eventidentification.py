#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:25:34 2025

### DESCRIPTION TO BE UPDATED ###
script that plots daily quicklooks for available radars
- Zdr corrected MXPol
- attenuation-corrected MIRA & BASTA
- calibrated BASTA (& MXPol)
- data quality flag (if DFR at top cloud != 0, )
- ERA5 isotherms
and generates spectrograms (MIRA)

@author: clerx
"""

# %% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pathlib import Path
from matplotlib.dates import HourLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from functools import partial
from src.aggregator import aggregate_campaign
from src.plotting import plot_spectrogram_MIRA, plot_spectrogram_MXPol, plot_DFR, plot_daily_quicklook, plot_MIRA_scans, meters_to_km
from src.radar_processing import Zdr_corr_xr, add_Kdp, cloudtop_dask, drop_nfft, prepare_MIRAdata, prepare_MXPoldata
from src.radar_processing import preprocess, preprocess_BASTA, preprocess_MIRA, preprocess_MXPol, safe_openmfdataset, correct_gas_attenuation, DFRcorrection
from src.utils import generate_date_range, safe_reindex, read_ERA_data, get_DFRdata, clean_dataframe, calc_alt, fix_encoding
from src.constants_input import start_date as startdate, end_date as enddate
from src.constants_input import MXPol_lost_frequency_times, MXPol_Zdr_corrections, radiosonde_dates, pltConfig, common_range, DFR_fn, att_fn, df_precip, vars_hydroclassif
from src.constants_input import radar_altitude, calibrationvalues, frequencies, nl,  dirs, base_dir, BASTAmode, latlon_Helmos, MIRA_lapserate
from src import os, re, glob, np, pd, xr, gc, plt, datetime, mdates, mcolors, warnings, pyart, time
from MXPol_hydroclassif import classify_hm, plot_classification

# %% dates, input directories & other constants
dates = generate_date_range(startdate, enddate)

# DFRs = get_DFRdata(DFR_files) # outdated - now using the merged file in which MIRA reflectivities are corrected
DFRs = pd.read_csv(DFR_fn, index_col=0)

drop_spectral = True
# input for MIRA spectrogram plotting
remove_noise = True
# noise_method = 'quantile'
noise_method = 'hildebrandsekhon'
stdev_threshold = 1

MXPol_lost_frequencies = [
    (pd.to_datetime(start, format='%Y%m%d-%H%M%S'),
     pd.to_datetime(end, format='%Y%m%d-%H%M%S'))
    for start, end in MXPol_lost_frequency_times
]


# %% what to do
# what = 'summaryQLs'           # generate daily summary quicklooks -- add MXPol 'profile' scans (elevation 0 or 177.7)!!
# generate MIRA spectrograms (spectral Ze and sLDR)
# what = 'MIRA_spectrograms'
# what = 'DFRcomparison'        # compare calculated DFR between MW results and updated algorithms for 2024.12.05
# what = 'DFRcorrection_plots'  # make plots of BASTA-MIRA DFR throughout the campaign
# what = 'DFR_dailyplots        # make daily BASTA-MIRA DFR plots
# what = 'MIRA_scans'           # generate MIRA quicklooks of scans for specific time intervals
# what = 'MXPol_spectrograms'   # generate MXPol spectrograms
# TO DO:
# what = 'WRF_plots'            # generate WRF plots
# what = 'triple_frequency'     # triple frequency analysis
# what = 'campaign_stats'       # full campaign statistics
what = 'MXPol_L1'             # add Zdr-correction and Kdp and hydroclassif to L1 files
# what = 'MIRA_tempcheck'       # compare MIRA to RS temperatures
# what = 'weekly_peaktree_plots' # weekly plots using peaktree & MIRA output
# what = 'attenuation_stats'    # atmospheric gas attenuation statistics


# %% create daily quicklooks
if what == 'summaryQLs':
    for date in dates:
        # delete any (memory-consuming) pre-existing variables that are defined in this cell
        for var in ['MXPol_data', 'MIRA_data', 'BASTA_data', 'DFR_correction']:
            if var in locals():
                del globals()[var]

        year, month, day = date.year, date.month, date.day
        print(f"Making daily summary QL for {year}-{month:02}-{day:02}")

        MXPol_filedir = os.path.join(
            dirs['MXPol'], f"{year}/{month:02}/{day:02}")
        MXPol_files = sorted(glob.glob(f"{MXPol_filedir}/*_Zdr.nc"))
        if MXPol_files:
            MXPol_files, _ = safe_openmfdataset(MXPol_files)
            MXPol_data = preprocess_MXPol(xr.open_mfdataset(MXPol_files,
                                                            combine='nested',
                                                            concat_dim='time',
                                                            chunks={'time': 3600}),
                                          Zdr_corr=True, drop_spectral=drop_spectral)

        MIRA_filedir = os.path.join(dirs['MIRA'], f"{year}/{month:02}/{day:02}")
        MIRA_files = sorted(glob.glob(f"{MIRA_filedir}/*_merged.nc"))
        if MIRA_files:
            MIRA_data = preprocess_MIRA(xr.open_mfdataset(MIRA_files,
                                                          combine='by_coords',
                                                          chunks={
                                                              'time': 3600},
                                                          preprocess=preprocess,
                                                          parallel=True), drop_spectral=drop_spectral)

        BASTA_files = sorted(glob.glob(os.path.join(
            dirs['BASTA'], f"BASTA_L1_{BASTAmode}_{year}{month:02}{day:02}*.nc")))
        if BASTA_files:
            BASTA_data = preprocess_BASTA(xr.open_mfdataset(
                BASTA_files, combine='by_coords'), drop_variables=True)

        dataset_times = []
        if 'MXPol_data' in locals() and MXPol_data is not None:
            dataset_times.append(MXPol_data['time'].values)
        if 'MIRA_data' in locals() and MIRA_data is not None:
            dataset_times.append(MIRA_data['time'].values)
        if 'BASTA_data' in locals() and BASTA_data is not None:
            dataset_times.append(BASTA_data['time'].values)

        # tmin = max(np.min(times) for times in dataset_times)
        # tmax = min(np.max(times) for times in dataset_times)
        tmin = np.datetime64(datetime(year, month, day))
        tmax = tmin + np.timedelta64(1, 'D')
        # start to end of available data resampled to every 5 seconds
        full_time = pd.date_range(tmin, tmax, freq='5s')

        if 'MXPol_data' in locals() and MXPol_data is not None:
            MXPol = safe_reindex(MXPol_data, 'time', full_time,
                                 tolerance='0.5s',  name='MXPol')
            MXPol = safe_reindex(MXPol, 'range', common_range,
                                 tolerance=25, name='MXPol')
        if 'MIRA_data' in locals() and MIRA_data is not None:
            MIRA = safe_reindex(MIRA_data, 'time', full_time,
                                tolerance='2.5s', name='MIRA')
            MIRA = safe_reindex(MIRA, 'range', common_range,
                                tolerance=25, name='MIRA')
        if 'BASTA_data' in locals() and BASTA_data is not None:
            BASTA = safe_reindex(BASTA_data, 'time', full_time,
                                 tolerance='1.5s', name='BASTA')
            BASTA = safe_reindex(BASTA, 'range', common_range,
                                 tolerance=25, name='BASTA')
            # correct BASTA for atmospheric gas attenuation & add DFR-corrected variables (MIRA-BASTA)
            BASTA['reflectivity_attn_corrected'] = xr.DataArray(correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequencies['BASTA'], tmin, tmax).values, dims=BASTA['reflectivity'].dims,
                                                                coords={dim: BASTA[dim] for dim in BASTA['reflectivity'].dims})
            BASTA = DFRcorrection(DFRs, BASTA)
            BASTA['reflectivity_attn_DFR_corrected'] = BASTA['reflectivity_attn_DFR_corrected'] + \
                calibrationvalues['BASTA'][0]

        datasets = {}
        if 'MXPol_data' in locals() and MXPol_data is not None:
            datasets['MXPol'] = MXPol
        if 'MIRA_data' in locals() and MIRA_data is not None:
            datasets['MIRA'] = MIRA
        if 'BASTA_data' in locals() and BASTA_data is not None:
            datasets['BASTA'] = BASTA

        active_datasets = {name: ds for name, ds in zip(
            datasets.keys(), datasets.values())}
        active_sizes = [ds.sizes for ds in active_datasets.values()]

        # check if all datasets have the same sizes
        if not all(size == active_sizes[0] for size in active_sizes):
            print(f"Error in dataset dimensions:{nl}" + nl.join(
                f"{name}: {ds.sizes}" for name, ds in active_datasets.items()))

        # import ERA5 data (2 days to avoid interpolation for the last timestep)
        ERA_data = xr.concat([read_ERA_data(date, dirs['ERA']), read_ERA_data(
            date + pd.Timedelta(days=1), dirs['ERA'])], dim='time').sortby('time')

        plot_daily_quicklook(datasets, ERA_data, DFRs,
                             f"{base_dir}/summaryQLs")


# %% compare MW-NC DFR correction for 2024-12-05
if what == 'DFRcomparison':
    for var in ['MIRA_data', 'BASTA_data', 'MIRA', 'BASTA']:
        if var in locals():
            del globals()[var]

    date = datetime(2024, 12, 5)
    year, month, day = date.year, date.month, date.day
    BASTA_files = sorted(glob.glob(os.path.join(
        dirs['BASTA'], f"BASTA_L1_{BASTAmode}_{year}{month:02}{day:02}*.nc")))
    if BASTA_files:
        BASTA_data = preprocess_BASTA(
            xr.open_mfdataset(BASTA_files, combine='by_coords'))

    dataset_times = []
    if MIRA_data is not None:
        dataset_times.append(MIRA_data['time'].values)
    if BASTA_data is not None:
        dataset_times.append(BASTA_data['time'].values)

    tmin = max(np.min(times) for times in dataset_times)
    tmax = min(np.max(times) for times in dataset_times)
    # start to end of available data resampled to every 5 seconds
    full_time = pd.date_range(tmin, tmax, freq='5s')

    if MIRA_data:
        MIRA = safe_reindex(MIRA_data, full_time,
                            tolerance='2.5s', name='MIRA')
        MIRA = MIRA.reindex(range=common_range, method='nearest')
    if BASTA_data:
        BASTA = safe_reindex(BASTA_data, full_time,
                             tolerance='1.5s', name='BASTA')
        BASTA = BASTA.reindex(range=common_range, method='nearest')
        # correct BASTA for atmospheric gas attenuation
        BASTA['reflectivity_attn_corrected'] = xr.DataArray(correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequencies['BASTA'], tmin, tmax).values, dims=BASTA['reflectivity'].dims,
                                                            coords={dim: BASTA[dim] for dim in BASTA['reflectivity'].dims})
        BASTA = DFRcorrection(DFRs, BASTA)
        BASTA['reflectivity_attn_DFR_corrected'] = BASTA['reflectivity_attn_DFR_corrected'] + \
            calibrationvalues['BASTA'][0]

    # read MW DFR file
    df = pd.read_csv(os.path.join(
        base_dir, f"radar_calib/DWR_Neto_{day}_{month}_wth_mask.csv"))
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    times = pd.to_datetime(BASTA['time'].values)
    corrections = BASTA['DFR_correction'].values

    mask = (~pd.isna(times)) & (~np.isnan(corrections))
    plt.plot(df['time'], df['mean_DWR'], 'b.',
             label='DFR correction (Morgane)')
    plt.plot(times[mask].values, corrections[mask],
             'r-', label='DFR correction (Nicole)')
    plt.legend()
    plt.savefig(
        f"{base_dir}/radar_calib/20241204_DFRcomparison_MW-NC.png", bbox_inches='tight')

    # compare to data in full DFR dataset (NC)
    plt.scatter(
        pd.to_datetime(df['time']),
        df['mean_DWR'],
        color='blue',
        edgecolors=(0, 0, 1, 0.5),  # semi-transparent blue outline
        linewidths=1,
        label='DFR MW'
    )
    plt.scatter(
        pd.to_datetime(DFRs.query(
            "time.dt.date == @pd.to_datetime('2024-12-05').date()")['time']),
        DFRs.query(
            "time.dt.date == @pd.to_datetime('2024-12-05').date()")['mean_DFR'],
        color='red',
        edgecolors=(1, 0, 0, 0.5),  # semi-transparent red outline
        linewidths=1,
        alpha=0.8,                  # overall transparency of fill (optional)
        label='DFR NC'
    )
    plt.ylabel('DFR at top cloud [dB]')
    plt.legend()
    plt.savefig(
        f"{base_dir}/radar_calib/20241204_DFRcomparison_MW-NC_rawdata.png", bbox_inches='tight')


# %% plotting MIRA-BASTA DFR's including precipitation flag
if what == 'DFRcorrection_plots':
    DFRs_all_daily = plot_DFR(DFRs, df_precip, clip=False, daily=True)
    DFRs_all_daily.savefig(f"{base_dir}/radar_calib/DFR_daily_MIRABASTA.png",
                           dpi=300, bbox_inches='tight', facecolor='w')
    DFRs_clip_daily = plot_DFR(DFRs, df_precip, clip=True, daily=True)
    DFRs_clip_daily.savefig(f"{base_dir}/radar_calib/DFR_daily_MIRABASTA_clipped.png",
                            dpi=300, bbox_inches='tight', facecolor='w')

    DFRs_all = plot_DFR(DFRs, df_precip, clip=False, daily=False)
    DFRs_all.savefig(f"{base_dir}/radar_calib/DFR_MIRABASTA.png",
                     dpi=300, bbox_inches='tight', facecolor='w')
    DFRs_clip = plot_DFR(DFRs, df_precip, clip=True, daily=False)
    DFRs_clip.savefig(f"{base_dir}/radar_calib/DFR_MIRABASTA_clipped.png",
                      dpi=300, bbox_inches='tight', facecolor='w')


# %% DFR daily plot
if what == 'DFR_dailyplots':
    outdir = f"{base_dir}/radar_calib/DFR/plots"
    DFR_files = sorted(glob.glob(os.path.join(
        base_dir, 'radar_calib/DFR/*.csv')))

    for f in DFR_files:
        if os.path.getsize(f) > 0:
            try:
                df = pd.read_csv(f)
                df['time'] = pd.to_datetime(df['time'])
            except pd.errors.EmptyDataError:
                continue

        if not df.empty:
            datestr = (f.split('/')[-1]).split('_')[0]
            date = pd.to_datetime((f.split('/')[-1]).split('_')[0])
            year, month, day = date.year, date.month, date.day
            df['mean_DFR_clipped'] = np.clip(df['mean_DFR'], -5, 25)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['time'], df['mean_DFR_clipped'], 'o',
                    color='r', label='clipped to [-5, 25 dB]')
            ax.plot(df['time'], df['mean_DFR'], 'o',
                    color='b', label='not clipped')
            ax.axhline(y=calibrationvalues['BASTA'][0], linewidth=0.8,
                       color='black', label='calibration value +/- 2 stdv')
            ax.fill_between([date, date+pd.Timedelta(days=1)], y1=calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA']
                            [1], y2=calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1], alpha=0.1, color='black')
            ax.set_xlim(date, date+pd.Timedelta(days=1))
            ax.set_ylim([-5, 25])
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_ylabel('DFR [dB]')
            plt.legend()
            plt.suptitle(
                f'MIRA-BASTA DFR on {year}-{month:02}-{day:02}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.savefig(
                f"{outdir}/{year}{month:02}{day:02}_MIRA-BASTA_DFR.png")
            print(f"Saved {year}{month:02}{day:02}_MIRA-BASTA_DFR.png")
            # plt.show()
            plt.close()


# %% MIRA scans
if what == 'MIRA_scans':
    for date in dates:
        year, month, day = date.year, date.month, date.day
        print(
            f"starting to plot scans for {year}-{month:02}-{day:02} at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        outputdir = f"{dirs['MIRA']}/quicklooks/scans/{year}/{month:02}/{day:02}"
        os.makedirs(outputdir, exist_ok=True)
        scans = sorted(glob.glob(os.path.join(
            f"{base_dir}/MIRA/mom/{year}/{month:02}/{day:02}/{year}{month:02}{day:02}_*.rhi*.znc")))
        scans = scans + sorted(glob.glob(os.path.join(
            f"{base_dir}/MIRA/mom/{year}/{month:02}/{day:02}/{year}{month:02}{day:02}_*.ppi*.znc")))

        if scans:
            for fn in scans:
                try:
                    plot_MIRA_scans(fn, outputdir, savefig=True)
                except Exception as e:
                    print(
                        f"Error while processing {os.path.basename(fn)}: {e}")
                    continue
        print(
            f"finished plotting scans for {year}-{month:02}-{day:02} at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# %% plot MIRA spectrograms
if what == 'MIRA_spectrograms':
    dates = generate_date_range(startdate, enddate)
    for date in dates:
        year, month, day = date.year, date.month, date.day
        file_dir = f"{dirs['MIRA']}/{year}/{month:02}/{day:02}"
        flist = sorted(glob.glob(f"{file_dir}/*_merged.nc"))
        # output_dir = f"{dirs['MIRA']}/spectrograms/raw/{year}/{month:02}/{day:02}"
        output_dir = f"/home/clerx/cleancloud/datapaper/{year}{month:02}{day:02}_spectrograms"
        os.makedirs(output_dir, exist_ok=True)

        for flte in flist:
            print(f"\nMaking MIRA spectrograms for {os.path.basename(flte)}")
            plot_spectrogram_MIRA(
                flte, output_dir, spq=1, plot_Z=True, plot_LDR=False, savefig=True, overwrite=True)
            gc.collect()


# %% MXPol spectrograms
if what == 'MXPol_spectrograms':
    dates = generate_date_range('2024-11-29', enddate)
    for date in dates:
        year, month, day = date.year, date.month, date.day
        MXPol_filedir = os.path.join(
            dirs['MXPol'], f"{year}/{month:02}/{day:02}")
        MXPol_files = sorted(glob.glob(f"{MXPol_filedir}/*_Zdr*.nc"))
        MXPol_files, _ = safe_openmfdataset(MXPol_files)
        if MXPol_files:
            # output_dir = f"{os.path.dirname(dirs['MXPol'])}/spectrograms/filtered/{year}/{month:02}/{day:02}" # for plots with noise filtering
            # raw plots
            output_dir = f"{os.path.dirname(dirs['MXPol'])}/spectrograms/raw/{year}/{month:02}/{day:02}"
            os.makedirs(output_dir, exist_ok=True)
            print(
                f"Making MXPol spectrograms for {year}-{month:02}-{day:02} ({len(MXPol_files)} files)")
            for filename in MXPol_files:
                plot_spectrogram_MXPol(
                    filename, output_dir, spq=1, remove_noise=False, savefig=True, overwrite=True)
                gc.collect()


# %% triple frequency analysis
if what == 'triple_frequency':
    dates = [datetime(2025, 1, 15)]
    for date in dates:
        for var in ['MXPol_data', 'MIRA_data', 'BASTA_data', 'DFR_correction']:
            if var in locals():
                del globals()[var]

        year, month, day = date.year, date.month, date.day

        MXPol_filedir = os.path.join(
            dirs['MXPol'], f"{year}/{month:02}/{day:02}")
        MXPol_files = sorted(glob.glob(f"{MXPol_filedir}/*_Zdr*.nc"))
        if MXPol_files:
            MXPol_files, _ = safe_openmfdataset(MXPol_files)
            MXPol_data = preprocess_MXPol(xr.open_mfdataset(MXPol_files,
                                                            combine='nested',
                                                            concat_dim='time',
                                                            chunks={'time': 3600},
                                                            preprocess=drop_nfft),
                                                            Zdr_corr=True)

        MIRA_filedir = os.path.join(
            dirs['MIRA'], f"{year}/{month:02}/{day:02}")
        MIRA_files = sorted(glob.glob(f"{MIRA_filedir}/*_merged.nc"))
        if MIRA_files:
            MIRA_data = preprocess_MIRA(xr.open_mfdataset(MIRA_files,
                                                          combine='by_coords',
                                                          chunks={'time': 3600},
                                                          preprocess=preprocess,
                                                          parallel=True))

        BASTA_files = sorted(glob.glob(os.path.join(
            dirs['BASTA'], f"BASTA_L1_{BASTAmode}_{year}{month:02}{day:02}*.nc")))
        if BASTA_files:
            BASTA_data = preprocess_BASTA(xr.open_mfdataset(BASTA_files, combine='by_coords'))

        dataset_times = []
        if 'MXPol_data' in locals() and MXPol_data is not None:
            dataset_times.append(MXPol_data['time'].values)
        if 'MIRA_data' in locals() and MIRA_data is not None:
            dataset_times.append(MIRA_data['time'].values)
        if 'BASTA_data' in locals() and BASTA_data is not None:
            dataset_times.append(BASTA_data['time'].values)

        # tmin = max(np.min(times) for times in dataset_times)
        # tmax = min(np.max(times) for times in dataset_times)
        tmin = np.datetime64(datetime(year, month, day))
        tmax = tmin + np.timedelta64(1, 'D')
        # start to end of available data resampled to every 5 seconds
        full_time = pd.date_range(tmin, tmax, freq='5s')

        if 'MXPol_data' in locals() and MXPol_data is not None:
            MXPol = safe_reindex(MXPol_data, 'time', full_time,
                                 tolerance='0.5s',  name='MXPol')
            MXPol = safe_reindex(MXPol, 'range', common_range,
                                 tolerance=25, name='MXPol')
        if 'MIRA_data' in locals() and MIRA_data is not None:
            MIRA = safe_reindex(MIRA_data, 'time', full_time,
                                tolerance='2.5s', name='MIRA')
            MIRA = safe_reindex(MIRA, 'range', common_range,
                                tolerance=25, name='MIRA')
        if 'BASTA_data' in locals() and BASTA_data is not None:
            BASTA = safe_reindex(BASTA_data, 'time', full_time,
                                 tolerance='1.5s', name='BASTA')
            BASTA = safe_reindex(BASTA, 'range', common_range,
                                 tolerance=25, name='BASTA')
            # correct BASTA for atmospheric gas attenuation & add DFR-corrected variables (MIRA-BASTA)
            BASTA['reflectivity_attn_corrected'] = xr.DataArray(correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequencies['BASTA'], tmin, tmax).values, dims=BASTA['reflectivity'].dims,
                                                                coords={dim: BASTA[dim] for dim in BASTA['reflectivity'].dims})
            BASTA = DFRcorrection(DFRs, BASTA)
            BASTA['reflectivity_attn_DFR_corrected'] = BASTA['reflectivity_attn_DFR_corrected'] + \
                calibrationvalues['BASTA'][0]

        datasets = {}
        if 'MXPol_data' in locals() and MXPol_data is not None:
            datasets['MXPol'] = MXPol
        if 'MIRA_data' in locals() and MIRA_data is not None:
            datasets['MIRA'] = MIRA
        if 'BASTA_data' in locals() and BASTA_data is not None:
            datasets['BASTA'] = BASTA

        active_datasets = {name: ds for name, ds in zip(
            datasets.keys(), datasets.values())}
        active_sizes = [ds.sizes for ds in active_datasets.values()]

        # check if all datasets have the same sizes
        if not all(size == active_sizes[0] for size in active_sizes):
            print(f"Error in dataset dimensions:{nl}" + nl.join(
                f"{name}: {ds.sizes}" for name, ds in active_datasets.items()))

        # import ERA5 data (2 days to avoid interpolation for the last timestep)
        ERA_data = xr.concat([read_ERA_data(date, dirs['ERA']), read_ERA_data(
            date + pd.Timedelta(days=1), dirs['ERA'])], dim='time').sortby('time')

        # plot_daily_quicklook(datasets, ERA_data, DFRs, output_dir=None, savefig=False)


# %% MXPol add Zdr correction + Kdp (for RHIs and sector scans) to files
if what == 'MXPol_L1':
    warnings.filterwarnings('ignore', message='.*TBB.*')
    os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pattern = re.compile(r"XPOL-(\d{8})-(\d{6})")
    excluded_files = []

    def process_file(f, RHIs, sectors, outdir, year, month, day):
        """Process a single MXPol L0 file -> L1. Returns f if excluded, else None."""
        input_match = pattern.match(os.path.basename(f))
        datetime_str = f"{input_match.group(1)}-{input_match.group(2)}"
        input_time = pd.to_datetime(datetime_str, format='%Y%m%d-%H%M%S')

        if any(start <= input_time <= end for start, end in MXPol_lost_frequencies):
            return None

        is_rhi_or_sector = f in RHIs or f in sectors
        if f in RHIs:
            outname = f"{outdir}/{input_match.group()}_RHI.nc"
        elif f in sectors:
            outname = f"{outdir}/{input_match.group()}_sector.nc"
        else:
            outname = f"{outdir}/{input_match.group()}_Zdr.nc"

        if os.path.exists(outname):
            return None

        try:
            t0 = time.time()
            ds = xr.open_dataset(f)
            ds = fix_encoding(ds)
            ds = Zdr_corr_xr(ds, MXPol_Zdr_corrections)

            if is_rhi_or_sector:
                try:
                    rainflag = True
                    mira = None
                    try:
                        mira = xr.open_dataset(f"{dirs['MIRA']}/{year}{month:02}{day:02}_small.nc")
                    except Exception as e:
                        print(f"no MIRA data for {year}-{month:02}-{day:02}: {e}\nskipping this day...")
                        rainflag = None

                    radar = classify_hm(f, rain_flag=rainflag, tempdata=mira if rainflag else None)
                    ds['Kdp'] = (('time', 'range'), radar.fields['Kdp']['data'].data)
                    ds['hydroclass'] = (('time', 'range'), radar.fields['hydro']['data'].data)
                    for var in vars_hydroclassif:
                        ds[var] = (('time', 'range'), radar.fields[var]['data'].data)
                except Exception as e:
                    print(f"Error while creating L1-file for {os.path.basename(outname)}: {e}")
                    return f

            ds.to_netcdf(outname)
            print(f"   created L1-file for {os.path.basename(outname)} at "
                f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                f"processing time: {time.time() - t0:.2f} s")
            ds.close()
            return None
        except Exception as e:
            print(f"Error while opening {os.path.basename(f)}: {e}")
            return f

    excluded_files = []

    for date in dates:
        year, month, day = date.year, date.month, date.day
        MXPol_filedir = os.path.join(os.path.dirname(dirs['MXPol']), f"L0_new/{year}/{month:02}/{day:02}")

        RHIs = sorted(glob.glob(f"{MXPol_filedir}/*_RHI.nc"))
        sectors = sorted(glob.glob(f"{MXPol_filedir}/*_sector.nc"))
        files = sorted(glob.glob(f"{MXPol_filedir}/*.nc"))

        if not files:
            continue

        print(f"processing {len(files)} files for {year}-{month:02}-{day:02}")
        outdir = f"{dirs['MXPol']}/{year}/{month:02}/{day:02}"
        os.makedirs(outdir, exist_ok=True)

        max_workers = 16
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, f, RHIs, sectors, outdir, year, month, day): f
                for f in files
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    excluded_files.append(result)

        gc.collect()

    with open(f"{dirs['MXPol']}/L1_new/excluded_files.txt", 'w') as f:
        for item in excluded_files:
            f.write(f"{item}\n")


# %% MXPol hydroclassif plots
if what == 'MXPol_hydroclassif':
    for date in dates:
        year, month, day = date.year, date.month, date.day
        MXPol_filedir = os.path.join(dirs['MXPol'], f"{year}/{month:02}/{day:02}")
        files = sorted(glob.glob(f"{MXPol_filedir}/*_RHI.nc")) + sorted(glob.glob(f"{MXPol_filedir}/*_sector.nc"))
        output_dir = f"{os.path.dirname(dirs['MXPol'])}/hydroclassif_plots/{year}/{month:02}/{day:02}"
        os.makedirs(output_dir, exist_ok=True)

        for filename in files:
            print(
                f"Making hydroclassif plot for {os.path.basename(filename)}")
            plot_classification(filename, output_dir, savefig=True)
            gc.collect()

# %% MIRA temperature check (+ comparison with RS data)
if what == 'MIRA_tempcheck':
    rs_dir = os.path.join(base_dir, 'radiosondes')

    def find_dirs(folder, partialname):
        base = Path(folder)
        matching_dirs = [str(d.resolve()) for d in base.iterdir()
                         if d.is_dir and partialname in d.name]

        return matching_dirs

    def read_RStxt(filename):
        """
        function to read radio sounding data and return a dataframe

        Parameters
        ----------
        filename : string
            file path of the radio sounding data (.txt format)

        Returns
        -------
        data_valid : pandas dataframe
            dataframe containing timestamps and numeric data of radio sounding input file

        """
        match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})', filename)
        if match:
            file_date = datetime.strptime(match.group(0), '%Y%m%d%H').date()

        # find # of lines to skip (up to & including 'Profile Data:')
        header_line = 0
        with open(filename, 'r', encoding='latin1') as f:
            for i, line in enumerate(f):
                if "Profile Data:" in line:
                    header_line = i + 1
                    break

        data_full = pd.read_csv(
            filename,
            encoding='latin1',
            sep='\t',
            header=header_line,
            skiprows=[header_line + 1],
            na_values=["-", "NA", "NaN", "null", ""]
        )

        # remove all rows after 'Tropopause:'
        tropopause_mask = data_full.iloc[:, 0].astype(
            str).str.startswith('Tropopause')
        if tropopause_mask.any():
            tropopause_idx = tropopause_mask.idxmax()
            data_full = data_full.loc[:tropopause_idx - 1]

        cols = {'Time': 'time_sec', 'UTC Time': 'time_UTC', 'P': 'P_hPa', 'T': 'T_degC',
                'Hu': 'RH_perc', 'Ws': 'Wsp_mps', 'Wd': 'Wd_deg', 'Lat.': 'lat_deg', 'Long.': 'lon_deg',
                'Geopot': 'elv_masl', 'Dewp.': 'Tdew_degC'}

        data_full.columns = data_full.columns.str.strip()
        data_full = data_full.rename(columns=cols)
        data_full = clean_dataframe(data_full).dropna(axis=1, how='all')

        for col in data_full.columns:
            if col not in ['time_UTC']:  # leave time_UTC for manual handling
                data_full[col] = pd.to_numeric(data_full[col], errors='coerce')

        data_full['time_UTC'] = pd.to_timedelta(
            data_full['time_UTC'].astype(str))
        data_full['time_UTC'] = pd.to_datetime(
            file_date) + data_full['time_UTC']

        return data_full

    def get_rs_vars(rsdata):
        maxidx = rsdata['elv_masl'].idxmax()
        subset = rsdata.iloc[:maxidx + 1].copy()
        subset = subset.dropna(
            subset=['elv_masl', 'T_degC', 'RH_perc', 'P_hPa', 'Wsp_mps', 'Wd_deg'])

        altitude = pd.to_numeric(subset['elv_masl']).values
        temp = subset['T_degC'].astype(float).values
        relhum = subset['RH_perc'].astype(float).values
        press = subset['P_hPa'].astype(float).values
        wspeed = subset['Wsp_mps'].astype(float).values
        wdir = subset['Wd_deg'].astype(float).values

        tmin = subset['time_UTC'].min().strftime('%Y-%m-%d %H:%M')
        tmax = subset['time_UTC'].iloc[-1].strftime('%Y-%m-%d %H:%M')

        return altitude, (temp, relhum, press, wspeed, wdir), (tmin, tmax)

    # more-or-less manual determination of lapse rate
    dates_rs = [datetime(2024, 11, 5), datetime(2024, 12, 5)]
    for date in dates_rs:
        year, month, day = date.year, date.month, date.day
        MIRA_filedir = os.path.join(
            dirs['MIRA'], f"{year}/{month:02}/{day:02}")

        folder = find_dirs(rs_dir, date.strftime('%Y%m%d'))[0]
        fn = glob.glob(f"{folder}/SOUNDING DATA/*.txt")
        rsdata = read_RStxt(fn[0])  # to look up radio sounding data

        MIRA_mmclx = sorted(glob.glob(f"{MIRA_filedir}/*_merged_mmclx.nc"))
        mmclx = xr.open_mfdataset(MIRA_mmclx, combine='by_coords', preprocess=preprocess, chunks={
                                  'time': 3600}, parallel=True)

        rstime = pd.to_datetime(rsdata['time_UTC'])
        rstime_sec = (rstime - rstime[0]).dt.total_seconds()
        t = pd.to_timedelta(
            np.interp(1845, rsdata['elv_masl'], rstime_sec), unit='s') + rstime[0]

        T_rs = np.interp(1845, rsdata['elv_masl'], rsdata['T_degC'])
        w_start = t - pd.Timedelta(minutes=5)
        w_end = t + pd.Timedelta(minutes=5)
        w_avg = mmclx['TEMP'].sel(time=slice(w_start, w_end)).isel(
            range=0).mean(skipna=True).values

        temp = mmclx['TEMP']
        z = mmclx['range']

        temp_grad = temp.differentiate(coord='range')
        mean_grad = temp_grad.mean(dim='time', skipna=True)
        # mean gradient: np.float32(-6.57895) deg/km --> moist (saturated) adiabatic lapse rate

    window = pd.Timedelta(minutes=10)
    profiles = []
    for date in radiosonde_dates:
        pddate = pd.to_datetime(date, format='%Y%m%d%H')
        year, month, day, hour = pddate.year, pddate.month, pddate.day, pddate.hour
        MIRA_filedir = os.path.join(
            dirs['MIRA'], f"{year}/{month:02}/{day:02}")

        folder = find_dirs(rs_dir, date)[0]
        try:
            fn = glob.glob(f"{folder}/SOUNDING DATA/*.txt")
            rsdata = read_RStxt(fn[0])
            idx_max = rsdata["elv_masl"].idxmax()
            rsdata = rsdata.loc[:idx_max]
        except Exception as e:
            print(
                f"radio sonde data for {year}-{month:02}-{day:02} not in correct format: {e}")
            continue
        MIRA_mmclx = sorted(glob.glob(f"{MIRA_filedir}/*_merged_mmclx.nc"))
        try:
            mmclx = xr.open_mfdataset(MIRA_mmclx, combine='by_coords', preprocess=preprocess, chunks={
                                      'time': 3600}, parallel=True)
            mmclx = mmclx.assign_coords(
                altitude=mmclx["range"] + radar_altitude)
        except Exception as e:
            print(
                f"MIRA mmclx data not available for {year}-{month:02}-{day:02}: {e}")
            continue
        if rsdata is not None and mmclx is not None:
            rstime = pd.to_datetime(rsdata['time_UTC'])
            rstime_sec = (rstime - rstime[0]).dt.total_seconds()
            t = pd.to_timedelta(
                np.interp(1845, rsdata['elv_masl'], rstime_sec), unit='s') + rstime[0]
            times.append(t)
            radar_window = mmclx.sel(time=slice(t - window, t + window))
            radar_mean = radar_window['TEMP'].mean(dim='time', skipna=True)
            radar_temp = np.interp(
                rsdata['elv_masl'], radar_mean['altitude'].values, radar_mean.values)
            profiles.append({
                "time": t,
                "altitude": rsdata['elv_masl'].values,
                "T_radiosonde": rsdata['T_degC'].values,
                "T_MIRA": radar_temp,
            })
        else:
            print(f"not all data available for {year}-{month:02}-{day:02}")
            continue

    range_grid = np.arange(0, 12e3, 25)
    sound_times = []
    Tr_all = []
    Tm_all = []
    for prof in profiles:
        z = prof["altitude"]
        Tr = prof["T_radiosonde"]
        Tm = prof["T_MIRA"]

        # interpolate to alt_grid
        Tr_interp = np.interp(range_grid, z, Tr, left=np.nan, right=np.nan)
        Tm_interp = np.interp(range_grid, z, Tm, left=np.nan, rnpight=np.nan)

        sound_times.append(prof["time"])
        Tr_all.append(Tr_interp)
        Tm_all.append(Tm_interp)

    Tr_all = np.array(Tr_all)
    Tm_all = np.array(Tm_all)
    sound_times = np.array(sound_times)
    ds = xr.Dataset(
        {"T_radiosonde": (("sounding", "altitude"), Tr_all),
         "T_MIRA":       (("sounding", "altitude"), Tm_all), },
        coords={
            "sounding": sound_times,
            "altitude": range_grid,
        },)
    ds["T_bias"] = ds["T_MIRA"] - ds["T_radiosonde"]

    # radiosonde vs. MIRA x-axis=date
    plt.figure(figsize=(8, 6))
    plt.scatter(
        x=np.repeat(ds["sounding"].values, len(ds["altitude"])),
        y=np.tile(ds["altitude"].values, len(ds["sounding"])),
        c=ds["T_bias"].values.flatten(),
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=10),
        s=10
    )
    plt.colorbar(label="Bias MIRA - radiosonde [°C]")
    plt.ylabel("Elevation [m a.s.l.]")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylim([1690, 5000])

    # radiosonde vs. MIRA x-axis=RS
    plt.figure(figsize=(8, 6))
    plt.imshow(
        ds["T_bias"].T,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=10),
        aspect="auto",
        origin="lower",
        extent=[0, len(ds["sounding"]), ds["altitude"].min(),
                ds["altitude"].max()]
    )
    plt.colorbar(label="Bias MIRA - radiosonde [°C]")
    plt.ylabel("Elevation [m a.s.l.]")
    plt.xlabel("Radiosonde index")
    plt.ylim([1690, 5000])

    # comparison plot (all T-profiles + bias plot)
    n_soundings = ds.dims['sounding']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # Left panel: All profiles with matching colors
    colors = plt.cm.viridis(np.linspace(0, 1, n_soundings))
    for i in range(n_soundings):
        rs_profile = ds["T_radiosonde"].isel(sounding=i)
        mira_profile = ds["T_MIRA"].isel(sounding=i)

        height_dim = rs_profile.dims[0]
        heights = rs_profile.coords[height_dim].values

        ax1.plot(heights, rs_profile.values,
                 color=colors[i], alpha=0.5, linewidth=1.5, linestyle='-')
        ax1.plot(heights, mira_profile.values,
                 color=colors[i], alpha=0.5, linewidth=1.5, linestyle='--')

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2,
               linestyle='-', label='Radiosonde'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='MIRA'),
    ]
    ax1.legend(handles=legend_elements, fontsize=11, loc='best')
    ax1.set_title('All temperature profiles\n(paired by color)',
                  fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlabel('Height [m a.s.l.]', fontsize=11)
    ax1.set_ylabel('Temperature [°C]', fontsize=11)
    ax1.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)

    # Right panel: Mean difference ± std
    diff = ds["T_MIRA"] - ds["T_radiosonde"]
    mean_diff = diff.mean(dim='sounding')
    std_diff = diff.std(dim='sounding')

    height_dim = mean_diff.dims[0]
    heights = mean_diff.coords[height_dim].values

    # Plot with height on x-axis, difference on y-axis
    ax2.plot(heights, mean_diff.values, color='darkred',
             linewidth=3, label='Mean difference')
    ax2.fill_between(heights,
                     (mean_diff - std_diff).values,
                     (mean_diff + std_diff).values,
                     color='red', alpha=0.2, label='±1 std')

    # Plot individual differences lightly
    for i in range(n_soundings):
        diff_profile = diff.isel(sounding=i)
        ax2.plot(heights, diff_profile.values,
                 color='gray', alpha=0.4, linewidth=0.5)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Temperature difference [°C]\n(MIRA - RS)', fontsize=11)
    ax2.set_xlabel('Height [m a.s.l.]', fontsize=11)
    ax2.set_ylim(-10, 10)
    ax2.set_title('Temperature bias & spread', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)

    plt.suptitle(f'RS-MIRA temperature comparison ({n_soundings} soundings)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{dirs['MIRA']}/temperatures_MIRAvsRS.png",
                dpi=300, bbox_inches='tight', facecolor='w')
    plt.show()


# %% campaign statistics
if what == 'campaign_stats':

    # for date in dates:
    #     datasets = load_zenithdata(date, dirs, DFRs, att_fn, BASTAmode, resample_time=True, ERA=False, peakTree=True)
    #     if 'BASTA' in datasets.keys():
    #         BASTA = datasets['BASTA']
    #     if 'MIRA' in datasets.keys():
    #         MIRA = datasets['MIRA']
    #     if 'MXPol' in datasets.keys():
    #         MXPol = datasets['MXPol']
    #     if 'peaktree' in datasets.keys():
    #         peaktree = datasets['peaktree']
    #     if 'ERA' in datasets.keys():
    #         ERA = datasets['ERA']

    # use the reusable aggregator for campaign-wide statistics for each radar.
    outdir = f"{base_dir}/campaign_stats"
    os.makedirs(outdir, exist_ok=True)

    # radars = list(frequencies.keys())
    radars = ['MIRA', 'BASTA', 'MXPol', 'peaktree']
    results = {}
    try:
        res = aggregate_campaign(
            radars=radars,
            dates=dates,
            dirs=dirs,
            DFRs=DFRs,
            att_fn=att_fn,
            BASTAmode=BASTAmode,
            range_grid=common_range,
            resample_time=False,
            stats=['mean', 'min', 'max', 'std', 'median', 'count'],
            output_dir=outdir,
            peakTree=True,
        )
    except Exception as e:
        print(f"Failed to create statistics file: {e}")

    print("Campaign aggregation finished. Results:")
    for r, out in results.items():
        print(f" {r}: {out}")


# %% weekly peaktree plots
if what == 'weekly_peaktree_plots':
    # cluster, client = initialize_dask_cluster()
    plotdates = ((datetime(2024, 10, 18), datetime(2024, 10, 25)),
                 (datetime(2024, 10, 25), datetime(2024, 11, 1)),
                 (datetime(2024, 11, 1), datetime(2024, 11, 8)),
                 (datetime(2024, 11, 8), datetime(2024, 11, 15)),
                 (datetime(2024, 11, 15), datetime(2024, 11, 23)),
                 (datetime(2024, 11, 23), datetime(2024, 12, 1)),
                 (datetime(2024, 12, 1), datetime(2024, 12, 8)),
                 (datetime(2024, 12, 8), datetime(2024, 12, 15)),
                 (datetime(2024, 12, 15), datetime(2024, 12, 22)),
                 (datetime(2024, 12, 22), datetime(2024, 12, 29)),
                 (datetime(2024, 12, 29), datetime(2025, 1, 5)),
                 (datetime(2025, 1, 5), datetime(2025, 1, 12)),
                 (datetime(2025, 1, 12), datetime(2025, 1, 18)),
                 (datetime(2025, 1, 18), datetime(2025, 1, 23))
                 )

    intp_times = ['10min', '5min', '1min']

    start = pd.Timestamp.now()
    mira = xr.open_mfdataset(sorted(glob.glob(f"{dirs['MIRA']}/*_small.nc")), combine='by_coords', parallel=True,
                             chunks={}, engine='h5netcdf'
                             ).chunk({'time': 120_000, 'range': -1})
    mira_dt = (mira.time.values[1] -
               mira.time.values[0]) / np.timedelta64(1, 's')
    elapsed_mira = pd.Timestamp.now() - start
    minutes, seconds = divmod(elapsed_mira.total_seconds(), 60)
    print(
        f"elapsed time for MIRA opening {int(minutes)} min {int(seconds):02} sec")
    start_basta = pd.Timestamp.now()
    basta = xr.open_mfdataset(sorted(glob.glob(f"{dirs['BASTA']}/BASTA_L1_25m*.nc")), combine='by_coords',
                              chunks={}, preprocess=partial(preprocess_BASTA, filter_vars=True),
                              parallel=True, coords='minimal'
                              ).chunk({'time': 60000, 'range': -1})
    basta = safe_reindex(basta, 'time', mira.time,
                         tolerance='1.5s', name='basta')
    elapsed_basta = pd.Timestamp.now() - start_basta
    minutes, seconds = divmod(elapsed_basta.total_seconds(), 60)
    print(
        f"elapsed time for BASTA opening {int(minutes)} min {int(seconds):02} sec")
    start_peaktree = pd.Timestamp.now()
    peaktree = xr.open_mfdataset(sorted(glob.glob(f"{dirs['peaktree']}/*_daily_*.nc4")), combine='by_coords',
                                 chunks={'time': 'auto', 'range': -1, 'nodes': -1}, engine='h5netcdf')
    peaktree = peaktree.reindex(
        range=mira.range, method='nearest', tolerance=25)
    peaktree = peaktree.reindex(time=mira.time, method='nearest')
    elapsed_peaktree = pd.Timestamp.now() - start_peaktree
    minutes, seconds = divmod(elapsed_peaktree.total_seconds(), 60)
    print(f"elapsed time for peaktree opening {int(minutes)} min {int(seconds):02} sec")

    for intp_time in intp_times:
        start_resampling = pd.Timestamp.now()
        # # using resample:
        # mr = mira.resample(time=intp_time).max(engine='flox')
        # bs = basta.resample(time=intp_time).max(engine='flox')
        # peaktree_peaks = peaktree['no_nodes'].resample(time=intp_time).max(engine='flox')
        # npks = mira.NPKg.resample(time=intp_time).max(engine='flox')
        # pw = mira.RMSg.resample(time=intp_time).max(engine='flox')
        # sk = mira.SKWg.resample(time=intp_time).max(engine='flox')
        # ldr = mira.LDRg.resample(time=intp_time).max(engine='flox')
        # using coarsen:
        t_bins = int(pd.to_timedelta(intp_time).total_seconds() / mira_dt)
        mr = mira.coarsen(time=t_bins, boundary='trim').max()
        bs = basta.coarsen(time=t_bins, boundary='trim').max()
        peaktree_peaks = peaktree['no_nodes'].coarsen(
            time=t_bins, boundary='trim').max()
        npks = mira.NPKg.coarsen(time=t_bins, boundary='trim').max()
        pw = mira.RMSg.coarsen(time=t_bins, boundary='trim').max()
        sk = mira.SKWg.coarsen(time=t_bins, boundary='trim').max()
        ldr = mira.LDRg.coarsen(time=t_bins, boundary='trim').max()
        cth = cloudtop_dask(mira.Z, total_height=500, rres=25)
        ctg = cloudtop_dask(mira.Z, total_height=500, gap_height=200, rres=25)
        # cth = cloudtop_dask(mira.Z, total_height=500, rres=31.1792)
        # ctg = cloudtop_continuous_dask(mira.Z, cth, gap_height=200, rres=31.192)
        cthb = cloudtop_dask(basta.reflectivity, total_height=500, rres=25)
        # cthbg = cloudtop_continuous_dask(basta.reflectivity, cthb, gap_height=200, rres=25)
        elapsed_resampling = pd.Timestamp.now() - start_resampling
        minutes, seconds = divmod(elapsed_resampling.total_seconds(), 60)
        print(
            f"elapsed time for resampling & cloud top calculation {int(minutes)} min {int(seconds):02} sec")

        for start, end in plotdates:
            start_compute = pd.Timestamp.now()
            # print(f"Making plots for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{(end.day-1):02} at {start_compute.strftime('%Y-%m-%d %H:%M:%S')}")
            # ms = mira.sel(time=slice(start, end)).resample(time=intp_time).max().compute()
            # bs = basta.sel(time=slice(start, end)).resample(time=intp_time).max().compute()
            sel_slice = slice(start, end)
            pt = peaktree_peaks.sel(time=sel_slice).chunk(
                {'time': -1}).compute()
            mrpk = npks.sel(time=sel_slice).chunk({'time': -1}).compute()
            mrpw = pw.sel(time=sel_slice).chunk({'time': -1}).compute()
            skw = sk.sel(time=sel_slice).chunk({'time': -1}).compute()
            cths = cth.sel(time=sel_slice).rolling(
                time=6, center=True).mean().compute()  # smooth using simple mean
            # take median for lower/gap-considering cloud top height to avoid outlier bias
            ctgs = ctg.sel(time=sel_slice).rolling(
                time=6, center=True).median().compute()
            ctbs = cthb.sel(time=sel_slice).rolling(
                time=6, center=True).median().compute()
            ldrs = ldr.sel(time=sel_slice).chunk({'time': -1}).compute()
            mrz = mr.sel(time=sel_slice).Z.chunk({'time': -1}).compute()
            mrv = mr.sel(time=sel_slice).VELg.chunk({'time': -1}).compute()
            bsz = bs.sel(time=sel_slice).reflectivity.chunk(
                {'time': -1}).compute()
            bsv = bs.sel(time=sel_slice).velocity.compute()
            elapsed_compute = pd.Timestamp.now() - start_compute
            minutes, seconds = divmod(elapsed_compute.total_seconds(), 60)
            print(
                f"elapsed time for selecting data & compute {int(minutes)} min {int(seconds):02} sec")

            # weekly plot with 5 panels
            start_plot = pd.Timestamp.now()
            fig, axs = plt.subplots(nrows=5, figsize=(20, 20), sharex=True)
            im1 = axs[0].pcolormesh(mrpk.time.values, mrpk.range.values, np.ceil(mrpk.T/2.),
                                    cmap=mcolors.ListedColormap(
                                        ["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"]),
                                    vmin=-0.5, vmax=5.5, shading='auto', rasterized=True)
            cbar1 = plt.colorbar(im1, ax=axs[0], shrink=0.8, pad=0.02)
            cbar1.set_label(pltConfig['no_nodes']['label'])
            axs[0].plot(cth.sel(time=sel_slice).time.values, cth.sel(
                time=sel_slice).values, color='tab:orange', linestyle='--', label='"raw" CTH')
            axs[0].plot(cths.time.values, cths.values, 'tab:brown',
                        label='10-min. mean CTH including multi-layer clouds')
            axs[0].plot(ctg.sel(time=sel_slice).time.values, ctg.sel(time=sel_slice).values,
                        color='mediumblue', linestyle='--', label='"raw" CTH below cloud gaps')
            axs[0].plot(ctgs.time.values, ctgs.values, 'k',
                        label='10-min. median lowest CTH (below cloud gaps)')
            axs[0].plot(cthb.sel(time=sel_slice).time.values, cthb.sel(
                time=sel_slice).values, color='tab:pink', linestyle='--', label='"raw" BASTA CTH')
            axs[0].legend(loc='best')

            im2 = axs[1].pcolormesh(pt.time.values, pt.range.values, np.ceil(pt.T/2.),
                                    cmap=mcolors.ListedColormap(
                                        ["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"]),
                                    vmin=-0.5, vmax=5.5, shading='auto', rasterized=True)
            cbar2 = plt.colorbar(im2, ax=axs[1], shrink=0.8, pad=0.02)
            cbar2.set_label(pltConfig['no_nodes']['label'])
            # axs[1].plot(cth.time.values, cth.values, color='tab:orange', linestyle='--', label='"raw" CTH')
            axs[1].plot(cths.time.values, cths.values, 'tab:brown',
                        label='10-min. mean CTH including multi-layer clouds')
            # axs[1].plot(ctg.time.values, ctg.values, color='mediumblue', linestyle='--', label='"raw" CTH below cloud gaps')
            axs[1].plot(ctgs.time.values, ctgs.values, 'k',
                        label='10-min. median lowest CTH (below cloud gaps)')

            im3 = mrpw.plot.imshow(x='time', vmin=pltConfig['RMSg']['vmin'], vmax=pltConfig['RMSg']['vmax'], cmap=pltConfig['RMSg']['cmap'],
                                   ax=axs[2], add_colorbar=False)
            cbar3 = plt.colorbar(im3, ax=axs[2], shrink=0.8, pad=0.02)
            cbar3.set_label(pltConfig['RMSg']['label'])

            im4 = skw.plot.imshow(x='time', vmin=pltConfig['SKWg']['vmin'], vmax=pltConfig['SKWg']['vmax'], cmap=pltConfig['SKWg']['cmap'],
                                  ax=axs[3], add_colorbar=False)
            cbar4 = plt.colorbar(im4, ax=axs[3], shrink=0.8, pad=0.02)
            cbar4.set_label(pltConfig['SKWg']['label'])

            im5 = ldrs.plot.imshow(x='time', vmin=pltConfig['LDRg']['vmin'], vmax=pltConfig['LDRg']['vmax'], cmap=pltConfig['LDRg']['cmap'],
                                   ax=axs[4], add_colorbar=False)
            cbar5 = plt.colorbar(im5, ax=axs[4], shrink=0.8, pad=0.02)
            cbar5.set_label(pltConfig['LDRg']['label'])
            axs[4].tick_params(axis='x')

            for ax in axs:
                ax.set_xlim(pd.Timestamp(ldrs.time.values[0]).round(
                    'h'), pd.Timestamp(ldrs.time.values[-1]).round('h'))
                ax.xaxis.set_minor_locator(HourLocator(byhour=range(0, 24, 6)))
                ax.yaxis.set_minor_locator(MultipleLocator(1e3))
                ax.grid(True, which='major', axis='both', alpha=0.7,
                        linestyle='-', linewidth=0.8, color='gray')
                ax.grid(True, which='minor', axis='both', alpha=0.2,
                        linestyle='--', linewidth=0.5, color='gray')
                ax.set_ylim([0, 12e3])
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.set_ylim([0, 12e3])
                ax.set_ylabel('height above radar')
                ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
                ax.set_xlabel('')

            fig.suptitle(
                f"MIRA / peaktree # of peaks, spectral width, skewness & LDR for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{(end.day-1):02}", fontweight='bold', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{end.year}{end.month:02}{(end.day-1):02}_{str(intp_time)}.png",
                        dpi=300, bbox_inches='tight', facecolor='w')
            plt.close()
            elapsed_plot1 = pd.Timestamp.now() - start_plot
            minutes, seconds = divmod(elapsed_plot1.total_seconds(), 60)
            print(
                f"elapsed time for plot 1 {int(minutes)} min {int(seconds):02} sec")

            # MIRA-BASTA reflectivity & doppler velocity
            fig, axs = plt.subplots(nrows=4, figsize=(20, 16), sharex=True)
            im1 = axs[0].pcolormesh(mrz.time.values, mrz.range.values, mrz.T.values, cmap=pltConfig['Z']['cmap'], vmin=pltConfig['Z']['vmin'],
                                    vmax=pltConfig['Z']['vmax'], shading='auto', rasterized=True)
            cbar1 = plt.colorbar(im1, ax=axs[0], shrink=0.8, pad=0.02)
            cbar1.set_label(pltConfig['Z']['label'])
            axs[0].plot(cth.sel(time=slice(start, end)).time.values, cth.sel(time=slice(start, end)).values, color='tab:orange',
                        linestyle='--', label='"raw" CTH')
            axs[0].plot(cths.time.values, cths.values, 'tab:brown',
                        label='10-min. mean CTH including multi-layer clouds')
            axs[0].plot(ctg.sel(time=slice(start, end)).time.values, ctg.sel(time=slice(start, end)).values, color='mediumblue',
                        linestyle='--', label='"raw" CTH below cloud gaps')
            axs[0].plot(ctgs.time.values, ctgs.values, 'k',
                        label='10-min. median lowest CTH (below cloud gaps)')
            axs[0].legend(loc='best')

            im2 = axs[1].pcolormesh(mrv.time.values, mrv.range.values, mrv.T.values, cmap=pltConfig['VELg']['cmap'], vmin=pltConfig['VELg']['vmin'],
                                    vmax=pltConfig['VELg']['vmax'], shading='auto', rasterized=True)
            cbar2 = plt.colorbar(im2, ax=axs[1], shrink=0.8, pad=0.02)
            cbar2.set_label(pltConfig['VELg']['label'])

            im3 = axs[2].pcolormesh(bsz.time.values, bsz.range.values, bsz.T.values, cmap=pltConfig['reflectivity']['cmap'],
                                    vmin=pltConfig['reflectivity']['vmin'], vmax=pltConfig['reflectivity']['vmax'], shading='auto')
            axs[2].plot(cthb.sel(time=slice(start, end)).time.values, cthb.sel(time=slice(start, end)).values, color='tab:pink',
                        linestyle='--', label='"raw" BASTA CTH')
            axs[2].plot(ctbs.sel(time=slice(start, end)).time.values, ctbs.sel(time=slice(start, end)).values, color='tab:purple',
                        label='"raw" BASTA CTH')
            cbar3 = plt.colorbar(im3, ax=axs[2], shrink=0.8, pad=0.02)
            cbar3.set_label(pltConfig['reflectivity']['label'])

            im4 = axs[3].pcolormesh(bsv.time.values, bsv.range.values, bsv.T.values, cmap=pltConfig['velocity']['cmap'],
                                    vmin=pltConfig['velocity']['vmin'], vmax=pltConfig['velocity']['vmax'], shading='auto')
            cbar4 = plt.colorbar(im4, ax=axs[3], shrink=0.8, pad=0.02)
            cbar4.set_label(pltConfig['velocity']['label'])

            for ax in axs:
                ax.set_xlim(pd.Timestamp(mrz.time.values[0]).round(
                    'h'), pd.Timestamp(mrz.time.values[-1]).round('h'))
                ax.xaxis.set_minor_locator(HourLocator(byhour=range(0, 24, 6)))
                ax.yaxis.set_minor_locator(MultipleLocator(1e3))
                ax.grid(True, which='major', axis='both', alpha=0.7,
                        linestyle='-', linewidth=0.8, color='gray')
                ax.grid(True, which='minor', axis='both', alpha=0.2,
                        linestyle='--', linewidth=0.5, color='gray')
                ax.set_ylim([0, 12e3])
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.set_ylim([0, 12e3])
                ax.set_ylabel('height above radar')
                ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
                ax.set_xlabel('')

            fig.suptitle(
                f"MIRA & BASTA reflectivity & Doppler velocity for {start.year}-{start.month:02}-{start.day:02} to {(end-pd.Timedelta(days=1)).year}-{(end-pd.Timedelta(days=1)).month:02}-{(end-pd.Timedelta(days=1)).day:02} ({intp_time}-resampling)", fontweight='bold', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{(end-pd.Timedelta(days=1)).year}{(end-pd.Timedelta(days=1)).month:02}{(end-pd.Timedelta(days=1)).day:02}_{intp_time}_Z_VEL.png", dpi=300, bbox_inches='tight', facecolor='w')
            plt.close()
            elapsed_plot2 = pd.Timestamp.now() - start_plot - elapsed_plot1
            minutes, seconds = divmod(elapsed_plot2.total_seconds(), 60)
            print(
                f"elapsed time for plot 2 {int(minutes)} min {int(seconds):02} sec")

            # # individual plots
            # # peaktree plot
            # if pt.time.size != 0 and not pt.isnull().all():
            #     fig, ax = plt.subplots(1, figsize=(24, 6))
            #     im = ax.pcolormesh(pt.time.values, pt.range.values, np.ceil(pt.T/2.), cmap=mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"]), vmin=-0.5, vmax=5.5, shading='auto')
            #     cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            #     cbar.set_label(pltConfig['no_nodes']['label'])
            #     ax.plot(cths.time.values, cths.values, label='10-min. mean CTH including multi-layer clouds')
            #     ax.plot(ctgs.time.values, ctgs.values, 'k', label='10-min. median lowest CTH (below cloud gaps)')
            #     ax.set_xlim(pd.Timestamp(pt.time.values[0]).round('h'), pd.Timestamp(pt.time.values[-1]).round('h'))
            #     ax.set_ylim([0, 12e3])
            #     ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            #     ax.legend(loc='best')
            #     plt.suptitle(f"peakTree peaks + cloud top height for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{(end.day-1):02}", fontweight='bold')
            # #     plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{end.year}{end.month:02}{end.day:02}_peaktree.png", dpi=300, bbox_inches='tight', facecolor='w')
            # #     plt.close()

            # # top plot of multi-panel figure including BASTA cth
            # fig, ax = plt.subplots(1, figsize=(20, 6))
            # im1 = ax.pcolormesh(mr.time.values, mr.range.values, np.ceil(mr.T/2.), cmap=mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"]), vmin=-0.5, vmax=5.5, shading='auto', rasterized=True)
            # cbar1 = plt.colorbar(im1, ax=axs[0], shrink=0.8, pad=0.02)
            # cbar1.set_label(pltConfig['no_nodes']['label'])
            # ax.plot(cth.sel(time=slice(start, end)).time.values, cth.sel(time=slice(start, end)).values, color='tab:orange', linestyle='--', label='"raw" CTH')
            # ax.plot(cths.time.sel(time=slice(start, end)).values, cths.sel(time=slice(start, end)).values, 'tab:brown', label='10-min. mean CTH including multi-layer clouds')
            # ax.plot(ctg.sel(time=slice(start, end)).time.values, ctg.sel(time=slice(start, end)).values, color='mediumblue', linestyle='--', label='"raw" CTH below cloud gaps')
            # ax.plot(ctgs.sel(time=slice(start, end)).time.values, ctgs.sel(time=slice(start, end)).values, 'k', label='10-min. median lowest CTH (below cloud gaps)')
            # ax.plot(cthb.sel(time=slice(start, end)).time.values, cthb.sel(time=slice(start, end)).values, color='tab:pink', linestyle=':', label='"raw" BASTA CTH')
            # ax.legend(loc='best')
            # ax.set_xlim(pd.Timestamp(mr.time.values[0]).round('h'), pd.Timestamp(mr.time.values[-1]).round('h'))

            # # mira peaks plot
            # if mr.time.size !=0 and not mr.isnull().all():
            #     fig, ax = plt.subplots(1, figsize=(24, 6))
            #     im = ax.pcolormesh(mr.time.values, mr.range.values, np.ceil(mr.T/2.), cmap=mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"]), vmin=-0.5, vmax=5.5, shading='auto')
            #     cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            #     cbar.set_label(pltConfig['no_nodes']['label'])
            #     ax.plot(cths.time.values, cths.values, label='10-min. mean CTH including multi-layer clouds')
            #     ax.plot(ctgs.time.values, ctgs.values, 'k', label='10-min. median lowest CTH (below cloud gaps)')
            #     ax.set_xlim(pd.Timestamp(mr.time.values[0]).round('h'), pd.Timestamp(mr.time.values[-1]).round('h'))
            #     ax.set_ylim([0, 12e3])
            #     ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            #     ax.legend(loc='best')
            #     plt.suptitle(f"MIRA peaks + cloud top height for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{end.day:02}", fontweight='bold')
            #     plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{end.year}{end.month:02}{end.day:02}_mirapeaks.png", dpi=300, bbox_inches='tight', facecolor='w')
            #     plt.close()

            # # spectral width plot
            # if mrpw.time.size !=0 and not mrpw.isnull().all():
            #     fig, ax = plt.subplots(1, figsize=(24, 6))
            #     im = mrpw.plot.imshow(x='time', vmin=0, vmax=2, cmap='Spectral')
            #     ax.set_xlim(pd.Timestamp(mrpw.time.values[0]).round('h'), pd.Timestamp(mrpw.time.values[-1]).round('h'))
            #     ax.set_ylim([0, 12e3])
            #     ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            #     ax.legend(loc='best')
            #     plt.suptitle(f"MIRA spectral width for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{end.day:02}", fontweight='bold')
            #     plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{end.year}{end.month:02}{end.day:02}_spectralwidth.png", dpi=300, bbox_inches='tight', facecolor='w')
            #     plt.close()

            # # skewness plot
            # if skw.time.size !=0 and not skw.isnull().all():
            #     fig, ax = plt.subplots(1, figsize=(24, 6))
            #     im = skw.plot.imshow(x='time', vmin=0, vmax=1, cmap='Spectral_r')
            #     ax.set_xlim(pd.Timestamp(skw.time.values[0]).round('h'), pd.Timestamp(skw.time.values[-1]).round('h'))
            #     ax.set_ylim([0, 12e3])
            #     ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            #     ax.legend(loc='best')
            #     plt.suptitle(f"MIRA skewness for {start.year}-{start.month:02}-{start.day:02} to {end.year}-{end.month:02}-{end.day:02}", fontweight='bold')
            #     plt.savefig(f"{base_dir}/summaryQLs/weekly_peaks_cth/{start.year}{start.month:02}{start.day:02}-{end.year}{end.month:02}{end.day:02}_skewness.png", dpi=300, bbox_inches='tight', facecolor='w')
            #     plt.close()

    # cloudtop = mira['cloud_top_height']
    # cloudtop_min = cloudtop.min().compute()
    # cloudtop_max = cloudtop.max().compute()
    # cloudtop_mean = cloudtop.mean().compute()

    # times = pd.to_datetime(mira['time'].values)
    # start_date = times.min().normalize()
    # end_date = times.max().normalize()
    # all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # days = [1, 10, 20]
    # major_ticks = pd.to_datetime([
    #     pd.Timestamp(year=dt.year, month=dt.month, day=d)
    #     for dt in pd.date_range(startdate, end_date, freq='MS')  # MS = Month Start
    #     for d in days
    #     if pd.Timestamp(year=dt.year, month=dt.month, day=1).replace(day=d) <= end_date
    # ])
    # add_dates = pd.to_datetime(['2025-01-25', '2024-10-18', '2024-10-20'])
    # major_ticks = major_ticks.union(add_dates).sort_values()

    # daily_stats, monthly_stats = daily_monthly_stats(cloudtop)
    # daily_dates = pd.to_datetime(daily_stats['time'].values)
    # monthly_dates = pd.to_datetime(monthly_stats['time'].values)

    # # no_peaks = mira['NPKg'].median(dim='range')
    # no_peaks_mira_max = np.ceil(mira['NPKg'].max(dim='range')/2.)
    # no_peaks_mira_max_hourly = no_peaks_mira_max.resample(time='24H').max()
    # no_peaks_pt_max = np.ceil(peaktree['no_nodes'].max(dim='range')/2.)
    # no_peaks_pt_max_hourly = no_peaks_pt_max.resample(time='24H').max()

    # # plot cloud top height + number of peaks (campaign-long plot comparing MIRA + peaktree # of peaks)
    # fig, ax = plt.subplots(figsize=(25, 6))
    # ax.plot(mira.time.values, mira['cloud_top_height'].values, '.', label='all')
    # ax.plot(daily_dates + pd.Timedelta(hours=12), daily_stats.sel(statistic='max').values, '.r', label='daily max')
    # ax.plot(daily_dates + pd.Timedelta(hours=12), daily_stats.sel(statistic='mean').values, '.c', label='daily mean')

    # for d in all_dates:
    #     night_start = d + pd.Timedelta(hours=18)
    #     night_end = d + pd.Timedelta(hours=30)
    #     ax.axvspan(night_start, night_end, color='grey', alpha=0.2)

    # ax.set_xlim([pd.to_datetime('2024-10-18'), pd.to_datetime('2025-01-25')])
    # ax.set_ylim([0, 12e3])
    # plt.xticks(rotation=45)
    # ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    # ax.legend(loc='upper left')
    # ax.grid(True)
    # ax2 = ax.twinx()  # Create secondary x-axis
    # ax2.plot(no_peaks_mira_max_hourly.time.values, no_peaks_mira_max_hourly.values, 'g*', label='MIRA peaks')
    # ax2.plot(no_peaks_pt_max_hourly.time.values, no_peaks_pt_max_hourly.values, 'k*', label='peaktree peaks')
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_ylim([0, 18])
    # ax2.legend(loc='upper right')
    # ax2.set_yticks(np.arange(0, 18, 3))
    # ax.set_xticks(major_ticks)

    # plt.suptitle('Cloud top height from MIRA data', fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(f"{MIRA_dir}/cloudtopheight.png", dpi=150, bbox_inches='tight', facecolor='w')

    # mask_below = mira['range'] < mira['cloud_top_height']
    # Z = mira['Z']#.chunk({'time': 20000})
    # cloudtop_idx = np.searchsorted(mira['range'].values, cloudtop.values)
    # cloudtop_idx = np.clip(cloudtop_idx, 0, len(mira['range']) - 1)

    # def Z_below_cloudtop(Z: xr.DataArray, cloudtop_idx: xr.DataArray, heights: xr.DataArray):
    #     range_idx = xr.DataArray(
    #         np.arange(Z.sizes["range"]),
    #         dims="range",
    #         coords={"range": Z["range"]}
    #     )
    #     mask = range_idx < cloudtop_idx

    #     Z_masked = Z.where(mask)

    #     min_ = Z_masked.min(dim="range")
    #     max_ = Z_masked.max(dim="range")
    #     mean_ = Z_masked.mean(dim="range")

    #     # NaN-safe max_height
    #     valid = Z_masked.notnull().any(dim="range")
    #     max_idx = Z_masked.fillna(-np.inf).argmax(dim="range")
    #     max_height = heights[max_idx].where(valid)

    #     return {
    #         "Z_below_cloudtop_min": min_,
    #         "Z_below_cloudtop_max": max_,
    #         "Z_below_cloudtop_mean": mean_,
    #         "Z_below_cloudtop_max_height": max_height,
    #     }

    # results = Z_below_cloudtop(mira['Z'], xr.DataArray(cloudtop_idx, dims=['time'], coords={'time': mira['time']}), mira['range'])
    # mira = mira.assign(results)

    # # mira['Z'].where(mask_below).plot.imshow(x='time', vmin=-40, vmax=30)

    # valid_mask = ~(np.isnan(mira['Z_below_cloudtop_max'].values) | np.isnan(mira['cloud_top_height'].values))
    # z_max = mira['Z_below_cloudtop_max'].values[valid_mask]
    # cth = mira['cloud_top_height'].values[valid_mask]

    # daily_Zmax, monthly_Zmax = daily_monthly_stats(mira['Z_below_cloudtop_max'])

    # # hexbin density plot (all data)
    # plt.figure(figsize=(10, 6))
    # hb = plt.hexbin(z_max, cth, gridsize=50, cmap='Blues', mincnt=100)
    # plt.xlabel('max. reflectivity below cloud top (dBZ)')
    # plt.ylabel('cloud top height (m)')
    # plt.title('Cloud top height vs. max. reflectivity')
    # plt.colorbar(hb, label='Count')
    # plt.show()

    # # shows that values below 0 dBz and 2k should maybe be masked?
    # # density_mask = (mira['Z_below_cloudtop_max'] > 0) & (mira['cloud_top_height'] > 2000)
    # density_mask = mira['cloud_top_height'] > 2000
    # z_max_filt = mira['Z_below_cloudtop_max'].where(density_mask & valid_mask)
    # cth_filt = mira['cloud_top_height'].where(density_mask & valid_mask)
    # plt.figure(figsize=(10, 6))
    # hb = plt.hexbin(z_max_filt.values, cth_filt.values, gridsize=50, cmap='Blues', mincnt=100)
    # plt.xlabel('max. reflectivity below cloud top (dBZ)')
    # plt.ylabel('cloud top height (m)')
    # plt.title('Cloud top height vs. max. reflectivity (filtered)')
    # plt.colorbar(hb, label='Count')
    # plt.show()

    # # scatter plot of daily maxima
    # plt.figure(figsize=(10,6))
    # plt.scatter(mira['Z_below_cloudtop_max'].resample(time='1D').max(), daily_stats.sel(statistic='max').values)
    # plt.xlabel('max. reflectivity below bloud top (dBZ)')
    # plt.ylabel('cloud top height (m)')
    # plt.title('Cloud top height vs. max. reflectivity (daily maxima)')

    # # hexbin plot of number of peaks vs. cloud top height
    # plt.figure(figsize=(10, 6))
    # hb = plt.hexbin(mira['NPKg'].reduce(np.nanmax, dim='range'), mira['cloud_top_height'], gridsize=50, cmap='Blues', mincnt=100)
    # plt.xlabel('number of peaks')
    # plt.xlim(0, 8)
    # plt.ylabel('cloud top height (m)')
    # plt.title('Cloud top height vs. number of peaks')
    # plt.colorbar(hb, label='Count')
    # plt.show()

    # df = pd.DataFrame(
    #     {'num_peaks': mira['NPKg'].reduce(np.nanmax, dim='range'),
    #      'cloud_top_height': cloudtop})

    # sns.violinplot(data=df, x='num_peaks', y='cloud_top_height')


# %% atmospheric gas attenuation statistics

att = xr.open_dataset(att_fn)
att_RS = xr.open_dataset(f"{os.path.dirname(dirs['ERA'])}/RS_attenuation.nc")

elevation = 15e3
pia_15km = att['PIA'].sel(elevation=elevation, method='nearest')

freqs = att['frequency'].values

for freq in freqs:
    pia_freq = pia_15km.sel(frequency=freq)

    min_val = pia_freq.min().values
    max_val = pia_freq.max().values

    print(f"Frequency {freq} GHz:")
    print(f"  Min PIA at 15 km: {min_val:.3f}")
    print(f"  Max PIA at 15 km: {max_val:.3f}")

rs_dates = pd.to_datetime(radiosonde_dates, format='%Y%m%d%H')
pia_stacked = att['PIA'].sel(elevation=slice(0, 15e3), time=rs_dates).stack(samples=('time', 'elevation'))

# PAMTRA-calculated two-way PIA for ERA5 data during radio soundings
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
for ax, f in zip(axes, freqs):
    values = pia_stacked.sel(frequency=f).values
    values = values[~np.isnan(values)]

    ax.hist(values, bins=50, color='blue')
    ax.set_xlabel(f"{f} GHz", fontweight='bold')
axes[0].set_ylabel("# of observations")
plt.tight_layout()
plt.suptitle("Two-way attenuation [dB] up to 15 km using ERA5 data during radio soundings", fontweight='bold')  
plt.show()
plt.savefig(f"{os.path.dirname(att_fn)}/att_histograms_ERA_duringRS.png", dpi=150, bbox_inches='tight', facecolor='w')

# PAMTRA-calculated two-way PIA during radio soundings
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
for ax, f in zip(axes, att_RS.frequency.values):
    data = att_RS['PIA'].sel(frequency=f).values
    data = data[~np.isnan(data)]

    ax.hist(data, bins=50, color='blue')
    ax.set_xlabel(f"{f} GHz")
axes[0].set_ylabel("# of observations")
plt.tight_layout()
plt.suptitle("Two-way attenuation [dB] during radio soundings", fontweight='bold')  
plt.show()
plt.suptitle("Two-way attenuation [dB] using radio sounding data", fontweight='bold')
plt.savefig(f"{os.path.dirname(att_fn)}/att_histograms_RS.png", dpi=150, bbox_inches='tight', facecolor='w')

# PAMTRA-calculated two-way PIA for full campaign
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, f in zip(axes, freqs):
    values = att['PIA'].sel(frequency=f).values
    values = values[~np.isnan(values)]

    ax.hist(values, bins=50, color='blue')
    ax.set_xlabel(f"{f} GHz", fontweight='bold')
axes[0].set_ylabel("# of observations")
plt.tight_layout()
plt.suptitle("Two-way attenuation [dB] up to 15 km using ERA5 data for full campaign", fontweight='bold')  
plt.show()
plt.savefig(f"{os.path.dirname(att_fn)}/att_histograms_ERA_fullcampaign.png", dpi=150, bbox_inches='tight', facecolor='w')

# plot combining ERA5 and RS data
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
for ax, f in zip(axes, freqs):
    RSvalues = att_RS['PIA'].where(att_RS['elevation'] <= 15e3).sel(frequency=f).values
    RSvalues = RSvalues[~np.isnan(RSvalues)]
    ERAvalues = pia_stacked.sel(frequency=f).values
    ERAvalues = ERAvalues[~np.isnan(ERAvalues)]
    ERA_all = att['PIA'].sel(elevation=slice(0, 15e3)).stack(samples=('time', 'elevation')).sel(frequency=f).values
    ERA_all = ERA_all[~np.isnan(ERA_all)]
    
    weights_RS = np.ones_like(RSvalues) / len(RSvalues)
    weights_ERA = np.ones_like(ERAvalues) / len(ERAvalues)
    weights_ERA_all = np.ones_like(ERA_all) / len(ERA_all)

    ax.hist(RSvalues, bins=50, weights=weights_RS, color='tab:blue', label='radiosonde data')
    ax.hist(ERAvalues, bins=50, weights=weights_ERA, histtype='step', color='tab:orange', linewidth=2, label='ERA5 data (during RS only)')
    ax.hist(ERA_all, bins=50, weights=weights_ERA_all, histtype='step', color='tab:green', linewidth=2, label='ERA5 data (all)')
    
    if f == freqs[-1]:
        ax.legend(loc='best')
    ax.set_xlabel(f"{f} GHz", fontweight='bold')
    ax.set_ylim([0, 0.075])

axes[0].set_ylabel("Fraction of observations")
plt.tight_layout()
plt.suptitle("Two-way attenuation [dB] up to 15 km using radiosonde and ERA5 data", fontweight='bold')
plt.savefig(f"{os.path.dirname(att_fn)}/att_histograms_combined.png", dpi=150, bbox_inches='tight', facecolor='w')


#%% temperature plots
def sort_lat(ds):
    return ds.sortby('latitude')

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

ERA_files = sorted(f for f in glob.glob(f"{dirs['ERA']}/*.nc"))
ERA_files = [f for f in ERA_files if 'large' not in f]
ERA_data = xr.open_mfdataset(ERA_files, combine='by_coords', preprocess=sort_lat)
altitudes = calc_alt(ERA_data['z'].mean(dim=['time', 'latitude', 'longitude']))
ERA_data = ERA_data.assign_coords(altitude=altitudes)
ERA_alt = ERA_data.swap_dims({'isobaricInhPa': 'altitude'})

MIRA_files = sorted(glob.glob(f"{dirs['MIRA']}/*_small.nc"))
MIRA_data = xr.open_mfdataset(MIRA_files, combine='by_coords')
MIRA_data['TEMP'] = MIRA_data['TEMP'].chunk({'time': 720, 'range': 50})

T_ERA = ERA_alt['t'].sel(altitude=radar_altitude, latitude=latlon_Helmos[0], longitude=latlon_Helmos[1], method='nearest') - 273.15
ERA_altval = ERA_alt['altitude'].sel(altitude=radar_altitude, method='nearest').values
# corrected 
if ERA_altval > radar_altitude:
    T_ERA_corr = T_ERA - (MIRA_lapserate * (ERA_altval - radar_altitude))
else:
    T_ERA_corr = T_ERA - (MIRA_lapserate * (radar_altitude - ERA_altval))

T_MIRA_corr = MIRA_data['TEMP'].isel(range=0) - (MIRA_lapserate * MIRA_data['range'].values[0])
T_MIRA_fill = (T_MIRA_corr.sel(time=slice(datetime(2024, 11, 8, 14), None))
               .chunk({'time': -1}).interpolate_na(dim='time', method='linear'))

fig, ax = plt.subplots(figsize=(20,5))
ax.plot(T_ERA.time.values, T_ERA.values, linewidth=2, color='tab:red', label=f"ERA5 ({ERA_altval:.1f} m)")
# ax.plot(T_ERA_corr.time.values, T_ERA_corr.values, linewidth=2, color='tab:red', alpha=0.7, label='ERA5 (matched to radar altitude)')
ax.plot(T_MIRA_fill.time.values, T_MIRA_fill.values, linewidth=1.5, color='tab:blue', label=f"MIRA ({radar_altitude:.1f} m)", rasterized=True)
ax.set_xticks(major_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.set_ylim([-8, 13])
ax.set_ylabel('temperature [°C]')
ax.axhline(y=0, linewidth=1, linestyle='-', color='k')
fig.suptitle('Temperature at the radar site throughout the campaign', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig(f"{os.path.dirname(dirs['ERA'])}/temperature_ERA5_MIRA.png", dpi=150, bbox_inches='tight', facecolor='w')

#%% first attempt for DFR spectrograms

year, month, day = date.year, date.month, date.day
file_dir_mira = f"{dirs['MIRA']}/{year}/{month:02}/{day:02}"
file_dir_mxpol = f"{dirs['MXPol']}/{year}/{month:02}/{day:02}"
flist_mira = sorted(glob.glob(f"{file_dir_mira}/*_merged.nc"))
flist_mxpol = sorted(glob.glob(f"{file_dir_mxpol}/*Zdr.nc"))
mira = prepare_MIRAdata(flist_mira)
mxpol = prepare_MXPoldata(flist_mxpol)
mxpol = mxpol.chunk({'time': 500})

tsel = datetime(2024, 12, 5, 12, 18)
rsel = 3000

validtimes_mask_mira = np.isfinite(mira['sZco']).any(dim=('range', 'doppler'))
validtimes_indices_mira = np.where(validtimes_mask_mira)[0]

validtimes_mask_mxpol = np.isfinite(mxpol['sZH']).any(dim=('range', 'nfft'))
validtimes_indices_mxpol = np.where(validtimes_mask_mxpol)[0]

spec_mira_time = mira.isel(time=validtimes_indices_mira).sel(
    time=tsel, method='nearest')['sZco']
validranges_mask_mira = np.isfinite(spec_mira_time).any(dim='doppler')
validranges_indices_mira = np.where(validranges_mask_mira)[0]
range_mira_values = spec_mira_time['range'].values[validranges_indices_mira]
rsel_mira_valid = range_mira_values[np.argmin(
    np.abs(range_mira_values - rsel))]

spec_mxpol_time = mxpol.isel(time=validtimes_indices_mxpol).sel(
    time=tsel, method='nearest')['sZH']
validranges_mxpol = np.isfinite(spec_mxpol_time).any(dim='nfft')
validranges_indices_mxpol = np.where(validranges_mxpol)[0]
range_mxpol_values = spec_mxpol_time['range'].values[validranges_indices_mxpol]
rsel_mxpol_valid = range_mxpol_values[np.argmin(
    np.abs(range_mxpol_values - rsel))]

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(-1*mira['doppler'].values, spec_mira_time.isel(
    range=validranges_indices_mira).sel(range=rsel_mira_valid).values, label='MIRA')
plt.plot(mxpol['sVel'].values, mxpol.isel(range=validranges_indices_mxpol).sel(
    range=rsel_mxpol_valid).values.T, label='MXPol')
ax.set_xlim([-10, 5])
ax.set_ylim([-60, 0])
ax.set_xlabel('Doppler velocity [m/s]')
ax.set_ylabel('Spectral Ze [dBsZ]')
ax.axvline(x=0, linestyle='--', color='k', alpha=0.5)
ax.grid(True, which='both')
fig.suptitle(
    f"Spectrograms for MIRA & MXPol {tsel.strftime('%Y-%m-%d %H:%M:%S')} ({rsel:.0f} m above radar)", fontweight='bold')
plt.tight_layout()
plt.legend(loc='best')
