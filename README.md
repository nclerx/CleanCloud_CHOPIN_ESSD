# CleanCloud_CHOPIN_ESSD

Code accompanying the data paper: *Multi-frequency radar measurements of clouds and precipitation collected during the 2024-2025 CHOPIN campaign at Mt. Helmos*, submitted to *Earth System Science Data (ESSD)*.

Repository: https://github.com/nclerx/CleanCloud_CHOPIN_ESSD

## Overview

This repository contains the processing pipeline used to generate the multi-frequency cloud radar dataset from the CHOPIN campaign (CLEANCLOUD project), conducted at Mount Helmos, Greece, in 2024. The pipeline processes and combines data from three radar systems operating at different frequencies:

- **MXPol** (X-band, 9.4 GHz)
- **MIRA** (Ka-band, 35.2 GHz)
- **BASTA** (W-band, 94.95 GHz)

The output is, as well as , a harmonised dataset (stored in .zarr-files) both in the native radar data resolution, as well as resampled onto a common spatiotemporal grid for multi-frequency cloud and precipitation retrievals.

## Repository structure

```
.
├── scripts/
│   ├── MXPol_L1_processing/
│   │   ├── MXPol_processing_L0.py                         # MXPol L0 processing
│   │   └── Zdr_calibration/
│   │       ├── 00_cfradial_netcdf_to_vaex_archive.py      # convert input Zdr-data (NetCDF) to vaex archive
│   │       ├── 01_compute_filtered_dataframe.ipynb        # filter data for calibration
│   │       ├── 02_compute_kriging.ipynb                   # kriging interpolation for calibration
│   │       └── 03_correction_for_all_scans.ipynb          # apply Zdr correction across all scans
│   ├── allradars_dataavailability.py                      # data availability overview across radars
│   ├── analyze_zarr.py                                    # analysis of campaign-wide data (stored in zarr-files)
│   ├── cloudtopDFR_BASTA-MIRA.py                          # determining cloudtop DFR (MIRA-BASTA) to estimate W-band attenuation
│   ├── ERA_plots.py                                       # plots of ERA5 reanalysis data
│   ├── eumetsat_download.py                               # download EUMETSAT satellite data
│   ├── MIRA_auxiliary.py                                  # MIRA auxiliary tools for data processing
│   ├── MIRA_dailyfiles.py                                 # create daily MIRA-files
│   ├── MIRA_mergezeniths.py                               # merge indidivual (minute-long) files of MIRA zenith scans into hourly .nc-files
│   ├── MXPol_filecorrection.py                            # MXPol file correction utilities
│   ├── MXPol_hydroclassif.py                              # MXPol hydrometeor classification
│   ├── pamtra_clearsky_radiosounding.py                   # PAMTRA - clear-sky vs. radiosounding calculations -- using PAMTRA-specific python environment
│   ├── process_plot_eventidentification.py                # various: MXPol L1 processing, create dailysummary quicklooks, generate spectrograms, 
│   ├── radar_calibration.py                               # radar intercalibration
│   └── zarr_fullcampaign.py                               # build full-campaign zarr archive
├── src/
│   ├── aggregator.py                                       # data aggregation functions
│   ├── constants_input.py                                  # constants and input parameters
│   ├── plotting.py                                         # shared plotting functions
│   ├── radar_processing.py                                 # core radar processing functions
│   ├── utils.py                                            # general utility functions
│   ├── zarr_utils.py                                       # zarr I/O helper functions
│   └── __init__.py
├── requirements.txt
└── README.md
```


## Installation

Set up the environment using conda/micromamba:

```bash
micromamba create -n chopin_processing -f environment.yml
micromamba activate chopin_processing
```

or with pip:

```bash
pip install -r requirements.txt
```

**Note:** `pyjacopo` is a separate module maintained in its own repository and is not listed in `requirements.txt`. Install it directly from: https://github.com/ltelab/pyjacopo

**PAMTRA:** Radiative transfer calculations use [PAMTRA](https://pamtra.readthedocs.io/), which requires a separate build/installation and is run in a dedicated environment, kept isolated from the main processing pipeline. See PAMTRA's own installation instructions.


## Usage

Most scripts were run interactively (e.g. in VS Code or a Jupyter environment) during development, but all `.py` scripts are also runnable directly from the command line, e.g.:

```bash
python scripts/MXPol_L1_processing/MXPol_processing_L0.py
```

The notebooks under `scripts/MXPol_L1_processing/Zdr_calibration/` (`01_compute_filtered_dataframe.ipynb`, `02_compute_kriging.ipynb`, `03_correction_for_all_scans.ipynb`) are intended to be run in sequence, following the numbering in the filenames.

There is no single pipeline entry point — scripts under `scripts/` correspond to distinct processing steps (per-radar L0/L1 processing, calibration, gridding, attenuation correction, campaign-wide archiving) that were run individually as part of generating the dataset. Shared functionality used across these scripts is in `src/`.


## Data

The processed dataset and other resulting files associated with this code is archived on Zenodo: https://doi.org/10.5281/zenodo.21245825.


## Key processing steps

- MXPol L1 data processing (Zdr-correction, hydrometeor classification)
- BASTA artefact masking (reflectivity- and Doppler velocity-based, strict/loose modes)
- MIRA calibration correction (LNA offset correction, see paper Section 3.1.2)
- Spatio-temporal regridding to a common 30-second / 25 m grid (linear interpolation for continuous variables, majority-vote mode for boolean flags)
- Atmospheric gas attenuation correction (BASTA data)
- Radar calibration
- BASTA attenuation correction using Ka-W cloudtop DFRs


## Citation

If you use this code or the associated dataset, please cite:

> Nicole Clerx, Romanos Foskinis, Morgane Weiss, Julien Delanoë, Athanasios Nenes, and Alexis Berne (2026). Multi-frequency radar measurements of clouds and precipitation collected during the 2024-2025 CHOPIN campaign at Mt. Helmos. *Earth System Science Data*. [DOI: to come](https://doi.org/)


## License

MIT


## Contact

Nicole Clerx — Environmental Remote Sensing Laboratory (LTE), EPFL — nicole.clerx@epfl.ch