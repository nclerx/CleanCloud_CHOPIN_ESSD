# CleanCloud_CHOPIN_ESSD

Code accompanying the data paper: *Multi-frequency radar measurements of clouds and precipitation collected during the 2024-2025 CHOPIN campaign at Mt. Helmos*, submitted to *Earth System Science Data (ESSD)*.

Repository: https://github.com/nclerx/CleanCloud_CHOPIN_ESSD

## Overview

This repository contains the processing pipeline used to generate the multi-frequency cloud radar dataset from the CHOPIN campaign (CLEANCLOUD project), conducted at Mt. Helmos, Greece, from October 2024 to January 2025. The pipeline processes and combines data from three radar systems operating at different frequencies:

- **MXPol** (X-band, 9.4 GHz)
- **MIRA** (Ka-band, 35.2 GHz)
- **BASTA** (W-band, 94.95 GHz)

The output is a harmonised dataset (stored in .zarr-files) both in the native radar resolution and resampled onto a common spatiotemporal grid for multi-frequency cloud and precipitation retrievals.

The file `howto.ipynb` contains a step-by-step example of how to read the various radar data files, how to inspect the metadata, and how to make basic plots.

## Repository structure

The code is organised in two main directories: `scripts` and `src`. The main data processing code is located in `scripts`, whereas `src` contains supporting modules (helper functions, frequently used constants, user input settings, etc.) that are shared across the scripts.

The main folder contains the files defining the (virtual) Python environments required to run the code (`environment*.yml` and `requirements*.txt`), as well as documentation on how to load and use the data (`howto.ipynb` and two `.md`-files describing the file format/variable naming etc.).

```
.
├── src/
│   ├── aggregator.py                                      # data aggregation functions
│   ├── constants_input.py                                 # constants and input parameters
│   ├── plotting.py                                        # shared plotting functions
│   ├── radar_processing.py                                # core radar processing functions
│   ├── utils.py                                           # general utility functions
│   ├── zarr_utils.py                                      # zarr I/O helper functions
│   └── __init__.py
├── scripts/
│   ├── ERA5/
│   │   ├── ERA5_download.py
│   │   ├── ERA5_grib2netCDF.py
│   │   └── ERA5_plots.py
│   ├── MXPol_L0_processing/
│   │   ├── MXPol_processing_L0.py                         # MXPol L0 processing
│   │   └── Zdr_calibration/
│   │       ├── 00_cfradial_netcdf_to_vaex_archive.py      # convert input Zdr-data (NetCDF) to vaex archive
│   │       ├── 01_compute_filtered_dataframe.ipynb        # filter data for calibration
│   │       ├── 02_compute_kriging.ipynb                   # kriging interpolation for calibration
│   │       └── 03_correction_for_all_scans.ipynb          # apply Zdr correction across all scans
│   ├── allradars_dataavailability.py                      # data availability overview across radars
│   ├── analyze_zarr.py                                    # analysis of campaign-wide data (stored in zarr-files)
│   ├── cloudtopDFR_BASTA-MIRA.py                          # determining cloudtop DFR (MIRA-BASTA) to estimate W-band attenuation
│   ├── eumetsat_download.py                               # download EUMETSAT satellite data
│   ├── MIRA_auxiliary.py                                  # MIRA auxiliary tools for data processing
│   ├── MIRA_dailyfiles.py                                 # create daily MIRA-files
│   ├── MIRA_mergezeniths.py                               # merge individual (minute-long) files of MIRA zenith scans into hourly .nc-files
│   ├── MXPol_filecorrection.py                            # MXPol file correction utilities
│   ├── MXPol_hydroclassif.py                              # MXPol hydrometeor classification
│   ├── pamtra_clearsky_radiosounding.py                   # PAMTRA - clear-sky vs. radiosounding calculations (using PAMTRA-specific python environment)
│   ├── process_plot_eventidentification.py                # various: MXPol L1 processing, create dailysummary quicklooks, generate spectrograms, etc.
│   ├── radar_calibration.py                               # radar intercalibration
│   └── zarr_fullcampaign.py                               # build full-campaign zarr archive
├── environment.yml                                        # conda/micromamba environment for the main processing (recommended)
├── requirements.txt                                       # pip alternative for the main processing (Python packages only)
├── environment_pamtra.yml                                 # conda/micromamba environment for PAMTRA (`pamtra_clearsky_radiosounding.py`)
├── requirements_pamtra.txt                                # pip alternative for PAMTRA (Python packages only)
├── howto.ipynb                                            # Jupyter notebook with a step-by-step guide for loading/plotting the various data
├── data_ActrisCloudnet_netcdf.md                          # file describing the various variables/attributes for the various .nc files on Cloudnet
├── data_Zenodo_zarr.md                                    # file describing the various variables/attributes for the .zarr files in the Zenodo repository
├── pyproject.toml                                         # package definition (enables `pip install -e .`)
├── LICENSE                                                # MIT license
└── README.md
```


## Installation

Two separate environments are used: one for the main processing pipeline and
one for the PAMTRA forward simulations.

### Main processing environment

Recommended (conda/micromamba):
```bash
micromamba create -n chopin_processing -f environment.yml
micromamba activate chopin_processing
```
Then install the repository itself (makes the shared modules in `src/` importable from all scripts):
```bash
pip install -e .
```

Alternatively, with pip:
```bash
pip install -r requirements.txt
```

In both cases, then install the repository itself (makes the shared modules in `src/` importable from all scripts):
```bash
   pip install -e .
```

`requirements.txt` only lists Python packages; system libraries (e.g. PROJ, netCDF/HDF5, eccodes) are not included and must be available on your system. The conda/micromamba route handles these automatically.

**pyjacopo:** `pyjacopo` is a separate module maintained in its own repository and is not included in `environment.yml` or `requirements.txt`. Install it into the active (main processing) environment following the instructions at https://github.com/ltelab/pyjacopo.


### PAMTRA environment (optional)

Only needed for `pamtra_clearsky_radiosounding.py`.

```bash
micromamba create -n chopin_processing_pamtra -f environment_pamtra.yml
micromamba activate chopin_processing_pamtra
```

or with pip: `pip install -r requirements_pamtra.txt`

**PAMTRA:** `PAMTRA` (https://pamtra.readthedocs.io/) itself requires a separate build/installation and is kept in this dedicated environment, isolated from the main processing pipeline. Follow PAMTRA's own installation instructions, then install it into this environment.


## Usage

### Configuration

Data paths and processing settings are defined in `src/constants_input.py`. Adapt the input/output directories there to your local setup before running any of the scripts.

Downloading ERA5 reanalysis data (`scripts/ERA5/ERA5_download.py`) requires a free account at the Copernicus Climate Data Store (CDS) and a personal access token stored in `~/.cdsapirc`. See the [CDS API setup guide](https://cds.climate.copernicus.eu/how-to-api) for instructions. Note that the ERA5 license must be accepted once on the CDS website before data can be downloaded.

Downloading EUMETSAT satellite data (`scripts/eumetsat_download.py`) requires a free EUMETSAT Data Store account and personal API credentials (consumer key and secret), available from your [EUMETSAT API key page](https://api.eumetsat.int/api-key/) after logging in. Provide them via the environment variables `EUMETSAT_CONSUMER_KEY` and `EUMETSAT_CONSUMER_SECRET` (see the script header).

### Running the scripts

Most scripts were run interactively (e.g. in VS Code or a Jupyter environment) during development, but all `.py` scripts are also runnable directly from the command line, e.g.:
```bash
   python scripts/MXPol_L0_processing/MXPol_processing_L0.py
```

The notebooks under `scripts/MXPol_L1_processing/Zdr_calibration/` (`01_compute_filtered_dataframe.ipynb`, `02_compute_kriging.ipynb`, `03_correction_for_all_scans.ipynb`) are intended to be run in sequence, following the numbering in the filenames.

There is no single pipeline entry point — scripts under `scripts/` correspond to distinct processing steps (per-radar L0/L1 processing, calibration, gridding, attenuation correction, campaign-wide archiving) that were run individually as part of generating the dataset. Shared functionality used across these scripts is in `src/`.


## Data

The processed dataset and other resulting files associated with this code are archived on Zenodo: https://doi.org/10.5281/zenodo.21245825.


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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Nicole Clerx — Environmental Remote Sensing Laboratory (LTE), EPFL — nicole.clerx@epfl.ch