# CleanCloud CHOPIN – zarr archives

This file describes the variables in the .zarr archives in the Zenodo record https://doi.org/10.5281/zenodo.21245825. The archives combine the data of the three radars over the full campaign (18 October 2024 – January 2025) at Mt. Helmos, Greece:

- **MXPol** (X-band, 9.4 GHz)
- **MIRA** (Ka-band, 35.2 GHz)
- **BASTA** (W-band, 94.95 GHz)

All variables are derived from the L1 NetCDF files available on the ACTRIS Cloudnet data portal (see `data_ActrisCloudnet_netcdf.md` for the instrument descriptions and the original variable definitions). Variables were renamed to a common naming scheme across the three radars (see the table below), and the corrections described in the paper were applied where indicated. The zarr files themselves contain only minimal metadata; this file is the reference for variable descriptions and units.

All archives are stored in Zarr format version 2. All times are in UTC. Reflectivities and dual-frequency ratios are given in dB(Z).


## Overview of the archives

| Archive | Contents | Resolution |
|---|---|---|
| `zarrs_no_resampling.zip` | `MXPol.zarr`, `MIRA.zarr`, `BASTA_12m5.zarr`, `BASTA_25m.zarr`, `BASTA_100m_18km.zarr` | native |
| `zarrs_resampled_30s_25m.zip` | `MXPol.zarr`, `MIRA.zarr`, `BASTA.zarr` | 30 s / 25 m common grid |
| `zarr_MIRAspectral.zip` | `MIRA.zarr`, including Doppler spectra | native |
| `zarr_MXPolspectral.zip` | `MXPol.zarr`, including Doppler spectra | native |

In the resampled archives, all three radars share the same `time` and `range` coordinates, so variables can be compared directly between radars. Continuous variables were linearly interpolated; boolean flags were resampled by majority vote.


## Variable naming

| zarr variable | MXPol (L1) | MIRA (L1) | BASTA (L1) |
|---|---|---|---|
| `reflectivity` | `Zh` | `Zg` (converted to dBZ) | `reflectivity` |
| `reflectivity_raw` | `Zh_full` | – | `raw_reflectivity` |
| `velocity` | `RVel` | `VELg` | `velocity` |
| `spectrumwidth` | `Sw` | `RMSg` | – |
| `snrh` / `snrv` | `SNRh` / `SNRv` | `SNRg` / `SNRcx` (co / cross channel) | – |
| `snrh_raw` / `snrv_raw` | `SNRh_full` / `SNRv_full` | – | – |
| `zdr` | `Zdr_corrected` | – | – |
| `rhohv` | `Rhohv` | – | – |
| `LDR` | – | `LDRg` (converted to dB) | – |
| `skewness` | – | `SKWg` | – |
| `numpeaks` | – | `NPKg` | – |
| `z_30s` / `z_120s` | – | – | `reflectivity_30s` / `reflectivity_120s` |
| `vel_30s` / `vel_120s` | – | – | `velocity_30s` / `velocity_120s` |
| `bg_30s` / `bg_120s` | – | – | `background_mask_30s` / `background_mask_120s` |


## Variables present in all archives

```
    time                            (time) datetime64[ns]                   time [UTC]
    range                           (range) float64                         range from the radar [m]
    is_precip                       (time) bool                             precipitation flag (True: precipitation) [-]
    zdi                             (time) float32                          height of the 0 degC isotherm (ERA5) [m]
```


## MXPol (native resolution)

File: `zarrs_no_resampling/MXPol.zarr` — dimensions: `time` (8467201), `range` (398)

```
    reflectivity                    (time, range) float64                   horizontal reflectivity factor, masked [dBZ]
    reflectivity_raw                (time, range) float64                   horizontal reflectivity factor, not masked [dBZ]
    zdr                             (time, range) float64                   differential reflectivity, Zdr-corrected [dB]
    rhohv                           (time, range) float64                   copolar correlation coefficient [-]
    velocity                        (time, range) float64                   mean Doppler velocity [m/s]
    spectrumwidth                   (time, range) float64                   Doppler spectrum width [m/s]
    snrh                            (time, range) float64                   signal-to-noise ratio, horizontal polarisation, masked [dB]
    snrv                            (time, range) float64                   signal-to-noise ratio, vertical polarisation, masked [dB]
    snrh_raw                        (time, range) float64                   signal-to-noise ratio, horizontal polarisation, not masked [dB]
    snrv_raw                        (time, range) float64                   signal-to-noise ratio, vertical polarisation, not masked [dB]
    DFR_XKa                         (time, range) float64                   dual-frequency ratio X-Ka (Z_X - Z_Ka) [dB]
    DFR_XW                          (time, range) float64                   dual-frequency ratio X-W (Z_X - Z_W) [dB]
```

Global attributes: `campaign`, `created_by`, `creation_date`, `radar`.


## MIRA (native resolution)

File: `zarrs_no_resampling/MIRA.zarr` — dimensions: `time` (1693441), `range` (477)

```
    reflectivity                    (time, range) float64                   equivalent reflectivity factor, LNA-offset corrected [dBZ]
    LDR                             (time, range) float64                   linear depolarisation ratio [dB]
    velocity                        (time, range) float64                   mean Doppler velocity [m/s]
    spectrumwidth                   (time, range) float64                   Doppler spectrum width [m/s]
    skewness                        (time, range) float64                   Doppler spectrum skewness [-]
    numpeaks                        (time, range) float32                   number of peaks in the Doppler spectrum [-]
    snrh                            (time, range) float64                   signal-to-noise ratio, co-channel [dB]
    snrv                            (time, range) float64                   signal-to-noise ratio, cross-channel [dB]
    cloudtop                        (time) float32                          cloud-top height [m]
    DFR_XKa                         (time, range) float64                   dual-frequency ratio X-Ka (Z_X - Z_Ka) [dB]
    DFR_KaW                         (time, range) float64                   dual-frequency ratio Ka-W (Z_Ka - Z_W) [dB]
```

Global attributes: `campaign`, `created_by`, `creation_date`, `radar`.


## BASTA (native resolution)

BASTA operated in three range-resolution modes, stored in separate archives:

| File | Mode | dimensions |
|---|---|---|
| `zarrs_no_resampling/BASTA_12m5.zarr` | 12.5 m | `time` (2822401), `range` (960) |
| `zarrs_no_resampling/BASTA_25m.zarr` | 25 m | `time` (2822401), `range` (720) |
| `zarrs_no_resampling/BASTA_100m_18km.zarr` | 100 m | `time` (2822401), `range` (180) |

Variables (all modes):

```
    reflectivity                    (time, range) float64                   radar reflectivity factor, calibrated and attenuation-corrected [dBZ]
    reflectivity_raw                (time, range) float64                   radar reflectivity factor before corrections [dBZ]
    velocity                        (time, range) float64                   Doppler velocity [m/s]
    background_mask                 (time, range) float32                   background noise mask (1: signal, 0: noise) [-]
    z_30s                           (time, range) float64                   radar reflectivity factor, 30 s average [dBZ]
    z_120s                          (time, range) float64                   radar reflectivity factor, 120 s average [dBZ]
    vel_30s                         (time, range) float64                   Doppler velocity, 30 s average [m/s]
    vel_120s                        (time, range) float64                   Doppler velocity, 120 s average [m/s]
    bg_30s                          (time, range) float32                   background noise mask, 30 s [-]
    bg_120s                         (time, range) float32                   background noise mask, 120 s [-]
    mask_artefacts_strict           (time, range) bool                      artefact mask, strict criteria (True: artefact) [-]   (25 m and 100 m modes)
    mask_artefacts_loose            (time, range) bool                      artefact mask, loose criteria (True: artefact) [-]    (25 m and 100 m modes)
    cloudtop                        (time) float32                          cloud-top height [m]
    DFR_KaW                         (time, range) float64                   dual-frequency ratio Ka-W (Z_Ka - Z_W) [dB]
    DFR_XW                          (time, range) float64                   dual-frequency ratio X-W (Z_X - Z_W) [dB]
    time_original                   (time) float64                          original BASTA time stamp before alignment to the common time axis [s]
```

Additional variables in `BASTA_25m.zarr` only. `elevation` and `frequency` also apply to the `12m5` and `100m_18km` modes. Since BASTA points vertically, `height_above_radar` equals `range` for all modes.

```
    cloudtopDFRs                    (time) float64                          cloud-top Ka-W dual-frequency ratio, used for W-band attenuation correction [dB]
    height_above_radar              (range) float64                         height above the radar [m]
    elevation                       (range) float64                         elevation angle [deg]
    frequency                       float64                                 radar frequency [GHz]
```

Global attributes (`BASTA_12m5.zarr` only, but valid for the `25m` and `100m_18km` modes as well): `campaign`, `created_by`, `creation_date`, `radar`.


## Resampled archives (30 s / 25 m)

Files: `zarrs_resampled_30s_25m/MXPol.zarr`, `MIRA.zarr`, `BASTA.zarr` — dimensions: `time` (282241), `range` (584)

The variables are the same as in the native-resolution archives, interpolated onto the common grid, with the following differences:

- **BASTA:** a single archive (`BASTA.zarr`) using the data from the 25m mode; it contains `cloudtopDFRs` and the artefact masks, but not `time_original`, `height_above_radar`, `elevation` or `frequency`.
- **MXPol, MIRA:** identical variable sets to the native archives.


## Spectral archives (native resolution)

Files: `zarr_MIRAspectral/MIRA.zarr` — dimensions: `time` (1693441), `range` (477), `doppler` (256)<br>
`zarr_MXPolspectral/MXPol.zarr` — dimensions: `time` (8467201), `range` (398), `doppler` (256)

These archives contain the same moments as the corresponding native-resolution archives (without the `_raw` variables), plus:

```
    doppler                         (doppler) float64                       Doppler velocity bins [m/s]
    sZh                             (time, range, doppler) float64          spectral reflectivity, horizontal polarisation / co-channel [dBZ]
    sZv                             (time, range, doppler) float64          spectral reflectivity, vertical polarisation / cross-channel [dBZ]
    sLDR                            (time, range, doppler) float64          spectral linear depolarisation ratio [dB]   (MIRA only)
```

For MIRA, the spectral variables are computed from the L1 Doppler spectra as described in `data_ActrisCloudnet_netcdf.md` (`sZco`, `sZcx`, `sLDR`).