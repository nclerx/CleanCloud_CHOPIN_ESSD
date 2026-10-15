# Cleancloud CHOPIN 

This file describes the variables in the .nc-files containing L1 radar data from the CleanCloud CHOPIN campaign that is available on the Cloudnet Actris Data Portal: https://cloudnet.fmi.fi/site/mt-helmos

The three radars of which the data is described here are:

- **MXPol** (X-band, 9.4 GHz) - Cloudnet instrument handle https://hdl.handle.net/21.12132/3.b553925cb93b4d66, LAPI MXPol weather radar
- **MIRA** (Ka-band, 35.2 GHz) - Cloudnet instrument handle https://hdl.handle.net/21.12132/3.0645493039fe4116, METEK MIRA-35S (MBR5) Doppler scanning cloud radar
- **BASTA** (W-band, 94.95 GHz) - Cloudnet instrument handle https://hdl.handle.net/21.12132/3.e9441c71f67f4c83, LATMOS BASTA Doppler non-scanning cloud radar


MXPol is a pulsed spectral, polarimetric scanning X-band (9.4 GHz) radar manufactured by ProSensing (Schneebeli et al.,2013 https://doi.org/10.1109/tgrs.2013.2287017). 

MIRA is a pulsed, polarimetric scanning Ka-band (35.2 GHz) radar (Görsdorf et al., 2015, https://doi.org/10.1175/JTECH-D-14-00066.1) manufactured by Metek (Meteorologische Messtechnik GmbH), Germany.

BASTA is a vertically pointing bi-static frequency-modulated continuous wave (FMCW) W-band cloud profiler developed by LATMOS (Laboratoire Atmosphères, Milieux, Observations Spatiales) in Guyancourt, France (Delanoë et al., 2016, https://doi.org/10.1175/jtech-d-15-0104.1).


## MXPol

Three different data files are available for MXPol, for the vertical profiling data (birdbath PPIs) - `_Zdr.nc`, for the RHI scans `_RHI.nc` and for the sector scans (partial PPIs) `_sector.nc`.

File naming convention is as follows:<br>
       `XPOL-[YYYYMMDD-HHMMSS]_[scantype].nc`

Data variables in the `_Zdr.nc` files:
```
    latitude               float64                          latitude [deg N]
    longitude              float64                          longitude [deg E]
    altitude               float64                          altitude [m a.s.l.]
    volume_number          float64                          volume number [-]
    time_coverage_start    |S32                             UTC time of first ray in file
    time_coverage_end      |S32                             UTC time of last ray in file
    time_reference         |S32                             UTC time reference [-]
    platform_type          |S32                             platform type [-]
    instrument_type        |S32                             instrument type [-]
    primary_axis           |S32                             primary axis [-]
    sweep_number           (sweep)                          sweep number[-]
    fixed_angle            (sweep) float64                  fixed angle [deg]
    sweep_start_ray_index  (sweep) float64                  index of first ray in sweep [-]
    sweep_end_ray_index    (sweep) float64                  index of last ray in sweep [-]
    sweep_mode             (sweep) |S32                     sweep mode [-]
    azimuth                (time) float64                   azimuth [deg]
    elevation              (time) float64                   elevation angle [deg]
    nyquist_velocity       (time) float64                   Nyquist velocity [m/s]
    Mask                   (time, range) float32            masked time steps identified as outliers [-]
    Zh                     (time, range) float32            horizontal reflectivity (masked) [dBZ]
    Zdr                    (time, range) float32            differential reflectivity (masked) [dB]
    Rhohv                  (time, range) float32            copolar correlation coefficient (masked) [-]
    RVel                   (time, range) float32            mean Doppler velocity (masked) [m/s]
    Sw                     (time, range) float32            spectral width (masked) [m/s]
    SNRh                   (time, range) float32            signal-to-noise ratio at horizontal polarisation (masked) [-]
    SNRv                   (time, range) float32            signal-to-noise ratio at vertical polarisation (masked) [-]
    Psidp                  (time, range) float32            total differential phase (masked) [deg]
    Signal_h               (time, range) float32            signal at horizontal polarisation (masked) [mW]
    Signal_v               (time, range) float32            signal at vertical polarisation (masked) [mW]
    Zh_full                (time, range) float32            horizontal reflectivity [dBZ]
    Zdr_full               (time, range) float32            differential reflectivity [dB]
    Rhohv_full             (time, range) float32            copolar correlation coefficient [-]
    RVel_full              (time, range) float32            mean Doppler velocity [m/s]
    Sw_full                (time, range) float32            spectral width [m/s]
    SNRh_full              (time, range) float32            signal-to-noise ratio  at horizontal polarisation [-]
    SNRv_full              (time, range) float32            signal-to-noise ratio  at vertical polarisation [-]
    Psidp_full             (time, range) float32            total differential phase [deg]
    Signal_h_full          (time, range) float32            signal at horizontal polarisation [mW]
    Signal_v_full          (time, range) float32            signal at vertical polarisation [mW]
    Zdr_corrected          (time, range) float64            corrected differential reflectivity (masked) [dB]
    sVel                   (nfft) float32                   velocity bins [m/s]
    sPowH                  (time, range, nfft) float32      spectral horizontal power (masked) [mW]
    sPowV                  (time, range, nfft) float32      spectral vertical power (masked) [mW]
    Noise_h                (time, range, nfft) float32      horizontal noise power (masked) [mW]
    Noise_v                (time, range, nfft) float32      vertical noise power (masked) [mW]
    sPowH_full             (time, range, nfft) float32      spectral horizontal power [mW]
    sPowV_full             (time, range, nfft) float32      spectral vertical power [mW]
    Noise_h_full           (time, range, nfft) float32      horizontal noise power [mW]
    Noise_v_full           (time, range, nfft) float32      vertical noise power [mW]
```

Data variables in the `_sector.nc` files:
```
    latitude               float64                          latitude [deg N]
    longitude              float64                          longitude [deg E]
    altitude               float64                          altitude [m a.s.l.]
    volume_number          float64                          volume number [-]
    time_coverage_start    |S32                             UTC time of first ray in file
    time_coverage_end      |S32                             UTC time of last ray in file
    time_reference         |S32                             UTC time reference [-]
    platform_type          |S32                             platform type [-]
    instrument_type        |S32                             instrument type [-]
    primary_axis           |S32                             primary axis [-]
    sweep_number           (sweep)                          sweep number[-]
    fixed_angle            (sweep) float64                  fixed angle [deg]
    sweep_start_ray_index  (sweep) float64                  index of first ray in sweep [-]
    sweep_end_ray_index    (sweep) float64                  index of last ray in sweep [-]
    sweep_mode             (sweep) |S32                     sweep mode [-]
    azimuth                (time) float64                   azimuth [deg]
    elevation              (time) float64                   elevation angle [deg]
    nyquist_velocity       (time) float64                   Nyquist velocity [m/s]
    Mask                   (time, range) float32            masked time steps identified as outliers [-]
    Zh                     (time, range) float32            horizontal reflectivity (masked) [dB]
    Zdr                    (time, range) float32            differential reflectivity (masked) [dB]
    Rhohv                  (time, range) float32            copolar correlation coefficient (masked) [-]
    RVel                   (time, range) float32            mean Doppler velocity (masked) [m/s]
    Sw                     (time, range) float32            spectral width (masked) [m/s]
    SNRh                   (time, range) float32            signal-to-noise ratio at horizontal polarisation (masked) [-]
    SNRv                   (time, range) float32            signal-to-noise ratio at vertical polarisation (masked) [-]
    Psidp                  (time, range) float32            total differential phase (masked) [deg]
    Zh_full                (time, range) float32            horizontal reflectivity [dB]
    Zdr_full               (time, range) float32            differential reflectivity [dB]
    Rhohv_full             (time, range) float32            copolar correlation coefficient [-]
    RVel_full              (time, range) float32            mean Doppler velocity [m/s]
    Sw_full                (time, range) float32            spectral width [m/s]
    SNRh_full              (time, range) float32            signal-to-noise ratio  at horizontal polarisation [-]
    SNRv_full              (time, range) float32            signal-to-noise ratio  at vertical polarisation [-]
    Psidp_full             (time, range) float32            total differential phase [deg]
    Zdr_corrected          (time, range) float64            corrected differential reflectivity (masked) [dB]
    Kdp                    (time, range) float64            specific differential phase [deg/km]
    hydroclass             (time, range) uint8              hydrometeor classification [-]
    height_over_iso0       (time, range) float64            height above 0 degree isotherm [m]
    hydroclass_entropy     (time, range) float64            hydroclass entropy (0: unambiguous, 1: uncertain) [-]
    proportion_AG          (time, range) float64            proportion aggregates [-]
    proportion_CR          (time, range) float64            proportion ice crystals [-]
    proportion_LR          (time, range) float64            proportion light rain [-]
    proportion_RP          (time, range) float64            proportion rimed particles [-]
    proportion_RN          (time, range) float64            proportion rain [-]
    proportion_VI          (time, range) float64            proportion vertically oriented ice [-]
    proportion_WS          (time, range) float64            proportion wet snow [-]
    proportion_MH          (time, range) float64            proportion melting hail [-]
    proportion_IH          (time, range) float64            proportion dry hail [-]
    sVel                   (nfft) float32                   velocity bins [m/s]
    sPowH                  (time, range, nfft) float32      spectral horizontal power (masked) [mW]
    sPowV                  (time, range, nfft) float32      spectral vertical power (masked) [mW]
    Noise_h                (time, range, nfft) float32      horizontal noise power (masked) [mW]
    Noise_v                (time, range, nfft) float32      vertical noise power (masked) [mW]
    sPowH_full             (time, range, nfft) float32      spectral horizontal power [mW]
    sPowV_full             (time, range, nfft) float32      spectral vertical power [mW]
    Noise_h_full           (time, range, nfft) float32      horizontal noise power [mW]
    Noise_v_full           (time, range, nfft) float32      vertical noise power [mW]
```

Data variables in the `_RHI.nc` files:
```
    latitude               float64                          latitude [deg N]
    longitude              float64                          longitude [deg E]
    altitude               float64                          altitude [m a.s.l.]
    volume_number          float64                          volume number [-]
    time_coverage_start    |S32                             UTC time of first ray in file
    time_coverage_end      |S32                             UTC time of last ray in file
    time_reference         |S32                             UTC time reference [-]
    platform_type          |S32                             platform type [-]
    instrument_type        |S32                             instrument type [-]
    primary_axis           |S32                             primary axis [-]
    sweep_number           (sweep)                          sweep number[-]
    fixed_angle            (sweep) float64                  fixed angle [deg]
    sweep_start_ray_index  (sweep) float64                  index of first ray in sweep [-]
    sweep_end_ray_index    (sweep) float64                  index of last ray in sweep [-]
    sweep_mode             (sweep) |S32                     sweep mode [-]
    azimuth                (time) float64                   azimuth [deg]
    elevation              (time) float64                   elevation angle [deg]
    nyquist_velocity       (time) float64                   Nyquist velocity [m/s]
    Mask                   (time, range) float32            masked time steps identified as outliers [-]
    Zh                     (time, range) float32            horizontal reflectivity (masked) [dB]
    Zdr                    (time, range) float32            differential reflectivity (masked) [dB]
    Rhohv                  (time, range) float32            copolar correlation coefficient (masked) [-]
    RVel                   (time, range) float32            mean Doppler velocity (masked) [m/s]
    Sw                     (time, range) float32            spectral width (masked) [deg]
    SNRh                   (time, range) float32            signal-to-noise ratio at horizontal polarisation (masked) [-]
    SNRv                   (time, range) float32            signal-to-noise ratio at vertical polarisation (masked) [-]
    Psidp                  (time, range) float32            total differential phase (masked) [deg]
    Signal_h               (time, range) float32            signal at horizontal polarisation (masked) [mW]
    Signal_v               (time, range) float32            signal at vertical polarisation (masked) [mW]
    Zh_full                (time, range) float32            horizontal reflectivity [dB]
    Zdr_full               (time, range) float32            differential reflectivity [dB]
    Rhohv_full             (time, range) float32            copolar correlation coefficient [-]
    RVel_full              (time, range) float32            mean Doppler velocity [m/s]
    Sw_full                (time, range) float32            spectral width [m/s]
    SNRh_full              (time, range) float32            signal-to-noise ratio  at horizontal polarisation [-]
    SNRv_full              (time, range) float32            signal-to-noise ratio  at vertical polarisation [-]
    Psidp_full             (time, range) float32            total differential phase [deg]
    Signal_h_full          (time, range) float32            signal at horizontal polarisation [mW]
    Signal_v_full          (time, range) float32            signal at vertical polarisation [mW]
    Zdr_corrected          (time, range) float64            corrected differential reflectivity (masked) [dB]
    Kdp                    (time, range) float64            specific differential phase [deg/km]
    hydroclass             (time, range) uint8              hydrometeor classification [-]
    height_over_iso0       (time, range) float64            height above 0 degree isotherm [m]
    hydroclass_entropy     (time, range) float64            hydroclass entropy (0: unambiguous, 1: uncertain) [-]
    proportion_AG          (time, range) float64            proportion aggregates [-]
    proportion_CR          (time, range) float64            proportion ice crystals [-]
    proportion_LR          (time, range) float64            proportion light rain [-]
    proportion_RP          (time, range) float64            proportion rimed particles [-]
    proportion_RN          (time, range) float64            proportion rain [-]
    proportion_VI          (time, range) float64            proportion vertically oriented ice [-]
    proportion_WS          (time, range) float64            proportion wet snow [-]
    proportion_MH          (time, range) float64            proportion melting hail [-]
    proportion_IH          (time, range) float64            proportion dry hail [-]
    sVel                   (nfft) float32                   velocity bins [m/s]
    sPowH                  (time, range, nfft) float32      spectral horizontal power (masked) [mW]
    sPowV                  (time, range, nfft) float32      spectral vertical power (masked) [mW]
    Noise_h                (time, range, nfft) float32      horizontal noise power (masked) [mW]
    Noise_v                (time, range, nfft) float32      vertical noise power (masked) [mW]
    sPowH_full             (time, range, nfft) float32      spectral horizontal power [mW]
    sPowV_full             (time, range, nfft) float32      spectral vertical power [mW]
    Noise_h_full           (time, range, nfft) float32      horizontal noise power [mW]
    Noise_v_full           (time, range, nfft) float32      vertical noise power [mW]
```


MXPol global attributes (example, fields are the same for all files but values might differ)
```
GPSCoordSystem            WGS84
Hemisphere                N
Latitude_unit             DegreesNorth
Latitude_value            38.0
Longitude_unit            DegreesEast
Longitude_value           22.11
Altitude_unit             MetersAboveSeaLevel
Altitude_value            1690.0
TimeAssignScan_unit       Seconds since 01-01-1970
TimeAssignScan_value      1733398234

RadarParameters           Radar NyquistVelocity AzOffset ElOffset RadarFreq BW3dB
Radar_name                MXPol
Radar_description         MXPol
NyquistVelocity_unit      MetersPerSecond
NyquistVelocity_value     11.390290957446808
AzOffset_unit             Degrees
AzOffset_value            190
AzOffset_description      Az. Offset already substracted
RadarFreq_unit            GigaHertz
RadarFreq_value           9.4
BW3dB_unit                Degrees
RBW3dB_value              1.3
BW3dB_description         Az. Offset already substracted

AcquisitionParameters     FilterBW PRI1 PRIT PulseWidth RangeResolution RangeToFirstGate NumRanges NoiseH NoiseV
FilterBW_unit             MegaHertz
FilterBW_value            5.0
PRI1_unit                 MicroSeconds
PRI1_value                700
PRI2_unit                 MicroSeconds
PRI2_value                700
PRI3_unit                 MicroSeconds
PRI3_value                1200
PRIT_unit                 MicroSeconds
PRIT_value                700
PulseWidth_unit           Microseconds
PulseWidth_value          0.20013845711889122
RangeResolution_unit      Meters
RangeResolution_value     30.0
RangeGateSpacing_unit     Meters
RangeGateSpacing_value    30.0
RangeToFirstGate_unit     Meters
RangeToFirstGate_value    376.95000000000005
NumRanges_unit            dimensionless
NumRanges_value           142
NoiseH(V)_explanation     Estimated noise in H(V) channels
NoiseH_unit               Milliwatts
NoiseH_value              2.366261656412887e-07
NoiseV_unit               Milliwatts
NoiseV_value              3.47741131202735e-07
Elevation_unit            Degreees
Elevation_value           89.70552309289354
MissingParam              -999.0
MissingData_explanation   Missing NetCDF parameters/attributes

MoreAcqParameters         ScanMode FFTLength CAve PostAve ClutterFilter SumPower
ScanMode                  FFT
FFT_length_unit           256
CAve_unit                 dimensionless
CAve_value                2
PostAve_unit              dimensionless
PostAve_value             2
ClutterFilter             OFF
ClutterFilterWidth_unit   MetersPerSecond
ClutterFilterWidth_value  0.0
SumPower                  OFF

CalibrationParameters     Zcal ZdrOffset
ZCal_dB                   8.86
ZdrOffset_dB              0

CensoringParameters       CensoringMethod SNRthPower SNRthPhase Rhohvth
CensoringMethod           SNRthPower_SNRthPhase_Rhohvth
SNRthPower_dB             -10.0
SNRthPhase_dB             10
Rhohvth                   0.5

PhidpParameters           PhidpEstMethod
PhidpEstParameters        MULTISTEP
KdpParameters             KdpEstMethod
KdpEstParameters          MULTISTEP
AttCorrParameters         AttCorrMethod
AttCorrMethod             HBORDAN

History                   Created 2026-06-29 23:11:36
Source                    Radar data processed at LTE-EPFL
Institution               Laboratoire de Teledetection Environnementale - Ecole Polytechnique Federale de Lausanne
ContactInformation        http://lte.epfl.ch
Elevation_correction      Yes
Conventions               CF/Radial
field_names               Zh, Zdr, Kdp, Phidp, Rhohv, ZhCorr, ZdrCorr, RVel, Sw, SNRh, SNRv, Psidp, sCC, sPowH, sPowV, sVel, Signal_h, Signal_v, Noise_h, Noise_v, Mask, Zh_full, Zdr_full, Rhohv_full, RVel_full, Sw_full, SNRh_full, SNRv_full, Psidp_full, sPowH_full, sPowV_full, Signal_h_full, Signal_v_full, Noise_h_full, Noise_v_full
history                   created by clerx on ltesrv7 at 2026-06-29T23:11:38Z using Py-ART
```


 ## MIRA
 Three different data files are available for MIRA, for the vertical profiling data - `[no_suffix].znc`, for the RHI scans `rhi[azimuth].znc` and for the sector scans (partial PPIs) `.ppi.znc`.

File naming convention is as follows:<br>
    if only profiling: <br>
        `[YYYYMMDD]_[HHMM].znc` (vertical profiling data, 5 second resolution), 1 file per hour<br>
    else:<br>
        `[YYYYMMDD]_[HHMM].znc` (vertical profiling data, 5 second resolution) with YYYYMMDD_HHMM the start date of the data<br>
        `[YYYYMMDD]_[HHMMSS].rhi[azimuth].znc`, RHI scans (azimuth in degrees, 5 second resolution), with YYYYYMMDD_HHMMSS the start date of the scan, 1 file per scan<br>
        `[YYYYMMDD]_[HHMMSS].ppi.znc`, sector scans (5 second resolution), with YYYYYMMDD_HHMMSS the start date of the scan, 1 file per scan<p>

Moments to be converted into dB as follows:<br>
`Zg, Zcx, Z, LDRg, LDRnormal, SNRg, SNRcx, RHO, PIA, AttnRain` <br>
`10 * log10(variable)`

Spectral variables to be converted in dB:<br>
`sZco, sZcx, sLDR` as follows:<br>
`sZco = (SPCco - HSDco) * RadarConst * (range/5000)^2 * SNRCorFaCo / npw1`
`sZcx = (SPCcx - HSDcx) * RadarConst * (range/5000)^2 * SNRCorFaCx / npw2`
`sLDR = sZcx / sZco`

Data variables in the `[no_suffix].znc` files:
```
    nfft             int32                              number of FFT points [count]
    prf              int32                              pulse repetition frequency [Hz]
    NyquistVelocity  float32                            speed of light  * PRF / (4 * frequency) [m/s]
    nave             int32                              number of spectral averages (not accounting for overlapping FFTs) [count]
    ovl              int32                              overlapping FFT flags [count]
    zrg              int32                              number of range gates [count]
    rg0              int32                              number of lowest range gates [count]
    drg              float32                            range resolution [m]
    lambda           float32                            wave length [m]
    microsec         (time) int32                       microsends [us]
    tpow             (time) float32                     average transmit power [W]
    npw1             (time) float32                     noise power co-channel [sum over the spectrum in DSP units]
    npw2             (time) float32                     noise power cross-channel [sum over the spectrum in DSP units]
    cpw1             (time) float32                     signal-to-noise ratio  of calibration signal co-channel [ratio]
    cpw2             (time) float32                     signal-to-noise ratio  of calibration signal cross-channel [ratio]
    grst             (time) int32                       general radar state [enumerate]
    azi              (time) float32                     azimuth angle, if > 370 azi-720 represents the middle of the averaging interval, if azi < 370 then azi represents the position of the end of the next data block [deg]
    elv              (time) float32                     elevation angle, if > 370 elv-720 represents the middle of the averaging interval, if elv < 370 then elv represents the position of the end of the next data block [deg]
    aziv             (time) float32                     azimuth angle velocity [deg/s]
    northangle       (time) float32                     north angle [deg]
    elvv             (time) float32                     elevation angle velocity [deg/s]
    LO_Frequency     (time) float32                     transmit frequency [MHz]
    DetuneFine       (time) float32                     detuning of the local oscillator detected from Txn footprint [kHz]
    MRMco            (time) float32                     radiometer signal Co [-]
    MRMcx            (time) float32                     radiometer signal Cx [-]
    RadarConst       (time) float32                     radar constant related to 5 km height, and 200 ns pulses. Z = SNR * RadarConst * (range/5 km)^2 * SNRCorFaCo (changes slightly in time due to the measured transmit power) [Z]
    SNRg             (time, range) float32              reflectivity SNRg [-]
    VELg             (time, range) float32              Doppler velocity VELg [m/s]
    RMSg             (time, range) float32              peak width RMSg [m/s]
    LDRg             (time, range) float32              linear depolarisation ratio LDRg [-]
    NPKg             (time, range) int8                 number of peaksg [-]
    SNRcx            (time, range) float32              reflectivity SNRg cross-channel [-]
    RHO              (time, range) float32              co-cross correlation [-]
    RHOwav           (time, range) float32              co-cross correlation peak-weighted [-]
    DPS              (time, range) float32              co-cross differential phase [deg]
    LDRnormal        (time, range) float32              LDR calculated by weighting both channels by the SNR of the more powerful one [-]
    HSDco            (time, range) float32              co-channel HSdiv noise level [DSP]
    HSDcx            (time, range) float32              cross-channel HSdiv noise level [DSP]
    Zg               (time, range) float32              reflectivity of all targets [Z]
    Zcx              (time, range) float32              cross-reflectivity [Z]
    Z                (time, range) float32              Mie-corrected reflectivity [Z]
    RR               (time, range) float32              rain rate [mm/h]
    LWC              (time, range) float32              liquid water content [g/m^3]
    Wcorr            (time, range) float32              vertical wind speed from left side of DSD [m/s]
    PIA              (time, range) float32              path-integrated attenuation due to rain [-]
    AttnRain         (time, range) float32              two-way attenuation due to rain per 1000 m [-]
    ISDRco           (time, range) float32              in-spectrum dynamic range co-channel [-]
    ISDRcx           (time, range) float32              in-spectrum dynamic range cross-channel [-]
    Saturatedco      (time, range) int8                 saturation flag co-channel [-]
    Saturatedcx      (time, range) int8                 saturation flag cross-channel [-]
    SNRCorFaCo       (time, range) float32              factor to correct co-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SNRCorFaCx       (time, range) float32              factor to correct cross-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SKWg             (time, range) float32              skewness [-]
    DropSize         (range, doppler) float32           drop size bins [mm]
    SPCco            (time, range, doppler) float32     Doppler spectrum co-channel [DSP]
    SPCcx            (time, range, doppler) float32     Doppler spectrum cross-channel [DSP]
    NvD              (time, range, doppler) float32     drop size distrubution [1/m^3 mm]
    SPCcocxRe        (time, range, doppler) float32     co-cross spectrum real part [DSP]
    SPCcocxIm        (time, range, doppler) float32     co-cross spectrum imaginary part [DSP]
```

Data variables in the `[azimuth].znc` files:
```
    nfft             int32                              number of FFT points [count]
    prf              int32                              pulse repetition frequency [Hz]
    NyquistVelocity  float32                            speed of light  * PRF / (4 * frequency) [m/s]
    nave             int32                              number of spectral averages (not accounting for overlapping FFTs) [count]
    ovl              int32                              overlapping FFT flags [count]
    zrg              int32                              number of range gates [count]
    rg0              int32                              number of lowest range gates [count]
    drg              float32                            range resolution [m]
    lambda           float32                            wave length [m]
    microsec         (time) int32                       microsends [us]
    tpow             (time) float32                     average transmit power [W]
    npw1             (time) float32                     noise power co-channel [sum over the spectrum in DSP units]
    npw2             (time) float32                     noise power cross-channel [sum over the spectrum in DSP units]
    cpw1             (time) float32                     SNR of calibration signal co-channel [ratio]
    cpw2             (time) float32                     SNR of calibration signal cross-channel [ratio]
    grst             (time) int32                       general radar state [enumerate]
    azi              (time) float32                     azimuth angle, if > 370 azi-720 represents the middle of the averaging interval, if azi < 370 then azi represents the position of the end of the next data block [deg]
    elv              (time) float32                     elevation angle, if > 370 elv-720 represents the middle of the averaging interval, if elv < 370 then elv represents the position of the end of the next data block [deg]
    aziv             (time) float32                     azimuth angle velocity [deg/s]
    northangle       (time) float32                     north angle [deg]
    elvv             (time) float32                     elevation angle velocity [deg/s]
    LO_Frequency     (time) float32                     transmit frequency [MHz]
    DetuneFine       (time) float32                     detuning of the local oscillator detected from Txn footprint [kHz]
    MRMco            (time) float32                     radiometer signal Co [-]
    MRMcx            (time) float32                     radiometer signal Cx [-]
    RadarConst       (time) float32                     radar constant related to 5 km height, and 200 ns pulses. Z = SNR * RadarConst * (range/5 km)^2 * SNRCorFaCo (changes slightly in time due to the measured transmit power) [Z]
    SNRg             (time, range) float32              reflectivity SNRg [-]
    VELg             (time, range) float32              Doppler velocity VELg [m/s]
    RMSg             (time, range) float32              peak width RMSg [m/s]
    LDRg             (time, range) float32              linear depolarisation ratio LDRg [-]
    NPKg             (time, range) int8                 number of peaksg [-]
    SNRcx            (time, range) float32              reflectivity SNRg cross-channel [-]
    RHO              (time, range) float32              co-cross correlation [-]
    RHOwav           (time, range) float32              co-cross correlation peak-weighted [-]
    DPS              (time, range) float32              co-cross differential phase [deg]
    LDRnormal        (time, range) float32              LDR calculated by weighting both channels by the SNR of the more powerful one [-]
    HSDco            (time, range) float32              co-channel HSdiv noise level [DSP]
    HSDcx            (time, range) float32              cross-channel HSdiv noise level [DSP]
    Zg               (time, range) float32              reflectivity of all targets [Z]
    Zcx              (time, range) float32              cross-reflectivity [Z]
    ISDRco           (time, range) float32              in-spectrum dynamic range co-channel [-]
    ISDRcx           (time, range) float32              in-spectrum dynamic range cross-channel [-]
    Saturatedco      (time, range) int8                 saturation flag co-channel [-]
    Saturatedcx      (time, range) int8                 saturation flag cross-channel [-]
    SNRCorFaCo       (time, range) float32              factor to correct co-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SNRCorFaCx       (time, range) float32              factor to correct cross-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SKWg             (time, range) float32              skewness [-]
    SPCco            (time, range, doppler) float32     Doppler spectrum co-channel [DSP]
    SPCcx            (time, range, doppler) float32     Doppler spectrum cross-channel [DSP]
    SPCcocxRe        (time, range, doppler) float32     co-cross spectrum real part [DSP]
    SPCcocxIm        (time, range, doppler) float32     co-cross spectrum imaginary part [DSP]
```

Data variables in the `.ppi.znc` files:
```
    nfft             int32                              number of FFT points [count]
    prf              int32                              pulse repetition frequency [Hz]
    NyquistVelocity  float32                            speed of light  * PRF / (4 * frequency) [m/s]
    nave             int32                              number of spectral averages (not accounting for overlapping FFTs) [count]
    ovl              int32                              overlapping FFT flags [count]
    zrg              int32                              number of range gates [count]
    rg0              int32                              number of lowest range gates [count]
    drg              float32                            range resolution [m]
    lambda           float32                            wave length [m]
    microsec         (time) int32                       microsends [us]
    tpow             (time) float32                     average transmit power [W]
    npw1             (time) float32                     noise power co-channel [sum over the spectrum in DSP units]
    npw2             (time) float32                     noise power cross-channel [sum over the spectrum in DSP units]
    cpw1             (time) float32                     SNR of calibration signal co-channel [ratio]
    cpw2             (time) float32                     SNR of calibration signal cross-channel [ratio]
    grst             (time) int32                       general radar state [enumerate]
    azi              (time) float32                     azimuth angle, if > 370 azi-720 represents the middle of the averaging interval, if azi < 370 then azi represents the position of the end of the next data block [deg]
    elv              (time) float32                     elevation angle, if > 370 elv-720 represents the middle of the averaging interval, if elv < 370 then elv represents the position of the end of the next data block [deg]
    aziv             (time) float32                     azimuth angle velocity [deg/s]
    northangle       (time) float32                     north angle [deg]
    elvv             (time) float32                     elevation angle velocity [deg/s]
    LO_Frequency     (time) float32                     transmit frequency [MHz]
    DetuneFine       (time) float32                     detuning of the local oscillator detected from Txn footprint [kHz]
    MRMco            (time) float32                     radiometer signal Co [-]
    MRMcx            (time) float32                     radiometer signal Cx [-]
    RadarConst       (time) float32                     radar constant related to 5 km height, and 200 ns pulses. Z = SNR * RadarConst * (range/5 km)^2 * SNRCorFaCo (changes slightly in time due to the measured transmit power) [Z]
    SNRg             (time, range) float32              reflectivity SNRg [-]
    VELg             (time, range) float32              Doppler velocity VELg [m/s]
    RMSg             (time, range) float32              peak width RMSg [m/s]
    LDRg             (time, range) float32              linear depolarisation ratio LDRg [-]
    NPKg             (time, range) int8                 number of peaksg [-]
    SNRcx            (time, range) float32              reflectivity SNRg cross-channel [-]
    RHO              (time, range) float32              co-cross correlation [-]
    RHOwav           (time, range) float32              co-cross correlation peak-weighted [-]
    DPS              (time, range) float32              co-cross differential phase [deg]
    LDRnormal        (time, range) float32              LDR calculated by weighting both channels by the SNR of the more powerful one [-]
    HSDco            (time, range) float32              co-channel HSdiv noise level [DSP]
    HSDcx            (time, range) float32              cross-channel HSdiv noise level [DSP]
    Zg               (time, range) float32              reflectivity of all targets [Z]
    Zcx              (time, range) float32              cross-reflectivity [Z]
    ISDRco           (time, range) float32              in-spectrum dynamic range co-channel [-]
    ISDRcx           (time, range) float32              in-spectrum dynamic range cross-channel [-]
    Saturatedco      (time, range) int8                 saturation flag co-channel [-]
    Saturatedcx      (time, range) int8                 saturation flag cross-channel [-]
    SNRCorFaCo       (time, range) float32              factor to correct co-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SNRCorFaCx       (time, range) float32              factor to correct cross-channel SNR based on receiver calibration measurement by Noisecom source [-]
    SKWg             (time, range) float32              skewness [-]
    SPCco            (time, range, doppler) float32     Doppler spectrum co-channel [DSP]
    SPCcx            (time, range, doppler) float32     Doppler spectrum cross-channel [DSP]
    SPCcocxRe        (time, range, doppler) float32     co-cross spectrum real part [DSP]
    SPCcocxIm        (time, range, doppler) float32     co-cross spectrum imaginary part [DSP]
```

MIRA global attributes (example, fields are the same for all files but values might differ)
```
convention                CF-1.0
location                  Mt. Helmos ski resort
Altitude                  1700m
Latitude                  38.0074250N
Longitude                 22.1962496E
system                    MBR5
title                     MIRA Cloud Radar Data
institution               METEK
source                    241205_114238.pds
reference                 Ka Band Cloud Radar MIRA, METEK GmbH www.metek.de
Copyright                 
Copyright_Owner           
notes                     It is accounted for the Receiver Bwd Loss in Radarconst*, and Z*.

ppar
    PRF                   5000
    PDR                   208
    SFT                   256
    AVC                   100
    IHP                   5
    CHG                   479
    POL                   208
    ATT                   1500
    TX                    0
    ADCGAIN0              0.00000
    ADCGAIN1              0.00000
    WND                   0
    POS                   -2147483312
    ADD                   128
    LEN                   0
    CAL                   208
    NOS                   0
    OF0                   56
    OF1                   55
    SWT                   50
    SUM                   0
    OSC                   16
    TST                   0
    COR                   1
    OFS                   0
    HSBN                  0
    HSBA                  1
    CALIBRPOWER_M         1.10000
    CALIBRSNR_M           280.000
    CALIBRPOWER_S         1.30000
    CALIBRSNR_S           290.000
    RAW_GATE1             5
    RAW_GATE2             482
    RAW                   0
    PRC                   1

hrd
    NAME                  241205_114238.pds
    OPER                  Thu Dec  5 11:42:38 2024
    DESCR                 Altitude=1700m:Latitude=38.0074250N:Longitude=22.1962496E
    PLACE                 Mt. Helmos ski resort
    SN                    fzk
    LEN_PPAR              140
    LEN_SRVI              -1
    PRF                   5000.00
    PULSE_WIDTH           2.08000e-07
    RX_PULSEWIDTH         2.08000e-07
    TX_PULSEWIDTH         2.08000e-07
    NFFT                  256
    AVC                   100
    NAVE                  200
    AVE                   5.12000
    C                     2.99800e+08
    XMT                   3.51500e+10
    FZK100                1.00000
    DELTA_H               31.1792
    MOM_H_MIN             155.896
    MOM_ZRG               477
    RAW_H_MIN             155.896
    RAW_ZRG               477
    DO_RAW                0
    VUAR                  21.3229
    VNY                   10.6615
    LA                    0.00852916
    POL                   1
    ATT                   0.00000
    STC_MAXGATE           0
    FFTWINDOWTYPE         0
    PCP                   336
    OVL                   1
```


## BASTA

Data variables are the same for all the BASTA files:
```
    elevation                      float32                  elevation above horizon [deg]
    carrier_frequency              float32                  radar carrier frequency [GHz]
    pulse_width                    float32                  radar pulse duration [us]
    gen_sampling_rate              float32                  sampling rate of the generation chirp [MHz]
    gen_chirp_starting_frequency   int32                    starting frequency of the generation chirp [kHz]
    gen_chirp_final_frequency      int32                    final frequency of the generation chirp [kHz]
    ref_sampling_rate              float32                  sampling rate of the echo signals [MHz]
    ref_chirp_starting_frequency   int32                    starting frequency of the reference chirp [kHz]
    ref_chirp_final_frequency      int32                    final frequency of the reference chirp [kHz]
    gate_length                    float32                  length of range gate [m]
    integration_time               float32                  pulse pair integration duration [s]
    decimation                     int32                    decimation of echo signals before pulse pair [-]
    ambiguous_velocity             float32                  ambiguous velocity [m/s]
    latitude                       (time) float32           latitude of radar [degN]
    longitude                      (time) float32           longitude of radar [degE]
    altitude                       (time) float32           altitude of antenna above mean sea level [m]
    power_v1                       (time) float32           power supply 1 (radar) voltage [V]
    power_i1                       (time) float32           power supply 1 (radar) current [A]
    power_v2                       (time) float32           power supply 2 (radar) voltage [V]
    power_i2                       (time) float32           power supply 2 (radar) current [A]
    radar_box_t                    (time) float32           radar box temperature [degC]
    radar_amplifier_t              (time) float32           radar amplifier temperature [degC]
    radar_air_flow                 (time) float32           radar air flow [L/min]
    radar_humidity                 (time) float32           radar humidity [%]
    radar_yaw                      (time) float32           radar yaw angle [deg]
    radar_pitch                    (time) float32           radar pitch angle [deg]
    radar_roll                     (time) float32           radar roll angle [deg]
    radar_transmitted_power        (time) float32           radar transmitted power [dBm]
    radar_oscillator               (time) int32             oscillator (0: VCO, 1: PLO)
    radar_vco_frequency            (time) int32             voltage controlled oscillator frequency [MHz]
    radar_phase_lock               (time) int32             radar phase lock [-]
    radar_thermal_cutoff_50        (time) int32             radar 50 degC thermal cutoff [-]
    peltier_set_t                  (time) float32           Peltier set temperature [degC]
    peltier_sensor_t               (time) float32           Peltier sensor temperature [degC]
    peltier_fet_t                  (time) float32           Peltier FET temperature [degC]
    peltier_output                 (time) float32           Peltier output [%]
    peltier_fan1_v                 (time) float32           Peltier fan 1 voltage [V]
    peltier_fan1_i                 (time) float32           Peltier fan 1 current [A]
    peltier_fan2_v                 (time) float32           Peltier fan 2 voltage [V]
    peltier_fan2_i                 (time) float32           Peltier fan 2 current [A]
    wind_blower_state              (time) int32             wind blower state [-]
    free_disk_space                (time) int32             free disk space [Mo]
    controller_error_code          (time) int32             controller error code [-]
    positioner_time                (time) float32           positioner GPS time [HHMMSS.mmm]
    positioner_azimuth             (time) float32           positioner azimuth [deg]
    positioner_elevation           (time) float32           positioner elevation [deg]
    soft_thermal_cutoff_threshold  (time) int32             temperature limit of soft cutoff [degC]
    raw_reflectivity               (time, range) float32    uncalibrated radar reflectivity factor [dB]
    raw_velocity                   (time, range) float32    raw Doppler velocity [m/s]
    calibration_table              (time) float32           calibration value accounting for power evolution [dB]
    n30s                           (time) int8              number of profiles used in the post processing 30 [-]
    n120s                          (time) int8              number of profiles used in the post processing 120 [-]
    real_integration_time_30s      (time) int8              time integration for each profile [s]
    real_integration_time_120s     (time) int8              time integration for each profile [s]
    reflectivity                   (time, range) float32    radar reflectivity factor [dBZ]
    velocity                       (time, range) float32    Doppler velocity [m/s]
    background_mask                (time, range) int8       background noise mask [-]
    reflectivity_30s               (time, range) float32    radar reflectivity factor 30s [dBZ]
    velocity_30s                   (time, range) float32    Doppler velocity 30 s [m/s]
    background_mask_30s            (time, range) int8       background noise mask 30s [-]
    reflectivity_120s              (time, range) float32    radar reflectivity factor 120s [dBZ]
    velocity_120s                  (time, range) float32    Doppler velocity 120 s [m/s]
    background_mask_120s           (time, range) int8       background noise mask 120s [-]
```

BASTA global attributes (example, fields are the same for all files but values might differ)
```
title                     BASTA L1 data
radar_mode_name           12m5
source                    Basta Cloud Radar
location                   Mt Helmos
positioner_heading        -1000
institution               LATMOS - CNRS
contact                   julien.delanoe@latmos.ipsl.fr
comment                   The echo signals are mixed with the emitted signal. Then, we apply a FFT.Finally,
                          a pulse pair processing is done.
history                   2013-10, BASTA FPGA
format_version            20
Conventions               CF-1.5
year                      2025
month                     1
day                       15
code_version              1.8.1
information               BASTA-mobile-LATMOS, Mt Helmos campaign
l1_processing_version     4.4
processing_info           data processed:2025-02-10
radar                     BASTA-LATMOS
```