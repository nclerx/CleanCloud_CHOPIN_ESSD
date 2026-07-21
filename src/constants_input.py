from . import os, glob, np, pd, matplotlib, mcolors, plt, datetime

from matplotlib.colors import LinearSegmentedColormap

radars = ['MIRA', 'BASTA', 'MXPol']

def skewness_cmap():
    n_neg = 100
    n_pos = 100
    
    hot_r = plt.cm.RdYlBu_r
    colors_pos_array = hot_r(np.linspace(0, 1, n_pos))
    cmap_neg = plt.cm.Blues
    colors_neg_array = cmap_neg(np.linspace(0, 1, n_neg))
    
    # colors_negative = ['darkblue', 'blue', 'lightblue', 'azure', 'white']
    # cmap_neg = LinearSegmentedColormap.from_list('neg', colors_negative, N=n_neg)
    # colors_neg_array = cmap_neg(np.linspace(0, 1, n_neg))
    
    colors_combined = np.vstack([colors_neg_array, colors_pos_array])
    return LinearSegmentedColormap.from_list('custom_skw', colors_combined)


BASTAmode = '25m'  # 12m5, 25m or 100m
BASTAmodes = ['12m5', '25m', '100m_18km']

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024'
dirs = {'MXPol': os.path.join(base_dir, 'MXPol/proc_data/L1_new'),
        'MIRA':  os.path.join(base_dir, 'MIRA/L1'),
        'BASTA': os.path.join(base_dir, f'BASTA/L1/{BASTAmode}'),
        'ERA': os.path.join(base_dir, 'ERA5/netcdf'),
        'peaktree': os.path.join(base_dir, 'peaktree'),
        'zarr': os.path.join(base_dir, 'zarr'),
        'HALO': os.path.join(base_dir, 'HALO'),
        'DEM': f"{base_dir}/eurodem.tif",
        }

# attenuation, precipitation and MIRA-BASTA DFR corrections
att_fn = os.path.join(base_dir, "ERA5", "ERA_attenuation_PAMTRA.nc")
df_precip = pd.read_excel(f"{base_dir}/precip_temp_stats.xlsx", index_col=0)
DFR_files = [f for f in glob.glob(os.path.join(base_dir, 'radar_calib/cloudtopDFR/*')) if os.path.isfile(f)]
DFR_fn = f"{base_dir}/radar_calib/cloudtopDFR/MIRA-BASTA_DFRs.csv"

latlon_Helmos = (38.007, 22.196)
radar_altitude = 1690 # m a.s.l.

start_date = '2024-10-18'
end_date = '2025-01-24'

G = 9.80665 # gravitational constant [m/s^2]
R_earth = 6378e3 # earth radius [m]
R_D = 287.06

RHOHV_thres = 0.5   # rho-HV threshold for calculating MXPol Kdp
SNRH_thres = 0      # SNR threshold for calculating MXPol Kdp

hmax_m = 10e3       # maximum plot height in meters
maskname = 'clean_mask'

common_range_spacing = 25 # meters
common_range = np.arange(common_range_spacing, 15e3, common_range_spacing) # 0-15 km elevation window at a 25 m resolution

temps = [-50, -40, -30, -20, -10, 0] # ° Celsius

MIRA_lapserate = -0.00657895 # degC/m - determined from MIRA temperature data

frequencies = {
    'MXPol': 9.41,
    'MIRA': 35.2,
    'BASTA': 94.95
}

calibrationvalues = {
    'MXPol': (1.66, 0.69), 
    'BASTA': (3.35, 0.91)
} # mean, standard deviation

cloudtop_params = (
    500,    # total height from top of cloud (m)
    200,    # selected height (m) / gap height
    # 0       # offset from top of cloud (m) - not necessary/used anymore since BASTA background mask now properly applied
)

cloudtop_criteria = {
    'DFR_variance_threshold': 4,
    'min_valid_points': 300,
    'min_correlation': 0.7,
    'min_values': 12, # time-moving window (minutes) to calculate DFR variance (for a 5 second time resolution)
    'time_window': 20, # time-moving window size (seconds), following Tridon et al. (2020)
    'range_window': 150, # range-moving window size (m), following Tridon et al. (2020)
    'range_W': (-30, -10), # as per Dias Neto et al. (2019)
    'range_X': (-20, -5), # as per Dias Neto et al. (2019)
}

all_variables_MIRA = ['nfft', 'prf', 'NyquistVelocity', 'nave', 'ovl', 'zrg', 'rg0',
 'drg', 'lambda', 'microsec', 'tpow', 'npw1', 'npw2', 'cpw1', 'cpw2', 'grst',
 'azi', 'elv', 'aziv', 'northangle', 'elvv', 'LO_Frequency', 'DetuneFine', 'SNRg',
 'VELg', 'RMSg', 'LDRg', 'NPKg', 'SNRcx', 'RHO', 'RHOwav', 'DPS', 'LDRnormal',
 'HSDco', 'HSDcx', 'Zg', 'Zcx', 'Z', 'RR', 'LWC', 'Wcorr', 'PIA', 'AttnRain',
 'ISDRco', 'ISDRcx', 'Saturatedco', 'Saturatedcx', 'SPCco', 'SPCcx', 'NvD',
 'DropSize', 'SPCcocxRe', 'SPCcocxIm', 'MRMco', 'MRMcx', 'RadarConst', 'SNRCorFaCo',
 'SNRCorFaCx', 'SKWg']
all_variables_MXPol = ['azimuth', 'elevation', 'Zh', 'Zdr', 'Rhohv', 'RVel', 'Sw', 
 'SNRh', 'SNRv', 'Psidp', 'sPowH', 'sPowV', 'sVel', 'Signal_h', 'Signal_v', 'Mask', 
 'Zh_full', 'Zdr_full', 'Rhohv_full', 'RVel_full', 'Sw_full', 'SNRh_full', 'SNRv_full', 
 'Psidp_full', 'Signal_h_full', 'Signal_v_full', 'sweep_number', 'fixed_angle', 
 'sweep_start_ray_index', 'sweep_end_ray_index', 'sweep_mode', 'nyquist_velocity', 
 'latitude', 'longitude', 'altitude', 'time_coverage_start', 'time_coverage_end', 
 'time_reference', 'volume_number', 'time', 'range']

variables_MIRA = ['elv', 'Z', 'VELg', 'RMSg', 'SKWg', 'SNRg', 'LDRg', 'RHO']
variables_MIRA_refl = ['Z', 'Zg', 'Zcx', 'LDRg']
variables_MXPol = ['time', 'range', 'nfft', 'Zh', 'Zdr', 'RVel', 'Rhohv', 'Sw', 'sPowH', 'sPowV', 'sVel', 'SNRh', 'SNRv']

variables = {
    'MXPol_all': ['azimuth', 'elevation', 'Zh', 'Zdr', 'Rhohv', 'RVel', 'Sw', 
                  'SNRh', 'SNRv', 'Psidp', 'sPowH', 'sPowV', 'sVel', 'Signal_h', 'Signal_v', 'Mask', 
                  'Zh_full', 'Zdr_full', 'Rhohv_full', 'RVel_full', 'Sw_full', 'SNRh_full', 'SNRv_full', 
                  'Psidp_full', 'Signal_h_full', 'Signal_v_full', 'sweep_number', 'fixed_angle', 
                  'sweep_start_ray_index', 'sweep_end_ray_index', 'sweep_mode', 'nyquist_velocity', 
                  'latitude', 'longitude', 'altitude', 'time_coverage_start', 'time_coverage_end', 
                  'time_reference', 'volume_number', 'time', 'range'],
    'MXPol': ['time', 'range', 'nfft', 'Zh', 'Zdr', 'RVel', 'Rhohv', 'Sw', 'sPowH', 'sPowV', 'sVel', 'SNRh', 'SNRv'],
    'MIRA_all': ['nfft', 'prf', 'NyquistVelocity', 'nave', 'ovl', 'zrg', 'rg0', 
                 'drg', 'lambda', 'microsec', 'tpow', 'npw1', 'npw2', 'cpw1', 'cpw2', 'grst',
                 'azi', 'elv', 'aziv', 'northangle', 'elvv', 'LO_Frequency', 'DetuneFine', 'SNRg',
                 'VELg', 'RMSg', 'LDRg', 'NPKg', 'SNRcx', 'RHO', 'RHOwav', 'DPS', 'LDRnormal',
                 'HSDco', 'HSDcx', 'Zg', 'Zcx', 'Z', 'RR', 'LWC', 'Wcorr', 'PIA', 'AttnRain',
                 'ISDRco', 'ISDRcx', 'Saturatedco', 'Saturatedcx', 'SPCco', 'SPCcx', 'NvD', 
                 'DropSize', 'SPCcocxRe', 'SPCcocxIm', 'MRMco', 'MRMcx', 'RadarConst', 'SNRCorFaCo', 
                 'SNRCorFaCx', 'SKWg'],
    'MIRA': ['elv', 'Z', 'VELg', 'RMSg', 'SKWg', 'SNRg', 'LDRg', 'RHO', 'NKPkg'],
    'BASTA_all': ['elevation', 'carrier_frequency', 'pulse_width', 'gen_sampling_rate', 'gen_chirp_starting_frequency', 
                  'gen_chirp_final_frequency', 'ref_sampling_rate', 'ref_chirp_starting_frequency', 'ref_chirp_final_frequency',
                  'gate_length', 'integration_time', 'decimation', 'ambiguous_velocity', 'latitude', 'longitude', 'altitude',
                  'power_v1', 'power_i1', 'power_v2', 'power_i2', 'radar_box_t', 'radar_amplifier_t', 'radar_air_flow',
                  'radar_humidity', 'radar_yaw', 'radar_pitch', 'radar_roll', 'radar_transmitted_power', 'radar_oscillator',
                  'radar_vco_frequency', 'radar_phase_lock', 'radar_thermal_cutoff_50', 'peltier_set_t', 'peltier_sensor_t',
                  'peltier_fet_t', 'peltier_output', 'peltier_fan1_v', 'peltier_fan1_i', 'peltier_fan2_v', 'peltier_fan2_i',
                  'wind_blower_state', 'free_disk_space', 'controller_error_code', 'positioner_time', 'positioner_azimuth',
                  'positioner_elevation', 'soft_thermal_cutoff_threshold', 'raw_reflectivity', 'raw_velocity', 'calibration_table',
                  'n30s', 'n120s', 'real_integration_time_30s', 'real_integration_time_120s', 'reflectivity', 'velocity',
                  'background_mask', 'reflectivity_30s', 'velocity_30s', 'background_mask_30s', 'reflectivity_120s', 'velocity_120s',
                  'background_mask_120s', 'clean_mask'],
    'BASTA': ['elevation', 'reflectivity', 'velocity', 'clean_mask', 'background_mask']

}

variables_to_plot = {
    'MXPol': ['Zh', 'RVel', 'Zdr'],
    'MIRA': ['Z', 'VELg', 'LDRg'],
    'BASTA': ['reflectivity', 'velocity'],
}
variables_type_map = {
    'reflectivity': ['reflectivity', 'Z', 'Zh', 'Zh_full', 'z', 'sZco', 'sZcx'],
    'velocity': ['velocity', 'VELg', 'velocity_corrected', 'RVel', 'v', 'skew'],
    'other': ['LDRg', 'Zdr', 'Rhohv', 'DFR_KaW', 'DFR_XKa', 'no_nodes', 'width', 'LDR', 'prominence', 'linear_depolarization_ratio', 'RMSg', 'ABL', 'turbulence', 'EDR', 'L', ]
}
variables_to_plot_MIRA = ['Z', 'VELg', 'LDRg']
variables_to_plot_BASTA = ['reflectivity', 'velocity']
variables_to_plot_MXPol = ['Zh', 'RVel', 'Zdr', 'Rhohv']
variables_to_plot_pyart = ['reflectivity', 'velocity', 'linear_depolarization_ratio']
variables_to_plot_pyart_all = ['reflectivity', 'velocity', 'linear_depolarization_ratio', 'number_of_peaks', 'skewness', 'spectrum_width', 'differential_phase']

variables_peaktree = ['no_nodes',  'width', 'prominence', 'skew'] # other available peakTree variables 'Z', 'v', 'LDR'

Zdr_fn = os.path.join(base_dir, 'MXPol/Zdr_calib_202606/corrections_CHOPIN_2024_full.txt') # updated 2026.06.18 after rerunning MXPol processing
MXPol_Zdr_corrections = pd.read_csv(Zdr_fn, header=0)
MXPol_Zdr_corrections['datetime'] = pd.to_datetime(MXPol_Zdr_corrections['datetime'], format="%Y-%m-%d %H:%M:%S")

att_fn = os.path.join(base_dir, 'ERA5/ERA_attenuation.nc')

nl = '\n'
headers = ['t_start', 't_end', 'CC_dB', 'std_dev', 'data_perc_used', 'h_min_max', 'lim_Ze_sensitivity', 'lim_gradient', 'solution_found', 'limits', 'slope', 'r_squared', 'rmse']

pltConfig = {
    'Z': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MIRA reflectivity [dBZ]', 'label_short_unit': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'Zh': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MXPol reflectivity [dBZ]', 'label_short_unit': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'Zdr': {'vmin': -0.5, 'vmax': 4, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MXPol Z$_{dr}$ [dBZ]', 'label_short_unit': 'Z$_{dr}$ [dBZ]', 'label_short': 'reflectivity'},
    'sZco': {'vmin': -35, 'vmax': 15, 'cmap': matplotlib.colormaps['viridis'], 'label': 'Spectral Ze [dBsZ]', 'label_short_unit': 'spectral Ze [dBsZ]', 'label_short': 'spectral reflectivity'},
    'reflectivity': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'BASTA reflectivity [dBZ]', 'label_short_unit': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'reflectivity_attn_DFR_corrected': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'BASTA reflectivity (corrected) [dBZ]', 'label_short': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'reflectivity_attn_corrected': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'BASTA reflectivity (attenuation-corrected) [dBZ]', 'label_short_unit': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'reflectivity_30s': {'vmin': -60, 'vmax': 30, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'BASTA reflectivity (30s) [dBZ]', 'label_short_unit': 'reflectivity [dBZ]', 'label_short': 'reflectivity'},
    'VELg': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'MIRA V$_{Dopp}$ [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'velocity': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'BASTA V$_{Dopp}$ [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'velocity_30s': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'BASTA V$_{Dopp}$ (30s) [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'RVel': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'MXPol V$_{Dopp}$ [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'v': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'Doppler velocity [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'velocity_corrected': {'vmin': -8, 'vmax': 5, 'cmap': matplotlib.colormaps['bwr'], 'norm': mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5), 'label': 'BASTA V$_{Dopp}$ [m s$^{-1}$]', 'label_short_unit': 'Doppler velocity [m s$^{-1}$]', 'label_short': 'Doppler velocity'},
    'Rhohv': {'vmin': 0, 'vmax': 1, 'cmap': matplotlib.colormaps['viridis_r'], 'label': r'MXPol H-V correlation $\rho_{HV}$', 'label_short_unit': r'$\rho_{HV}$', 'label_short': r'$\rho_{HV}$'},
    'LDRg': {'vmin': -35, 'vmax': -5, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MIRA LDR [dB]', 'label_short_unit': 'LDR [dB]', 'label_short': 'LDR'},
    'linear_depolarization_ratio': {'vmin': -35, 'vmax': -5, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MIRA LDR [dB]', 'label_short_unit': 'LDR [dB]', 'label_short': 'LDR'},
    'LDR': {'vmin': -35, 'vmax': -5, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'Linear depolarisation ratio [dB]', 'label_short_unit': 'LDR [DB]', 'label_short': 'LDR'},
    'sLDR': {'vmin': -35, 'vmax': -5, 'cmap': matplotlib.colormaps['plasma'], 'label': 'sLDR [dBs]', 'label_short_unit': 'spectral LDR [dBs]', 'label_short': 'spectral LDR'},
    'differential_phase': {'vmin': -180, 'vmax': 180, 'cmap': matplotlib.colormaps['Spectral_r'], 'label': r'Differential phase $\Phi_{DP}$ [$^\circ$]', 'label_short_unit': r'$\Phi_{DP}$ [$^\circ$]', 'label_short': r'$\Phi_{DP}$'},
    'DPS': {'vmin': -180, 'vmax': 180, 'cmap': matplotlib.colormaps['viridis_r'], 'label': r'MXPol $\Phi_{DP}$ [$^\circ$]', 'label_short_unit': r'$\Phi_{DP}$ [$^\circ$]', 'label_short': r'$\Phi_{DP}$'},
    'SKWg': {'vmin': -1, 'vmax': 2.5, 'cmap': matplotlib.colormaps['RdYlBu_r'], 'norm': mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2.5), 'label': 'Skewness [-]', 'label_short_unit': 'skewness', 'label_short': 'skewness'},
    'SkewH': {'vmin': -1, 'vmax': 2.5, 'cmap': matplotlib.colormaps['RdYlBu_r'], 'norm': mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2.5), 'label': 'MXPol skewness [-]', 'label_short_unit': 'skewness', 'label_short': 'skewness'},
    'skewness': {'vmin': -1, 'vmax': 2.5, 'cmap':matplotlib.colormaps['RdYlBu_r'], 'norm': mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2.5), 'label': 'Skewness', 'label_short_unit': 'skewness', 'label_short': 'skewness'},
    'skew': {'vmin': -1, 'vmax': 2.5, 'cmap': matplotlib.colormaps['RdYlBu_r'], 'norm': mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2.5), 'label': 'Skewness', 'label_short_unit': 'skewness', 'label_short': 'skewness'},
    'RMSg': {'vmin': 0, 'vmax': 3, 'cmap': matplotlib.colormaps['viridis_r'], 'label': r'Doppler spectrum width [m s$^{-1}$]', 'label_short_unit': 'spectrum width [m s$^{-1}$]', 'label_short': 'spectrum width'},
    'width': {'vmin': 0, 'vmax': 2.5, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'Doppler spectrum width [m s$^{-1}$]', 'label_short_unit': 'spectrum width [m s$^{-1}$]', 'label_short': 'spectrum width'},
    'SNRg': {'vmin': -360, 'vmax': 360, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MIRA SNR [dB]', 'label_short_unit': 'SNR [dB]', 'label_short': 'SNR'},
    'SNRh': {'vmin': -360, 'vmax': 360, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MXPol SNRh [dB]', 'label_short_unit': 'SNRh [dB]', 'label_short': 'SNRh'},
    'SNRv': {'vmin': -360, 'vmax': 360, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'MXPol SNRv [dB]', 'label_short_unit': 'SNRv [dB]', 'label_short': 'SNRv'},
    'DFR_KaW': {'vmin': 0, 'vmax': 15, 'cmap':matplotlib.colormaps['viridis_r'], 'label': r'$DFR_{Ka-W}$ [dB]', 'label_short_unit': r'$DFR_{Ka-W}$ [dB]', 'label_short': r'$DFR_{Ka-W}$'},
    'DFR_KaW_unc': {'vmin': 0, 'vmax': 15, 'cmap':matplotlib.colormaps['viridis_r'], 'label': r'$DFR_{Ka-W}$ (uncorrected) [dB]', 'label_short_unit': r'$DFR_{Ka-W}$ [dB]', 'label_short': r'uncorrected $DFR_{Ka-W}$'},
    'DFR_XKa': {'vmin': 0, 'vmax': 15, 'cmap':matplotlib.colormaps['viridis_r'], 'label': r'$DFR_{X-Ka}$ [dB]', 'label_short_unit': r'$DFR_{X-Ka}$ [dB]', 'label_short':  r'$DFR_{X-Ka}$'}, 
    'no_nodes': {'vmin': -0.5, 'vmax': 5.5, 'cmap': mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat'), 'label': 'Number of peaks', 'label_short_unit': 'number of peaks', 'label_short': 'number of peaks'},
    'NPKg': {'vmin': -0.5, 'vmax': 5.5, 'cmap': mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat'), 'label': 'Number of peaks', 'label_short_unit': 'number of peaks', 'label_short': 'number of peaks'},
    'number_of_peaks': {'vmin': -0.5, 'vmax': 5.5, 'cmap': mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat'), 'label': 'Number of peaks', 'label_short_unit': 'number of peaks', 'label_short': 'number of peaks'},
    'numpeaks': {'vmin': -0.5, 'vmax': 5.5, 'cmap': mcolors.ListedColormap(["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat'), 'label': 'Number of peaks', 'label_short_unit': 'number of peaks', 'label_short': 'number of peaks'},
    'prominence': {'vmin': 0, 'vmax': 2.5, 'cmap': matplotlib.colormaps['viridis_r'], 'label': 'Prominence of peak above threshold [dBZ]', 'label_short_unit': 'peak prominence [dBZ]', 'label_short': 'peak prominence'},
    'EDR': {'vmin': -6, 'vmax': -1, 'cmap': matplotlib.colormaps['jet'], 'label': r'Dissipation rate of turbulent kinetic energy ε [$m^2 s^{-3}$]', 'label_short_unit': 'EDR [$m^2 s^{-3}$]', 'label_short': 'EDR'},
    'L': {'vmin': 0, 'vmax': 4000, 'cmap': matplotlib.colormaps['jet'], 'label': r'Length scale of scattering volume dimension per single sample [$m^2 s^{-3}$]', 'label_short_unit': 'L [$m^2 s^{-3}$]', 'label_short': 'L'},
    'ABL': {'vmin': 0, 'vmax': 9,'cmap': matplotlib.colormaps['tab10_r'], 'label': None, 'label_short_unit': 'ABL classification'},
    'turbulence': {'vmin': 0, 'vmax': 6, 'cmap': matplotlib.colormaps['tab10_r'].resampled(7), 'label': None, 'label_short_unit': 'turbulence', 'label_short': 'turbulence'},
}

radiosonde_dates = ['2024102309', '2024102509', '2024102609', '2024102616', '2024102915', '2024103109',
         '2024110209', '2024110509', '2024110515', '2024111511', '2024111609', '2024111709',
         '2024111809', '2024111815', '2024111909', '2024111915', '2024112009', '2024112109',
         '2024112115', '2024112209', '2024112315', '2024112409', '2024112515', '2024112609',
         '2024112615', '2024112815', '2024112909', '2024112915', '2024113009', '2024113015',
         '2024120115', '2024120209', '2024120215', '2024120509', '2024120515', '2024120609', 
         '2024120615', '2024121315', '2025010815', #'2025012509', '2025012709'
         ]

MXPol_lost_frequency_times = [
     ('20241129-00000', '20241129-111859'),
     ('20241209-061913', '20241209-062252'),
     ('20241210-045727', '20241210-065155'),
     ('20241211-005731', '20241211-131323'),
     ('20241212-065545', '20241212-065937'),
     ('20241213-090121', '20241214-220406'),
     ('20241215-205044', '20241216-084431'),
     ('20241217-014915', '20241220-094708'),
     ('20241222-081854', '20241222-083141'),
     ('20241224-123639', '20241224-123914'),
     ('20250103-072540', '20250103-074247'),
     ('20250104-021318', '20250106-062100'), # from here onwards maybe some files already OK (or bad) before the mentioned files
     ('20250107-010046', '20250107-062514'),
     ('20250111-013936', '20250114-063819'), 
     ('20250115-194048', '20250118-144705'), # plenty of files not recorded properly (0 bytes)
     ('20250120-071620', '20250121-120002', )
 ]

WRF_sims = pd.read_csv(os.path.join(f"{base_dir}/WRF/overview.csv"))
WRF_sims['start'] = pd.to_datetime(WRF_sims['start'], format='%d.%m.%y %H:%M')
WRF_sims['end'] = pd.to_datetime(WRF_sims['end'], format='%d.%m.%y %H:%M')
WRF_sims['period'] = WRF_sims['end'] - WRF_sims['start']

metek_fieldmapping = {
    'Zg': 'reflectivity',
    'Zcx': 'reflectivity_cx',
    'VELg': 'velocity',
    'RMSg': 'spectrum_width',
    'SNRg': 'signal_to_noise_ratio',
    'SNRcx': 'signal_to_noise_ratio_cx',
    'LDRg': 'linear_depolarization_ratio',
    'RHO': 'cross_correlation_ratio',
    'RHO_wav': 'cross_correlation_ratio_wav',
    'DPS': 'differential_phase',
    'SKWg': 'skewness',
    'NPKg': 'number_of_peaks',
    'SNRCorFaCo': 'snr_correction_factor_co',
    'SNRCorFaCx': 'snr_correction_factor_cx',
}

metek_fieldmetadata = {
    'reflectivity': {
        'units': 'dBZ',
        'long_name': 'Equivalent reflectivity factor',
        'standard_name': 'equivalent_reflectivity_factor',
        'valid_min': -80.0,
        'valid_max': 80.0,
    },
    'velocity': {
        'units': 'm/s', 
        'long_name': 'Mean Doppler velocity',
        'standard_name': 'radial_velocity_of_scatterers_away_from_instrument',
        'valid_min': -50.0,
        'valid_max': 50.0,
    },
    'spectrum_width': {
        'units': 'm/s',
        'long_name': 'Doppler spectrum width', 
        'standard_name': 'doppler_spectrum_width',
        'valid_min': 0.0,
        'valid_max': 20.0,
    },
    'signal_to_noise_ratio': {
        'units': 'dB',
        'long_name': 'Signal-to-noise ratio',
        'valid_min': -20.0,
        'valid_max': 80.0,
    },
    'linear_depolarization_ratio': {
        'units': 'dB',
        'long_name': 'Linear depolarization ratio',
        'valid_min': -40.0,
        'valid_max': 0.0,
    },
    'cross_correlation_ratio': {
        'units': '',
        'long_name': 'Cross-correlation coefficient',
        'valid_min': 0.0,
        'valid_max': 1.0,
    },
    'differential_phase': {
        'units': 'degrees',
        'long_name': 'Differential phase shift',
    },
    'skewness': {
        'units': '',
        'long_name': 'Doppler spectrum skewness',
    },
}

metek_attrmapping = {
    # Institution/source info
    'institution': 'institution',
    'source': 'source', 
    'references': 'references',
    'comment': 'comment',
    'history': 'history',
    
    # Instrument info
    'instrument_name': 'instrument_name',
    'instrument_type': 'instrument_type',
    'platform_type': 'platform_type',
    'primary_axis': 'primary_axis',
    'data_level': 'data_level',
    'scan_type': 'scan_type',
    'scan_name': 'scan_name',
    
    # Location - Metek specific names
    'Latitude': 'latitude',
    'lat': 'latitude', 
    'Longitude': 'longitude',
    'lon': 'longitude',
    'Altitude': 'altitude',
    'alt': 'altitude',
    'height': 'altitude',
    
    # Time info
    'time_coverage_start': 'time_coverage_start',
    'time_coverage_end': 'time_coverage_end',
    'time_reference': 'time_reference',
    
    # Radar parameters
    'frequency': 'frequency',
    'wavelength': 'wavelength', 
    'pulse_width': 'pulse_width',
    'prt': 'prt',
    'prt_mode': 'prt_mode',
    'nyquist_velocity': 'nyquist_velocity',
    'unambiguous_range': 'unambiguous_range',
    'beamwidth_h': 'radar_beam_width_h',
    'beamwidth_v': 'radar_beam_width_v',
    'antenna_gain': 'radar_antenna_gain',
    
    # Processing info
    'signal_processor': 'signal_processor',
    'calibration_constant': 'radar_constant',
    'receiver_gain': 'receiver_gain',
    'noise_floor': 'noise_floor',
    'range_resolution': 'range_resolution',
    'gate_spacing': 'range_resolution',
    
    # Quality control
    'qc_procedure': 'qc_procedure',
    'qc_comment': 'qc_comment',
    
    # Metek specific
    'software_version': 'signal_processor_version',
    'config_file': 'instrument_parameters',
    'measurement_type': 'scan_type',
}
# coordinate mappings
metek_coordattrs = {
    'time': {
        'standard_name': 'time',
        'long_name': 'Time in seconds since volume start',
        'units': 'seconds since start_time',
        'calendar': 'gregorian',
    },
    'range': {
        'standard_name': 'projection_range_coordinate', 
        'long_name': 'Range to center of gate',
        'units': 'm',
        'spacing_is_constant': 'true',
        'meters_to_center_of_first_gate': 'first_gate_distance',
        'meters_between_gates': 'gate_spacing',
    },
    'azimuth': {
        'standard_name': 'ray_azimuth_angle',
        'long_name': 'Azimuth angle from true north',
        'units': 'degrees',
        'positive': 'clockwise_from_north',
    },
    'elevation': {
        'standard_name': 'ray_elevation_angle', 
        'long_name': 'Elevation angle from horizontal plane',
        'units': 'degrees',
        'positive': 'up',
    },
    'latitude': {
        'standard_name': 'latitude',
        'long_name': 'Latitude of radar',
        'units': 'degrees_north',
    },
    'longitude': {
        'standard_name': 'longitude',
        'long_name': 'Longitude of radar', 
        'units': 'degrees_east',
    },
    'altitude': {
        'standard_name': 'altitude',
        'long_name': 'Altitude of radar above mean sea level',
        'units': 'm',
        'positive': 'up',
    }
}

PBLdays = ['2024-10-18', '2024-10-29', '2024-10-30', '2024-10-31', '2024-11-01', '2024-11-02', '2024-11-03', 
           '2024-11-04', '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08', '2024-11-09', '2024-11-10', 
           '2024-11-25', '']


default_variable_maps = {
    'MIRA': {
        'reflectivity': 'Z',
        'velocity': 'VELg',
        'snrh': 'SNRg',
        'snrv': 'SNRcx',
        'spectrumwidth': 'RMSg',
        'numpeaks': 'NPKg',
        'skewness': 'SKWg',
        'LDR': 'LDRg',
        'cloudtop': 'cloud_top_height_continuous_mean',
        'doppler': 'doppler',
        'sZh': 'sZco',
        'sZv': 'sZcx',
        'sLDR': 'sLDR',
    },
    'BASTA': {
        # prefer attenuation+DFR corrected reflectivity if available
        'reflectivity': 'reflectivity_attn_DFR_corrected',
        'reflectivity_raw': 'reflectivity',
        'time_original': 'time_original',
        'velocity': 'velocity',
        'background_mask': 'background_mask',
        'cloudtop': 'cloud_top_height_continuous_mean',
        'z_30s': 'reflectivity_30s',
        'vel_30s': 'velocity_30s',
        'bg_30s': 'background_mask_30s',
        'z_120s': 'reflectivity_120s',
        'vel_120s': 'velocity_120s',
        'bg_120s': 'background_mask_120s'
    },
    'MXPol': {
        'reflectivity': 'Zh',
        'reflectivity_raw': 'Zh_full',
        'velocity': 'RVel',
        'snrh': 'SNRh',
        'snrv': 'SNRv',
        'snrh_raw': 'SNRh_full',
        'snrv_raw': 'SNRv_full',
        'spectrumwidth': 'Sw',
        'rhohv': 'Rhohv',
        'zdr': 'Zdr',
        'kdp': 'Kdp',
        'phidp': 'Phidp',
        'doppler': 'nfft',
        'sZh': 'sZH',
        'sZv': 'sZV',
    },
    'peaktree': {
        'reflectivity': 'Z',
        'velocity': 'v',
        'spectrumwidth': 'width',
        'numpeaks': 'no_nodes',
        'skewness': 'skew',
        'LDR': 'LDR',
        'cloudtop': 'cloud_top_height_continuous_mean',
        'doppler': 'vel'
    }
}

doppler_dim_map = {
    'MIRA': 'doppler',
    'MXPol': 'nfft',
    'peaktree': None,
    'BASTA': None,
}

variable_groups = {
    'reflectivity': ['reflectivity_mean', 'reflectivity_min', 'reflectivity_max', 
                     'reflectivity_std', 'reflectivity_median', 'reflectivity_count'],
    'velocity': ['velocity_mean', 'velocity_min', 'velocity_max', 
                 'velocity_std', 'velocity_median', 'velocity_count'],
    'spectrumwidth': ['spectrum_width_mean', 'spectrum_width_min', 'spectrum_width_max', 
                       'spectrum_width_std', 'spectrum_width_median', 'spectrum_width_count'],
    'numpeaks': ['num_peaks_mean', 'num_peaks_min', 'num_peaks_max', 
                  'num_peaks_std', 'num_peaks_median', 'num_peaks_count'],
    'cloudtop': ['cloudtop_mean', 'cloudtop_min', 'cloudtop_max', 
                 'cloudtop_std', 'cloudtop_median', 'cloudtop_count']
}

zarr_variables = {
    'MIRA': {
        "reflectivity": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "velocity": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "skewness": dict(dims=("time", "range"), range=(-2.5, 2.5), precision=0.001),
        "spectrumwidth": dict(dims=("time", "range"), range=(0, 6), precision=0.001),
        "LDR": dict(dims=("time", "range"), range=(-35, 5), precision=0.1),
        "numpeaks": dict(dims=("time", "range"), range=(1, 10), dtype=np.uint8),
        "cloudtop": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "snrh": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "snrv": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "zdi": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "DFR_XKa": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "DFR_KaW": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "is_precip": dict(dims=("time", ), dtype=bool),
        "sZh": dict(dims=("time", "range", "doppler"), range=(-45, 15), precision=0.01),
        "sZv": dict(dims=("time", "range", "doppler"), range=(-45, 15), precision=0.01),
        "sLDR": dict(dims=("time", "range", "doppler"), range=(-35, 5), precision=0.01)
        },
    'MXPol': {
        "reflectivity": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "reflectivity_raw": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "velocity": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "spectrumwidth": dict(dims=("time", "range"), range=(0, 3), precision=0.01),
        "zdr": dict(dims=("time", "range"), range=(0, 5), precision=0.01),
        "rhohv": dict(dims=("time", "range"), range=(0, 1), precision = 0.01),
        "snrh": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "snrv": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "snrh_raw": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "snrv_raw": dict(dims=('time', 'range'), range=(-30, 70), precision=0.1),
        "zdi": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "DFR_XKa": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "DFR_XW": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "is_precip": dict(dims=("time", ), dtype=bool),
        "sZh": dict(dims=("time", "range", "doppler"), range=(-45, 15), precision=0.01),
        "sZv": dict(dims=("time", "range", "doppler"), range=(-45, 15), precision=0.01),
        },
    'BASTA': {
        "reflectivity": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "reflectivity_raw": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "time_original": dict(dims=("time",), dtype='datetime', units="seconds since 1970-01-01T00:00:00Z"),
        "z_30s": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "z_120s": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "velocity": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "vel_30s": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "vel_120s": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "cloudtop": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "zdi": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "DFR_KaW": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "DFR_XW": dict(dims=("time", "range"), range=(0, 15), precision=0.1),
        "is_precip": dict(dims=("time", ), dtype=bool),
        "background_mask": dict(dims=("time", "range"), dtype=np.uint8),
        "bg_30s": dict(dims=("time", "range"), dtype=np.uint8),
        "bg_120s": dict(dims=("time", "range"), dtype=np.uint8),
        },
    'peaktree': {
        "reflectivity": dict(dims=("time", "range"), range=(-60, 45), precision=0.1),
        "velocity": dict(dims=("time", "range"), range=(-10, 10), precision=0.1),
        "skewness": dict(dims=("time", "range"), range=(-2.5, 2.5), precision=0.01),
        "spectrumwidth": dict(dims=("time", "range"), range=(0, 3), precision=0.01),
        "LDR": dict(dims=("time", "range"), range=(-35, 5), precision=0.1),
        "numpeaks": dict(dims=("time", "range"), range=(1, 10), dtype=np.uint8),
        "cloudtop": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "zdi": dict(dims=("time",), range=(0, 15000), dtype=np.uint16),
        "is_precip": dict(dims=("time", ), dtype=bool)
        }
    }

radar_ranges = {
    'MXPol': [  376.95,   406.95,   436.95,   466.95,   496.95,   526.95,
         556.95,   586.95,   616.95,   646.95,   676.95,   706.95,
         736.95,   766.95,   796.95,   826.95,   856.95,   886.95,
         916.95,   946.95,   976.95,  1006.95,  1036.95,  1066.95,
        1096.95,  1126.95,  1156.95,  1186.95,  1216.95,  1246.95,
        1276.95,  1306.95,  1336.95,  1366.95,  1396.95,  1426.95,
        1456.95,  1486.95,  1516.95,  1546.95,  1576.95,  1606.95,
        1636.95,  1666.95,  1696.95,  1726.95,  1756.95,  1786.95,
        1816.95,  1846.95,  1876.95,  1906.95,  1936.95,  1966.95,
        1996.95,  2026.95,  2056.95,  2086.95,  2116.95,  2146.95,
        2176.95,  2206.95,  2236.95,  2266.95,  2296.95,  2326.95,
        2356.95,  2386.95,  2416.95,  2446.95,  2476.95,  2506.95,
        2536.95,  2566.95,  2596.95,  2626.95,  2656.95,  2686.95,
        2716.95,  2746.95,  2776.95,  2806.95,  2836.95,  2866.95,
        2896.95,  2926.95,  2956.95,  2986.95,  3016.95,  3046.95,
        3076.95,  3106.95,  3136.95,  3166.95,  3196.95,  3226.95,
        3256.95,  3286.95,  3316.95,  3346.95,  3376.95,  3406.95,
        3436.95,  3466.95,  3496.95,  3526.95,  3556.95,  3586.95,
        3616.95,  3646.95,  3676.95,  3706.95,  3736.95,  3766.95,
        3796.95,  3826.95,  3856.95,  3886.95,  3916.95,  3946.95,
        3976.95,  4006.95,  4036.95,  4066.95,  4096.95,  4126.95,
        4156.95,  4186.95,  4216.95,  4246.95,  4276.95,  4306.95,
        4336.95,  4366.95,  4396.95,  4426.95,  4456.95,  4486.95,
        4516.95,  4546.95,  4576.95,  4606.95,  4636.95,  4666.95,
        4696.95,  4726.95,  4756.95,  4786.95,  4816.95,  4846.95,
        4876.95,  4906.95,  4936.95,  4966.95,  4996.95,  5026.95,
        5056.95,  5086.95,  5116.95,  5146.95,  5176.95,  5206.95,
        5236.95,  5266.95,  5296.95,  5326.95,  5356.95,  5386.95,
        5416.95,  5446.95,  5476.95,  5506.95,  5536.95,  5566.95,
        5596.95,  5626.95,  5656.95,  5686.95,  5716.95,  5746.95,
        5776.95,  5806.95,  5836.95,  5866.95,  5896.95,  5926.95,
        5956.95,  5986.95,  6016.95,  6046.95,  6076.95,  6106.95,
        6136.95,  6166.95,  6196.95,  6226.95,  6256.95,  6286.95,
        6316.95,  6346.95,  6376.95,  6406.95,  6436.95,  6466.95,
        6496.95,  6526.95,  6556.95,  6586.95,  6616.95,  6646.95,
        6676.95,  6706.95,  6736.95,  6766.95,  6796.95,  6826.95,
        6856.95,  6886.95,  6916.95,  6946.95,  6976.95,  7006.95,
        7036.95,  7066.95,  7096.95,  7126.95,  7156.95,  7186.95,
        7216.95,  7246.95,  7276.95,  7306.95,  7336.95,  7366.95,
        7396.95,  7426.95,  7456.95,  7486.95,  7516.95,  7546.95,
        7576.95,  7606.95,  7636.95,  7666.95,  7696.95,  7726.95,
        7756.95,  7786.95,  7816.95,  7846.95,  7876.95,  7906.95,
        7936.95,  7966.95,  7996.95,  8026.95,  8056.95,  8086.95,
        8116.95,  8146.95,  8176.95,  8206.95,  8236.95,  8266.95,
        8296.95,  8326.95,  8356.95,  8386.95,  8416.95,  8446.95,
        8476.95,  8506.95,  8536.95,  8566.95,  8596.95,  8626.95,
        8656.95,  8686.95,  8716.95,  8746.95,  8776.95,  8806.95,
        8836.95,  8866.95,  8896.95,  8926.95,  8956.95,  8986.95,
        9016.95,  9046.95,  9076.95,  9106.95,  9136.95,  9166.95,
        9196.95,  9226.95,  9256.95,  9286.95,  9316.95,  9346.95,
        9376.95,  9406.95,  9436.95,  9466.95,  9496.95,  9526.95,
        9556.95,  9586.95,  9616.95,  9646.95,  9676.95,  9706.95,
        9736.95,  9766.95,  9796.95,  9826.95,  9856.95,  9886.95,
        9916.95,  9946.95,  9976.95, 10006.95, 10036.95, 10066.95,
       10096.95, 10126.95, 10156.95, 10186.95, 10216.95, 10246.95,
       10276.95, 10306.95, 10336.95, 10366.95, 10396.95, 10426.95,
       10456.95, 10486.95, 10516.95, 10546.95, 10576.95, 10606.95,
       10636.95, 10666.95, 10696.95, 10726.95, 10756.95, 10786.95,
       10816.95, 10846.95, 10876.95, 10906.95, 10936.95, 10966.95,
       10996.95, 11026.95, 11056.95, 11086.95, 11116.95, 11146.95,
       11176.95, 11206.95, 11236.95, 11266.95, 11296.95, 11326.95,
       11356.95, 11386.95, 11416.95, 11446.95, 11476.95, 11506.95,
       11536.95, 11566.95, 11596.95, 11626.95, 11656.95, 11686.95,
       11716.95, 11746.95, 11776.95, 11806.95, 11836.95, 11866.95,
       11896.95, 11926.95, 11956.95, 11986.95, 12016.95, 12046.95,
       12076.95, 12106.95, 12136.95, 12166.95, 12196.95, 12226.95,
       12256.95, 12286.95],
    'MIRA': [  155.896 ,   187.0752,   218.2544,   249.4336,   280.6128,
         311.792 ,   342.9712,   374.1504,   405.3296,   436.5088,
         467.688 ,   498.8672,   530.0464,   561.2256,   592.4048,
         623.584 ,   654.7632,   685.9424,   717.1216,   748.3008,
         779.48  ,   810.6592,   841.8384,   873.0176,   904.1968,
         935.376 ,   966.5552,   997.7344,  1028.9136,  1060.0928,
        1091.272 ,  1122.4512,  1153.6304,  1184.8096,  1215.9888,
        1247.168 ,  1278.3472,  1309.5264,  1340.7056,  1371.8848,
        1403.064 ,  1434.2432,  1465.4224,  1496.6016,  1527.7808,
        1558.96  ,  1590.1392,  1621.3184,  1652.4976,  1683.6768,
        1714.856 ,  1746.0352,  1777.2144,  1808.3936,  1839.5728,
        1870.752 ,  1901.9312,  1933.1104,  1964.2896,  1995.4688,
        2026.648 ,  2057.8271,  2089.0063,  2120.1855,  2151.3647,
        2182.544 ,  2213.7231,  2244.9023,  2276.0815,  2307.2607,
        2338.44  ,  2369.6191,  2400.7983,  2431.9775,  2463.1567,
        2494.336 ,  2525.5151,  2556.6943,  2587.8735,  2619.0527,
        2650.232 ,  2681.4111,  2712.5903,  2743.7695,  2774.9487,
        2806.128 ,  2837.3071,  2868.4863,  2899.6655,  2930.8447,
        2962.024 ,  2993.2031,  3024.3823,  3055.5615,  3086.7407,
        3117.92  ,  3149.099 ,  3180.2783,  3211.4575,  3242.6367,
        3273.816 ,  3304.995 ,  3336.1743,  3367.3535,  3398.5327,
        3429.712 ,  3460.891 ,  3492.0703,  3523.2495,  3554.4287,
        3585.608 ,  3616.787 ,  3647.9663,  3679.1455,  3710.3247,
        3741.504 ,  3772.683 ,  3803.8623,  3835.0415,  3866.2207,
        3897.4   ,  3928.579 ,  3959.7583,  3990.9375,  4022.1167,
        4053.296 ,  4084.475 ,  4115.6543,  4146.8335,  4178.0127,
        4209.192 ,  4240.371 ,  4271.5503,  4302.7295,  4333.9087,
        4365.088 ,  4396.267 ,  4427.4463,  4458.6255,  4489.8047,
        4520.984 ,  4552.163 ,  4583.3423,  4614.5215,  4645.7007,
        4676.88  ,  4708.059 ,  4739.2383,  4770.4175,  4801.5967,
        4832.776 ,  4863.955 ,  4895.1343,  4926.3135,  4957.4927,
        4988.672 ,  5019.851 ,  5051.0303,  5082.2095,  5113.3887,
        5144.568 ,  5175.747 ,  5206.9263,  5238.1055,  5269.2847,
        5300.464 ,  5331.643 ,  5362.8223,  5394.0015,  5425.1807,
        5456.36  ,  5487.539 ,  5518.7183,  5549.8975,  5581.0767,
        5612.256 ,  5643.435 ,  5674.6143,  5705.7935,  5736.9727,
        5768.152 ,  5799.331 ,  5830.5103,  5861.6895,  5892.8687,
        5924.048 ,  5955.227 ,  5986.4062,  6017.5854,  6048.7646,
        6079.944 ,  6111.123 ,  6142.3022,  6173.4814,  6204.6606,
        6235.84  ,  6267.019 ,  6298.198 ,  6329.3774,  6360.5566,
        6391.736 ,  6422.915 ,  6454.094 ,  6485.2734,  6516.4526,
        6547.632 ,  6578.811 ,  6609.99  ,  6641.1694,  6672.3486,
        6703.528 ,  6734.707 ,  6765.886 ,  6797.0654,  6828.2446,
        6859.424 ,  6890.603 ,  6921.782 ,  6952.9614,  6984.1406,
        7015.32  ,  7046.499 ,  7077.678 ,  7108.8574,  7140.0366,
        7171.216 ,  7202.395 ,  7233.574 ,  7264.7534,  7295.9326,
        7327.112 ,  7358.291 ,  7389.47  ,  7420.6494,  7451.8286,
        7483.008 ,  7514.187 ,  7545.366 ,  7576.5454,  7607.7246,
        7638.904 ,  7670.083 ,  7701.262 ,  7732.4414,  7763.6206,
        7794.8   ,  7825.979 ,  7857.158 ,  7888.3374,  7919.5166,
        7950.696 ,  7981.875 ,  8013.054 ,  8044.2334,  8075.4126,
        8106.592 ,  8137.771 ,  8168.95  ,  8200.129 ,  8231.309 ,
        8262.488 ,  8293.667 ,  8324.846 ,  8356.025 ,  8387.205 ,
        8418.385 ,  8449.5625,  8480.742 ,  8511.922 ,  8543.102 ,
        8574.279 ,  8605.459 ,  8636.639 ,  8667.818 ,  8698.996 ,
        8730.176 ,  8761.355 ,  8792.535 ,  8823.713 ,  8854.893 ,
        8886.072 ,  8917.252 ,  8948.43  ,  8979.609 ,  9010.789 ,
        9041.969 ,  9073.146 ,  9104.326 ,  9135.506 ,  9166.686 ,
        9197.863 ,  9229.043 ,  9260.223 ,  9291.402 ,  9322.58  ,
        9353.76  ,  9384.939 ,  9416.119 ,  9447.297 ,  9478.477 ,
        9509.656 ,  9540.836 ,  9572.014 ,  9603.193 ,  9634.373 ,
        9665.553 ,  9696.73  ,  9727.91  ,  9759.09  ,  9790.27  ,
        9821.447 ,  9852.627 ,  9883.807 ,  9914.986 ,  9946.164 ,
        9977.344 , 10008.523 , 10039.703 , 10070.881 , 10102.061 ,
       10133.24  , 10164.42  , 10195.598 , 10226.777 , 10257.957 ,
       10289.137 , 10320.314 , 10351.494 , 10382.674 , 10413.854 ,
       10445.031 , 10476.211 , 10507.391 , 10538.57  , 10569.748 ,
       10600.928 , 10632.107 , 10663.287 , 10694.465 , 10725.645 ,
       10756.824 , 10788.004 , 10819.182 , 10850.361 , 10881.541 ,
       10912.721 , 10943.898 , 10975.078 , 11006.258 , 11037.4375,
       11068.615 , 11099.795 , 11130.975 , 11162.154 , 11193.332 ,
       11224.512 , 11255.691 , 11286.871 , 11318.049 , 11349.229 ,
       11380.408 , 11411.588 , 11442.766 , 11473.945 , 11505.125 ,
       11536.305 , 11567.482 , 11598.662 , 11629.842 , 11661.021 ,
       11692.199 , 11723.379 , 11754.559 , 11785.738 , 11816.916 ,
       11848.096 , 11879.275 , 11910.455 , 11941.633 , 11972.8125,
       12003.992 , 12035.172 , 12066.35  , 12097.529 , 12128.709 ,
       12159.889 , 12191.066 , 12222.246 , 12253.426 , 12284.605 ,
       12315.783 , 12346.963 , 12378.143 , 12409.322 , 12440.5   ,
       12471.68  , 12502.859 , 12534.039 , 12565.217 , 12596.396 ,
       12627.576 , 12658.756 , 12689.934 , 12721.113 , 12752.293 ,
       12783.473 , 12814.65  , 12845.83  , 12877.01  , 12908.189 ,
       12939.367 , 12970.547 , 13001.727 , 13032.906 , 13064.084 ,
       13095.264 , 13126.443 , 13157.623 , 13188.801 , 13219.98  ,
       13251.16  , 13282.34  , 13313.518 , 13344.697 , 13375.877 ,
       13407.057 , 13438.234 , 13469.414 , 13500.594 , 13531.773 ,
       13562.951 , 13594.131 , 13625.311 , 13656.49  , 13687.668 ,
       13718.848 , 13750.027 , 13781.207 , 13812.385 , 13843.564 ,
       13874.744 , 13905.924 , 13937.102 , 13968.281 , 13999.461 ,
       14030.641 , 14061.818 , 14092.998 , 14124.178 , 14155.357 ,
       14186.535 , 14217.715 , 14248.895 , 14280.074 , 14311.252 ,
       14342.432 , 14373.611 , 14404.791 , 14435.969 , 14467.148 ,
       14498.328 , 14529.508 , 14560.686 , 14591.865 , 14623.045 ,
       14654.225 , 14685.402 , 14716.582 , 14747.762 , 14778.941 ,
       14810.119 , 14841.299 , 14872.479 , 14903.658 , 14934.836 ,
       14966.016 , 14997.195 ],
    'BASTA_25m': np.arange(25, 18e3+25, 25),
    'BASTA_12m5': np.arange(12.5, 12e3+12.5, 12.5),
    'BASTA_100m_18km': np.arange(100, 18e3+100, 100)
}

doppler_dims = {
    'MIRA': [-10.66145, -10.578157, -10.494865, -10.411572, -10.32828, -10.2449875, -10.161695, -10.0784025, -9.99511, 
                    -9.911818, -9.828525, -9.745232, -9.66194, -9.578647, -9.495355, -9.412062, -9.328769, -9.245477, -9.162184, 
                    -9.078892, -8.995599, -8.912306, -8.829014, -8.745721, -8.662429, -8.579136, -8.495843, -8.412551, -8.329258, 
                    -8.245966, -8.162673, -8.07938, -7.996088, -7.912795, -7.829503, -7.74621, -7.662917, -7.579625, -7.496332, 
                    -7.41304, -7.329747, -7.2464542, -7.163162, -7.0798693, -6.9965773, -6.9132843, -6.8299913, -6.7466993, -6.6634064, 
                    -6.5801144, -6.4968214, -6.4135284, -6.3302364, -6.2469435, -6.1636515, -6.0803585, -5.9970655, -5.9137735, -5.8304806, 
                    -5.7471886, -5.6638956, -5.5806026, -5.4973106, -5.4140177, -5.3307247, -5.2474327, -5.1641407, -5.080847, -4.997555, 
                    -4.914263, -4.830969, -4.747677, -4.664385, -4.581093, -4.497799, -4.414507, -4.331215, -4.247921, -4.164629, 
                    -4.081337, -3.998043, -3.914751, -3.831459, -3.748167, -3.6648731, -3.581581, -3.498289, -3.4149952, -3.3317032, 
                    -3.2484112, -3.1651173, -3.0818253, -2.9985332, -2.9152412, -2.8319473, -2.7486553, -2.6653633, -2.5820694, -2.4987774, 
                    -2.4154854, -2.3321915, -2.2488995, -2.1656075, -2.0823154, -1.9990215, -1.9157295, -1.8324375, -1.7491436, 
                    -1.6658516, -1.5825596, -1.4992657, -1.4159737, -1.3326817, -1.2493896, -1.1660957, -1.0828037, -0.9995117, -0.9162178, 
                    -0.8329258, -0.7496338, -0.6663399, -0.58304787, -0.49975586, -0.41646385, -0.33316994, -0.24987793, -0.16658592, 
                    -0.08329201, 0.0, 0.08329258, 0.16658516, 0.24987775, 0.33317032, 0.4164629, 0.4997555, 0.58304805, 0.66634065, 
                    0.74963325, 0.8329258, 0.9162184, 0.999511, 1.0828036, 1.1660961, 1.2493887, 1.3326813, 1.4159739, 1.4992665, 1.582559, 
                    1.6658516, 1.7491442, 1.8324368, 1.9157294, 1.999022, 2.0823145, 2.1656072, 2.2488997, 2.3321922, 2.415485, 2.4987774, 
                    2.58207, 2.6653626, 2.748655, 2.8319478, 2.9152403, 2.998533, 3.0818255, 3.165118, 3.2484107, 3.3317032, 3.414996, 
                    3.4982884, 3.5815809, 3.6648736, 3.748166, 3.8314588, 3.9147513, 3.998044, 4.0813365, 4.164629, 4.2479215, 4.3312144, 
                    4.414507, 4.4977994, 4.581092, 4.6643844, 4.7476773, 4.83097, 4.9142623, 4.997555, 5.0808473, 5.16414, 5.2474327, 
                    5.330725, 5.4140177, 5.49731, 5.580603, 5.6638956, 5.747188, 5.8304806, 5.913773, 5.997066, 6.0803585, 6.163651, 
                    6.2469435, 6.330236, 6.413529, 6.4968214, 6.580114, 6.6634064, 6.746699, 6.829992, 6.9132843, 6.996577, 7.0798693, 
                    7.1631618, 7.2464547, 7.329747, 7.4130397, 7.496332, 7.5796247, 7.6629176, 7.74621, 7.8295026, 7.912795, 7.996088, 
                    8.07938, 8.162673, 8.245966, 8.329258, 8.412551, 8.495843, 8.579136, 8.662429, 8.745721, 8.829014, 8.912306, 8.995599, 
                    9.078892, 9.162184, 9.245477, 9.328769, 9.412062, 9.495355, 9.578647, 9.66194, 9.745232, 9.828525, 9.911818, 9.99511, 
                    10.0784025, 10.161695, 10.2449875, 10.32828, 10.411572, 10.494865, 10.578157],
    'MXPol': [-11.301304  , -11.212317  , -11.123331  , -11.034345  , -10.945357  , -10.856371  , -10.767385  , -10.678398  ,
              -10.589411  , -10.500424  , -10.411438  , -10.322452  ,  -10.233464  , -10.144478  , -10.055491  ,  -9.966505  ,
              -9.877518  ,  -9.788531  ,  -9.699545  ,  -9.610558  ,   -9.521571  ,  -9.432585  ,  -9.343598  ,  -9.254611  ,
              -9.165625  ,  -9.076638  ,  -8.987652  ,  -8.898664  ,   -8.809678  ,  -8.720692  ,  -8.631705  ,  -8.542718  ,
              -8.453732  ,  -8.364745  ,  -8.275759  ,  -8.186771  ,   -8.097785  ,  -8.008799  ,  -7.9198117 ,  -7.830825  ,
              -7.7418385 ,  -7.6528516 ,  -7.563865  ,  -7.4748783 ,   -7.385892  ,  -7.296905  ,  -7.2079186 ,  -7.118932  ,
              -7.0299454 ,  -6.9409585 ,  -6.851972  ,  -6.762985  ,   -6.673999  ,  -6.585012  ,  -6.496025  ,  -6.4070387 ,
              -6.318052  ,  -6.2290654 ,  -6.1400785 ,  -6.051092  ,   -5.9621053 ,  -5.873119  ,  -5.784132  ,  -5.6951456 ,
              -5.6061587 ,  -5.5171723 ,  -5.4281855 ,  -5.339199  ,   -5.250212  ,  -5.161226  ,  -5.072239  ,  -4.9832525 ,
              -4.8942657 ,  -4.805279  ,  -4.7162924 ,  -4.6273055 ,   -4.538319  ,  -4.449332  ,  -4.360346  ,  -4.271359  ,
              -4.1823726 ,  -4.0933857 ,  -4.0043993 ,  -3.9154124 ,   -3.8264258 ,  -3.7374392 ,  -3.6484525 ,  -3.559466  ,
              -3.4704792 ,  -3.3814926 ,  -3.292506  ,  -3.2035193 ,   -3.1145327 ,  -3.025546  ,  -2.9365594 ,  -2.8475728 ,
              -2.7585862 ,  -2.6695995 ,  -2.580613  ,  -2.4916263 ,   -2.4026394 ,  -2.3136528 ,  -2.224666  ,  -2.1356795 ,
              -2.0466928 ,  -1.9577062 ,  -1.8687196 ,  -1.779733  ,   -1.6907463 ,  -1.6017597 ,  -1.512773  ,  -1.4237864 ,
              -1.3347998 ,  -1.2458131 ,  -1.1568264 ,  -1.0678397 ,   -0.9788531 ,  -0.8898665 ,  -0.80087984,  -0.7118932 ,
              -0.62290657,  -0.5339199 ,  -0.44493324,  -0.3559466 ,   -0.26695994,  -0.1779733 ,  -0.08898665,   0.        ,
              0.08898665,   0.1779733 ,   0.26695994,   0.3559466 ,    0.44493324,   0.5339199 ,   0.62290657,   0.7118932 ,
              0.80087984,   0.8898665 ,   0.9788531 ,   1.0678397 ,   1.1568264 ,   1.2458131 ,   1.3347998 ,   1.4237864 ,
              1.512773  ,   1.6017597 ,   1.6907463 ,   1.779733  ,   1.8687196 ,   1.9577062 ,   2.0466928 ,   2.1356795 ,
              2.224666  ,   2.3136528 ,   2.4026394 ,   2.4916263 ,   2.580613  ,   2.6695995 ,   2.7585862 ,   2.8475728 ,
              2.9365594 ,   3.025546  ,   3.1145327 ,   3.2035193 ,   3.292506  ,   3.3814926 ,   3.4704792 ,   3.559466  , 
              3.6484525 ,   3.7374392 ,   3.8264258 ,   3.9154124 ,   4.0043993 ,   4.0933857 ,   4.1823726 ,   4.271359  ,
              4.360346  ,   4.449332  ,   4.538319  ,   4.6273055 ,   4.7162924 ,   4.805279  ,   4.8942657 ,   4.9832525 ,
              5.072239  ,   5.161226  ,   5.250212  ,   5.339199  ,   5.4281855 ,   5.5171723 ,   5.6061587 ,   5.6951456 ,
              5.784132  ,   5.873119  ,   5.9621053 ,   6.051092  ,   6.1400785 ,   6.2290654 ,   6.318052  ,   6.4070387 ,
              6.496025  ,   6.585012  ,   6.673999  ,   6.762985  ,   6.851972  ,   6.9409585 ,   7.0299454 ,   7.118932  ,
              7.2079186 ,   7.296905  ,   7.385892  ,   7.4748783 ,  7.563865  ,   7.6528516 ,   7.7418385 ,   7.830825  ,
              7.9198117 ,   8.008799  ,   8.097785  ,   8.186771  ,   8.275759  ,   8.364745  ,   8.453732  ,   8.542718  ,
              8.631705  ,   8.720692  ,   8.809678  ,   8.898664  ,   8.987652  ,   9.076638  ,   9.165625  ,   9.254611  ,
              9.343598  ,   9.432585  ,   9.521571  ,   9.610558  ,   9.699545  ,   9.788531  ,   9.877518  ,   9.966505  ,
              10.055491  ,  10.144478  ,  10.233464  ,  10.322452  , 10.411438  ,  10.500424  ,  10.589411  ,  10.678398  ,
              10.767385  ,  10.856371  ,  10.945357  ,  11.034345  , 11.123331  ,  11.212317  ,  11.301304  ,  11.390291  ],
}

time_bins_radars = {
    'MIRA': pd.date_range(start_date, end_date, freq='5s'),
    'MXPol': pd.date_range(start_date, end_date, freq='1s'),
    'BASTA': pd.date_range(start_date, end_date, freq='3s'),
}

ranges = {
    'BASTA': {'res': 25,
              'tol': 12.5},
    'MIRA': {'res': 31.18,
             'tol': 16},
    'MXPol': {'res': 30,
              'tol': 15},    
}

plotdates_weekly = ((datetime(2024, 10, 18), datetime(2024, 10, 25)),
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

HALO = {'ABL': {
            0: 'no signal',
            1: 'stable/neutral',
            2: 'unstable',
            3: 'non-turbulent',
            4: 'convective mixing',
            5: 'wind shear',
            6: 'intermittent',
            7: 'in cloud',
            8: 'cloud-driven',
            9: 'precipitation'
        },
        'turbulence': {
            0: 'no signal',
            1: 'non-turbulent',
            2: 'surface-connected',
            3: 'cloud-driven',
            4: 'in cloud',
            5: 'unconnected',
            6: 'no retrieval'
        },
        'aerosols': {
            0: 'no aerosols',
            1: 'aerosol layer'
        }
}

tickstart = pd.Timestamp('2024-10-1')
tickend = pd.Timestamp('2025-01-25')
# days = [1, 10, 20]
tickdays = [1, 15]

date_ticks = pd.to_datetime([
    pd.Timestamp(year=dt.year, month=dt.month, day=d)
    for dt in pd.date_range(tickstart+pd.Timedelta(days=16), tickend, freq='MS')  # MS = Month Start
    for d in tickdays
    if pd.Timestamp(year=dt.year, month=dt.month, day=1).replace(day=d) <= tickend
])

add_dates = pd.to_datetime(['2025-01-25', '2024-10-14'])
date_ticks = date_ticks.union(add_dates).sort_values()

vars_hydroclassif = ['height_over_iso0', 'hydroclass_entropy', 'proportion_AG', 'proportion_CR', 'proportion_LR', 'proportion_RP', 
                     'proportion_RN', 'proportion_VI', 'proportion_WS', 'proportion_MH', 'proportion_IH']

scantimes_MXPol = {
    'Zdr': 123*2,
    'RHI': 49*2,
    'sector': 43*2,
    }