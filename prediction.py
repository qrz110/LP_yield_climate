import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import re
# File paths (relative paths since .py is in the project directory)
IRRIGATION_RESIDUALS = "panel_regression_results/irrigation_model3_detailed_residuals.csv"
RAINFED_RESIDUALS = "panel_regression_results/rainfed_model3_detailed_residuals.csv"
IRRIGATION_COEFF = "panel_regression_results/irrigation_coefficients.csv"
RAINFED_COEFF = "panel_regression_results/rainfed_coefficients.csv"
CLIMATE_CHANGE = "yield_weather_analysis/climate_diff_final.csv"

def read_data():
    """Read all data files"""
    print("Reading data files...")
    files_to_check = [IRRIGATION_RESIDUALS, RAINFED_RESIDUALS, IRRIGATION_COEFF, RAINFED_COEFF, CLIMATE_CHANGE]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
    irrigation_res = pd.read_csv(IRRIGATION_RESIDUALS)
    rainfed_res = pd.read_csv(RAINFED_RESIDUALS)
    irrigation_codes = irrigation_res['CODE'].unique()
    rainfed_codes = rainfed_res['CODE'].unique()
    print(f"Irrigation unique CODEs: {len(irrigation_codes)}")
    print(f"Rainfed unique CODEs: {len(rainfed_codes)}")
    irrigation_coeff = pd.read_csv(IRRIGATION_COEFF)
    rainfed_coeff = pd.read_csv(RAINFED_COEFF)
    climate_diff = pd.read_csv(CLIMATE_CHANGE)
    print(f"Climate data shape: {climate_diff.shape}")
    print(f"Climate data columns: {climate_diff.columns.tolist()}")
    return irrigation_res, rainfed_res, irrigation_coeff, rainfed_coeff, climate_diff

def plot_coefficients(irrigation_coeff, rainfed_coeff):
    """Plot coefficients with error bars"""
    print("Creating coefficient plots...")
    variables = ['gs_tmin_mean', 'prep_total', 'gs_vpd_mean']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    irr_data = irrigation_coeff[irrigation_coeff['Variable'].isin(variables)]
    irr_effects = (np.exp(irr_data['Coefficient'])-1)*100
    ax1.errorbar(range(len(irr_data)), irr_effects, 
                yerr=(np.exp(irr_data['Std_Error'])-1)*100, 
                fmt='o', capsize=5, capthick=2)
    for i, effect in enumerate(irr_effects):
        ax1.text(i, effect + 0.3, f'{effect:.1f}', 
                ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax1.set_xticks(range(len(irr_data)))
    ax1.set_xticklabels(irr_data['Variable'], rotation=45)
    ax1.set_title('Irrigated maize')
    ax1.set_ylabel('Yield Effect (%)')
    ax1.grid(False)
    ax1.set_ylim(-10, 10)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    rain_data = rainfed_coeff[rainfed_coeff['Variable'].isin(variables)]
    rain_effects = (np.exp(rain_data['Coefficient'])-1)*100
    ax2.errorbar(range(len(rain_data)), rain_effects, 
                yerr=(np.exp(rain_data['Std_Error'])-1)*100, 
                fmt='o', capsize=5, capthick=2)
    for i, effect in enumerate(rain_effects):
        ax2.text(i, effect + 0.3, f'{effect:.1f}', 
                ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax2.set_xticks(range(len(rain_data)))
    ax2.set_xticklabels(rain_data['Variable'], rotation=45)
    ax2.set_title('Rainfed maize')
    ax2.set_ylabel('Yield Effect (%)')
    ax2.grid(False)
    ax2.set_ylim(-10, 10)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs('prediction_results', exist_ok=True)
    plt.savefig('prediction_results/coefficients_comparison.pdf', dpi=300, bbox_inches='tight')
    return irr_data, rain_data

def calculate_ratios(climate_diff, irrigation_coeff, rainfed_coeff, irrigation_res, rainfed_res):
    """Calculate climate and time ratios for both SSP scenarios"""
    print("Calculating climate and time ratios for SSP245 and SSP585...")
    def get_coeff_dict(coeff_df):
        return dict(zip(coeff_df['Variable'], coeff_df['Coefficient']))
    irr_coeff_dict = get_coeff_dict(irrigation_coeff)
    rain_coeff_dict = get_coeff_dict(rainfed_coeff)
    ssp_scenarios = ['SSP245', 'SSP585']
    all_results = []
    for ssp in ssp_scenarios:
        print(f"Processing {ssp} scenario...")
        climate_ssp = climate_diff[climate_diff['ssp'] == ssp].copy()
        climate_ssp['gs_tmin_mean'] = climate_ssp['gs_tmin_mean_diff']
        climate_ssp['gs_vpd_mean'] = climate_ssp['gs_vpd_mean_diff'] * 10
        climate_ssp['prep_total'] = (climate_ssp['gs_prep_total'] + climate_ssp['ngs_prep_total'] )/ 10
        print(f"{ssp} data points: {len(climate_ssp)}")
        irrigation_codes = irrigation_res['CODE'].unique()
        for code in irrigation_codes:
            climate_row = climate_ssp[climate_ssp['CODE'] == code]
            if len(climate_row) > 0:
                row = climate_row.iloc[0]
                beta_tmin = irr_coeff_dict.get('gs_tmin_mean', 0)
                beta_prep = irr_coeff_dict.get('prep_total', 0)
                beta_vpd = irr_coeff_dict.get('gs_vpd_mean', 0)
                delta_tmin = row['gs_tmin_mean']
                delta_prep = row['prep_total']
                delta_vpd = row['gs_vpd_mean']
                climate_effect = (np.exp(beta_tmin * delta_tmin + 
                                   beta_prep * delta_prep + 
                                   beta_vpd * delta_vpd) - 1) * 100
                # Tmin effect
                Tmin_effect = (np.exp(beta_tmin * delta_tmin ) - 1) * 100
                # VPD effect
                VPD_effect = (np.exp(beta_vpd * delta_vpd) - 1) * 100
                # Province (first two characters of CODE)
                province = str(code)[:2]
                all_results.append({
                    'CODE': code,
                    'SSP': ssp,
                    'Province': province,
                    'climate_effect': climate_effect,
                    'Tmin_effect': Tmin_effect,
                    'VPD_effect': VPD_effect,
                    'Type': 'Irrigation'
                })
        rainfed_codes = rainfed_res['CODE'].unique()
        for code in rainfed_codes:
            climate_row = climate_ssp[climate_ssp['CODE'] == code]
            if len(climate_row) > 0:
                row = climate_row.iloc[0]
                beta_tmin = rain_coeff_dict.get('gs_tmin_mean', 0)
                beta_prep = rain_coeff_dict.get('prep_total', 0)
                beta_vpd = rain_coeff_dict.get('gs_vpd_mean', 0)
                delta_tmin = row['gs_tmin_mean']
                delta_prep = row['prep_total']
                delta_vpd = row['gs_vpd_mean']
                climate_effect = (np.exp(beta_tmin * delta_tmin + 
                                   beta_prep * delta_prep + 
                                   beta_vpd * delta_vpd) - 1) * 100
                Tmin_effect = (np.exp(beta_tmin * delta_tmin ) - 1) * 100
                VPD_effect = (np.exp(beta_vpd * delta_vpd) - 1) * 100
                province = str(code)[:2]
                all_results.append({
                    'CODE': code,
                    'SSP': ssp,
                    'Province': province,
                    'climate_effect': climate_effect,
                    'Tmin_effect': Tmin_effect,
                    'VPD_effect': VPD_effect,
                    'Type': 'Rainfed'
                })
    results_df = pd.DataFrame(all_results)
    print(f"Total results calculated: {len(results_df)}")
    return results_df

def plot_boxplots(results_df):
    """Create boxplots by province with SSP245 and SSP585 side by side"""
    print("Creating combined boxplots by province for SSP245 and SSP585...")
    province_mapping = {
        '14': 'Shanxi',
        '61': 'Shaanxi', 
        '64': 'Ningxia',
        '15': 'Inner Mongolia',
        '41': 'Henan',
        '62': 'Gansu'
    }
    ratios = ['VPD_effect', 'Tmin_effect', 'climate_effect']
    titles = ['VPD_effect', 'Tmin_effect', 'Climate effect']
    fig, axes = plt.subplots(2, 3, figsize=(25, 10))
    axes = axes.ravel()
    ssp245_color = '#FFB366'  # Light orange
    ssp585_color = '#D73027'  # Red
    median_color = '#4A4A4A'  # Dark gray
    for i, (ratio, title) in enumerate(zip(ratios, titles)):
        ax_irr = axes[i]
        irrigation_data = results_df[results_df['Type'] == 'Irrigation'].copy()     
        if len(irrigation_data) > 0:
            irrigation_data['Province_Name'] = irrigation_data['Province'].map(province_mapping)
            loess_plateau_data = irrigation_data.copy()
            loess_plateau_data['Province_Name'] = 'Loess Plateau'
            irrigation_data_with_loess = pd.concat([irrigation_data, loess_plateau_data], ignore_index=True)
            province_order = ['Loess Plateau', 'Shanxi', 'Shaanxi', 'Ningxia', 'Inner Mongolia', 'Henan', 'Gansu']
            positions = []
            box_data = []
            colors = []
            labels = []
            for j, province in enumerate(province_order):
                province_data = irrigation_data_with_loess[irrigation_data_with_loess['Province_Name'] == province]
                ssp245_data = province_data[province_data['SSP'] == 'SSP245'][ratio].dropna()
                if len(ssp245_data) > 0:
                    positions.append(j * 3 + 1)
                    box_data.append(ssp245_data)
                    colors.append(ssp245_color)
                    if j == 0:  # Only add legend labels once
                        labels.append('SSP245')
                    else:
                        labels.append('')
                ssp585_data = province_data[province_data['SSP'] == 'SSP585'][ratio].dropna()
                if len(ssp585_data) > 0:
                    positions.append(j * 3 + 1.8)
                    box_data.append(ssp585_data)
                    colors.append(ssp585_color)
                    if j == 0:  # Only add legend labels once
                        labels.append('SSP585')
                    else:
                        labels.append('')
            if box_data:
                bp = ax_irr.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor(median_color)
                    patch.set_linewidth(0.5)
                for element in ['whiskers', 'caps', 'medians']:
                    for item in bp[element]:
                        item.set_color(median_color)
                        item.set_linewidth(0.5)
                for flier in bp['fliers']:
                    flier.set_marker('o')
                    flier.set_markerfacecolor("blue")
                    flier.set_markeredgecolor("blue")
                    flier.set_markersize(2)
                for pos, data in zip(positions, box_data):
                    if len(data) > 0:
                        mean_val = data.mean()
                        ax_irr.plot(pos, mean_val, 'o', color=median_color, markersize=2, markeredgewidth=0.5)
                        ax_irr.text(pos, mean_val + 0.5, f'{mean_val:.1f}', 
                                  ha='center', va='bottom', fontsize=6, color='black', weight='bold')
                x_ticks = [j * 3 + 1.4 for j in range(len(province_order))]
                ax_irr.set_xticks(x_ticks)
                ax_irr.set_xticklabels(province_order, rotation=45)
                if i == 0:
                    legend_elements = [Patch(facecolor=ssp245_color, edgecolor=median_color, label='SSP245'),
                                     Patch(facecolor=ssp585_color, edgecolor=median_color, label='SSP585')]
                    ax_irr.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        ax_irr.set_title(f'(a) Irrigation maize - {title}', loc='left', fontsize=11)
        ax_irr.set_ylabel(f'{ratio} (%)', fontsize=10)
        ax_irr.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax_irr.grid(False)
        if ratio == 'climate_effect':
            ax_irr.set_ylim(-15.5, 15.5)
        elif ratio == 'Tmin_effect':
            ax_irr.set_ylim(0, 20)
        else:
            ax_irr.set_ylim(-20, 0)
        ax_rain = axes[i + 3]
        rainfed_data = results_df[results_df['Type'] == 'Rainfed'].copy()
        if len(rainfed_data) > 0:
            rainfed_data['Province_Name'] = rainfed_data['Province'].map(province_mapping)
            loess_plateau_data = rainfed_data.copy()
            loess_plateau_data['Province_Name'] = 'Loess Plateau'
            rainfed_data_with_loess = pd.concat([rainfed_data, loess_plateau_data], ignore_index=True)
            positions = []
            box_data = []
            colors = []
            for j, province in enumerate(province_order):
                province_data = rainfed_data_with_loess[rainfed_data_with_loess['Province_Name'] == province]
                ssp245_data = province_data[province_data['SSP'] == 'SSP245'][ratio].dropna()
                if len(ssp245_data) > 0:
                    positions.append(j * 3 + 1)
                    box_data.append(ssp245_data)
                    colors.append(ssp245_color)
                ssp585_data = province_data[province_data['SSP'] == 'SSP585'][ratio].dropna()
                if len(ssp585_data) > 0:
                    positions.append(j * 3 + 1.8)
                    box_data.append(ssp585_data)
                    colors.append(ssp585_color)
            if box_data:
                bp = ax_rain.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor(median_color)
                    patch.set_linewidth(0.5)
                for element in ['whiskers', 'caps', 'medians']:
                    for item in bp[element]:
                        item.set_color(median_color)
                        item.set_linewidth(0.5)
                for flier in bp['fliers']:
                    flier.set_marker('o')
                    flier.set_markerfacecolor("blue")
                    flier.set_markeredgecolor("blue")
                    flier.set_markersize(2)
                for pos, data in zip(positions, box_data):
                    if len(data) > 0:
                        mean_val = data.mean()
                        ax_rain.plot(pos, mean_val, 'o', color=median_color, markersize=2, markeredgewidth=0.5)
                        ax_rain.text(pos, mean_val + 0.5, f'{mean_val:.1f}', 
                                   ha='center', va='bottom', fontsize=6, color='black', weight='bold')
                x_ticks = [j * 3 + 1.4 for j in range(len(province_order))]
                ax_rain.set_xticks(x_ticks)
                ax_rain.set_xticklabels(province_order, rotation=45)
        ax_rain.set_title(f'(b) Rainfed maize - {title}', loc='left', fontsize=11)
        ax_rain.set_ylabel(f'{ratio} (%)', fontsize=10)
        ax_rain.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax_rain.grid(False)
        if ratio == 'climate_effect':
            ax_rain.set_ylim(-15.5, 15.5)
        elif ratio == 'Tmin_effect':
            ax_rain.set_ylim(0, 20)
        else:
            ax_rain.set_ylim(-20, 0)
    plt.tight_layout()
    os.makedirs('prediction_results', exist_ok=True)
    plt.savefig('prediction_results/ratios_boxplot_combined_SSP.pdf', dpi=300, bbox_inches='tight')
    print("Combined boxplot saved as ratios_boxplot_combined_SSP.pdf")

def save_results(results_df):
    """Save results to CSV"""
    print("Saving results to CSV...")
    os.makedirs('prediction_results', exist_ok=True)
    results_df.to_csv('prediction_results/climate_impact_results.csv', index=False, encoding='utf-8-sig')
    irrigation_results = results_df[results_df['Type'] == 'Irrigation']
    rainfed_results = results_df[results_df['Type'] == 'Rainfed']
    irrigation_results.to_csv('prediction_results/irrigation_climate_impact.csv', index=False, encoding='utf-8-sig')
    rainfed_results.to_csv('prediction_results/rainfed_climate_impact.csv', index=False, encoding='utf-8-sig')
    print("Results saved successfully!")
    print("\nSummary Statistics:")
    print("Irrigation Results:")
    print(irrigation_results[['climate_effect', 'Tmin_effect', 'VPD_effect']].describe())
    print("\nRainfed Results:")
    print(rainfed_results[['climate_effect', 'Tmin_effect', 'VPD_effect']].describe())


def plot_maps(out_dir):
    tif_path = "inputdata/climate_outputs/maps/*.tif"
    shp_path = "inputdata/shp/Loess_plateau_boundary_wgs1984.shp"
    os.makedirs(out_dir, exist_ok=True)
    def _read_and_clip_raster(tif_file, geom, raster_crs, var, period):
        with rasterio.open(tif_file) as src:
            clipped_arr, clipped_transform = mask(
                dataset=src,
                shapes=[mapping(geom)],
                crop=True,
                nodata=src.nodata
            )
            meta = src.meta.copy()
        if var == 'vpd':
            conversion_factor = 10
        elif var == 'prep':
            if period == 'annual':
                conversion_factor = 366 / 10
            elif period == 'growing':
                conversion_factor = 214 / 10
            elif period == 'non_growing':
                conversion_factor = 152 / 10
            else:
                conversion_factor = 1
        else:
            conversion_factor = 1
        arr = clipped_arr[0] * conversion_factor
        arr = np.where(arr == 0, np.nan, arr)
        nodata_val = meta.get("nodata", None)
        if nodata_val is not None:
            arr = np.where(arr == nodata_val, np.nan, arr)
        return arr, clipped_transform
    def _get_extent(arr_shape, transform):
        (xmin, ymax) = transform * (0, 0)
        (xmax, ymin) = transform * (arr_shape[1], arr_shape[0])
        return (xmin, xmax, ymin, ymax)
    def _calculate_stats(data):

        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return np.nan, np.nan
        mean_val = np.mean(valid_data)
        se_val = np.std(valid_data) 
        return mean_val, se_val
    def _parse_filename(filename):
        basename = os.path.basename(filename)
        parts = basename.replace('.tif', '').split('_')
        ssp = None
        for part in parts:
            if part.startswith('ssp'):
                ssp = part
                break
        if 'heat' in basename:
            if 'rh' in basename:
                var = 'rh'
            elif 'tmax' in basename:
                var = 'tmax'
            elif 'tmean' in basename:
                var = 'tmean'
            elif 'tmin' in basename:
                var = 'tmin'
            elif 'vpd' in basename:
                var = 'vpd'
        elif 'prep' in basename:
            var = 'prep'
        else:
            var = 'unknown'
        if 'non_growing' in basename:
            period = 'non_growing'
        elif 'growing' in basename:
            period = 'growing'
        elif 'annual' in basename:
            period = 'annual'
        else:
            period = 'unknown'
        return ssp, var, period
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile no elements.")
    try:
        geom = gdf.union_all()
    except AttributeError:
        geom = gdf.unary_union
    tif_files = glob.glob(tif_path)
    if not tif_files:
        raise ValueError(f"No tif files found in {tif_path}.")
    data_dict = {}
    raster_crs = None
    with rasterio.open(tif_files[0]) as src:
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

        try:
            geom = gdf.union_all()
        except AttributeError:
            geom = gdf.unary_union
    for tif_file in tif_files:
        ssp, var, period = _parse_filename(tif_file)
        if var == 'unknown':
            print(f"   Skip unknown variable: {tif_file}")
            continue
        if var not in data_dict:
            data_dict[var] = {}
        if ssp not in data_dict[var]:
            data_dict[var][ssp] = {}
        
        arr, transform = _read_and_clip_raster(tif_file, geom, raster_crs, var, period)
        data_dict[var][ssp][period] = {'data': arr, 'transform': transform}
    cmap = plt.cm.BrBG
    cmap_half2 = mcolors.LinearSegmentedColormap.from_list(
        'BrBG_blue', cmap(np.linspace(0.5, 1, 256))
    )
    cmap = plt.cm.RdBu_r
    cmap_half1 = mcolors.LinearSegmentedColormap.from_list(
        'RdBu_r', cmap(np.linspace(0.5, 1, 256))
    )
    var_info = {
        'rh': {'name': 'Relative Humidity', 'unit': '%', 'cmap': 'coolwarm_r'},
        'tmax': {'name': 'Maximum Temperature', 'unit': '°C', 'cmap': 'Reds'},
        'tmean': {'name': 'Mean Temperature', 'unit': '°C', 'cmap': 'Reds'},
        'tmin': {'name': 'Minimum Temperature', 'unit': '°C', 'cmap': 'Reds'},
        'vpd': {'name': 'Vapor Pressure Deficit', 'unit': 'hPa', 'cmap':cmap_half1 },
        'prep': {'name': 'Precipitation', 'unit': 'cm', 'cmap':cmap_half2}
    }
    for var in data_dict.keys():
        if var not in var_info:
            continue 
        print(f"\nGenerating image for variable {var}...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'{var_info[var]["name"]} Difference', fontsize=16, fontweight='bold')
        subplot_titles = [
            'SSP2-4.5 Annual', 'SSP2-4.5 April-October',
            'SSP5-8.5 Annual', 'SSP5-8.5 April-October'
        ]
        data_order = [
            ('ssp245', 'annual'), ('ssp245', 'growing'),
            ('ssp585', 'annual'), ('ssp585', 'growing')
        ]
        all_data = []
        valid_data = []
        transforms = []
        for ssp, period in data_order:
            if ssp in data_dict[var] and period in data_dict[var][ssp]:
                data = data_dict[var][ssp][period]['data']
                transform = data_dict[var][ssp][period]['transform']
                all_data.append(data)
                transforms.append(transform)
                valid_vals = data[~np.isnan(data)]
                if len(valid_vals) > 0:
                    valid_data.extend(valid_vals)
            else:
                all_data.append(None)
                transforms.append(None)
        if not valid_data:
            print(f"Variable {var} has no valid data, skipping...")
            plt.close(fig)
            continue
        if len(valid_data) > 0:
            data_min, data_max = np.min(valid_data), np.max(valid_data)
            if data_min < 0 and data_max > 0:
                vabs = max(abs(data_min), abs(data_max))
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = data_min, data_max
            print(f"Variable {var} data range: [{data_min:.3f}, {data_max:.3f}] -> color range: [{vmin:.3f}, {vmax:.3f}]")
        else:
            vmin, vmax = -1, 1 
        cmap = plt.cm.get_cmap(var_info[var]['cmap'])
        cmap.set_bad(color='white', alpha=0.0)  
        im = None
        for i, ((ssp, period), data, transform) in enumerate(zip(data_order, all_data, transforms)):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            if data is not None:
                extent = _get_extent(data.shape, transform)
                im = ax.imshow(data, cmap=cmap, extent=extent, origin='upper', 
                            vmin=vmin, vmax=vmax, aspect='auto')
                
                mean_val, se_val = _calculate_stats(data)
                if not np.isnan(mean_val):
                    stats_text = f'{mean_val:.2f} ± {se_val:.2f}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        verticalalignment='top', horizontalalignment='left')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(subplot_titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.tick_params(labelsize=9)
        if im is not None:
            plt.tight_layout()
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(f'Difference ({var_info[var]["unit"]})', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        
        out_fig = os.path.join(out_dir, f'{var}_difference_comparison.pdf')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        print(f"Saving image: {out_fig}")
        plt.close(fig)

def main():
    """Main analysis function"""
    try:
        # Read all data
        irrigation_res, rainfed_res, irrigation_coeff, rainfed_coeff, climate_diff = read_data()
        # Plot coefficients
        plot_coefficients(irrigation_coeff, rainfed_coeff)
        # Calculate ratios
        results_df = calculate_ratios(climate_diff, irrigation_coeff, rainfed_coeff, 
                                    irrigation_res, rainfed_res) 
        # Create boxplots
        plot_boxplots(results_df)
        # Save results
        save_results(results_df)
        # Plot maps
        out_dir = 'prediction_results/maps'
        plot_maps(out_dir)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check file paths and data formats.")

if __name__ == "__main__":
    main()
