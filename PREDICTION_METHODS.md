# Climate Impact Projection and Spatial Visualization
## Supplementary Methods Documentation

**Project:** Loess Plateau Climate Impact Assessment  
**Analysis Framework:** Regression-based Climate Scenario Projection  
**Last Updated:** November 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Framework](#theoretical-framework)
3. [Data Requirements](#data-requirements)
4. [Climate Impact Calculation](#climate-impact-calculation)
5. [Visualization Methods](#visualization-methods)
6. [Output Files](#output-files)
7. [Usage Instructions](#usage-instructions)
8. [References](#references)

---

## 1. Overview

This module implements a projection framework to quantify future climate change impacts on agricultural yields using regression coefficients from historical panel analysis. The approach follows the established methodology in climate impact assessment literature (Lobell & Field, 2007; Challinor et al., 2014), combining econometric estimates with climate model projections.

### Key Features

- **Scenario-based Projections**: SSP2-4.5 and SSP5-8.5 climate scenarios (2041-2050 vs baseline) 
- **System-specific Assessment**: Separate projections for irrigated and rainfed maize systems
- **Decomposition Analysis**: Individual and combined effects of temperature, precipitation, and VPD
- **Spatial Mapping**: High-resolution climate change difference maps

### Workflow

```
Historical Regression → Climate Projections → Impact Calculation → Visualization
(from regression.py)   (from climate models)   (this module)      (maps & plots)
```

---

## 2. Theoretical Framework

### 2.1 Impact Assessment Equation

Climate impact on yield is calculated using the log-linear regression relationship:

```
ΔY/Y = exp(Σ βⱼ·ΔXⱼ) - 1
```

Where:
- `ΔY/Y`: Proportional change in yield (%)
- `βⱼ`: Regression coefficient for climate variable j (from Model 3)
- `ΔXⱼ`: Projected change in climate variable j (future - baseline)

### 2.2 Additive Decomposition

The total climate effect can be decomposed into contributions from individual variables:

**Temperature Effect**:
```
ΔY_Tmin = [exp(β_Tmin × ΔTmin) - 1] × 100
```

**Vapor Pressure Deficit Effect**:
```
ΔY_VPD = [exp(β_VPD × ΔVPD) - 1] × 100
```

**Precipitation Effect** (implicit):
```
ΔY_Precip = ΔY_total - ΔY_Tmin - ΔY_VPD
```

**Combined Effect**:
```
ΔY_total = [exp(β_Tmin·ΔTmin + β_Precip·ΔPrecip + β_VPD·ΔVPD) - 1] × 100
```

### 2.3 Uncertainty Sources

1. **Statistical Uncertainty**: Standard errors from regression (not propagated in point estimates)
2. **Climate Model Uncertainty**: GCM ensemble mean used (conservative approach)
3. **Adaptation**: Assumes no changes in cultivar or management (static technology)
4. **Non-linearities**: Linear approximation may underestimate extreme impacts

### 2.4 Interpretation

- **Positive values**: Yield increase due to climate change
- **Negative values**: Yield decrease (climate penalty)
- **Magnitude**: Percentage change relative to baseline yield levels

---

## 3. Data Requirements

### 3.1 Input Files

#### Regression Outputs (from `regression.py`)

**Coefficient Files**:
- `panel_regression_results/irrigation_coefficients.csv`
- `panel_regression_results/rainfed_coefficients.csv`

Required columns:
- `Variable`: Climate variable name
- `Coefficient`: Regression coefficient (β)
- `Std_Error`: Standard error (optional, for uncertainty analysis)

**Residual Files** (for county identification):
- `panel_regression_results/irrigation_model3_detailed_residuals.csv`
- `panel_regression_results/rainfed_model3_detailed_residuals.csv`

Required columns:
- `CODE`: County identifier
- `Year`: Year (not used in projection)

#### Climate Projections

**Tabular Data**:
- `yield_weather_analysis/climate_diff_final.csv`

Required columns:
- `CODE`: County identifier (6-digit)
- `ssp`: Scenario name ('SSP245' or 'SSP585')
- `gs_tmin_mean_diff`: Change in growing season minimum temperature (°C)
- `gs_vpd_mean_diff`: Change in growing season VPD (kPa)
- `gs_prep_total`: Change in growing season precipitation (mm)
- `ngs_prep_total`: Change in non-growing season precipitation (mm)

**Spatial Data** (for mapping):
- `climate_outputs/maps/*.tif`: GeoTIFF files with climate differences

Naming convention: `{var}_{scenario}_{period}_difference.tif`
- `var`: tmin, tmax, tmean, vpd, prep, rh
- `scenario`: ssp245, ssp585
- `period`: annual, growing, non_growing

**Boundary Shapefile**:
- `Loess_plateau_boundary_wgs1984.shp`: Study region boundary for clipping

### 3.2 Data Preprocessing

#### Unit Conversions

The module applies automatic unit conversions to match regression model units:

| Variable | Input Unit | Output Unit | Conversion |
|----------|------------|-------------|------------|
| `gs_tmin_mean` | °C | °C | No conversion (difference) |
| `gs_vpd_mean` | kPa | hPa | × 10 |
| `prep_total` | mm | cm | ÷ 10 |

#### Precipitation Aggregation

Total precipitation combines growing and non-growing season:
```
prep_total = (gs_prep_total + ngs_prep_total) / 10  # mm → cm


## 4. Climate Impact Calculation

### 4.1 Algorithm Overview

For each county and scenario:

1. **Load coefficients** for irrigation status (irrigated/rainfed)
2. **Extract climate changes** (ΔTmin, ΔPrecip, ΔVPD)
3. **Calculate impacts**:
   - Total effect: exp(β₁·ΔTmin + β₂·ΔPrecip + β₃·ΔVPD) - 1
   - Tmin effect: exp(β₁·ΔTmin) - 1
   - VPD effect: exp(β₃·ΔVPD) - 1
4. **Convert to percentage**: × 100
5. **Aggregate by province** using first 2 digits of CODE

### 4.2 Implementation Details

#### County-level Projection

```python
climate_effect = (np.exp(beta_tmin * delta_tmin + 
                         beta_prep * delta_prep + 
                         beta_vpd * delta_vpd) - 1) * 100
```

**Properties**:
- Non-linear transformation preserves proportional effects
- Allows for interaction between climate variables
- Bounded interpretation (percentage change)

#### Provincial Aggregation

Counties grouped by province code:

| Province Code | Province Name |
|---------------|---------------|
| 14 | Shanxi |
| 15 | Inner Mongolia |
| 41 | Henan |
| 61 | Shaanxi |
| 62 | Gansu |
| 64 | Ningxia |

**Aggregation Method**: Distribution of county-level effects preserved (no averaging in calculation).

### 4.3 Statistical Outputs

For each (County × Scenario × Irrigation) combination:

| Output Variable | Description | Unit |
|----------------|-------------|------|
| `climate_effect` | Total climate impact | % |
| `Tmin_effect` | Temperature-only impact | % |
| `VPD_effect` | VPD-only impact | % |
| `Province` | Province code | - |
| `Type` | Irrigation/Rainfed | - |
| `SSP` | Scenario name | - |


## 5. Visualization Methods

### 5.1 Coefficient Comparison Plot

**Purpose**: Display regression coefficients with uncertainty for both irrigation systems

**Design**:
- Two-panel figure (irrigated vs rainfed)
- Error bars: ± 1 standard error (transformed to % scale)
- Reference line at y=0 (no effect)
- Text labels showing point estimates

**Formula for Error Bars**:
```
SE_transformed = [exp(SE) - 1] × 100
```

**Output**: `prediction_results/coefficients_comparison.pdf`

**Interpretation**: Larger effects and smaller error bars indicate stronger climate sensitivity.

### 5.2 Impact Distribution Boxplots

**Purpose**: Show distribution of projected impacts across counties, grouped by province and scenario

**Design**:
- 6-panel layout: 3 variables × 2 irrigation types
- Side-by-side SSP245 (orange) and SSP585 (red) boxes
- "Loess Plateau" aggregate includes all counties
- Mean values overlaid (black dots with labels)
- Y-axis scaled by variable:
  - Climate effect: -15.5% to +15.5%
  - Tmin effect: 0% to +20%
  - VPD effect: -20% to 0%

**Statistical Elements**:
- Box: 25th, 50th (median), 75th percentiles
- Whiskers: 1.5 × IQR (interquartile range)
- Outliers: Individual points beyond whiskers
- Mean: Calculated across all counties in group

**Output**: `prediction_results/ratios_boxplot_combined_SSP.pdf`

**Interpretation**:
- Box width: Spread of impacts across counties
- Median vs mean: Asymmetry in distribution
- SSP585 vs SSP245: Difference in warming levels (~2°C vs ~4°C)

### 5.3 Spatial Climate Difference Maps

**Purpose**: Visualize spatial patterns of climate change across the Loess Plateau

**Design**:
- 4-panel layout per variable: (SSP245 Annual, SSP245 Growing Season, SSP585 Annual, SSP585 Growing Season)
- Diverging colormaps:
  - Temperature/VPD: Red (increase)
  - Precipitation: Blue (increase) to brown (decrease)
- Symmetric color scale for variables with positive and negative changes
- Text annotation: Mean ± SE for each panel

**Clipping**: Raster data masked to Loess Plateau boundary

**Output**: `prediction_results/maps/{var}_difference_comparison.pdf`

Variables mapped:
- `tmin`: Minimum temperature (°C)
- `tmax`: Maximum temperature (°C)
- `tmean`: Mean temperature (°C)
- `vpd`: Vapor pressure deficit (hPa)
- `prep`: Precipitation (cm)
- `rh`: Relative humidity (%)

**Spatial Statistics**:
```
Mean: μ = Σ(valid_pixels) / n
SE: σ = SD(valid_pixels) / √n
```

**Interpretation**: Spatial heterogeneity in climate change drives variation in crop impacts.

---

## 6. Output Files

### 6.1 CSV Output Structure

#### Main Results File

**File**: `prediction_results/climate_impact_results.csv`

Columns:
- `CODE`: County identifier
- `SSP`: Scenario (SSP245 or SSP585)
- `Province`: Province code (14, 15, 41, 61, 62, 64)
- `climate_effect`: Total yield impact (%)
- `Tmin_effect`: Temperature-only impact (%)
- `VPD_effect`: VPD-only impact (%)
- `Type`: Irrigation or Rainfed

**Rows**: N_counties × 2_scenarios × 2_systems ≈ 400-600 rows

#### System-specific Files

**Files**:
- `prediction_results/irrigation_climate_impact.csv`
- `prediction_results/rainfed_climate_impact.csv`

Same structure as main file, filtered by irrigation type.

### 6.2 Summary Statistics

**Interpretation**:
- **Mean**: Average impact across all counties
- **Std**: Spatial variability in impacts
- **Min/Max**: Range of impacts (most vulnerable vs least vulnerable)
- **Percentiles**: Distribution shape

### 6.3 Visualization Outputs

| File | Type | Dimensions | Description |
|------|------|------------|-------------|
| `coefficients_comparison.pdf` | Figure | 10×4 in | Regression coefficients |
| `ratios_boxplot_combined_SSP.pdf` | Figure | 25×10 in | Impact distributions by province |
| `{var}_difference_comparison.pdf` | Figure | 10×8 in | Spatial climate maps (6 variables) |

**Resolution**: All figures saved at 300 DPI (publication quality)

---

## 7. Usage Instructions

### 7.1 Basic Execution

```python
# Run complete projection pipeline
python prediction.py
```

**Prerequisites**: Must run `regression.py` first to generate coefficient files.

**Expected Runtime**: 2-5 minutes (depends on map resolution)

### 7.2 Configuration and Customization

#### File Paths

Modify at top of script (lines 15-19):

```python
IRRIGATION_RESIDUALS = "panel_regression_results/irrigation_model3_detailed_residuals.csv"
RAINFED_RESIDUALS = "panel_regression_results/rainfed_model3_detailed_residuals.csv"
IRRIGATION_COEFF = "panel_regression_results/irrigation_coefficients.csv"
RAINFED_COEFF = "panel_regression_results/rainfed_coefficients.csv"
CLIMATE_CHANGE = "yield_weather_analysis/climate_diff_final.csv"
```

#### Plot Appearance

**Boxplot Colors** (lines 171-173):
```python
ssp245_color = '#FFB366'  # Light orange
ssp585_color = '#D73027'  # Red
median_color = '#4A4A4A'  # Dark gray
```

**Y-axis Limits** (lines 240-301):
```python
if ratio == 'climate_effect':
    ax.set_ylim(-15.5, 15.5)
elif ratio == 'Tmin_effect':
    ax.set_ylim(0, 20)
else:  # VPD_effect
    ax.set_ylim(-20, 0)
```

**Map Colormaps** (lines 441-447):
```python
var_info = {
    'tmin': {'name': 'Minimum Temperature', 'unit': '°C', 'cmap': 'Reds'},
    'vpd': {'name': 'Vapor Pressure Deficit', 'unit': 'hPa', 'cmap': custom_cmap},
    'prep': {'name': 'Precipitation', 'unit': 'cm', 'cmap': custom_cmap2}
}
```

### 7.4 Quality Control Checks

#### Data Availability

```python
from prediction import read_data

irrigation_res, rainfed_res, irrigation_coeff, rainfed_coeff, climate_diff = read_data()

print(f"Irrigation counties: {irrigation_res['CODE'].nunique()}")
print(f"Rainfed counties: {rainfed_res['CODE'].nunique()}")
print(f"Climate data counties: {climate_diff['CODE'].nunique()}")
print(f"SSP scenarios: {climate_diff['ssp'].unique()}")
```

#### Coefficient Validity

```python
# Check that coefficients are for Model 3 variables
expected_vars = ['gs_tmin_mean', 'prep_total', 'gs_vpd_mean']
irrigation_coeff_vars = irrigation_coeff['Variable'].tolist()
rainfed_coeff_vars = rainfed_coeff['Variable'].tolist()

assert all(v in irrigation_coeff_vars for v in expected_vars), "Missing irrigation coefficients"
assert all(v in rainfed_coeff_vars for v in expected_vars), "Missing rainfed coefficients"
```

#### Impact Range Check

```python
from prediction import read_data, calculate_ratios

irrigation_res, rainfed_res, irrigation_coeff, rainfed_coeff, climate_diff = read_data()
results_df = calculate_ratios(climate_diff, irrigation_coeff, rainfed_coeff, 
                               irrigation_res, rainfed_res)

# Check for unrealistic impacts (likely data errors)
extreme_impacts = results_df[abs(results_df['climate_effect']) > 50]
if len(extreme_impacts) > 0:
    print(f"Warning: {len(extreme_impacts)} counties with >50% impact")
    print(extreme_impacts[['CODE', 'SSP', 'Type', 'climate_effect']])
```

---

## 8. References

### Climate Impact Assessment Methods


1. **Lobell, D. B. & Field, C. B.** (2007). Global scale climate–crop yield relationships and the impacts of recent warming. *Environmental Research Letters*, 2(1), 014002.

2. **Challinor, A. J., Watson, J., Lobell, D. B., Howden, S. M., Smith, D. R., & Chhetri, N.** (2014). A meta-analysis of crop yield under climate change and adaptation. *Nature Climate Change*, 4(4), 287-291.

3. **Zhao, C., Liu, B., Piao, S., Wang, X., Lobell, D. B., Huang, Y., ... & Asseng, S.** (2017). Temperature increase reduces global yields of major crops in four independent estimates. *Proceedings of the National Academy of Sciences*, 114(35), 9326-9331.

### Climate Scenarios (CMIP6/SSPs)
4. **Williams E, Funk C, Peterson P, Tuholske C.** (2024). High resolution climate change observations and projections for the evaluation of heat-related extremes. Scientific Data 11, 261.

5. **O'Neill, B. C., Tebaldi, C., Van Vuuren, D. P., Eyring, V., Friedlingstein, P., Hurtt, G., ... & Sanderson, B. M.** (2016). The scenario model intercomparison project (ScenarioMIP) for CMIP6. *Geoscientific Model Development*, 9(9), 3461-3482.

6. **Riahi, K., Van Vuuren, D. P., Kriegler, E., Edmonds, J., O'neill, B. C., Fujimori, S., ... & Tavoni, M.** (2017). The shared socioeconomic pathways and their energy, land use, and greenhouse gas emissions implications: an overview. *Global Environmental Change*, 42, 153-168.

### Statistical Transformation

7. **Kennedy, P.** (2008). *A Guide to Econometrics* (6th ed.). Wiley-Blackwell. (Chapter on interpretation of log-linear models)

### Visualization Best Practices

8. **Tufte, E. R.** (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.

9. **Franconeri, S. L., Padilla, L. M., Shah, P., Zacks, J. M., & Hullman, J.** (2021). The science of visual data communication: What works. *Psychological Science in the Public Interest*, 22(3), 110-161.

### Agricultural Adaptation

10. **Rosenzweig, C., Elliott, J., Deryng, D., Ruane, A. C., Müller, C., Arneth, A., ... & Jones, J. W.** (2014). Assessing agricultural risks of climate change in the 21st century in a global gridded crop model intercomparison. *Proceedings of the National Academy of Sciences*, 111(9), 3268-3273.

11. **Porter, J. R., Xie, L., Challinor, A. J., Cochrane, K., Howden, S. M., Iqbal, M. M., ... & Travasso, M. I.** (2014). Food security and food production systems. In *Climate Change 2014: Impacts, Adaptation, and Vulnerability* (pp. 485-533). Cambridge University Press.

---

## Technical Notes

### Software Requirements

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.21.0
- matplotlib ≥ 3.4.0
- seaborn ≥ 0.11.0
- geopandas ≥ 0.10.0
- rasterio ≥ 1.2.0
- shapely ≥ 1.8.0

### Performance Considerations

- **Computation Time**: 2-5 minutes for full pipeline
- **Memory Usage**: ~1-2 GB RAM (dominated by raster processing)
- **Disk Space**: ~50 MB for all outputs (maps are largest files)

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data Sources                      │
├──────────────────┬──────────────────┬──────────────────────┤
│ Regression Output│ Climate Scenarios│ Spatial Boundaries   │
│ (coefficients)   │ (GCM projections)│ (shapefiles)         │
└────────┬─────────┴────────┬─────────┴──────────┬───────────┘
         │                  │                    │
         ▼                  ▼                    ▼
    ┌─────────────────────────────────────────────────┐
    │         calculate_ratios()                      │
    │  • Load coefficients (β)                        │
    │  • Load climate changes (ΔX)                    │
    │  • Compute: exp(Σ β·ΔX) - 1                    │
    │  • Decompose by variable                        │
    └─────────┬───────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────┐
    │         Results DataFrame                       │
    │  County × Scenario × System × Effects           │
    └─────────┬───────────────────────────────────────┘
              │
              ├─────────────┬──────────────┬───────────┐
              ▼             ▼              ▼           ▼
         ┌─────────┐  ┌──────────┐  ┌──────────┐ ┌────────┐
         │Boxplots │  │CSV Files │  │Coef Plot │ │  Maps  │
         │(6-panel)│  │(3 files) │  │(2-panel) │ │(6 vars)│
         └─────────┘  └──────────┘  └──────────┘ └────────┘
```

### Known Limitations

1. **Static Technology**: No adaptation or technological progress assumed
2. **Linear Effects**: Potential threshold effects not captured
3. **Missing Counties**: Climate data availability may limit spatial coverage

### Future Enhancements

- [ ] Bootstrap confidence intervals for projected impacts
- [ ] Multi-model climate ensemble integration
- [ ] Adaptation scenario modeling (cultivar switching, irrigation expansion)
- [ ] Interactive web-based visualization (Plotly/Dash)
- [ ] Gridded (non-county) impact assessment for higher resolution

---


## Citation

If you use this projection framework in your research, please cite:

> Qin, R. (2025). Climate Impact Projection and Spatial Visualization Framework: 
> Regression-based Climate Scenario Assessment for the Loess Plateau. 
> Loess Plateau Climate Impact Assessment Project.

And cite the underlying regression analysis:

> Qin, R. (2025). Panel Regression Analysis for Climate-Yield Relationships: 
> Methodological Framework and Implementation. 
> Loess Plateau Climate Impact Assessment Project.

For questions or issues, contact: [rongzhuqin@gmail.com]

---

**Document Version:** 1.0  
**Code Version:** prediction.py  
**Last Review Date:** November 26, 2025

