# Panel Regression Analysis for Climate-Yield Relationships
## Supplementary Methods Documentation

**Project:** Loess Plateau Climate Impact Assessment  
**Analysis Framework:** Fixed-effects Panel Regression with Robust Standard Errors  
**Last Updated:** November 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Framework](#theoretical-framework)
3. [Data Requirements](#data-requirements)
4. [Model Specifications](#model-specifications)
5. [Statistical Methods](#statistical-methods)
6. [Diagnostic Procedures](#diagnostic-procedures)
7. [Output Metrics](#output-metrics)
8. [Usage Instructions](#usage-instructions)
9. [References](#references)

---

## 1. Overview

This module implements a comprehensive panel regression framework to quantify the relationships between climate variables and agricultural yields in the Loess Plateau region. The analysis employs fixed-effects models with clustered standard errors, following best practices established in agricultural economics and climate impact assessment literature (Schlenker & Roberts, 2009; Lobell et al., 2011).

### Key Features

- **Five Model Specifications**: Comprehensive model comparison including year fixed effects, county fixed effects, and temporal trend models
- **Robust Inference**: Driscoll-Kraay standard errors accounting for spatial and temporal correlation
- **Multiple Diagnostics**: VIF, residual analysis, leverage detection, and out-of-sample validation
- **Separate Analysis**: Independent estimation for irrigated and rainfed agricultural systems

---

## 2. Theoretical Framework

### 2.1 Conceptual Model

Agricultural yield is modeled as a function of climate variables, controlling for time-invariant spatial heterogeneity and common temporal shocks:

```
Y_it = f(Climate_it) + α_i + λ_t + ε_it
```

Where:
- `Y_it`: Log-transformed yield for county i in year t (kg/ha)
- `Climate_it`: Vector of weather variables during growing season
- `α_i`: County fixed effects (time-invariant characteristics)
- `λ_t`: Year fixed effects (common temporal shocks)
- `ε_it`: Idiosyncratic error term

### 2.2 Log Transformation

Yields are log-transformed (`USE_LOG_YIELD = True`) for three reasons:

1. **Multiplicative Effects**: Climate impacts are proportional to yield levels
2. **Variance Stabilization**: Reduces heteroskedasticity common in yield data
3. **Interpretation**: Coefficients represent semi-elasticities (% change per unit)

**Elasticity Calculation**:
```
Elasticity (%) = [exp(β) - 1] × 100
```

### 2.3 Climate Variables

Three primary climate variables based on agronomic theory:

| Variable | Unit | Agronomic Rationale |
|----------|------|---------------------|
| `gs_tmin_mean` | °C | Minimum temperature affects photosynthesis and respiration rates |
| `prep_total` | cm | Total precipitation (growing + non-growing season) determines water availability |
| `gs_vpd_mean` | hPa | Vapor pressure deficit drives evapotranspiration and water stress |

**Note**: Variables are NOT standardized (`STANDARDIZE_WEATHER = False`) to preserve interpretability and facilitate climate scenario projections.

---

## 3. Data Requirements

### 3.1 Input File Structure

**Primary Data File**: `yield_weather_analysis/yield_weather_matched.csv`

Required columns:
- `Year`: Integer (1980-2022)
- `CODE`: County identifier (6-digit administrative code)
- `Yield_t_ha`: Yield in metric tons per hectare (converted to kg/ha)
- `gs_tmin_mean`: Growing season mean minimum temperature (°C)
- `gs_vpd_mean`: Growing season mean VPD (kPa, converted to hPa)
- `gs_prep_total`: Growing season precipitation (mm, converted to cm)
- `ngs_prep_total`: Non-growing season precipitation (mm, converted to cm)

### 3.2 Spatial Classification

**Irrigation Status Shapefiles**:
- `county_LP_irri.shp`: Counties with substantial irrigation infrastructure
- `county_LP_rain.shp`: Rainfed agricultural counties

### 3.3 Data Cleaning

Automatic removal of:
1. Counties with < 10,000 observations (68 counties excluded)
2. Non-positive yield values (for log transformation)
3. Missing values in key variables (listwise deletion)

---

## 4. Model Specifications

### Model 1: Full Fixed Effects (Primary Model)

**Formula**:
```
log(Y_it) = β₁·TMin_it + β₂·Precip_it + β₃·VPD_it + α_i + λ_t + ε_it
```

**Features**:
- County fixed effects: Control for time-invariant heterogeneity
- Year fixed effects: Control for common shocks (policy, prices, technology)
- **Standard Errors**: Driscoll-Kraay (accounts for spatial & temporal correlation)

**Interpretation**: Climate effects identified from within-county deviations from county-specific means, net of common year effects.

### Model 2: County Fixed Effects Only

**Formula**:
```
log(Y_it) = β₁·TMin_it + β₂·Precip_it + β₃·VPD_it + α_i + ε_it
```

**Features**:
- No time controls
- Identifies climate effects including temporal trends
- Useful for long-run climate change impacts

**Caveat**: Coefficients may capture confounding temporal trends (e.g., technological progress).

### Model 3: Linear Time Trend (Recommended for Projection)

**Formula**:
```
log(Y_it) = β₁·TMin_it + β₂·Precip_it + β₃·VPD_it + β₄·(Year-2001) + α_i + ε_it
```

**Features**:
- Linear time trend instead of year dummies
- Allows extrapolation to future years
- **Used for climate scenario projections**

**Rationale**: Balances controlling for technological change while maintaining climate signal for prediction.

### Model 4: Early Period Subsample (1980-2000)

**Formula**: Same as Model 3
**Sample**: Years 1980-2000 only

**Purpose**: Test temporal stability of climate-yield relationships

### Model 5: Recent Period Subsample (2001-2022)

**Formula**: Same as Model 3
**Sample**: Years 2001-2022 only

**Purpose**: 
- Assess if relationships changed with modern agricultural practices
- More relevant for near-term projections

---

## 5. Statistical Methods

### 5.1 Estimation Strategy

**Base Estimation**: Ordinary Least Squares with dummy variables for fixed effects

**Standard Error Correction** (Priority Order):

1. **Driscoll-Kraay (Preferred)**: 
   - Accounts for cross-sectional and temporal dependence
   - Bartlett kernel with automatic bandwidth selection
   - Implementation: `linearmodels.PanelOLS` with `cov_type='driscoll-kraay'`

2. **Two-Way Clustering (Fallback)**:
   - Clusters by county × year
   - Conservative inference when DK fails

3. **Site Clustering (Final Fallback)**:
   - Clusters by county only
   - Accounts for within-county serial correlation

### 5.2 Multicollinearity Assessment

**Variance Inflation Factor (VIF)** calculated for all predictors:

```
VIF_j = 1 / (1 - R²_j)
```

Where R²_j is from regressing X_j on all other predictors.

**Diagnostic Thresholds**:
- VIF < 5: No concern
- 5 ≤ VIF < 10: Moderate multicollinearity (acceptable)
- VIF ≥ 10: High multicollinearity (warning issued)

### 5.3 Joint Hypothesis Testing

**Wald Test** for joint significance of all climate variables:

```
H₀: β₁ = β₂ = β₃ = 0
H₁: At least one β_j ≠ 0
```

**Test Statistic**: 
```
W = β'V⁻¹β ~ χ²(k)
```

Where V is the robust variance-covariance matrix and k is number of restrictions.

### 5.4 Multiple Testing Correction

**Benjamini-Hochberg False Discovery Rate (FDR)** applied to p-values:

```
q_i = min(p_i × m / rank_i, 1)
```

Where:
- `p_i`: Original p-value for variable i
- `m`: Total number of tests
- `rank_i`: Rank of p_i among all p-values

**Interpretation**: q-value is the expected proportion of false positives among all rejections.

---

## 6. Diagnostic Procedures

### 6.1 Residual Diagnostics

#### Standardized Residuals
```
z_it = ε_it / σ_ε
```

**Quality Checks**:
- % |z| > 2: Should be ≈ 5% (2σ rule)
- % |z| > 3: Should be ≈ 0.3% (3σ rule)

#### Autocorrelation
**ACF-1** (first-order autocorrelation) calculated within each county:

```
ρ₁ = Corr(ε_it, ε_i,t-1)
```

**Interpretation**: Median ACF-1 across counties indicates persistence in shocks.

### 6.2 Influence Diagnostics

#### Leverage
```
h_i = X_i(X'X)⁻¹X_i'
```

**Threshold**: h_i > 2p/n indicates high leverage observation

#### Cook's Distance
```
D_i = (r_i² / p) × (h_i / (1-h_i)²)
```

Where r_i is the standardized residual.

**Threshold**: D_i > 4/n suggests influential observation

### 6.3 Model Comparison Metrics

#### Within R² (Primary)
```
R²_within = 1 - SSE / SST_within
```

**Interpretation**: Proportion of within-county, within-year variance explained.

#### Inclusive R² (Alternative)
Full model R² from OLS with all dummy variables included.

**ΔR²**: Increment in R² from adding weather variables to baseline model with only fixed effects.

#### Out-of-Sample R²

**Model 1**: Leave-One-County-Out Cross-Validation (LOCO)
```
R²_OOS = 1 - Σ(y_test - ŷ_test)² / Σ(y_test - ȳ_test)²
```

**Models 2-3**: Leave-One-Year-Out Cross-Validation (LOYO)

**Purpose**: Assess prediction performance and overfitting.

---

## 7. Output Metrics

### 7.1 Coefficient Table

For each model and variable, the following are reported:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `Coefficient` | Raw regression coefficient (β) | Log-point change in yield per unit change in predictor |
| `Std_Error` | Robust standard error | Precision of estimate (accounting for clustering) |
| `t_value` | t-statistic | β / SE |
| `p_value` | Two-tailed p-value | Probability under H₀: β = 0 |
| `q_value_BH` | FDR-adjusted p-value | Controls false discovery rate |
| `CI_2.5%`, `CI_97.5%` | 95% confidence interval | Plausible range for β |
| `Std_Beta` | Standardized coefficient | (β × σ_x) / σ_y, comparable across variables |
| `Partial_R2` | Partial R² | t² / (t² + df_resid), unique contribution |
| `Elasticity_%(per_unit)` | Semi-elasticity | [exp(β) - 1] × 100, % yield change per unit |

### 7.2 Model Evaluation Metrics

**Fit Statistics**:
- R² (Within): Primary measure for FE models
- Adjusted R²: Penalizes additional parameters
- F-statistic: Overall model significance

**Error Metrics** (Log Scale):
- RMSE: Root mean squared error
- MAE: Mean absolute error  
- RSE: Relative standard error (RMSE / σ_y)

**Error Metrics** (Original Scale):
When `USE_LOG_YIELD = True`, back-transformed metrics also reported:
- RMSE_original: In kg/ha
- MAE_original: In kg/ha
- MAPE_original: Mean absolute percentage error

**Information Criteria**:
- AIC: Akaike Information Criterion
- BIC: Bayesian Information Criterion

### 7.3 Detailed Residual Outputs

**CSV File**: `{region}_model{n}_detailed_residuals.csv`

Contains observation-level information:
- True values (log and original scale)
- Fitted values (log and original scale)
- Residuals (log and original scale)
- Standardized residuals
- Leverage values
- Cook's distance
- Percentage errors

**Purpose**: Enable external validation and diagnostic visualization.

---

## 8. Usage Instructions

### 8.1 Basic Execution

```python
from regression import ImprovedPanelRegression

# Initialize analyzer
analyzer = ImprovedPanelRegression()

# Run complete analysis pipeline
analyzer.run_analysis()
```

### 8.2 Configuration Options

Edit `Config` class (lines 23-51):

```python
class Config:
    # Essential paths
    MATCHED_DATA_FILE = "yield_weather_analysis/yield_weather_matched.csv"
    OUTPUT_DIR = "panel_regression_results"
    IRRIGATION_SHP = "path/to/county_LP_irri.shp"
    RAINFED_SHP = "path/to/county_LP_rain.shp"
    
    # Key methodological choices
    USE_LOG_YIELD = True              # Recommended: True
    STANDARDIZE_WEATHER = False       # For projection: False
    USE_CLUSTERED_SE = True           # Recommended: True
    TRY_DRISCOLL_KRAAY = True        # Recommended: True
    
    # Weather variables
    CUSTOM_WEATHER_VARS = [
        'gs_tmin_mean',   # °C
        'prep_total',     # cm  
        'gs_vpd_mean',    # hPa
    ]
    USE_CUSTOM_VARS = True           # Recommended: True
```

### 8.3 Output Files

**Directory**: `panel_regression_results/`

#### Coefficient Tables
- `irrigation_model{1-5}_coef_enhanced.csv`: Full coefficient details with elasticities
- `rainfed_model{1-5}_coef_enhanced.csv`: Rainfed system results

#### Model Comparison
- `{region}_all_models_comparison.csv`: Side-by-side coefficient comparison
- `{region}_model_evaluations.csv`: R², OOS R², fit statistics

#### Diagnostics
- `{region}_error_metrics.csv`: RMSE, MAE, MAPE for all models
- `{region}_residual_diagnostics.csv`: ACF-1, leverage, outlier percentages
- `{region}_wald_joint_tests.csv`: Joint significance tests

#### Detailed Output
- `{region}_model{n}_detailed_residuals.csv`: Observation-level residuals

#### Visualizations
- `{region}_regression_diagnostics.pdf`: 8-panel diagnostic plot
- `{region}_all_models_coefficients.pdf`: Coefficient comparison across models


## 9. References

### Methodological Foundation

1. **Schlenker, W. & Roberts, M. J.** (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. *Proceedings of the National Academy of Sciences*, 106(37), 15594-15598.

2. **Lobell, D. B., Schlenker, W., & Costa-Roberts, J.** (2011). Climate trends and global crop production since 1980. *Science*, 333(6042), 616-620.

3. **Burke, M. & Emerick, K.** (2016). Adaptation to climate change: Evidence from US agriculture. *American Economic Journal: Economic Policy*, 8(3), 106-140.

4. **Jonathan Proctor et al.** (2025).Climate change increases the interannual variance of summer crop yields globally through changes in temperature and water supply.Sci. Adv.11,eady3575.

5. **Hsiao, C.** (2014). *Analysis of Panel Data* (3rd ed.). Cambridge University Press.

### Agricultural Applications

6. **Ortiz-Bobea, A., Ault, T. R., Carrillo, C. M., Chambers, R. G., & Lobell, D. B.** (2021). Anthropogenic climate change has slowed global agricultural productivity growth. *Nature Climate Change*, 11(4), 306-312.

7. **Ray, D. K., Gerber, J. S., MacDonald, G. K., & West, P. C.** (2015). Climate variation explains a third of global crop yield variability. *Nature Communications*, 6(1), 1-9.

---

## Technical Notes

### Software Requirements

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.21.0
- statsmodels ≥ 0.13.0
- linearmodels ≥ 4.27
- scikit-learn ≥ 1.0.0
- geopandas ≥ 0.10.0
- matplotlib ≥ 3.4.0
- scipy ≥ 1.7.0

### Performance Considerations

- **Computation Time**: ~5-15 minutes for full analysis (depends on sample size)
- **Memory Usage**: ~2-4 GB RAM for typical dataset (100+ counties, 40+ years)
- **Parallelization**: Not currently implemented; single-threaded execution

### Known Limitations

1. **Balanced Panel Assumption**: Counties with missing years handled via listwise deletion
2. **Linear Functional Form**: No explicit temperature thresholds or nonlinearities
3. **Contemporaneous Effects**: No lagged weather variables
4. **Homogeneous Coefficients**: Same climate response across all counties (within irrigation category)

### Future Enhancements

- [ ] Implement temperature binning approach (Schlenker & Roberts, 2009)
- [ ] Add distributed lag models for weather effects
- [ ] Support for heterogeneous treatment effects (quantile regression)
- [ ] Spatial econometric extensions (spatial lag/error models)
- [ ] Bootstrapped confidence intervals as alternative to asymptotic inference

---

## Citation

If you use this analysis framework in your research, please cite:

> [Rongzhu Qin]. (2025). Panel Regression Analysis for Climate-Yield Relationships: 
> Methodological Framework and Implementation. Loess Plateau Climate Impact Assessment Project.

For questions or issues, contact: [rongzhuqin@gmail.com]

---

**Document Version:** 1.0  
**Code Version:** regression.py
**Last Review Date:** November 26, 2025

