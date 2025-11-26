# LP_yield_climate
# Climate-Yield Analysis for the Loess Plateau

Panel regression and climate impact projection for maize yield.

### Scripts

**`regression.py`**  
Panel regression with fixed-effects models (5 specifications). Estimates historical climate-yield relationships.

**`prediction.py`**  
Climate impact projection for SSP2-4.5 and SSP5-8.5 scenarios.

### Data

**`./inputdata/climate_outputs/maps/`**  
Gridded climate difference maps (36 GeoTIFF files).

**`./inputdata/shp/`**  
Shapefiles for irrigated/rainfed counties and study region boundary.

**`./inputdata/yield_weather_analysis/`**  
Analysis-ready datasets including historical yield-weather panel data and future climate changes.

### Results

**`./panel_regression_results/`**  
Regression coefficients, model comparisons, diagnostics, and figures (40+ files).

**`./prediction_results/`**  
Climate impact estimates, spatial maps, and distribution boxplots.

### Documentation

**`REGRESSION_METHODS.md`**  
Methodological guide for panel regression (theoretical framework, model specs, diagnostics).

**`PREDICTION_METHODS.md`**  
Documentation for climate projection (impact equations, decomposition, visualization).

## Usage

```bash
python regression.py    # Step 1: Estimate coefficients
python prediction.py    # Step 2: Project impacts
```

## Requirements

Python >= 3.8, pandas, numpy, statsmodels, linearmodels, geopandas, matplotlib, rasterio

---

