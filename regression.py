# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from pathlib import Path
from statsmodels.formula.api import ols 
from scipy.stats import chi2
from linearmodels.panel import PanelOLS
from patsy import dmatrices

warnings.filterwarnings('ignore')


class Config:
    """Configuration class"""
    MATCHED_DATA_FILE = "inputdata/yield_weather_analysis/yield_weather_matched.csv"
    OUTPUT_DIR = "panel_regression_results"
    IRRIGATION_SHP = "inputdata/shp/county_LP_irri.shp"
    RAINFED_SHP = "inputdata/shp/county_LP_rain.shp"
    YIELD_COLUMN = "Yield_t_ha"  # Original yield column name
    ALPHA = 0.05  # Significance level


    USE_LOG_YIELD = True            # Whether to use log transformation for yield
    STANDARDIZE_WEATHER = False     # Whether to standardize weather variables  
    USE_YEAR_FE = False              # Whether to use year fixed effects
    USE_CLUSTERED_SE = True          # Whether to use clustered standard errors
    TRY_DRISCOLL_KRAAY = True       # Whether to use Driscoll-Kraay standard errors
    TRY_TWOWAY_CLUSTER = True       # Whether to use two-way cluster standard errors
    DK_BANDWIDTH = None              # Bandwidth for Driscoll-Kraay standard errors
    VIF_THRESHOLD = 10.0            # VIF threshold for multicollinearity warning


    FIGSIZE = (14, 10)              # Figure size
    DPI = 300                       # DPI for figure
    CUSTOM_WEATHER_VARS = [
        'gs_tmin_mean',   # Growing season mean for daily minimum temperature 
        'prep_total',     # Growing season cumulative precipitation (gs + ngs)
        'gs_vpd_mean',    # Growing season mean for daily vapor pressure deficit
    ]
    USE_CUSTOM_VARS = True          # Whether to use custom weather variable list


class ImprovedPanelRegression:
    def __init__(self):
        self.config = Config()
        self.data = None
        self.irrigation_codes = None
        self.rainfed_codes = None
        self.irrigation_data = None
        self.rainfed_data = None
        self.results = {}
        self.scaler = StandardScaler()
        self.yield_column_name = self.config.YIELD_COLUMN

    def load_data(self):

        try:
            self.data = pd.read_csv(self.config.MATCHED_DATA_FILE)
            self.data["Yield_t_ha" ] = self.data["Yield_t_ha" ]*1000 # Convert yield from  t/ha to kg/ha
            self.data['gs_vpd_mean'] = self.data['gs_vpd_mean']*10 # Convert vapor pressure deficit from kPa to hPa
            self.data['ngs_vpd_mean'] = self.data['ngs_vpd_mean']*10 # Convert vapor pressure deficit from kPa to hPa
            self.data['gs_prep_total'] = self.data['gs_prep_total']/10 # Convert precipitation from mm to cm
            self.data['ngs_prep_total'] = self.data['ngs_prep_total']/10 # Convert precipitation from mm to cm
        except FileNotFoundError:
            print(f"Error: Could not find {self.config.MATCHED_DATA_FILE}")
            return False
        try:
            irrigation_shp = gpd.read_file(self.config.IRRIGATION_SHP)
            self.irrigation_codes = set(irrigation_shp['CODE'].astype(str))
        except FileNotFoundError:
            print(f"Error: Could not find {self.config.IRRIGATION_SHP}")
            return False
        try:
            rainfed_shp = gpd.read_file(self.config.RAINFED_SHP)
            self.rainfed_codes = set(rainfed_shp['CODE'].astype(str))
        except FileNotFoundError:
            print(f"Error: Could not find {self.config.RAINFED_SHP}")
            return False

        return True

    def prepare_data(self):
        self.data['CODE'] = self.data['CODE'].astype(str)
        if 'province_id' not in self.data.columns:
            self.data['province_id'] = self.data['CODE'].str[:2]
        if 'prep_total' not in self.data.columns:
            if 'gs_prep_total' in self.data.columns and 'ngs_prep_total' in self.data.columns:
                self.data['prep_total'] = self.data['gs_prep_total'] + self.data['ngs_prep_total']
        # Remove codes with less than 10,000 records based on data summary
        codes_to_remove = [
            '140106', '140107', '140109', '140110', '140202', '140212', '140302', '140402', '140411', '140525',
            '150102', '150103', '150104', '150202', '150203', '150204', '150205', '150303', '150304', '150921', 
            '150924', '152921', '410102', '410302', '410303', '410304', '410305', '410311', '410822', '411222', 
            '411281', '140981', '610102', '610103', '610104', '610111', '610113', '610114', '610203', '610303', 
            '610402', '610403', '610422', '610481', '610522', '610632', '610923', '620103', '620104', '620105', 
            '620111', '620403', '620623', '622921', '622927', '140702', '140882', '630102', '630103', '630104',
            '632322', '140781', '610304', '632122', '632123', '632126', '632523', '640104', '640105', '640425',
             '610116', '610204', '620503', '621021'
        ]
        original_count = len(self.data)
        self.data = self.data[~self.data['CODE'].isin(codes_to_remove)]
        removed_count = original_count - len(self.data)
        if removed_count > 0:
            print(f"Removed {removed_count} observations from codes with <10,000 records")
        raw_vars = self.config.CUSTOM_WEATHER_VARS
        for v in raw_vars:
            if v in self.data.columns:
                self.data[f'{v}_raw'] = self.data[v]
        if self.config.USE_LOG_YIELD:
            positive_yield_mask = self.data[self.config.YIELD_COLUMN] > 0
            original_count = len(self.data)
            self.data = self.data[positive_yield_mask]
            removed_count = original_count - len(self.data)
            if removed_count > 0:
                print(f"removed {removed_count} observations with non yield")
            self.data['log_yield'] = np.log(self.data[self.config.YIELD_COLUMN])
            self.yield_column_name = 'log_yield'
            print(f"Applied log transformation. Using column: {self.yield_column_name}")
        else:
            self.yield_column_name = self.config.YIELD_COLUMN
            print(f"Using original yield column: {self.yield_column_name}")
        irrigation_mask = self.data['CODE'].isin(self.irrigation_codes)
        rainfed_mask = self.data['CODE'].isin(self.rainfed_codes)
        self.irrigation_data = self.data[irrigation_mask].copy()
        self.rainfed_data = self.data[rainfed_mask].copy()
        self.data['irrigation_dummy'] = self.data['CODE'].isin(self.irrigation_codes).astype(int)
        if self.config.STANDARDIZE_WEATHER:
            print("standardizing weather variables...")
            self._standardize_weather_variables()
        else:
            print("using original units of weather variables (not standardized)")

    def get_weather_variables(self, data):
        if self.config.USE_CUSTOM_VARS:
            weather_vars = []
            for var in self.config.CUSTOM_WEATHER_VARS:
                if var in data.columns:
                    weather_vars.append(var)
                else:
                    print(f"Warning: Variable {var} not found in data")
            return weather_vars
        else:
            exclude_cols = ['Year', self.config.YIELD_COLUMN, 'CODE', 'log_yield', 'irrigation_dummy']
            weather_vars = [col for col in data.columns if col not in exclude_cols]
            return weather_vars

    

    def _standardize_weather_variables(self):
        weather_vars = self.get_weather_variables(self.data)
        if not weather_vars:
            print("No weather variables found for standardization")
            return
        all_weather_data = self.data[weather_vars].dropna()
        mu = all_weather_data.mean()
        sd = all_weather_data.std(ddof=0).replace(0, np.nan)
        def _zscore(df, weather_vars):
            df = df.copy()
            for v in weather_vars:
                if v in df.columns:
                    df[v] = (df[v] - mu[v]) / sd[v]
            return df
        if not self.irrigation_data.empty:
            self.irrigation_data = _zscore(self.irrigation_data, weather_vars)
        if not self.rainfed_data.empty:
            self.rainfed_data = _zscore(self.rainfed_data, weather_vars)

    def _check_multicollinearity(self, data, weather_vars):
        vif_results = {}
        try:
            X_vars = ['Year'] + weather_vars if not self.config.USE_YEAR_FE else weather_vars
            X_data = data[X_vars].dropna()
            if X_data.empty:
                print(" No clean data available for VIF calculation")
                return vif_results
            X_with_const = sm.add_constant(X_data)
            high_vif_vars = []
            for i, var in enumerate(X_with_const.columns):
                if var != 'const':
                    vif = variance_inflation_factor(X_with_const.values, i)
                    vif_results[var] = vif
                    if vif > self.config.VIF_THRESHOLD:
                        status = "HIGH"
                        high_vif_vars.append(var)
                    elif vif > 5:
                        status = "MODERATE"
                    else:
                        status = "OK"
                    print(f"{var}: {vif:.2f} ({status})")
            if high_vif_vars:
                print(f"Variables with high multicollinearity (VIF > {self.config.VIF_THRESHOLD}): {', '.join(high_vif_vars)}")
            else:
                print(f" No severe multicollinearity detected (all VIF < {self.config.VIF_THRESHOLD})")
        except Exception as e:
            print(f"Error calculating VIF: {str(e)}")
        return vif_results

    def _to_series(self,x, index_like):
        if x is None:
            return pd.Series(dtype=float)
        if hasattr(x, "index"):
            return x
        try:
            return pd.Series(x, index=index_like.index)
        except Exception:
            return pd.Series(x)

    def _extract_param_series(self, res):
        params = getattr(res, "params", None)
        se = getattr(res, "bse", None)
        if se is None or isinstance(se, (float, int)):
            se = getattr(res, "std_errors", None)
        tvals = getattr(res, "tvalues", None)
        if tvals is None or isinstance(tvals, (float, int)):
            tvals = getattr(res, "tstats", None)
        pvals = getattr(res, "pvalues", None)
        params = self._to_series(params, params)
        se = self._to_series(se, params)
        tvals = self._to_series(tvals, params)
        pvals = self._to_series(pvals, params)
        return params, se, tvals, pvals

    def _safe_conf_int(self, res, var):
        try:
            ci = res.conf_int()
            if isinstance(ci, pd.DataFrame):
                if var in ci.index:
                    lower_names = ['lower', 'ci_lower', 'lower bound', 'lower 2.5%', '2.5%', 'lower_ci', 0]
                    upper_names = ['upper', 'ci_upper', 'upper bound', 'upper 97.5%', '97.5%', 'upper_ci', 1]
                    lower_col = next((c for c in lower_names if c in ci.columns), None)
                    upper_col = next((c for c in upper_names if c in ci.columns), None)
                    if lower_col is not None and upper_col is not None:
                        return float(ci.loc[var, lower_col]), float(ci.loc[var, upper_col])
                    if ci.shape[1] >= 2:
                        row = ci.loc[var]
                        return float(row.iloc[0]), float(row.iloc[1])
            elif isinstance(ci, pd.Series):
                pass
        except Exception:
            pass
        return np.nan, np.nan

    def _get_model_stat(self, model, stat_name, prefer='within'):
        """
        Get model statistics; for PanelOLS, default to return within R² as rsquared.
        prefer: 'within'|'overall' to control the scope of rsquared in PanelOLS.
        """
        is_panel = hasattr(model, 'rsquared_within') or 'Panel' in type(model).__name__
        if stat_name == 'rsquared':
            if is_panel:
                if prefer == 'within' and hasattr(model, 'rsquared_within'):
                    return model.rsquared_within
                if prefer == 'overall' and hasattr(model, 'rsquared_overall'):
                    return model.rsquared_overall
                return getattr(model, 'rsquared', None)
            else:
                return getattr(model, 'rsquared', None)
        if stat_name in ('rsquared_within', 'rsquared_overall', 'rsquared_between'):
            return getattr(model, stat_name, None)
        if stat_name == 'rsquared_adj':
            if is_panel:
                return getattr(model, 'rsquared_adj', None)
            return getattr(model, 'rsquared_adj', None)
        if stat_name in ('fstatistic', 'f_statistic'):
            if hasattr(model, 'f_statistic'):
                fs = model.f_statistic
                return fs.stat if hasattr(fs, 'stat') else fs
            return getattr(model, 'fvalue', None)
        if stat_name == 'f_pvalue':
            if hasattr(model, 'f_statistic'):
                fs = model.f_statistic
                return getattr(fs, 'pval', None)
            return getattr(model, 'f_pvalue', None)
        if stat_name == 'nobs':
            return getattr(model, 'nobs', getattr(getattr(model, 'model', None), 'nobs', None))
        if stat_name == 'aic':
            return getattr(model, 'aic', None)
        if stat_name == 'bic':
            return getattr(model, 'bic', None)
        if stat_name in ('df_model', 'df_resid'):
            return getattr(model, stat_name, None)
        return getattr(model, stat_name, None)

    def _get_model_attributes(self, model):
        if hasattr(model, 'fitted_values'):
            fitted = model.fitted_values
            if hasattr(model, 'wresids'):
                residuals = model.wresids
            else:
                actual = model.dependent.values2d.flatten() if hasattr(model, 'dependent') else None
                if actual is not None:
                    residuals = actual - fitted
                else:
                    raise ValueError("Cannot compute residuals for this Panel model")
        elif hasattr(model, 'fittedvalues'): 
            fitted = model.fittedvalues
            residuals = model.resid
        else:
            available_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            fitted_attrs = [attr for attr in available_attrs if 'fitted' in attr.lower()]
            resid_attrs = [attr for attr in available_attrs if 'resid' in attr.lower()]
            raise ValueError(f"Unknown model type: {type(model)}. Available fitted attrs: {fitted_attrs}, resid attrs: {resid_attrs}")
        return fitted, residuals

    def _get_model_se(self, model, param_name):
        if hasattr(model, 'std_errors'):  # PanelEffectsResults (linearmodels)
            return model.std_errors[param_name] if param_name in model.std_errors.index else None
        elif hasattr(model, 'bse'):  # RegressionResults (statsmodels)
            return model.bse[param_name] if param_name in model.bse.index else None
        else:
            return None

    def _bh_fdr(self, pvals):
        p = pd.Series(pvals).astype(float)
        m = p.notna().sum()
        if m == 0:
            return pd.Series(np.nan, index=p.index)
        order = p.sort_values().index
        ranks = pd.Series(range(1, m + 1), index=p[order].index)
        q = p.copy()
        q.loc[order] = (p.loc[order] * m / ranks).cummin().clip(upper=1.0)
        return q

    def _pick_ci(self, ci_df, var_name, beta_val, se_val):
        lo = hi = np.nan
        if ci_df is not None:
            if isinstance(ci_df, pd.DataFrame) and (var_name in ci_df.index):
                
                lower_names = ['lower', 'ci_lower', 'lower bound', 'lower 2.5%', '2.5%', 'lower_ci']
                upper_names = ['upper', 'ci_upper', 'upper bound', 'upper 97.5%', '97.5%', 'upper_ci']
                lower_col = next((c for c in lower_names if c in ci_df.columns), None)
                upper_col = next((c for c in upper_names if c in ci_df.columns), None)
                try:
                    if lower_col is not None and upper_col is not None:
                        lo = float(ci_df.loc[var_name, lower_col])
                        hi = float(ci_df.loc[var_name, upper_col])
                    elif ci_df.shape[1] >= 2:
                        
                        row = ci_df.loc[var_name]
                        lo = float(row.iloc[0])
                        hi = float(row.iloc[1])
                except Exception:
                    lo = hi = np.nan
    
        if (not np.isfinite(lo)) or (not np.isfinite(hi)):
            if np.isfinite(se_val):
                lo = beta_val - 1.96 * se_val
                hi = beta_val + 1.96 * se_val
            else:
                lo = hi = np.nan
        return lo, hi

    def _make_coef_table(self, model, data, weather_vars, year_var_in_model='Year', use_robust=True):
        y = data[self.yield_column_name].to_numpy()
        y_std = float(np.nanstd(y, ddof=0)) if np.isfinite(np.nanstd(y, ddof=0)) else np.nan
        params, se_s, t_s, p_s = self._extract_param_series(model)
        try:
            ci = model.conf_int()
        except Exception:
            ci = None
        vars_to_show = list(weather_vars)
        if year_var_in_model in params.index:
            vars_to_show.append(year_var_in_model)
        x_sd = {}
        for v in weather_vars:
            if v in data.columns:
                s = np.nanstd(data[v].to_numpy(), ddof=0)
                x_sd[v] = float(s) if np.isfinite(s) else np.nan
            else:
                x_sd[v] = np.nan
        rows = []
        for v in vars_to_show:
            if v not in params.index:
                continue
            beta = float(params.get(v, np.nan))
            se   = float(se_s.get(v, np.nan))
            tval = float(t_s.get(v, np.nan))
            pval = float(p_s.get(v, np.nan))
            lo, hi = self._pick_ci(ci, v, beta, se)
            try:
                df_resid = float(getattr(model, 'df_resid', np.nan))
                partial_r2 = tval * tval / (tval * tval + df_resid) if np.isfinite(tval) and np.isfinite(df_resid) and df_resid > 0 else np.nan
            except Exception:
                partial_r2 = np.nan
            if v in x_sd and np.isfinite(x_sd[v]) and x_sd[v] > 0 and np.isfinite(y_std) and y_std > 0:
                std_beta = beta * (x_sd[v] / y_std)
            else:
                std_beta = np.nan
            elasticity = np.nan
            if self.config.USE_LOG_YIELD and (v in weather_vars) and np.isfinite(beta):
                elasticity = (np.exp(beta) - 1.0) * 100.0
            rows.append({
                'Variable': v,
                'Coefficient': beta,
                'Std_Error': se,
                't_value': tval,
                'p_value': pval,
                'CI_2.5%': lo,
                'CI_97.5%': hi,
                'Std_Beta': std_beta,
                'Partial_R2': partial_r2,
                'Elasticity_%(per_unit)': elasticity
            })
        df = pd.DataFrame(rows).set_index('Variable')
        if not df.empty:
            df['q_value_BH'] = self._bh_fdr(df['p_value'])

            def _stars(x):
                return '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''

            df['Sig(p)'] = df['p_value'].apply(_stars)
            df['Sig(q)'] = df['q_value_BH'].apply(_stars)
        return df


    def _compute_error_metrics(self, base_model, robust_model, data, save_detailed_table=True, output_dir=None, region_name=None):
        y_true = data[self.yield_column_name].values
        fitted, resid = self._get_model_attributes(base_model)
        leverage = base_model.get_influence().hat_matrix_diag
        cooks_d = base_model.get_influence().cooks_distance[0]
        yhat = np.asarray(fitted).reshape(-1, )
        resid = np.asarray(resid).reshape(-1, )
        rmse = float(np.sqrt(np.nanmean(resid ** 2)))
        mae = float(np.nanmean(np.abs(resid)))
        sd_y = float(np.nanstd(y_true, ddof=0))
        rse = float(rmse / sd_y) if sd_y > 0 else np.nan
        metrics = {
            'scale': 'log' if self.config.USE_LOG_YIELD else 'original',
            'RMSE': rmse,
            'MAE': mae,
            'RSE': rse
        }
        if self.config.USE_LOG_YIELD:
            y_true_original = np.exp(y_true)
            yhat_original = np.exp(yhat)
            resid_original = y_true_original - yhat_original
            rmse_original = float(np.sqrt(np.nanmean(resid_original ** 2)))
            mae_original = float(np.nanmean(np.abs(resid_original)))
            mape_original = float(np.nanmean(np.abs(resid_original / y_true_original)) * 100)
            metrics.update({
                'RMSE_original': rmse_original,
                'MAE_original': mae_original,
                'MAPE_original': mape_original
            })
        else:
            y_true_original = y_true
            yhat_original = yhat
            resid_original = resid
            mape_original = float(np.nanmean(np.abs(resid_original / y_true_original)) * 100)
            metrics['MAPE'] = mape_original
        if save_detailed_table and output_dir and region_name:
            try:
                from pathlib import Path
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                detailed_table = pd.DataFrame({
                    'Year': data['Year'].values,
                    'CODE': data['CODE'].values,
                    'True_Value_Log': y_true,
                    'Fitted_Value_Log': yhat,
                    'Residual_Log': resid,
                    'True_Value_Original': y_true_original,
                    'Fitted_Value_Original': yhat_original,
                    'Residual_Original': resid_original,
                    'Standardized_Residual': resid / np.std(resid) if np.std(resid) > 0 else resid,
                    'Absolute_Error_Log': np.abs(resid),
                    'Absolute_Error_Original': np.abs(resid_original),
                    'Squared_Error_Log': resid ** 2,
                    'Squared_Error_Original': resid_original ** 2,
                    'Leverage': leverage,
                    'Cooks_Distance': cooks_d
                })
                if np.all(y_true_original > 0):
                    detailed_table['Relative_Error_Original'] = resid_original / y_true_original
                    detailed_table['Percentage_Error_Original'] = (resid_original / y_true_original) * 100
                detailed_file = output_path / f'{region_name.lower()}_detailed_residuals.csv'
                detailed_table.to_csv(detailed_file, index=False)
            except Exception as e:
                print(f"save detailed table error: {str(e)}")
        return metrics

    def _acf1_by_group(self, df):
        vals = df.sort_values('Year')['resid'].values
        if len(vals) < 3:
            return np.nan
        v = np.var(vals)
        if v <= 0:
            return np.nan
        return float(np.corrcoef(vals[1:], vals[:-1])[0, 1])

    def _residual_panel_diagnostics(self, base_ols_model, robust_model, data):
        """
        - base_ols_model: base（statsmodels OLS）from _fit_ols
        - robust_model:   robust（DK/cluster）for reporting
        Return dict + several Series
        """
        fitted_r, resid_r = self._get_model_attributes(base_ols_model)
        yhat = np.asarray(fitted_r).reshape(-1, )
        resid = np.asarray(resid_r).reshape(-1, )
        out = {}
        res_std = resid / (np.nanstd(resid, ddof=0) if np.nanstd(resid, ddof=0) > 0 else 1.0)
        out['pct_|z|>2'] = float(np.mean(np.abs(res_std) > 2.0))
        out['pct_|z|>3'] = float(np.mean(np.abs(res_std) > 3.0))
        infl = None
        try:
            infl = base_ols_model.get_influence()
            leverage = pd.Series(infl.hat_matrix_diag, name='leverage')
            cooks = pd.Series(infl.cooks_distance[0], name='cooks_d')
            out['leverage_mean'] = float(leverage.mean())
            out['pct_high_leverage(>2p/n)'] = float(np.mean(leverage > (2 * len(base_ols_model.params) / len(leverage))))
            out['pct_high_cooks(>4/n)'] = float(np.mean(cooks > (4.0 / len(cooks))))
        except Exception:
            out['leverage_mean'] = np.nan
            out['pct_high_leverage(>2p/n)'] = np.nan
            out['pct_high_cooks(>4/n)'] = np.nan
            leverage = cooks = None
        df_res = data[['CODE', 'Year']].copy()
        df_res['resid'] = resid
        acf1 = df_res.groupby('CODE').apply(self._acf1_by_group)
        out['acf1_median_by_county'] = float(np.nanmedian(acf1.values))
        out['acf1_pct_pos'] = float(np.nanmean(acf1.values > 0))
        return out, res_std, leverage, cooks

    def _wald_joint_test(self, model, var_list):
        """
        Return {'stat': chi2, 'df': k, 'pval': p}
        Suitable for statsmodels or linearmodels; prioritize using model's robust covariance.
        """
        if hasattr(model, 'cov_params'):
            cov = model.cov_params()
        elif hasattr(model, 'cov'):
            cov = model.cov
        else:
            return {'note': 'no covariance available'}
        params = getattr(model, 'params', None)
        if params is None:
            return {'note': 'no params'}
        params = params.dropna()
        S = [v for v in var_list if v in params.index]
        k = len(S)
        if k == 0:
            return {'note': 'no variables to test'}
        beta = params.loc[S].values
        if hasattr(cov, 'loc'):
            V = cov.loc[S, S].values
        else:
            idx = [list(params.index).index(v) for v in S]
            V = cov[np.ix_(idx, idx)]
        try:
            stat = float(beta.T @ np.linalg.inv(V) @ beta)
            pval = float(1 - chi2.cdf(stat, k))
            return {'stat': stat, 'df': k, 'pval': pval}
        except Exception as e:
            return {'note': f'wald failed: {e}'}

    def _oos_r2_by_group(self, formula, data, group_col):
        """
        Generic grouped leave-one-out cross-validation OOS R².
        Use test fold variance (adaptively for each fold), overall R² use weighted aggregation by fold sample size.
        """
        groups = data[group_col].unique()
        sse_total, sst_total = 0.0, 0.0
        n_total = 0
        for g in groups:
            train = data[data[group_col] != g]
            test = data[data[group_col] == g]
            if len(test) < 10 or len(train) < 50:
                continue
            try:
                m = smf.ols(formula, data=train).fit()
                y_hat = m.predict(test)
                y = test[self.yield_column_name]
                sse = float(((y - y_hat) ** 2).sum())
                sst = float(((y - y.mean()) ** 2).sum())
                if sst > 0:
                    sse_total += sse
                    sst_total += sst
                    n_total += len(test)
            except Exception:
                continue
        if sst_total <= 0 or n_total == 0:
            return None
        return 1.0 - sse_total / sst_total

    def inclusive_r2(self, formula_full, formula_base, data):
        r2b = smf.ols(formula_base, data=data).fit().rsquared
        r2f = smf.ols(formula_full, data=data).fit().rsquared
        return float(r2f), float(r2f - r2b)

    def _evaluate_models(self, clean_data, weather_vars, base1, m1, base2, m2, base3, m3, base4, m4, base5, m5):
        """
        Generate evaluation metrics for 3 main models:
        - Recommended R²（FE space）：priority model.rsquared（linearmodels），otherwise _get_model_stat
        - ΔR²（inclusive FE standard）：inclusive R²(full) - inclusive R²(baseline)
        - OOS R²：Model1 LOCO(CODE)；Model2/3 LOYO(Year)
        """
        evals = {}
        # Model1: County FE + Year FE  —— use clean_data
        r2m1 = self._get_model_stat(m1, 'rsquared')
        form1_full = f"{self.yield_column_name} ~ " + " + ".join(['C(Year)'] + weather_vars + ['C(CODE)'])
        form1_base = f"{self.yield_column_name} ~ C(Year) + C(CODE)"
        inc1, del1 = self.inclusive_r2(form1_full, form1_base, clean_data)
        oos1 = self._oos_r2_by_group(form1_full, clean_data, 'CODE')
        evals['model1'] = {
            'r2_main': r2m1, 'r2_main_label': 'FE-space R2',
            'r2_inclusive': inc1, 'delta_inclusive_weather': del1,
            'oos_r2': oos1, 'oos_scheme': 'LOCO(CODE)'
        }
        # Model2: Only County FE  —— use clean_data
        r2m2 = self._get_model_stat(m2, 'rsquared')
        form2_full = f"{self.yield_column_name} ~ " + " + ".join(weather_vars + ['C(CODE)'])
        form2_base = f"{self.yield_column_name} ~ C(CODE)"
        inc2, del2 = self.inclusive_r2(form2_full, form2_base, clean_data)
        oos2 = self._oos_r2_by_group(form2_full, clean_data, 'Year')
        evals['model2'] = {
            'r2_main': r2m2, 'r2_main_label': 'FE-space R2',
            'r2_inclusive': inc2, 'delta_inclusive_weather': del2,
            'oos_r2': oos2, 'oos_scheme': 'LOYO(Year)'
        }
        # Model3: Linear Year + County FE  —— note use data with Yr column
        data_with_Yr = clean_data.copy()
        data_with_Yr['Yr'] = data_with_Yr['Year']-2001
        r2m3 = self._get_model_stat(m3, 'rsquared')
        form3_full = f"{self.yield_column_name} ~ Yr + " + " + ".join(weather_vars + ['C(CODE)'])
        form3_base = f"{self.yield_column_name} ~ Yr + C(CODE)"
        inc3, del3 = self.inclusive_r2(form3_full, form3_base, data_with_Yr)
        oos3 = self._oos_r2_by_group(form3_full, data_with_Yr, 'Year')
        evals['model3'] = {
            'r2_main': r2m3, 'r2_main_label': 'FE-space R2',
            'r2_inclusive': inc3, 'delta_inclusive_weather': del3,
            'oos_r2': oos3, 'oos_scheme': 'LOYO(Year)'
        }
        # Model4: County FE + Year FE 
        data_1980_2000 = clean_data[(clean_data['Year'] >= 1980) & (clean_data['Year'] <= 2000)]
        data_1980_2000['Yr'] = data_1980_2000['Year']-1990
        data_2001_2022 = clean_data[(clean_data['Year'] >= 2001) & (clean_data['Year'] <= 2022)]
        data_2001_2022['Yr'] = data_2001_2022['Year']-2011.5
        r2m4 = self._get_model_stat(m4, 'rsquared')
        form4_full = f"{self.yield_column_name} ~Yr+ " + " + ".join(weather_vars + ['C(CODE)'])
        form4_base = f"{self.yield_column_name} ~ Yr + C(CODE)"
        inc4, del4 = self.inclusive_r2(form4_full, form4_base, data_1980_2000)
        oos4 = self._oos_r2_by_group(form4_full, data_1980_2000, 'CODE')
        evals['model4'] = {
            'r2_main': r2m4, 'r2_main_label': 'FE-space R2',
            'r2_inclusive': inc4, 'delta_inclusive_weather': del4,
            'oos_r2': oos4, 'oos_scheme': 'LOCO(CODE)'
        }
        # Model5: County FE + Year FE  —— use clean_data
        r2m5 = self._get_model_stat(m5, 'rsquared')
        form5_full = f"{self.yield_column_name} ~ Yr+ " + " + ".join(weather_vars + ['C(CODE)'])
        form5_base = f"{self.yield_column_name} ~ Yr + C(CODE)"
        inc5, del5 = self.inclusive_r2(form5_full, form5_base, data_2001_2022)
        oos5 = self._oos_r2_by_group(form5_full, data_2001_2022, 'CODE')
        evals['model5'] = {
            'r2_main': r2m5, 'r2_main_label': 'FE-space R2',
            'r2_inclusive': inc5, 'delta_inclusive_weather': del5,
            'oos_r2': oos5, 'oos_scheme': 'LOCO(CODE)'
        }
        return evals


    def _fit_ols(self, formula, data):
        """
        return (base, robust)
        - base: statsmodels OLS regular estimation (convenient to take influence/residuals)
        - robust: try Driscoll–Kraay (PanelOLS), otherwise two-way cluster approximation, otherwise site-cluster
        """
        # First run base OLS (including FE virtual variables in formula; inclusive R² uses it)
        base = smf.ols(formula, data=data).fit()
        robust = base
        if not self.config.USE_CLUSTERED_SE:
            return base, robust
        has_interaction = (':' in formula) or ('*' in formula)
        if self.config.TRY_DRISCOLL_KRAAY and (not has_interaction):
            try:
                has_year_fe = 'C(Year)' in formula
                y, X = dmatrices(formula, data=data, return_type='dataframe')
                yname = y.columns[0]
                drop_cols = ('C(CODE)', 'C(Year)', 'Intercept')
                X = X[[c for c in X.columns if not c.startswith(drop_cols)]]
                if 'Year' in X.columns:
                    X = X.rename(columns={'Year': 'Year_reg'})
                idx = data[['CODE', 'Year']].rename(columns={'Year': 'Year_idx'}).reset_index(drop=True)
                df = pd.concat([idx, y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
                df = df.set_index(['CODE', 'Year_idx']).sort_index()
                bw = getattr(self.config, 'DK_BANDWIDTH', None)
                mod = PanelOLS(df[yname], df[X.columns],
                            entity_effects=True,
                            time_effects=has_year_fe)
                robust = mod.fit(cov_type='driscoll-kraay',
                                 bandwidth=bw,
                                 kernel='bartlett')
                print(f"Using Driscoll–Kraay SE (linearmodels){' | bw='+str(bw) if bw is not None else ''}")
                return base, robust
            except Exception as e:
                print(f"DK failed -> {type(e).__name__}: {e}")
        try:
            both = data[['CODE', 'Year']].astype(str).agg('-'.join, axis=1)
            robust = smf.ols(formula, data=data).fit(
                cov_type='cluster', cov_kwds={'groups': both}
            )
            print("Using two-way (site×year) clustered SE (approx.)")
            return base, robust
        except Exception as e:
            print(f"Two-way cluster failed -> {type(e).__name__}: {e}")
        try:
            robust = smf.ols(formula, data=data).fit(
                cov_type='cluster', cov_kwds={'groups': data['CODE']}
            )
            print("Using site-clustered SE")
        except Exception as e:
            print(f"Site-cluster failed -> {type(e).__name__}: {e}")
            print("Fallback to conventional SE")
        return base, robust


    def _run_diagnostics(self, model, data):
        """Improved diagnostic function"""
        diagnostics = {}
        try:
            fitted, residuals = self._get_model_attributes(model)
            diagnostics['r_squared'] = self._get_model_stat(model, 'rsquared')
            diagnostics['adj_r_squared'] = self._get_model_stat(model, 'rsquared_adj')
            diagnostics['f_statistic'] = self._get_model_stat(model, 'fstatistic')
            diagnostics['f_pvalue'] = self._get_model_stat(model, 'f_pvalue')
            diagnostics['n_obs'] = self._get_model_stat(model, 'nobs')
            diagnostics['aic'] = self._get_model_stat(model, 'aic')
            diagnostics['bic'] = self._get_model_stat(model, 'bic')
            model_type = type(model).__name__
            diagnostics['model_type'] = model_type
            if 'Panel' in model_type:
                diagnostics['r_squared_overall'] = getattr(model, 'rsquared_overall', None)
                diagnostics['r_squared_within'] = getattr(model, 'rsquared_within', None)
                diagnostics['r_squared_between'] = getattr(model, 'rsquared_between', None)
                diagnostics['entity_effects'] = getattr(model, 'entity_effects', None)
                diagnostics['time_effects'] = getattr(model, 'time_effects', None)
            if hasattr(model, 'resid') and hasattr(model, 'model'):
                try:
                    _, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
                    diagnostics['bp_test'] = {
                        'p_value': bp_pvalue,
                        'heteroskedasticity': bp_pvalue < self.config.ALPHA
                    }
                    dw_stat = durbin_watson(model.resid)
                    diagnostics['dw_test'] = {
                        'statistic': dw_stat,
                        'autocorrelation': abs(dw_stat - 2) > 0.5
                    }
                except Exception as e:
                    print(f"   Warning: Diagnostics test failed: {str(e)}")
                    diagnostics['bp_test'] = {'note': f'test failed: {str(e)}'}
                    diagnostics['dw_test'] = {'note': f'test failed: {str(e)}'}
            else:
                diagnostics['bp_test'] = {'note': 'not applicable to panel model'}
                diagnostics['dw_test'] = {'note': 'not applicable to panel model'}
            try:
                test_residuals = residuals[:5000] if len(residuals) > 5000 else residuals
                _, shapiro_p = stats.shapiro(test_residuals)
                diagnostics['normality_test'] = {
                    'shapiro_p': shapiro_p,
                    'normal': shapiro_p > self.config.ALPHA,
                    'sample_size': len(test_residuals)
                }
            except Exception as e:
                diagnostics['normality_test'] = {'note': f'test failed: {str(e)}'}
        except Exception as e:
            print(f"Warning: Diagnostics calculation error: {str(e)}")
            diagnostics['error'] = str(e)
        return diagnostics

    def create_visualization(self, result, save_plots=True):
        """create improved visualization charts"""
        if result is None:
            return
        model = result['base1']  
        weather_vars = result['weather_vars']
        region_name = result['region_name']
        fitted, residuals = self._get_model_attributes(model)
        fig = plt.figure(figsize=self.config.FIGSIZE)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Panel Regression Diagnostics - {region_name}', fontsize=16, fontweight='bold')

        # 1. residuals vs fitted values
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(fitted, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)

        # 2. Q-Q plot
        ax2 = fig.add_subplot(gs[0, 1])
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality)')
        ax2.grid(True, alpha=0.3)

        # 3. residuals histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
        ax3.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.4f}')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Residual Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. weather variable coefficients plot
        ax4 = fig.add_subplot(gs[1, :2])
        weather_coefs, weather_names, weather_pvals = [], [], []
        for var in weather_vars:
            if var in model.params.index:
                weather_coefs.append(model.params[var])
                weather_names.append(var)
                weather_pvals.append(model.pvalues[var])
        colors = ['red' if p < self.config.ALPHA else 'steelblue' for p in weather_pvals]
        ax4.barh(range(len(weather_coefs)), weather_coefs, color=colors, alpha=0.7)
        ax4.set_xlabel('Coefficients')
        ax4.set_ylabel('Weather Variables')
        ax4.set_title('Weather Variable Coefficients\n(Red = Significant at 5% level)')
        ax4.set_yticks(range(len(weather_names)))
        ax4.set_yticklabels(weather_names)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. standardized residuals
        ax5 = fig.add_subplot(gs[1, 2])
        standardized_resid = residuals / residuals.std()
        ax5.scatter(fitted, standardized_resid, alpha=0.6, s=20)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax5.axhline(y=2, color='orange', linestyle=':', alpha=0.8, label='±2σ')
        ax5.axhline(y=-2, color='orange', linestyle=':', alpha=0.8)
        ax5.set_xlabel('Fitted Values')
        ax5.set_ylabel('Standardized Residuals')
        ax5.set_title('Standardized Residuals')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. leverage values plot (only for statsmodels models)
        ax6 = fig.add_subplot(gs[2, 0])
        try:
            if hasattr(model, 'get_influence'):
                leverage = model.get_influence().hat_matrix_diag
                ax6.scatter(range(len(leverage)), leverage, alpha=0.6, s=20)
                ax6.axhline(y=2 * len(model.params) / len(leverage), color='red', linestyle='--',
                            label=f'2p/n = {2 * len(model.params) / len(leverage):.4f}')
                ax6.set_xlabel('Observation')
                ax6.set_ylabel('Leverage')
                ax6.set_title('Leverage Values')
                ax6.legend()
            else:
                ax6.text(0.5, 0.5, "Leverage Values\nNot available for\nPanel models",
                         ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Leverage Values')
        except Exception as e:
            ax6.text(0.5, 0.5, f"Leverage Values\nError: {str(e)}",
                     ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Leverage Values')
        ax6.grid(True, alpha=0.3)

        # 7. Cook's Distance (only for statsmodels models)
        ax7 = fig.add_subplot(gs[2, 1])
        try:
            if hasattr(model, 'get_influence'):
                influence = model.get_influence()
                cooks_d = influence.cooks_distance[0]
                ax7.scatter(range(len(cooks_d)), cooks_d, alpha=0.6, s=20)
                ax7.axhline(y=4 / len(cooks_d), color='red', linestyle='--',
                            label=f'4/n = {4 / len(cooks_d):.4f}')
                ax7.set_xlabel('Observation')
                ax7.set_ylabel("Cook's Distance")
                ax7.set_title("Cook's Distance")
                ax7.legend()
            else:
                ax7.text(0.5, 0.5, "Cook's Distance\nNot available for\nPanel models",
                         ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title("Cook's Distance")
        except Exception as e:
            ax7.text(0.5, 0.5, f"Cook's Distance\nError: {str(e)}",
                     ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title("Cook's Distance")
        ax7.grid(True, alpha=0.3)

        # 8. annual yield trend
        ax8 = fig.add_subplot(gs[2, 2])
        data = result['data']
        yield_col = self.yield_column_name
        yearly_mean = data.groupby('Year')[yield_col].mean()
        ax8.plot(yearly_mean.index, yearly_mean.values, marker='o', linewidth=2, markersize=4)
        ax8.set_xlabel('Year')
        ax8.set_ylabel('Mean Yield')
        ax8.set_title('Annual Yield Trend')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0.8,1.8)

        if save_plots:
            output_dir = Path(self.config.OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True)
            filename = f'{region_name.lower()}_regression_diagnostics.pdf'
            plt.savefig(output_dir / filename, dpi=self.config.DPI, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def create_all_models_visualization(self, result):
        """create visualization charts for all models"""
        if result is None:
            return
        region_name = result['region_name']
        weather_vars = self.config.CUSTOM_WEATHER_VARS
        models_to_show = ['model1_robust', 'model2_robust', 'model3_robust', 'model4_robust', 'model5_robust']
        model_descriptions = {
            'model1_robust': 'with year fixed effects',
            'model2_robust': 'without time control',
            'model3_robust': 'linear year trend',
            'model4_robust': '1980-1999 subsample',
            'model5_robust': '2000-2022 subsample'
        }
        valid_models = {}
        for model_key in models_to_show:
            model = result.get(model_key)
            if model:
                valid_models[model_key] = model
        if not valid_models:
            return

        # 1. coefficients comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{region_name} - all models coefficients comparison', fontsize=16, fontweight='bold')
        for i, var in enumerate(weather_vars):
            ax = axes[i]
            model_names = []
            coefficients = []
            errors = []
            colors = []
            color_map = {'model1_robust': 'blue', 'model2_robust': 'red', 'model3_robust': 'green',
                         'model4_robust': 'orange', 'model5_robust': 'purple'}
            for model_key, model in valid_models.items():
                if var in model.params.index:
                    model_names.append(model_key)
                    coefficients.append(model.params[var])
                    se = self._get_model_se(model, var)
                    errors.append(se if se is not None else 0)
                    colors.append(color_map.get(model_key, 'gray'))
            if coefficients:
                ax.bar(model_names, coefficients, yerr=errors,
                       capsize=5, alpha=0.7, color=colors)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'{var}')
                ax.set_ylabel('coefficients')
                ax.tick_params(axis='x', rotation=45)
                for j, (model_key, model) in enumerate(valid_models.items()):
                    if var in model.params.index:
                        pval = model.pvalues[var] if hasattr(model, 'pvalues') and var in model.pvalues.index else None
                        if pval and pval < 0.001:
                            ax.text(j, coefficients[model_names.index(model_key)], '***',
                                    ha='center', va='bottom')
                        elif pval and pval < 0.01:
                            ax.text(j, coefficients[model_names.index(model_key)], '**',
                                    ha='center', va='bottom')
                        elif pval and pval < 0.05:
                            ax.text(j, coefficients[model_names.index(model_key)], '*',
                                    ha='center', va='bottom')
        plt.tight_layout()
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'{region_name.lower()}_all_models_coefficients.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_detailed_results(self, result):
        """save detailed results"""
        if result is None:
            return
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        region_name = result['region_name']
        model = result['model3_robust']
        model_robust = result.get('model3_robust', model)
        liner_data = result['liner_data']  # get cleaned data
        coef_rows = []
        params, se_s, t_s, p_s = self._extract_param_series(model_robust)
        for var in params.index:
            coef = params[var]
            se = se_s.get(var, np.nan)
            tval = t_s.get(var, np.nan)
            pval = p_s.get(var, np.nan)
            lo, hi = self._safe_conf_int(model_robust, var)
            coef_rows.append({
                'Variable': var,
                'Coefficient': coef,
                'Std_Error': se,
                'T_Statistic': tval,
                'P_Value': pval,
                'Conf_Lower': lo,
                'Conf_Upper': hi,
                'Significant': bool(pval < self.config.ALPHA) if pd.notnull(pval) else False
            })
        pd.DataFrame(coef_rows).to_csv(output_dir / f'{region_name.lower()}_coefficients.csv', index=False)
        try:
            ev = result.get('evaluations', {})
            wald = result.get('wald', {})
            coefs = result.get('coef_tables', {})
            em = result.get('error_metrics', {})
            rd = result.get('resid_diag', {})
            for k, df in coefs.items():
                if df is not None and not df.empty:
                    df.to_csv(output_dir / f'{region_name.lower()}_{k}_coef_enhanced.csv', encoding='utf-8')
            if em:
                pd.DataFrame(em).to_csv(output_dir / f'{region_name.lower()}_error_metrics.csv', encoding='utf-8')
            if rd:
                pd.DataFrame(rd).to_csv(output_dir / f'{region_name.lower()}_residual_diagnostics.csv', encoding='utf-8')
            if ev:
                pd.DataFrame(ev).to_csv(output_dir / f'{region_name.lower()}_model_evaluations.csv', encoding='utf-8')
            if wald:
                pd.DataFrame(wald).to_csv(output_dir / f'{region_name.lower()}_wald_joint_tests.csv', encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not save enhanced outputs: {e}")

    def display_detailed_regression_results(self):
        model_descriptions = {
            'model1': 'with year fixed effects',
            'model2': 'no time controls',
            'model3': 'linear year trend',
            'model4': '1980-1999 subsample',
            'model5': '2000-2022 subsample'
        }
        for region_name in ['Irrigation', 'Rainfed']:
            result_key = region_name.lower()
            if result_key not in self.results:
                continue
            result = self.results[result_key]
            models_to_show = ['model1_robust', 'model2_robust', 'model3_robust', 'model4_robust', 'model5_robust']
            weather_vars = self.config.CUSTOM_WEATHER_VARS
            valid_models = {}
            for model_key in models_to_show:
                model = result.get(model_key)
                if model:
                    valid_models[model_key] = model
                    _ = model_descriptions[model_key.replace('_robust', '')] if model_key.endswith('_robust') else model_descriptions.get(model_key, model_key)
            for var in weather_vars:
                _ = var  
            self._save_all_models_to_csv(region_name, valid_models)

    def _save_all_models_to_csv(self, region_name, valid_models):
        """save all models' detailed result to CSV file"""
        outdir = Path(self.config.OUTPUT_DIR)
        outdir.mkdir(exist_ok=True)
        weather_vars = self.config.CUSTOM_WEATHER_VARS
        comparison_data = []
        for var in weather_vars:
            row_data = {'Variable': var}
            if var == 'gs_tmin_mean':
                row_data['Unit'] = 'degree Celsius' if not self.config.STANDARDIZE_WEATHER else 'standardized'
            elif var == 'gs_vpd_mean':
                row_data['Unit'] = 'hPa' if not self.config.STANDARDIZE_WEATHER else 'standardized'
            elif var == 'prep_total':
                row_data['Unit'] = 'cm' if not self.config.STANDARDIZE_WEATHER else 'standardized'
            for model_key, model in valid_models.items():
                if var in model.params.index:
                    coef = model.params[var]
                    se = self._get_model_se(model, var)
                    pval = model.pvalues[var] if hasattr(model, 'pvalues') and var in model.pvalues.index else None
                    sig = '***' if pval and pval < 0.001 else '**' if pval and pval < 0.01 else '*' if pval and pval < 0.05 else ''
                    row_data[f'{model_key}_coef'] = coef
                    row_data[f'{model_key}_se'] = se
                    row_data[f'{model_key}_pval'] = pval
                    row_data[f'{model_key}_sig'] = sig
                else:
                    row_data[f'{model_key}_coef'] = np.nan
                    row_data[f'{model_key}_se'] = np.nan
                    row_data[f'{model_key}_pval'] = np.nan
                    row_data[f'{model_key}_sig'] = ''
            comparison_data.append(row_data)
        df_comparison = pd.DataFrame(comparison_data)
        comparison_file = outdir / f"{region_name.lower()}_all_models_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)

    def run_panel_regression(self, data, region_name):
        weather_vars = self.get_weather_variables(data)
        if not weather_vars:
            print("No weather variables found!")
            return None
        regression_vars = [self.yield_column_name, 'Year', 'CODE'] + weather_vars
        clean_data = data[regression_vars].dropna()
        if clean_data.empty:
            print("No clean data available for regression!")
            return None
        vif_results = self._check_multicollinearity(clean_data, weather_vars)
        rhs_common = ['C(Year)'] + weather_vars + ['C(CODE)']
        formula = f"{self.yield_column_name} ~ " + " + ".join(rhs_common)
        base1, model1_robust = self._fit_ols(formula, clean_data)
        rhs_no_year = weather_vars + ['C(CODE)']
        formula_no_year = f"{self.yield_column_name} ~ " + " + ".join(rhs_no_year)
        base2, model2_robust = self._fit_ols(formula_no_year, clean_data)
        liner_data = clean_data.copy()
        liner_data['Yr'] = liner_data['Year']-2001
        formula_linear_year = f"{self.yield_column_name} ~ Yr + " + " + ".join(weather_vars + ['C(CODE)'])
        base3, model3_robust = self._fit_ols(formula_linear_year, liner_data)
        data_1980_2000 = clean_data[(clean_data['Year'] >= 1980) & (clean_data['Year'] <= 2000)]
        data_1980_2000['Yr'] = data_1980_2000['Year']-1990
        if len(data_1980_2000) > 50:
            base4, model4_robust = self._fit_ols(formula_linear_year, data_1980_2000)
        else:
            print("   Insufficient data for 1980-2000 model")
            model4_robust = None
        data_2001_2022 = clean_data[(clean_data['Year'] >= 2001) & (clean_data['Year'] <= 2022)]
        data_2001_2022['Yr'] = data_2001_2022['Year']-2011.5
        if len(data_2001_2022) > 50:
            base5, model5_robust = self._fit_ols(formula_linear_year, data_2001_2022)
        else:
            print("   Insufficient data for 2001-2022 model")
            model5_robust = None
        diagnostics = self._run_diagnostics(model3_robust, clean_data)
        diagnostics['vif'] = vif_results
        evaluations = self._evaluate_models(
            clean_data, weather_vars,
            base1, model1_robust,
            base2, model2_robust,
            base3, model3_robust,
            base4, model4_robust,
            base5, model5_robust
        )
        wald1 = self._wald_joint_test(model1_robust, weather_vars)
        wald2 = self._wald_joint_test(model2_robust, weather_vars)
        wald3 = self._wald_joint_test(model3_robust, weather_vars)
        wald4 = self._wald_joint_test(model4_robust, weather_vars)
        wald5 = self._wald_joint_test(model5_robust, weather_vars)
        coef_tables = {
            'model1': self._make_coef_table(model1_robust, clean_data, weather_vars, year_var_in_model='Year'),
            'model2': self._make_coef_table(model2_robust, clean_data, weather_vars, year_var_in_model='Year'),
            'model3': self._make_coef_table(model3_robust, liner_data, weather_vars, year_var_in_model='Yr'),
            'model4': self._make_coef_table(model4_robust, data_1980_2000, weather_vars, year_var_in_model='Year'),
            'model5': self._make_coef_table(model5_robust, data_2001_2022, weather_vars, year_var_in_model='Year'),
        }
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        error_metrics = {
            'model1': self._compute_error_metrics(base1, model1_robust, clean_data, 
                                                save_detailed_table=True, 
                                                output_dir=output_dir, 
                                                region_name=f"{region_name}_model1"),
            'model2': self._compute_error_metrics(base2, model2_robust, clean_data, 
                                                save_detailed_table=True, 
                                                output_dir=output_dir, 
                                                region_name=f"{region_name}_model2"),
            'model3': self._compute_error_metrics(base3, model3_robust, liner_data, 
                                                save_detailed_table=True, 
                                                output_dir=output_dir, 
                                                region_name=f"{region_name}_model3"),
            'model4': self._compute_error_metrics(base4, model4_robust, data_1980_2000, 
                                                save_detailed_table=True, 
                                                output_dir=output_dir, 
                                                region_name=f"{region_name}_model4"),
            'model5': self._compute_error_metrics(base5, model5_robust, data_2001_2022, 
                                                save_detailed_table=True, 
                                                output_dir=output_dir, 
                                                region_name=f"{region_name}_model5"),
        }
        resid_diag = {}
        resid_diag['model1'], _, _, _ = self._residual_panel_diagnostics(base1, model1_robust, clean_data)
        resid_diag['model2'], _, _, _ = self._residual_panel_diagnostics(base2, model2_robust, clean_data)
        resid_diag['model3'], _, _, _ = self._residual_panel_diagnostics(base3, model3_robust, liner_data)
        resid_diag['model4'], _, _, _ = self._residual_panel_diagnostics(base4, model4_robust, data_1980_2000)
        resid_diag['model5'], _, _, _ = self._residual_panel_diagnostics(base5, model5_robust, data_2001_2022)
        result = {
            'model1_robust': model1_robust,  # Year FE + County FE
            'model2_robust': model2_robust,  # County FE only (no time control)
            'model3_robust': model3_robust,  # Linear Year trend + County FE
            'model4_robust': model4_robust,  # 1980-1999 subsample
            'model5_robust': model5_robust,  # 2000-2022 subsample
            'base1': base1, 'base2': base2, 'base3': base3, 'base4': base4, 'base5': base5,
            'data': clean_data,
            'liner_data': liner_data,
            'data_1980_2000': data_1980_2000 if model4_robust else None,
            'data_2001_2022': data_2001_2022 if model5_robust else None,
            'weather_vars': weather_vars,
            'diagnostics': diagnostics,
            'formula': formula,
            'formula_no_year': formula_no_year,
            'formula_linear_year': formula_linear_year,
            'region_name': region_name,
            'evaluations': evaluations,
            'wald': {'model1': wald1, 'model2': wald2, 'model3': wald3, 'model4': wald4, 'model5': wald5},
            'coef_tables': coef_tables,
            'error_metrics': error_metrics,
            'resid_diag': resid_diag
        }
        return result

    def run_analysis(self):
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return
        self.prepare_data()
        irrigation_result = self.run_panel_regression(self.irrigation_data, "Irrigation")
        if irrigation_result:
            self.results['irrigation'] = irrigation_result
            self.create_visualization(irrigation_result)
            self.create_all_models_visualization(irrigation_result)
            self.save_detailed_results(irrigation_result)
        rainfed_result = self.run_panel_regression(self.rainfed_data, "Rainfed")
        if rainfed_result:
            self.results['rainfed'] = rainfed_result
            self.create_visualization(rainfed_result)
            self.create_all_models_visualization(rainfed_result)
            self.save_detailed_results(rainfed_result)
        self.display_detailed_regression_results()

if __name__ == "__main__":
    analyzer = ImprovedPanelRegression()
    analyzer.run_analysis()
