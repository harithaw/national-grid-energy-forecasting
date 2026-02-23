# Sri Lanka National Grid – Daily Energy Generation Forecasting
## Project Report

---

## 1. Dataset and Pre-processing

### 1.1 Data Source

The dataset is sourced from the **Ceylon Electricity Board (CEB) / Public Utilities Commission of Sri Lanka (PUCSL)** daily generation records. It is stored locally as:

```
data/merged-generation-profile-2017-2026.csv
```

The file consolidates daily energy generation figures (in GWh/day) from multiple generation sources across the Sri Lanka national grid, covering **2017-01-01 to 2026-02-22** — a span of nine years and approximately 3,340 days.

### 1.2 Dataset Size

| Dimension | Value |
|-----------|-------|
| Total rows (days) | 3,340 |
| Raw columns | 12 |
| Engineered columns (final) | 29 |
| Date range | 2017-01-01 → 2026-02-22 |
| Missing days (gaps) | 0 (after reindexing) |

### 1.3 Raw Features

The raw CSV contains per-source daily generation columns:

| Raw Column | Renamed To | Generation Type |
|---|---|---|
| Solar (Energy in GWh/day) | `Solar` | Utility-scale solar |
| Rooftop Solar (Est.) (Energy in GWh/day) | `Rooftop_Solar` | Distributed rooftop solar |
| Solar (Estimated) (Energy in GWh/day) | `Solar_Estimated` | Estimated solar contribution |
| Mini Hydro (Telemetered) (Energy in GWh/day) | `Mini_Hydro_Telemetered` | Small hydro, metered |
| Mini Hydro (Estimated) (Energy in GWh/day) | `Mini_Hydro_Estimated` | Small hydro, estimated |
| Biomass and Waste Heat (Energy in GWh/day) | `Biomass_Waste` | Biomass/waste co-generation |
| Wind (Energy in GWh/day) | `Wind` | Wind farms |
| Major Hydro (Energy in GWh/day) | `Major_Hydro` | Large hydro reservoirs |
| Oil (IPP) (Energy in GWh/day) | `Oil_IPP` | Independent power producers (oil) |
| Oil (CEB) (Energy in GWh/day) | `Oil_CEB` | CEB-owned oil plants |
| Coal (Energy in GWh/day) | `Coal` | Coal power station (Norochcholai) |

### 1.4 Target Variable

```
Total_Generation  (GWh/day)
```

Computed as the sum of all generation sources:

```
Total_Generation = Solar_Total + Mini_Hydro + Biomass_Waste + Wind
                 + Major_Hydro + Oil_IPP + Oil_CEB + Coal
```

Where:
- `Solar_Total = Solar + Solar_Estimated + Rooftop_Solar`
- `Mini_Hydro  = Mini_Hydro_Telemetered + Mini_Hydro_Estimated`

**Target statistics (2017–2026):**

| Statistic | Value (GWh/day) |
|-----------|----------------|
| Mean | 42.40 |
| Std dev | 6.49 |
| Min | 24.95 |
| Median | 41.57 |
| 75th percentile | 45.93 |
| Max | 90.41 |

**Yearly mean generation (upward trend visible):**

| Year | Mean GWh/day |
|------|-------------|
| 2017 | 36.99 |
| 2018 | 39.55 |
| 2019 | 39.67 |
| 2020 | 38.90 |
| 2021 | 43.06 |
| 2022 | 42.35 |
| 2023 | 42.09 |
| 2024 | 44.86 |
| 2025 | 52.00 |
| 2026 (Jan–Feb) | 57.40 |

The rising trend in 2025–2026 reflects ongoing capacity additions (primarily solar) in the Sri Lanka grid.

**Mean contribution per source (GWh/day):**

| Source | Mean GWh/day | Share (approx.) |
|---|---|---|
| Coal | 14.28 | 33.7% |
| Major Hydro | 13.34 | 31.5% |
| Oil IPP | 4.47 | 10.6% |
| Oil CEB | 4.79 | 11.3% |
| Solar Total | 2.37 | 5.6% |
| Mini Hydro | 1.55 | 3.7% |
| Wind | 1.47 | 3.5% |
| Biomass/Waste | 0.15 | 0.4% |

### 1.5 Pre-processing Steps

All pre-processing is implemented in `preprocess.py → load_and_preprocess()`.

**Step 1 — Date parsing and sorting**
The `Date (GMT+5:30)` column is parsed to `datetime`, set as the index, and the DataFrame is sorted chronologically.

**Step 2 — Full date range reindexing**
A complete daily date range from `min(Date)` to `max(Date)` is created via `pd.date_range`. The DataFrame is reindexed to this range, inserting `NaN` for any missing calendar days. This guarantees a gap-free regular daily time series required by Darts.

**Step 3 — Missing value imputation**
For each raw generation column:
- Values recorded before that column's first valid observation are set to `0.0` (the source did not yet exist).
- Remaining `NaN` values are filled using **time-based linear interpolation** (`interpolate(method="time")`).
- Results are clipped to `≥ 0` and any residual `NaN` is filled with `0`.

**Step 4 — Composite feature construction**
Aggregate columns (`Solar_Total`, `Mini_Hydro`, `Total_Generation`) are computed from their component parts.

**Step 5 — Cyclical calendar features (future covariates)**
Sinusoidal encodings ensure the model captures smooth periodicity without imposing ordinal ordering:

| Feature | Formula |
|---|---|
| `Sin_DayOfYear` | sin(2π × day_of_year / 365.25) |
| `Cos_DayOfYear` | cos(2π × day_of_year / 365.25) |
| `Sin_Month` | sin(2π × month / 12) |
| `Cos_Month` | cos(2π × month / 12) |
| `Sin_DayOfWeek` | sin(2π × day_of_week / 7) |
| `Cos_DayOfWeek` | cos(2π × day_of_week / 7) |
| `IsWeekend` | 1 if Saturday or Sunday, else 0 |
| `TrendIndex` | Integer index 0, 1, 2, … (global trend) |

**Step 6 — Rolling statistical features (past covariates)**
Computed on `Total_Generation` to capture recent momentum and volatility:

| Feature | Window | Description |
|---|---|---|
| `Roll7_Mean` | 7 days | Weekly rolling mean |
| `Roll30_Mean` | 30 days | Monthly rolling mean |
| `Roll90_Mean` | 90 days | Quarterly rolling mean |
| `Roll365_Mean` | 365 days (min 30) | Annual rolling mean |
| `Roll30_Std` | 30 days | Monthly rolling standard deviation |
| `Monthly_Avg` | Full history | Mean for each calendar month (grouped) |

**Step 7 — Year-over-year delta feature**

```python
YoY_Delta = Total_Generation(t) - Total_Generation(t - 365)
```

Values in the first year (where the lag is unavailable) are filled with `0.0`. This feature captures whether generation is running above or below the same period last year — a key signal for structural growth and seasonal anomalies.

---

## 2. Machine Learning Model

### 2.1 Model Architecture

The model used is a **LightGBM-based time series forecasting model** from the [Darts](https://unit8co.github.io/darts/) library:

```
darts.models.LightGBMModel
```

LightGBM (Light Gradient Boosting Machine) is a gradient-boosted decision tree algorithm. In the Darts `GlobalForecastingModel` framework, it is wrapped as a **tabular regression** problem: the model receives a sliding window of lagged target values and covariate values as features and predicts the next time step's output. This is also called the **DIRECT** forecasting strategy.

**Why LightGBM was chosen:**
- Strong performance on tabular, non-stationary time series.
- Handles mixed numerical features (lags, calendar, rolling stats) natively.
- Computationally efficient for rolling retrain regimes.
- Interpretable via SHAP feature importances.
- No requirement for stationarity or explicit differencing.

### 2.2 Covariate Structure

| Covariate Type | Features | Count |
|---|---|---|
| **Past covariates** (only known up to present) | `Solar_Total`, `Mini_Hydro`, `Biomass_Waste`, `Wind`, `Major_Hydro`, `Oil_IPP`, `Oil_CEB`, `Coal`, `Roll7_Mean`, `Roll30_Mean`, `Roll90_Mean`, `Roll365_Mean`, `Roll30_Std`, `Monthly_Avg`, `YoY_Delta` | 15 |
| **Future covariates** (known for all future dates) | `Sin_DayOfYear`, `Cos_DayOfYear`, `Sin_Month`, `Cos_Month`, `Sin_DayOfWeek`, `Cos_DayOfWeek`, `IsWeekend`, `TrendIndex` | 8 |

### 2.3 Lag Structure (tuned)

| Lag Type | Values |
|---|---|
| Target lags | −365, −182, −90, −28, −14, −7, −3, −2, −1 |
| Past covariate lags | −1, −2, −7, −14, −28 |
| Future covariate range | [−7, +1] |

The target lags explicitly include annual (−365), semi-annual (−182), and quarterly (−90) lookbacks, enabling the model to capture strong seasonality typical of hydro-dominated grids.

---

## 3. Model Training and Evaluation

### 3.1 Train / Validation / Test Split

A strict **chronological split** is applied — no data from the future leaks into training.

| Split | Date Range | Days |
|---|---|---|
| **Train** | 2017-01-01 → 2023-12-31 | 2,556 |
| **Validation** | 2024-01-01 → 2024-12-31 | 366 |
| **Test** | 2025-01-01 → 2026-02-22 | 418 |

For final test evaluation, the model is first refitted on **Train + Validation** (2017–2024) then assessed on the held-out 2025–2026 test set.

### 3.2 Hyperparameter Tuning

Hyperparameters were optimised using **Optuna** (v3.x) with the **Tree-structured Parzen Estimator (TPE)** sampler. The study is persisted to a SQLite database (`artifacts/optuna_study.db`, study name: `lgbm_forecast_v3`) allowing resumable trials.

**Tuning regime:**
- Objective: minimise validation RMSE using `retrain=True` rolling historical forecasts over 2024.
- 30 trials completed.
- Best result at **trial #29**.

**Search space:**

| Parameter | Range / Choices |
|---|---|
| `num_leaves` | 31 – 255 |
| `n_estimators` | 300 – 1,500 |
| `learning_rate` | 0.005 – 0.10 (log-uniform) |
| `max_depth` | −1, 6, 8, 10, 12 |
| `min_child_samples` | 5 – 40 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha` | 1e-4 – 1.0 (log-uniform) |
| `reg_lambda` | 1e-4 – 1.0 (log-uniform) |
| `target_lags` | Fixed + optional lags at −182 and −730 |
| `past_cov_depth` | shallow / medium / deep |
| `train_length` | 760, 900, 1095, 1460 days |
| `stride` | 7, 14, 30 days |

**Best hyperparameters (trial #29):**

| Parameter | Value |
|---|---|
| `num_leaves` | 65 |
| `n_estimators` | 615 |
| `learning_rate` | 0.006958 |
| `max_depth` | 10 |
| `min_child_samples` | 15 |
| `subsample` | 0.8144 |
| `colsample_bytree` | 0.9854 |
| `reg_alpha` | 0.004792 |
| `reg_lambda` | 0.402349 |
| `target_lags` | [−365, −182, −90, −28, −14, −7, −3, −2, −1] |
| `past_cov_lags` | [−1, −2, −7, −14, −28] |
| `future_cov_lags` | [−7, +1] |
| `train_length` | 760 days |
| `stride` | 30 days |

### 3.3 Evaluation Strategy

The test set is evaluated using **adaptive rolling retraining**:

```
retrain=True, train_length=760 days, stride=30 days
```

Every 30 days (monthly), the model is fully retrained from scratch on the most recent 760 days of data before producing the next 30 forecasts. This mirrors a realistic deployment scenario where an operator retrains the model periodically as new data arrives.

This strategy was used consistently during both Optuna tuning and final evaluation, preventing any mismatch between tuning and production regimes.

### 3.4 Performance Metrics

**Validation set (2024, retrain=False, fitted on 2017–2023):**

| Metric | Value |
|---|---|
| RMSE | 0.9655 GWh/day |
| MAE | 0.6607 GWh/day |
| MAPE | 1.53% |

**Test set (2025–2026, rolling retrain):**

| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | **2.9861 GWh/day** | Average magnitude of prediction error |
| **MAE** | **2.0437 GWh/day** | Median-like error; less sensitive to outliers |
| **MAPE** | **4.07%** | Relative error; 4% of ~42 GWh/day mean |
| **MSE** | 8.9169 GWh²/day² | Squared error (penalises large errors more) |
| **R²** | **0.7866** | 78.7% of variance in daily generation explained |

### 3.5 Interpretation of Results

**Validation RMSE (0.97) vs Test RMSE (2.99):**
The significant gap between validation and test error is expected and meaningful:
- The validation period (2024) overlaps with the training distribution (2017–2023), even though it is held out.
- The test period (2025–2026) represents a structurally different grid: total generation jumped from ~44.9 GWh/day (2024 mean) to ~52.0 GWh/day (2025), a **+16% year-over-year increase**. This is driven by rapid solar capacity additions and growing demand. No model trained on historical data can perfectly anticipate step-changes in grid capacity.

**MAPE of 4.07%:**
For operational grid planning, MAPE under 5% on a one-day-ahead forecast is generally considered acceptable. The Sri Lanka grid operates at 40–60 GWh/day, so a 2 GWh/day MAE corresponds to roughly ±5% of typical daily output — within the tolerance of most dispatch planning systems.

**R² of 0.79:**
The model explains approximately 79% of the variance in daily total generation. The remaining 21% is attributable to unpredictable factors such as unplanned plant outages, sudden weather events, and demand spikes — none of which are encoded in the feature set.

**Artifacts produced:**
- `artifacts/prediction_plot.png` — Daily actual vs predicted time series + residual bar chart (test period)
- `artifacts/monthly_comparison.png` — Monthly average actual vs predicted grouped bar chart
- `artifacts/actual_vs_predicted.png` — Scatter plot of actual vs predicted, coloured by error magnitude

---

## 4. Explainability and Interpretation

### 4.1 Method: SHAP (SHapley Additive exPlanations)

Explainability is implemented via the **Darts `ShapExplainer`** wrapper, which internally uses the `shap` library's `TreeExplainer` — optimised for tree-based models like LightGBM.

SHAP values are grounded in cooperative game theory. For each prediction, each feature is assigned a contribution (in GWh/day) representing how much that feature pushed the model output above or below the expected baseline. SHAP values are:
- **Consistent**: if a feature's impact on the model increases, its SHAP value increases.
- **Locally accurate**: the sum of all SHAP values equals the model output minus the baseline.
- **Comparable across features and samples**.

**Background dataset:** 2020-01-01 → 2024-12-31 (5 years, ~1,827 days). This is wide enough to cover the maximum lag of 730 days used by the model.

**Foreground dataset:** The most recent 800 days (approximately 2023-11–2026-02), representing recent grid behaviour.

**Artifacts produced:**
- `artifacts/shap_summary.png` — Beeswarm plot: each dot is one day, x-axis = SHAP value (GWh/day impact), colour = feature value (high/low). Shows the global distribution of feature influence.
- `artifacts/shap_local.png` — Bar chart for the most recent test day: top 20 features ranked by absolute SHAP value, showing whether each feature is pushing generation up (red) or down (blue).

### 4.2 What the Model Has Learned

Based on SHAP analysis, the model has learned the following key behaviours:

**Lagged total generation is the dominant signal.**
The lags `lag_-1` (yesterday's generation), `lag_-7` (same day last week), `lag_-365` (same day last year), and `lag_-28` are consistently among the top contributors. This reflects the strong autocorrelation in grid output — generation levels persist day-to-day, repeat weekly (week-day demand patterns), and repeat annually (seasonal hydrology and solar irradiance).

**Rolling means encode medium-term momentum.**
`Roll365_Mean` and `Roll30_Mean` have high SHAP magnitudes. The 365-day rolling mean acts as a slowly-drifting baseline — when the grid is structurally producing more than the recent year's average (e.g., after new capacity comes online), the model adjusts its forecast upward. The 30-day mean captures shorter-term episodes such as a dry spell reducing hydro output.

**YoY_Delta captures structural growth.**
The year-over-year delta feature has consistently positive SHAP values in 2025–2026, reflecting the accelerating generation growth driven by solar additions. Without this feature, the model would under-forecast during periods of structural capacity expansion.

**Major hydro and coal are high-information past covariates.**
`Major_Hydro` and `Coal` contribute strongly. Major hydro is weather-sensitive (reservoir levels depend on rainfall) and seasonally variable. Coal output is relatively stable but its lagged values signal whether the grid is running in base-load or peaking mode.

**Calendar features encode known demand cycles.**
`Sin_DayOfYear` and `Cos_DayOfYear` capture the annual cycle: peak demand during the hot dry season (Feb–April) and reduced hydro generation. `IsWeekend` captures lower industrial demand on Saturdays and Sundays.

**TrendIndex captures long-run capacity growth.**
The monotonically increasing trend index captures the steady upward drift in generation levels over 2017–2026 that is not explained by seasonal factors alone.

### 4.3 Most Influential Features (Global Ranking)

Based on mean absolute SHAP values across the foreground dataset:

| Rank | Feature | Domain Justification |
|---|---|---|
| 1 | `lag_-1` (yesterday's generation) | Strong daily autocorrelation |
| 2 | `Roll365_Mean` | Long-run baseline / capacity level |
| 3 | `lag_-365` (same day last year) | Annual seasonality (hydrology, solar) |
| 4 | `Major_Hydro (lag)` | Weather-driven hydro variability |
| 5 | `Roll30_Mean` | Short-term momentum |
| 6 | `lag_-7` (same day last week) | Day-of-week demand patterns |
| 7 | `YoY_Delta` | Structural capacity growth |
| 8 | `Coal (lag)` | Base-load / thermal dispatch state |
| 9 | `TrendIndex` | Long-run growth trend |
| 10 | `Sin_DayOfYear` / `Cos_DayOfYear` | Annual seasonal cycle |

### 4.4 Alignment with Domain Knowledge

The model's learned behaviour aligns well with grid-domain expectations:

- **Hydro-dependence**: Major Hydro accounts for ~31% of mean generation and is the largest single source of volatility (dependent on monsoon rainfall). The model correctly weights hydro-related features highly.
- **Annual seasonality**: Sri Lanka experiences two monsoon seasons (South-West: May–September; North-East: October–January). The annual lag (−365) and sinusoidal calendar features align with this pattern.
- **Thermal base-load**: Coal operates near constant output as base load; its lagged values correctly signal grid-wide utilisation levels.
- **Solar growth**: The rising `TrendIndex` and positive `YoY_Delta` in 2025–2026 align with the documented rapid solar capacity expansion in Sri Lanka during this period.
- **Weekend effect**: `IsWeekend` being a significant feature aligns with lower industrial electricity demand on weekends — a universal grid characteristic.

---

## 5. Critical Discussion

### 5.1 Limitations of the Model

**Forecast horizon limitations.**
The model uses `output_chunk_length=1` (one-step-ahead) with auto-regression for longer horizons. Auto-regressive forecasting compounds errors — each future step feeds back as an input, so 90-day forecasts carry substantially more uncertainty than 1-day forecasts. The current architecture is optimised for short-range operational forecasting (1–7 days ahead), not long-range planning.

**No weather inputs.**
The model has no direct access to meteorological data (rainfall, temperature, cloud cover, wind speed). This is a fundamental limitation: hydro generation depends directly on reservoir inflow (rainfall), and solar generation depends on irradiance. The rolling features and lagged generation values serve as proxies but cannot capture sudden weather anomalies such as an unexpected drought or an unusually cloudy month.

**Static covariate assumptions for future forecast.**
When generating the 90-day forward forecast, future values of past covariates (energy source contributions) are unknown. The current implementation tiles the last observed day's values forward as a constant — a necessary but simplistic assumption. Source mix proportions are estimated from the trailing 90-day average.

**Structural break sensitivity.**
The test period (2025–2026) exhibits a structural break: generation grew 16% YoY, faster than any prior period. The model partially adapts via the rolling retrain and `YoY_Delta` feature, but a sudden capacity addition not reflected in the training window can cause systematic under-forecasting until the retrain window catches up.

**Increasing uncertainty over the forecast horizon.**
The 90-day forward forecast uses auto-regression: each future step's prediction is fed back as an input for the next step, compounding uncertainty. The 80% confidence band (p10–p90) widens noticeably beyond ~30 days, and the interval should be interpreted with increasing caution as the horizon extends. Short-range forecasts (1–7 days) remain tight and operationally reliable.

### 5.2 Data Quality Issues

**Estimated and telemetered readings mixed.**
Mini Hydro and Solar both have a "Telemetered" and an "Estimated" column. Estimated values may reflect modelled or imputed data from the source (CEB), introducing a layer of uncertainty that is not distinguished by the model.

**Column availability changes over time.**
Some generation sources (e.g., rooftop solar) were not metered in 2017. The preprocessing sets pre-first-observation values to zero, which is correct physically but means the model may see abnormally low historical values for these sources — potentially distorting rolling features.

**Single data source.**
All data comes from one official source (CEB/PUCSL). If the source reports data with errors, delays, or revisions, there is no cross-validation against an independent measurement.

**No public holiday data.**
Sri Lanka observes Poya (full moon) public holidays every month — approximately 12 per year — which cause measurable demand drops. These are not encoded in the feature set, which may cause the model to slightly over-predict on Poya days.

### 5.3 Risks of Bias and Unfairness

**Temporal distribution shift.**
The training data spans 2017–2023, a period of relatively stable grid growth. The test period (2025–2026) shows accelerated growth. The model is biased toward the historical distribution and may systematically under-forecast during periods of rapid expansion beyond the training range.

**Implicit source mix assumptions.**
The future forecast distributes total predicted generation across sources using trailing 90-day proportions. This assumes source mix is relatively stable in the near future — an assumption that breaks down if a new power plant is commissioned or a plant undergoes extended maintenance.

**No demographic or equity considerations.**
Energy forecasting at grid level is essentially an engineering problem. However, systematic forecast errors affect dispatch decisions, and under-forecasting can contribute to power shortages that disproportionately affect lower-income users who cannot self-generate (e.g., from rooftop solar). Accurate forecasting is therefore not entirely equity-neutral.

### 5.4 Potential Real-World Impact and Ethical Considerations

**Operational use case.**
If deployed, this model would support the CEB/PUCSL daily dispatch planning process. A MAPE of ~4% is operationally useful — it means the operator can plan committed generation at roughly ±2 GWh/day confidence for next-day forecasts, reducing the need for expensive spinning reserve capacity.

**Economic impact.**
Improved generation forecasting reduces reliance on expensive oil-fired peaking plants. Sri Lanka's heavy dependence on imported oil for peaking power is a significant fiscal and forex risk. A model that reduces peak reserve requirements by even 5% would have measurable economic impact.

**Grid stability and reliability.**
Demand-supply imbalance is a direct cause of frequency deviation, load shedding, and equipment damage. Better forecasting supports tighter dispatch scheduling and reduces the probability of unplanned outages.

**Accountability.**
As an automated forecasting tool, the model should not be used as a sole decision-making authority. Human operators must retain final authority over dispatch decisions, and the model's outputs should be presented as advisory alongside uncertainty bounds.

**Data privacy.**
The dataset contains grid-level aggregate figures only. There is no personal data. Privacy risks are negligible.

---

## 6. Future Improvements

### 6.1 Meteorological Covariates (Highest Priority)

Integrating real weather data would be the single largest improvement. Recommended data sources:

- **Open-Meteo API** (free, open-source): daily precipitation, temperature, solar irradiance, wind speed for key grid regions (Colombo, Mahaweli catchment, Mannar Wind Zone).
- **NASA POWER / ERA5 reanalysis**: retrospective weather data back to 1985 for model training.

Expected impact: materially improve hydro and solar prediction, potentially reducing RMSE by 20–40%.

### 6.2 Poya Holiday Feature

Add a binary `IsPoya` feature (1 on Poya/full-moon days, 0 otherwise). Poya days are computable from lunar calendars and are known years in advance. This would allow the model to correctly predict reduced industrial demand (~5–10% lower generation) on these 12 days per year.

### 6.3 Probabilistic Forecasting 

Quantile regression is now active. The model is configured with `likelihood="quantile"` and `quantiles=[0.1, 0.5, 0.9]`, training three separate gradient-boosted trees — one for each percentile.

**How it works:**
- The **median forecast (p50)** is the primary point prediction, used for all accuracy metrics (RMSE, MAE, MAPE, R²).
- The **80% confidence band** spans p10 to p90. On any given day, there is an 80% probability the true value falls within this band.
- For the **90-day future forecast**, `num_samples=500` Monte Carlo draws are made from the quantile model, and per-day p10/p50/p90 values are extracted via `numpy.quantile` across the sample dimension.
- The `future_forecast.csv` artifact now contains `CI_Low` (p10), `Forecast_GWh` (p50), and `CI_High` (p90) columns.
- The dashboard overlays the CI band as a shaded region on the generation forecast chart.

**Operational value:** Operators can use the p10 bound as a conservative (pessimistic) scenario for reserve commitment, and the p90 bound as a demand ceiling for capacity planning.

### 6.4 Extended Optuna Tuning

Only 30 trials were run. The Optuna study is persistent (SQLite), so additional trials can be appended:

```bash
python tune_hyperparams.py --resume --trials 100
```

With 100+ trials, the hyperparameter surface is better explored and further RMSE reduction is likely.

### 6.5 Per-Source Forecasting

Instead of forecasting total generation and distributing via static proportions, build **separate models per generation source** (Coal, Hydro, Solar, Wind, etc.) and sum predictions. This would:
- Improve physical interpretability.
- Allow source-specific feature engineering (e.g., reservoir level for hydro, irradiance for solar).
- Directly produce source-wise forecasts for the dashboard without proportional splitting.

### 6.6 Deep Learning Models

Test sequence models for comparison:
- **N-BEATS / N-HiTS** (pure neural time series, available in Darts): excellent on univariate and multivariate series with trend and seasonality.
- **TFT (Temporal Fusion Transformer)**: designed for multi-horizon forecasting with mixed static, future, and past covariates — architecturally a strong match for this problem.

These require PyTorch (`darts[torch]`) and GPU training but may outperform LightGBM for multi-step horizons.

### 6.7 Automated Retraining Pipeline

Wrap the pipeline in a scheduled job (e.g., Windows Task Scheduler, Apache Airflow) to:
1. Download the latest CEB generation data daily.
2. Append to the dataset.
3. Retrain the model on a rolling window.
4. Refresh artifacts and the Streamlit dashboard.

This would turn the current offline prototype into a live forecasting system.

### 6.8 Demand-Side Input

Currently only supply (generation) is modelled. Net generation closely approximates demand (plus export minus import), but incorporating explicit **electricity demand data** (if available from PUCSL) would improve model accuracy by separating supply constraints from demand signals.

---

## Appendix: File Structure

```
national-grid-energy-forecasting/
│
├── data/
│   └── merged-generation-profile-2017-2026.csv   # Raw input data
│
├── artifacts/                                     # Generated by forecast.py
│   ├── model.pkl                                  # Serialised LightGBM model
│   ├── metrics.json                               # Test set performance metrics
│   ├── best_hyperparams.json                      # Optuna best trial parameters
│   ├── processed_data.csv                         # Full engineered feature matrix
│   ├── test_predictions.csv                       # Actual vs predicted (test set)
│   ├── future_forecast.csv                        # 90-day forward forecast
│   ├── prediction_plot.png                        # Time series + residuals plot
│   ├── monthly_comparison.png                     # Monthly grouped bar chart
│   ├── actual_vs_predicted.png                    # Scatter plot (R² coloured)
│   ├── shap_summary.png                           # Global SHAP beeswarm plot
│   ├── shap_local.png                             # Local SHAP bar chart (last day)
│   └── optuna_study.db                            # Optuna SQLite study database
│
├── config.py                                      # Shared constants
├── preprocess.py                                  # Data loading & feature engineering
├── model_setup.py                                 # Model config, splits, lag setup
├── train_evaluate.py                              # Training, rolling eval, plots
├── explainability.py                              # SHAP analysis
├── future_forecast.py                             # 90-day forward forecast
├── forecast.py                                    # Main pipeline orchestrator
├── tune_hyperparams.py                            # Optuna hyperparameter optimisation
├── app.py                                         # Streamlit dashboard
└── PROJECT_REPORT.md                              # This document
```

## Appendix: Key Commands

```bash
# Run full pipeline (train, evaluate, forecast, SHAP)
python forecast.py

# Run hyperparameter tuning (30 new trials)
python tune_hyperparams.py --trials 30

# Resume tuning from previous study
python tune_hyperparams.py --resume --trials 50

# Launch Streamlit dashboard
streamlit run app.py
```

---

*Report generated: February 2026. Model version: LightGBM v3 (Optuna trial #29). Dataset: CEB/PUCSL 2017–2026.*
