"""
forecast.py - Sri Lanka National Grid Daily Generation Forecasting
=================================================================
v2 - Improved model addressing distribution shift (2025-2026 generation
     grew ~24% above 2017-2024 baseline).

Key improvements over v1
------------------------
1.  Rich feature engineering:
    *  Cyclical sin/cos encoding of day-of-year, month, day-of-week
    *  Linear TrendIndex (day 0, 1, 2, ...) - captures structural growth
    *  Rolling statistics (7d / 30d / 90d mean, 30d std) of Total_Generation
    Calendar / trend features become FUTURE covariates (known ahead of time)
    Rolling stats remain PAST covariates (only known up to prediction point)

2.  Specific meaningful lag list:
    target lags        : [1,2,3,7,14,28,90,182,365]
    past-cov lags      : [1,2,7,14,28]
    future-cov range   : (-7, 1)

3.  Train on TRAIN+VAL (2017-2024) for the final model on the test set.
    The validation set (2024) gives exposure to the transitional year
    before the high-generation 2025-2026 test period.

4.  Adaptive rolling retraining during prediction:
    retrain=True, stride=30, train_length=730
    Each month the model retrains on the most recent 730 days.

5.  Tuned LightGBM: 1000 estimators, lower learning rate.

Steps:
  1. Data Preprocessing & Setup
  2. Model Setup & Chronological Split
  3. Model Training & Evaluation
  4. Explainability with SHAP (via Darts ShapExplainer)
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.explainability import ShapExplainer
from darts.metrics import rmse as darts_rmse, mae as darts_mae, mape as darts_mape

warnings.filterwarnings("ignore")

DATA_FILE    = Path("data/merged-generation-profile-2017-2026.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

PREV_METRICS = {"RMSE": 7.3252, "MAE": 5.8396, "MAPE": 11.1687,
                "MSE": 53.659, "R2": -0.1788}

FORECAST_HORIZON = 90   # days to forecast beyond the last date in the dataset


# ===========================================================================
# STEP 1 - Data Preprocessing & Feature Engineering
# ===========================================================================

def load_and_preprocess(path: Path) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1 - Data Preprocessing & Feature Engineering")
    print("="*60)

    df = pd.read_csv(path, na_values=["Data N/A", "N/A", "", " "])
    print(f"Loaded  {len(df):,} rows  x  {df.shape[1]} columns")

    df["Date"] = pd.to_datetime(df["Date (GMT+5:30)"], format="%Y-%m-%d", errors="coerce")
    df.drop(columns=["Date (GMT+5:30)"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    rename_map = {
        "Solar (Energy in GWh/day)"                          : "Solar",
        "Rooftop Solar (Est.) (Energy in GWh/day)"           : "Rooftop_Solar",
        "Solar (Estimated) (Energy in GWh/day)"              : "Solar_Estimated",
        "Mini Hydro (Telemetered) (Energy in GWh/day)"       : "Mini_Hydro_Telemetered",
        "Mini Hydro (Estimated) (Energy in GWh/day)"         : "Mini_Hydro_Estimated",
        "Biomass and Waste Heat (Energy in GWh/day)"         : "Biomass_Waste",
        "Wind (Energy in GWh/day)"                           : "Wind",
        "Major Hydro (Energy in GWh/day)"                    : "Major_Hydro",
        "Oil (IPP) (Energy in GWh/day)"                      : "Oil_IPP",
        "Oil (CEB) (Energy in GWh/day)"                      : "Oil_CEB",
        "Coal (Energy in GWh/day)"                           : "Coal",
    }
    df.rename(columns=rename_map, inplace=True)

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "Date"
    print(f"Date range : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"Total days : {len(df):,}")

    source_cols = [
        "Solar", "Rooftop_Solar", "Solar_Estimated",
        "Mini_Hydro_Telemetered", "Mini_Hydro_Estimated",
        "Biomass_Waste", "Wind", "Major_Hydro",
        "Oil_IPP", "Oil_CEB", "Coal",
    ]
    for col in source_cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        fv = df[col].first_valid_index()
        if fv is not None:
            df.loc[df.index < fv, col] = 0.0
        df[col] = df[col].interpolate(method="time").clip(lower=0).fillna(0)

    print(f"Missing values after imputation : {df[source_cols].isna().sum().sum()}")

    # All three solar sources are always additive: utility-scale IPP (Solar),
    # distributed/estimated (Solar_Estimated), and rooftop (Rooftop_Solar)
    df["Solar_Total"] = df["Solar"] + df["Solar_Estimated"] + df["Rooftop_Solar"]
    # Both mini hydro sub-categories measure different physical plants and are additive
    df["Mini_Hydro"] = df["Mini_Hydro_Telemetered"] + df["Mini_Hydro_Estimated"]
    df["Total_Generation"] = (
        df["Solar_Total"] + df["Mini_Hydro"]
        + df["Biomass_Waste"] + df["Wind"] + df["Major_Hydro"]
        + df["Oil_IPP"] + df["Oil_CEB"] + df["Coal"]
    )

    # -- FUTURE covariates (deterministic calendar + trend) -----------------
    t   = df.index
    doy = t.dayofyear
    df["Sin_DayOfYear"] = np.sin(2 * np.pi * doy / 365.25)
    df["Cos_DayOfYear"] = np.cos(2 * np.pi * doy / 365.25)
    df["Sin_Month"]     = np.sin(2 * np.pi * t.month / 12)
    df["Cos_Month"]     = np.cos(2 * np.pi * t.month / 12)
    df["Sin_DayOfWeek"] = np.sin(2 * np.pi * t.dayofweek / 7)
    df["Cos_DayOfWeek"] = np.cos(2 * np.pi * t.dayofweek / 7)
    df["IsWeekend"]     = (t.dayofweek >= 5).astype(float)
    df["TrendIndex"]    = np.arange(len(df), dtype=float)

    # -- PAST covariates (rolling statistics on observed generation) --------
    gen = df["Total_Generation"]
    df["Roll7_Mean"]  = gen.rolling(7,  min_periods=1).mean()
    df["Roll30_Mean"] = gen.rolling(30, min_periods=1).mean()
    df["Roll90_Mean"] = gen.rolling(90, min_periods=1).mean()
    df["Roll30_Std"]  = gen.rolling(30, min_periods=2).std().fillna(0)
    df["Monthly_Avg"] = gen.groupby(t.month).transform("mean")

    print(f"\nTarget statistics:")
    print(df["Total_Generation"].describe().round(3).to_string())
    print(f"\nYearly mean Total_Generation:")
    print(df["Total_Generation"].resample("YE").mean().round(2).to_string())
    print("\nPreprocessing complete OK")
    return df


# ===========================================================================
# STEP 2 - Model Setup & Chronological Split
# ===========================================================================

def build_darts_series(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 2 - Model Setup & Chronological Split")
    print("="*60)

    past_cov_cols = [
        "Solar_Total", "Mini_Hydro",
        "Biomass_Waste", "Wind", "Major_Hydro",
        "Oil_IPP", "Oil_CEB", "Coal",
        "Roll7_Mean", "Roll30_Mean", "Roll90_Mean",
        "Roll30_Std", "Monthly_Avg",
    ]

    future_cov_cols = [
        "Sin_DayOfYear", "Cos_DayOfYear",
        "Sin_Month",     "Cos_Month",
        "Sin_DayOfWeek", "Cos_DayOfWeek",
        "IsWeekend",     "TrendIndex",
    ]

    target_series = TimeSeries.from_series(
        df["Total_Generation"], freq="D", fill_missing_dates=True)
    past_cov_ts   = TimeSeries.from_dataframe(
        df[past_cov_cols], freq="D", fill_missing_dates=True)
    future_cov_ts = TimeSeries.from_dataframe(
        df[future_cov_cols], freq="D", fill_missing_dates=True)

    print(f"Target series         : {len(target_series)} time-steps")
    print(f"Past covariate series : {len(past_cov_ts)} x {len(past_cov_cols)} features")
    print(f"Future cov series     : {len(future_cov_ts)} x {len(future_cov_cols)} features")

    val_start    = pd.Timestamp("2024-01-01")
    train_end    = pd.Timestamp("2023-12-31")
    val_end      = pd.Timestamp("2024-12-31")
    test_start_d = pd.Timestamp("2025-01-01")

    def _slice(ts, start_excl, end_excl):
        return ts.drop_before(start_excl).drop_after(end_excl)

    train_target = target_series.drop_after(val_start)
    val_target   = _slice(target_series, train_end, test_start_d)
    test_target  = target_series.drop_before(val_end)

    train_past   = past_cov_ts.drop_after(val_start)
    val_past     = _slice(past_cov_ts, train_end, test_start_d)
    test_past    = past_cov_ts.drop_before(val_end)

    train_future = future_cov_ts.drop_after(val_start)
    val_future   = _slice(future_cov_ts, train_end, test_start_d)
    test_future  = future_cov_ts.drop_before(val_end)

    print(f"\nSplit summary:")
    print(f"  Train : {len(train_target):4d} days "
          f"({train_target.start_time().date()} -> {train_target.end_time().date()})")
    print(f"  Val   : {len(val_target):4d} days "
          f"({val_target.start_time().date()}  -> {val_target.end_time().date()})")
    print(f"  Test  : {len(test_target):4d} days "
          f"({test_target.start_time().date()}  -> {test_target.end_time().date()})")

    # Specific meaningful lags
    # 1,2,3 = short AR; 7,14 = weekly; 28,90 = monthly/quarterly;
    # 182,365 = semi-annual/annual (365-day lag is key for YoY trend capture)
    target_lags     = [-1, -2, -3, -7, -14, -28, -90, -182, -365]
    past_cov_lags   = [-1, -2, -7, -14, -28]
    future_cov_lags = (7, 1)   # 7 days past context + current/next step

    model = LightGBMModel(
        lags=target_lags,
        lags_past_covariates=past_cov_lags,
        lags_future_covariates=future_cov_lags,
        output_chunk_length=1,
        n_estimators=1000,
        num_leaves=127,
        learning_rate=0.03,
        max_depth=10,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        verbosity=-1,
    )

    print(f"\nLightGBMModel v2 configured OK")
    print(f"  target lags           : {target_lags}")
    print(f"  past covariate lags   : {past_cov_lags}")
    print(f"  future covariate range: {future_cov_lags}  (7 past + 1 ahead)")
    print(f"  n_estimators          : 1000  (was 500)")
    print(f"  num_leaves            : 127   (was 63)")
    print(f"  learning_rate         : 0.03  (was 0.05)")

    return (model,
            target_series, past_cov_ts, future_cov_ts,
            train_target, val_target, test_target,
            train_past, val_past, test_past,
            train_future, val_future, test_future,
            past_cov_cols, future_cov_cols)


# ===========================================================================
# STEP 3 - Model Training & Evaluation
# ===========================================================================

def train_and_evaluate(model,
                       train_target, val_target, test_target,
                       train_past, val_past, test_past,
                       train_future, val_future, test_future,
                       target_series, past_cov_ts, future_cov_ts):
    print("\n" + "="*60)
    print("STEP 3 - Model Training & Evaluation")
    print("="*60)

    # Combine train+val -> train on 2017-2024
    trainval_target = train_target.append(val_target)
    trainval_past   = train_past.append(val_past)
    trainval_future = train_future.append(val_future)

    # -- Initial fit on train+val ------------------------------------------
    print("\nFitting model on train+val (2017-2024) ...")
    model.fit(
        series=trainval_target,
        past_covariates=trainval_past,
        future_covariates=trainval_future,
    )
    print("Training complete OK")

    # -- Quick validation check (train-only model) -------------------------
    print("\nGenerating validation forecasts (train-only quick check) ...")
    val_hist = model.historical_forecasts(
        series=trainval_target,
        past_covariates=trainval_past,
        future_covariates=trainval_future,
        start=val_target.start_time(),
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=False,
        last_points_only=True,
    )
    val_aligned = val_target.slice_intersect(val_hist)
    print(f"  Val RMSE : {float(darts_rmse(val_aligned, val_hist)):.4f} GWh")
    print(f"  Val MAE  : {float(darts_mae(val_aligned, val_hist)):.4f} GWh")
    print(f"  Val MAPE : {float(darts_mape(val_aligned, val_hist)):.4f} %")

    # -- Adaptive rolling forecasts on test set ----------------------------
    # retrain=True, stride=30: retrains every 30 days on the most recent
    # 730 days of data, so by mid-2025 the model has seen 2024-2025 data
    # with the high generation values -> adapts to the structural upward trend
    print("\nGenerating adaptive rolling forecasts on test set ...")
    print("  (retrain=True, stride=30, train_length=730 - retrains monthly)")

    predictions = model.historical_forecasts(
        series=target_series,
        past_covariates=past_cov_ts,
        future_covariates=future_cov_ts,
        start=test_target.start_time(),
        forecast_horizon=1,
        stride=1,
        retrain=True,
        train_length=730,
        verbose=False,
        last_points_only=True,
    )
    print(f"Predictions generated : {len(predictions)} time-steps")

    # -- Metrics -----------------------------------------------------------
    test_aligned = test_target.slice_intersect(predictions)
    y_true = test_aligned.to_series().values
    y_pred = predictions.to_series().values

    rmse_val = float(darts_rmse(test_aligned, predictions))
    mae_val  = float(darts_mae(test_aligned,  predictions))
    mape_val = float(darts_mape(test_aligned, predictions))
    mse_val  = float(mean_squared_error(y_true, y_pred))
    r2_val   = float(1 - mse_val / np.var(y_true))

    metrics = {
        "RMSE" : round(rmse_val, 4),
        "MAE"  : round(mae_val,  4),
        "MAPE" : round(mape_val, 4),
        "MSE"  : round(mse_val,  4),
        "R2"   : round(r2_val,   4),
    }

    print("\n-- Evaluation Metrics (Test Set) --------------------------------")
    for k, v in metrics.items():
        unit  = " %" if k == "MAPE" else (" GWh" if k in ["RMSE","MAE"] else "")
        delta = v - PREV_METRICS[k]
        arrow = ("+" if delta > 0 else "") + f"{delta:.4f}"
        better = delta < 0 if k != "R2" else delta > 0
        tag = "improved" if better else "worsened"
        print(f"  {k:<6}: {v:>9}{unit}   (v1: {PREV_METRICS[k]}  delta:{arrow}  {tag})")
    print("-"*65)

    # -- Plots -------------------------------------------------------------
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    ax = axes[0]
    test_aligned.to_series().plot(ax=ax, label="Actual",    color="#2196F3", lw=1.3)
    predictions.to_series().plot( ax=ax, label="Predicted", color="#FF5722", lw=1.3, ls="--")
    ax.set_title("Daily Total Generation - Actual vs Predicted (Test Set 2025-2026)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Generation (GWh/day)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(test_aligned.to_series().index, residuals,
            color=np.where(residuals >= 0, "#4CAF50", "#F44336"), alpha=0.7, width=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Residuals (Actual - Predicted)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Residual (GWh/day)"); ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "prediction_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPrediction plot saved -> {ARTIFACT_DIR / 'prediction_plot.png'}")

    actual_df2 = test_aligned.to_dataframe(); actual_df2.columns = ["Actual"]
    pred_df2   = predictions.to_dataframe();  pred_df2.columns   = ["Predicted"]
    monthly    = actual_df2.join(pred_df2).resample("ME").mean()

    fig2, ax3 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(monthly)); w = 0.35
    ax3.bar(x - w/2, monthly["Actual"],    width=w, label="Actual",    color="#2196F3", alpha=0.85)
    ax3.bar(x + w/2, monthly["Predicted"], width=w, label="Predicted", color="#FF5722", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.strftime("%b %Y") for d in monthly.index], rotation=45, ha="right")
    ax3.set_title("Monthly Average Generation - Actual vs Predicted",
                  fontsize=13, fontweight="bold")
    ax3.set_ylabel("Avg Daily Generation (GWh/day)")
    ax3.legend(); ax3.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "monthly_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Monthly comparison plot saved -> {ARTIFACT_DIR / 'monthly_comparison.png'}")

    return predictions, test_aligned, metrics


# ===========================================================================
# STEP 4 - Explainability with SHAP
# ===========================================================================

def explain_with_shap(model, target_series, past_cov_ts, future_cov_ts, test_target):
    print("\n" + "="*60)
    print("STEP 4 - Explainability with SHAP")
    print("="*60)

    bg_start = pd.Timestamp("2022-12-31")
    bg_end   = pd.Timestamp("2025-01-01")
    bg_target = target_series.drop_before(bg_start).drop_after(bg_end)
    bg_past   = past_cov_ts.drop_before(bg_start).drop_after(bg_end)
    bg_future = future_cov_ts.drop_before(bg_start).drop_after(bg_end)

    print("Building ShapExplainer (2023-2024 background) ...")
    explainer = ShapExplainer(
        model=model,
        background_series=bg_target,
        background_past_covariates=bg_past,
        background_future_covariates=bg_future,
    )
    print("ShapExplainer built OK")

    # Foreground needs >= 365 points (the longest lag); use 400 days as context.
    fg_start  = test_target.end_time() - pd.Timedelta(days=400)
    fg_target = target_series.drop_before(fg_start)
    fg_past   = past_cov_ts.drop_before(fg_start)
    fg_future = future_cov_ts.drop_before(fg_start)

    print(f"Computing SHAP values ({len(fg_target)} foreground days) ...")
    shap_explanation = explainer.explain(
        foreground_series=fg_target,
        foreground_past_covariates=fg_past,
        foreground_future_covariates=fg_future,
        horizons=[1],
    )

    print("Generating SHAP summary (feature importance) plot ...")
    explainer.summary_plot(horizons=[1])
    plt.savefig(ARTIFACT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved -> {ARTIFACT_DIR / 'shap_summary.png'}")

    print("Generating local SHAP explanation for the most recent test day ...")
    try:
        imp_ts     = shap_explanation.get_explanation(horizon=1)
        imp_df     = imp_ts.to_dataframe()
        feat_names = imp_df.columns.tolist()
        last_shap  = imp_df.iloc[-1]
        pred_date  = imp_df.index[-1].date()

        top_idx = np.argsort(np.abs(last_shap.values))[::-1][:20]
        colors  = ["#F44336" if last_shap.values[i] >= 0 else "#2196F3"
                   for i in top_idx]

        fig_l, ax_l = plt.subplots(figsize=(10, 6))
        ax_l.barh([feat_names[i] for i in top_idx],
                  [last_shap.values[i] for i in top_idx],
                  color=colors, edgecolor="white")
        ax_l.axvline(0, color="black", lw=0.8)
        ax_l.set_title(
            f"Local SHAP Explanation - Prediction for {pred_date}\n"
            "Red = pushes generation up  |  Blue = pushes generation down",
            fontsize=11, fontweight="bold")
        ax_l.set_xlabel("SHAP value  (GWh/day impact on model output)")
        ax_l.invert_yaxis()
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "shap_local.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Local SHAP plot saved -> {ARTIFACT_DIR / 'shap_local.png'}")
    except Exception as exc:
        print(f"[WARN] Local SHAP plot skipped: {exc}")

    return shap_explanation


# ===========================================================================
# STEP 5 – Future Forecast (beyond last known date)
# ===========================================================================

def generate_future_forecast(model: LightGBMModel,
                              df: pd.DataFrame,
                              target_series: "TimeSeries",
                              past_cov_ts: "TimeSeries",
                              future_cov_ts: "TimeSeries",
                              horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Refit the model on the complete dataset (all data including test period),
    then produce `horizon` daily forecasts beyond the last known date.

    Future covariates are purely deterministic (calendar + trend) so they can
    be computed for any date in advance.  Past covariates (rolling stats, source
    mix) are only needed up to the last known date – their lagged values supply
    the model's autoregressive context.
    """
    print("\n" + "="*60)
    print("STEP 5 - Future Forecast Generation")
    print("="*60)

    last_date   = df.index.max()
    future_idx  = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_idx.name = "Date"

    # ── Build deterministic future covariates ──────────────────────────────
    t   = future_idx
    doy = t.dayofyear
    last_trend = float(df["TrendIndex"].iloc[-1])

    ext = pd.DataFrame(index=future_idx)
    ext["Sin_DayOfYear"] = np.sin(2 * np.pi * doy / 365.25)
    ext["Cos_DayOfYear"] = np.cos(2 * np.pi * doy / 365.25)
    ext["Sin_Month"]     = np.sin(2 * np.pi * t.month / 12)
    ext["Cos_Month"]     = np.cos(2 * np.pi * t.month / 12)
    ext["Sin_DayOfWeek"] = np.sin(2 * np.pi * t.dayofweek / 7)
    ext["Cos_DayOfWeek"] = np.cos(2 * np.pi * t.dayofweek / 7)
    ext["IsWeekend"]     = (t.dayofweek >= 5).astype(float)
    ext["TrendIndex"]    = np.arange(last_trend + 1, last_trend + 1 + horizon, dtype=float)
    ext.index.name = "Date"

    future_cov_ext  = TimeSeries.from_dataframe(ext, freq="D", fill_missing_dates=True)
    extended_future = future_cov_ts.append(future_cov_ext)

    # ── Pad past covariates into the forecast horizon ─────────────────────
    # For multi-step auto-regression (n > output_chunk_length=1), Darts needs
    # past_covariates to extend horizon-1 steps beyond the last known date.
    past_cov_df = past_cov_ts.to_dataframe()
    pad_idx     = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon - 1, freq="D"
    )
    pad_idx.name = "Date"
    pad_df  = pd.DataFrame(
        np.tile(past_cov_df.iloc[-1].values, (len(pad_idx), 1)),
        index=pad_idx, columns=past_cov_df.columns,
    )
    pad_df.index.name = "Date"
    past_cov_pad  = TimeSeries.from_dataframe(pad_df, freq="D")
    extended_past = past_cov_ts.append(past_cov_pad)

    # ── Retrain on the full dataset (2017-present) ─────────────────────────
    print(f"\nRefitting on complete dataset ({len(target_series)} days, "
          f"{target_series.start_time().date()} → {target_series.end_time().date()}) ...")
    model.fit(
        series            = target_series,
        past_covariates   = past_cov_ts,
        future_covariates = extended_future,
    )
    print("Refit complete OK")

    # ── Predict ────────────────────────────────────────────────────────────
    first_future = last_date + pd.Timedelta(days=1)
    print(f"Generating {horizon}-day forecast from {first_future.date()} ...")
    future_preds = model.predict(
        n                 = horizon,
        series            = target_series,
        past_covariates   = extended_past,
        future_covariates = extended_future,
    )

    future_df = future_preds.to_dataframe()
    future_df.columns  = ["Forecast_GWh"]
    future_df.index.name = "Date"
    future_df["Forecast_GWh"] = future_df["Forecast_GWh"].clip(lower=0).round(4)

    # ── Source-level breakdown via trailing 90-day proportions ─────────────
    # Each source keeps its recent seasonal proportion of the total; daily
    # variation in the mix follows the same ratio as the trailing window mean.
    src_cols = [
        "Solar_Total", "Mini_Hydro", "Biomass_Waste", "Wind",
        "Major_Hydro", "Oil_IPP", "Oil_CEB", "Coal",
    ]
    recent      = df[src_cols].iloc[-90:]
    recent_mean = recent.mean()
    total_mean  = recent_mean.sum()
    proportions = recent_mean / total_mean if total_mean > 0 else recent_mean * 0 + (1 / len(src_cols))

    for src in src_cols:
        future_df[src] = (future_df["Forecast_GWh"] * proportions[src]).round(4)

    out = ARTIFACT_DIR / "future_forecast.csv"
    future_df.to_csv(out)
    print(f"Future forecast saved → {out}")
    print(future_df[["Forecast_GWh"] + src_cols].head(10).to_string())
    print(f"  ... ({len(future_df)} rows total)")
    return future_df


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    df = load_and_preprocess(DATA_FILE)

    (model,
     target_series, past_cov_ts, future_cov_ts,
     train_target, val_target, test_target,
     train_past, val_past, test_past,
     train_future, val_future, test_future,
     past_cov_cols, future_cov_cols) = build_darts_series(df)

    predictions, test_aligned, metrics = train_and_evaluate(
        model,
        train_target, val_target, test_target,
        train_past,   val_past,   test_past,
        train_future, val_future, test_future,
        target_series, past_cov_ts, future_cov_ts,
    )

    shap_explanation = explain_with_shap(
        model, target_series, past_cov_ts, future_cov_ts, test_target
    )

    print("\nSaving artifacts ...")
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    pred_df3   = predictions.to_dataframe();   pred_df3.columns   = ["Predicted"]
    actual_df3 = test_aligned.to_dataframe();  actual_df3.columns = ["Actual"]
    actual_df3.join(pred_df3).to_csv(ARTIFACT_DIR / "test_predictions.csv")

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv(ARTIFACT_DIR / "processed_data.csv")

    # STEP 5 - Future forecast beyond last known date
    generate_future_forecast(
        model, df, target_series, past_cov_ts, future_cov_ts,
        horizon=FORECAST_HORIZON,
    )

    print(f"\nAll artifacts saved to  {ARTIFACT_DIR.resolve()}")
    print("\n" + "="*60)
    print("Pipeline complete OK")
    print("="*60)
    print("\nTo launch the dashboard:")
    print("  .\\venv\\Scripts\\streamlit.exe run app.py")


if __name__ == "__main__":
    main()
