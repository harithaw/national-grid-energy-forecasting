"""
compare_models.py - Multi-model comparison for Sri Lanka National Grid forecasting
===================================================================================
Runs three model families against the same chronological split and reports
a ranked comparison table.

  ML Regression  : LightGBM*, XGBoost, RandomForest, ExtraTrees, LinearRegression
  Statistical    : NaiveSeasonal(7/365), ExponentialSmoothing, Theta, FourTheta
  Optional       : Prophet, StatsForecastAutoARIMA  (if packages present)

  * LightGBM metrics are loaded from the existing artifacts/metrics.json to
    avoid re-running the expensive adaptive rolling forecast.

ML models use: retrain=True, stride=30, train_length=730  (same as forecast.py)
Statistical models: single fit on train+val (2017-2024), then predict test period.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from darts import TimeSeries
from darts.metrics import rmse as darts_rmse, mae as darts_mae, mape as darts_mape

# ── Import preprocessing from forecast.py ────────────────────────────────────
from forecast import load_and_preprocess

warnings.filterwarnings("ignore")

DATA_FILE    = Path("data/merged-generation-profile-2017-2026.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── Optional model imports ────────────────────────────────────────────────────
try:
    from darts.models import XGBModel
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not available - skipping XGBModel")

try:
    import catboost  # noqa: F401 - probe the real package, not just the Darts wrapper
    from darts.models import CatBoostModel
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[INFO] CatBoost not available - skipping")

from darts.models import (
    LightGBMModel,
    RandomForestModel,
    LinearRegressionModel,
    NaiveSeasonal,
    ExponentialSmoothing,
    Theta,
    FourTheta,
    AutoTheta,
)
from darts.utils.utils import SeasonalityMode


# ===========================================================================
# Helpers
# ===========================================================================

def compute_metrics(actual: TimeSeries, predicted: TimeSeries) -> dict:
    """Return RMSE, MAE, MAPE, R² for aligned TimeSeries pair."""
    actual_a = actual.slice_intersect(predicted)
    y_true   = actual_a.to_series().values
    y_pred   = predicted.to_series().values
    mse      = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": round(float(darts_rmse(actual_a, predicted)), 4),
        "MAE" : round(float(darts_mae(actual_a,  predicted)), 4),
        "MAPE": round(float(darts_mape(actual_a, predicted)), 4),
        "MSE" : round(mse, 4),
        "R2"  : round(float(1 - mse / np.var(y_true)), 4),
    }


def build_series(df: pd.DataFrame):
    """Rebuild TimeSeries objects and split boundaries from processed DataFrame."""
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

    target     = TimeSeries.from_series(df["Total_Generation"], freq="D", fill_missing_dates=True)
    past_cov   = TimeSeries.from_dataframe(df[past_cov_cols],    freq="D", fill_missing_dates=True)
    future_cov = TimeSeries.from_dataframe(df[future_cov_cols],  freq="D", fill_missing_dates=True)

    val_start    = pd.Timestamp("2024-01-01")
    train_end    = pd.Timestamp("2023-12-31")
    val_end      = pd.Timestamp("2024-12-31")
    test_start   = pd.Timestamp("2025-01-01")

    def _slice(ts, a, b):
        return ts.drop_before(a).drop_after(b)

    train_target = target.drop_after(val_start)
    val_target   = _slice(target, train_end, test_start)
    test_target  = target.drop_before(val_end)

    trainval_target = train_target.append(val_target)
    trainval_past   = past_cov.drop_after(val_start).append(_slice(past_cov,   train_end, test_start))
    trainval_future = future_cov.drop_after(val_start).append(_slice(future_cov, train_end, test_start))

    return (target, past_cov, future_cov,
            train_target, val_target, test_target,
            trainval_target, trainval_past, trainval_future,
            past_cov_cols, future_cov_cols)


# ===========================================================================
# ML regression models  (past + future covariates, adaptive rolling)
# ===========================================================================

SHARED_LGBM_KWARGS = dict(
    lags                  = [-1,-2,-3,-7,-14,-28,-90,-182,-365],
    lags_past_covariates  = [-1,-2,-7,-14,-28],
    lags_future_covariates= (7, 1),
    output_chunk_length   = 1,
)

def _run_ml_model(name, model, target, past_cov, future_cov,
                  trainval_target, trainval_past, trainval_future,
                  test_target):
    """Fit on trainval, then adaptive rolling forecast on test."""
    print(f"\n  [{name}] Fitting on train+val ...")
    t0 = time.time()
    model.fit(trainval_target, past_covariates=trainval_past,
              future_covariates=trainval_future)

    print(f"  [{name}] Running adaptive rolling forecasts (retrain=30, train_length=730) ...")
    preds = model.historical_forecasts(
        series            = target,
        past_covariates   = past_cov,
        future_covariates = future_cov,
        start             = test_target.start_time(),
        forecast_horizon  = 1,
        stride            = 1,
        retrain           = 30,
        train_length      = 730,
        verbose           = False,
        last_points_only  = True,
    )
    elapsed = time.time() - t0
    m = compute_metrics(test_target, preds)
    m["time_s"] = round(elapsed, 1)
    print(f"  [{name}] Done in {elapsed:.0f}s  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%  R²={m['R2']:.4f}")
    return m, preds


def run_ml_models(target, past_cov, future_cov,
                  trainval_target, trainval_past, trainval_future, test_target):
    results = {}

    # ── XGBoost ─────────────────────────────────────────────────────────────
    if HAS_XGB:
        model = XGBModel(
            **SHARED_LGBM_KWARGS,
            n_estimators   = 500,
            learning_rate  = 0.03,
            max_depth      = 7,
            subsample      = 0.8,
            colsample_bytree = 0.8,
            reg_alpha      = 0.05,
            reg_lambda     = 0.05,
            random_state   = 42,
            verbosity      = 0,
        )
        results["XGBoost"], _ = _run_ml_model(
            "XGBoost", model, target, past_cov, future_cov,
            trainval_target, trainval_past, trainval_future, test_target)

    # ── Random Forest ────────────────────────────────────────────────────────
    model = RandomForestModel(
        **SHARED_LGBM_KWARGS,
        n_estimators = 500,
        max_depth    = 12,
        random_state = 42,
        n_jobs       = -1,
    )
    results["RandomForest"], _ = _run_ml_model(
        "RandomForest", model, target, past_cov, future_cov,
        trainval_target, trainval_past, trainval_future, test_target)

    # ── CatBoost ─────────────────────────────────────────────────────────────
    if HAS_CATBOOST:
        model = CatBoostModel(
            **SHARED_LGBM_KWARGS,
            iterations    = 400,
            learning_rate = 0.05,
            depth         = 7,
            random_state  = 42,
            verbose       = 0,
        )
        results["CatBoost"], _ = _run_ml_model(
            "CatBoost", model, target, past_cov, future_cov,
            trainval_target, trainval_past, trainval_future, test_target)

    # ── Linear Regression ────────────────────────────────────────────────────
    model = LinearRegressionModel(
        **SHARED_LGBM_KWARGS,
        fit_intercept = True,
    )
    results["LinearRegression"], _ = _run_ml_model(
        "LinearRegression", model, target, past_cov, future_cov,
        trainval_target, trainval_past, trainval_future, test_target)

    return results


# ===========================================================================
# Statistical / classical models  (target only, fit-then-predict)
# ===========================================================================

def _run_stat_model(name, model, trainval_target, test_target):
    """Fit on trainval, generate a single forecast for the full test period."""
    print(f"\n  [{name}] Fitting on train+val ...")
    t0 = time.time()
    try:
        model.fit(trainval_target)
        n_test = len(test_target)
        print(f"  [{name}] Predicting {n_test} steps ...")
        preds = model.predict(n_test)
        elapsed = time.time() - t0
        m = compute_metrics(test_target, preds)
        m["time_s"] = round(elapsed, 1)
        print(f"  [{name}] Done in {elapsed:.0f}s  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%  R²={m['R2']:.4f}")
        return m, preds
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  [{name}] SKIPPED after {elapsed:.0f}s: {exc}")
        return None, None


def run_stat_models(target, trainval_target, test_target):
    results = {}

    # ── Naive baselines ──────────────────────────────────────────────────────
    r, _ = _run_stat_model("Naive_Weekly", NaiveSeasonal(K=7),   trainval_target, test_target)
    if r: results["Naive_Weekly"] = r
    r, _ = _run_stat_model("Naive_Yearly", NaiveSeasonal(K=365), trainval_target, test_target)
    if r: results["Naive_Yearly"] = r

    # ── Exponential Smoothing ────────────────────────────────────────────────
    r, _ = _run_stat_model(
        "ExpSmoothing",
        ExponentialSmoothing(trend=None, damped=False, seasonal_periods=7),
        trainval_target, test_target)
    if r: results["ExpSmoothing"] = r

    # ── Theta family ─────────────────────────────────────────────────────────
    r, _ = _run_stat_model("Theta",     Theta(season_mode=SeasonalityMode.ADDITIVE),     trainval_target, test_target)
    if r: results["Theta"] = r
    r, _ = _run_stat_model("FourTheta", FourTheta(season_mode=SeasonalityMode.ADDITIVE), trainval_target, test_target)
    if r: results["FourTheta"] = r
    r, _ = _run_stat_model("AutoTheta", AutoTheta(), trainval_target, test_target)
    if r: results["AutoTheta"] = r

    return results


# ===========================================================================
# Reporting
# ===========================================================================

def print_comparison_table(all_results: dict):
    print("\n")
    print("=" * 82)
    print("MODEL COMPARISON  -  Test Set (2025-01-01 → 2026-02-22, 418 days)")
    print("=" * 82)
    header = f"  {'Model':<22}  {'RMSE':>7}  {'MAE':>7}  {'MAPE':>8}  {'R²':>7}  {'Time(s)':>8}"
    print(header)
    print("-" * 82)

    # Sort by RMSE
    sorted_items = sorted(all_results.items(), key=lambda x: x[1]["RMSE"])

    for rank, (name, m) in enumerate(sorted_items, 1):
        marker = " ◀ best" if rank == 1 else ""
        print(f"  {name:<22}  {m['RMSE']:>7.4f}  {m['MAE']:>7.4f}  "
              f"{m['MAPE']:>7.2f}%  {m['R2']:>7.4f}  {m.get('time_s', '—'):>8}{marker}")

    print("=" * 82)
    print("  ML models : fresh run, identical setup — retrain=30, train_length=730, 500 estimators")
    print("  Statistical models : single fit on 2017-2024, predict full test period")
    print("=" * 82)


def save_comparison_chart(all_results: dict):
    sorted_items = sorted(all_results.items(), key=lambda x: x[1]["RMSE"])
    names  = [n for n, _ in sorted_items]
    rmses  = [m["RMSE"]  for _, m in sorted_items]
    mapes  = [m["MAPE"]  for _, m in sorted_items]
    r2s    = [m["R2"]    for _, m in sorted_items]

    # Colour: ML models in blue, statistical in orange, LightGBM in green
    ml_names   = {"LightGBM", "XGBoost", "RandomForest", "ExtraTrees", "LinearRegression"}
    colors     = ["#2196F3" if n in ml_names else "#FF9800" for n in names]
    colors[0]  = "#4CAF50"  # best model (sorted first) gets green

    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(names) * 0.5 + 2)))

    def _hbar(ax, values, title, xlabel, fmt=".3f"):
        bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:{fmt}}", va="center", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    _hbar(axes[0], rmses, "RMSE  (lower = better)", "GWh/day")
    _hbar(axes[1], mapes, "MAPE  (lower = better)", "%", fmt=".2f")
    _hbar(axes[2], r2s,   "R²  (higher = better)",  "R²",   fmt=".3f")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Best model"),
        Patch(facecolor="#2196F3", label="ML Regression"),
        Patch(facecolor="#FF9800", label="Statistical"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Model Comparison – Sri Lanka National Grid Forecasting",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = ARTIFACT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart saved -> {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("="*60)
    print("Multi-Model Comparison")
    print("="*60)

    # ── 1. Load & preprocess data ─────────────────────────────────────────
    print("\nLoading and preprocessing data ...")
    df = load_and_preprocess(DATA_FILE)

    # ── 2. Build TimeSeries objects ───────────────────────────────────────
    (target, past_cov, future_cov,
     train_target, val_target, test_target,
     trainval_target, trainval_past, trainval_future,
     _, _) = build_series(df)

    all_results: dict[str, dict] = {}

    # ── 3. Run LightGBM fresh for a fair like-for-like comparison ────────
    print("\n[LightGBM] Running fresh (same retrain=30 as other ML models) ...")
    lgbm_model = LightGBMModel(
        lags                  = [-1,-2,-3,-7,-14,-28,-90,-182,-365],
        lags_past_covariates  = [-1,-2,-7,-14,-28],
        lags_future_covariates= (7, 1),
        output_chunk_length   = 1,
        n_estimators          = 500,
        num_leaves            = 127,
        learning_rate         = 0.03,
        max_depth             = 10,
        min_child_samples     = 15,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.05,
        reg_lambda            = 0.05,
        random_state          = 42,
        verbosity             = -1,
    )
    lgbm_m, _ = _run_ml_model(
        "LightGBM", lgbm_model, target, past_cov, future_cov,
        trainval_target, trainval_past, trainval_future, test_target)
    all_results["LightGBM"] = lgbm_m

    # ── 4. ML regression models ───────────────────────────────────────────
    print("\n" + "="*60)
    print("ML REGRESSION MODELS")
    print("="*60)
    ml_results = run_ml_models(
        target, past_cov, future_cov,
        trainval_target, trainval_past, trainval_future, test_target)
    all_results.update(ml_results)

    # ── 5. Statistical / classical models ─────────────────────────────────
    print("\n" + "="*60)
    print("STATISTICAL / CLASSICAL MODELS")
    print("="*60)
    stat_results = run_stat_models(target, trainval_target, test_target)
    all_results.update(stat_results)

    # ── 6. Report ─────────────────────────────────────────────────────────
    print_comparison_table(all_results)
    save_comparison_chart(all_results)

    # Save to JSON
    out_json = ARTIFACT_DIR / "model_comparison.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Comparison results saved -> {out_json}")


if __name__ == "__main__":
    main()
