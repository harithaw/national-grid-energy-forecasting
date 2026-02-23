"""Model configuration, lag setup, and chronological data splitting."""

from __future__ import annotations

import json

import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel

from config import ARTIFACT_DIR


def build_darts_series(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 2 - Model Setup & Chronological Split")
    print("="*60)

    past_cov_cols = [
        "Solar_Total", "Mini_Hydro",
        "Biomass_Waste", "Wind", "Major_Hydro",
        "Oil_IPP", "Oil_CEB", "Coal",
        "Roll7_Mean", "Roll30_Mean", "Roll90_Mean", "Roll365_Mean",
        "Roll30_Std", "Monthly_Avg",
        "YoY_Delta",
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

    _hp_path = ARTIFACT_DIR / "best_hyperparams.json"
    if _hp_path.exists():
        with open(_hp_path) as _f:
            _hp = json.load(_f)
        target_lags     = _hp["target_lags"]
        past_cov_lags   = _hp["past_cov_lags"]
        future_cov_lags = tuple(_hp["future_cov_lags"])
        lgbm_kwargs = dict(
            n_estimators     = _hp["n_estimators"],
            num_leaves       = _hp["num_leaves"],
            learning_rate    = _hp["learning_rate"],
            max_depth        = _hp["max_depth"],
            min_child_samples= _hp["min_child_samples"],
            subsample        = _hp["subsample"],
            colsample_bytree = _hp["colsample_bytree"],
            reg_alpha        = _hp["reg_alpha"],
            reg_lambda       = _hp["reg_lambda"],
        )
        retrain_train_length = int(_hp.get("train_length", 730))
        retrain_stride       = int(_hp.get("stride", 30))
        _src = (f"tuned (Optuna trial #{_hp.get('_optuna_trial_no','?')}, "
                f"val RMSE={_hp.get('_val_rmse','?')} GWh)")
    else:
        target_lags     = [-1, -2, -3, -7, -14, -28, -90, -182, -365]
        past_cov_lags   = [-1, -2, -7, -14, -28]
        future_cov_lags = (7, 1)
        lgbm_kwargs = dict(
            n_estimators=1000, num_leaves=127, learning_rate=0.03,
            max_depth=10, min_child_samples=15, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
        )
        retrain_train_length = 730
        retrain_stride       = 30
        _src = "defaults (run tune_hyperparams.py to optimise)"

    QUANTILES = [0.1, 0.5, 0.9]

    model = LightGBMModel(
        lags=target_lags,
        lags_past_covariates=past_cov_lags,
        lags_future_covariates=future_cov_lags,
        output_chunk_length=1,
        likelihood="quantile",
        quantiles=QUANTILES,
        random_state=42,
        verbosity=-1,
        **lgbm_kwargs,
    )

    print(f"\nLightGBMModel configured OK  [{_src}]")
    print(f"  quantiles             : {QUANTILES}  (10th / 50th / 90th percentile)")
    print(f"  target lags           : {target_lags}")
    print(f"  past covariate lags   : {past_cov_lags}")
    print(f"  future covariate range: {future_cov_lags}")
    print(f"  n_estimators          : {lgbm_kwargs['n_estimators']}")
    print(f"  num_leaves            : {lgbm_kwargs['num_leaves']}")
    print(f"  learning_rate         : {lgbm_kwargs['learning_rate']}")
    print(f"  retrain train_length  : {retrain_train_length}")
    print(f"  retrain stride        : {retrain_stride}")

    return (model,
            target_series, past_cov_ts, future_cov_ts,
            train_target, val_target, test_target,
            train_past, val_past, test_past,
            train_future, val_future, test_future,
            past_cov_cols, future_cov_cols,
            retrain_train_length, retrain_stride,
            QUANTILES)
