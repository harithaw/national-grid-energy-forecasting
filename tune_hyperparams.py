""" 
tune_hyperparams.py – Optuna hyperparameter search for the LightGBM forecast model
====================================================================================
Strategy
--------
Each trial trains a model via adaptive rolling historical_forecasts on 2024 with
retrain=True and a fixed train_length window — the SAME evaluation regime used
on the test set.  This is slower (~30-60 s/trial) but optimises directly for
what we actually care about: rolling forecast quality.

The search covers:
  - All major LightGBM hyperparameters
  - Target lag structure (whether to include -730 / -182)
  - Past covariate lag depth
  - Future covariate lookahead window
  - Retraining strategy (train_length, stride)

Best parameters are written to:
    artifacts/best_hyperparams.json

forecast.py automatically picks them up on the next run.

Usage
-----
    # 30 trials (default, ~15-30 min)
    .\\venv\\Scripts\\python.exe tune_hyperparams.py

    # Custom trial count
    .\\venv\\Scripts\\python.exe tune_hyperparams.py --trials 50

    # Resume a previous study
    .\\venv\\Scripts\\python.exe tune_hyperparams.py --resume
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import rmse as darts_rmse

from forecast import load_and_preprocess

warnings.filterwarnings("ignore")

DATA_FILE    = Path("data/merged-generation-profile-2017-2026.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
STUDY_DB     = str(ARTIFACT_DIR / "optuna_study.db")

# Current baseline on the VALIDATION set (2024) for reference
# (run with retrain=False – this is what each trial measures)
BASELINE_RMSE = None   # filled after first trial or can preset manually
CURRENT_CFG = {
    "num_leaves": 127, "n_estimators": 1000, "learning_rate": 0.03,
    "max_depth": 10,   "min_child_samples": 15, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 0.05,
    "target_lags": [-1,-2,-3,-7,-14,-28,-90,-182,-365],
    "past_cov_lags": [-1,-2,-7,-14,-28],
    "future_cov_lags": [7, 1],
}


# ---------------------------------------------------------------------------
# Data preparation (done ONCE outside the objective)
# ---------------------------------------------------------------------------

def prepare_data():
    """Load, preprocess, and build all Darts TimeSeries objects needed."""
    df = load_and_preprocess(DATA_FILE)

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

    target_ts   = TimeSeries.from_series(df["Total_Generation"], freq="D")
    past_cov_ts = TimeSeries.from_dataframe(df[past_cov_cols], freq="D")
    fut_cov_ts  = TimeSeries.from_dataframe(df[future_cov_cols], freq="D")

    train_end   = pd.Timestamp("2023-12-31")
    val_start   = pd.Timestamp("2024-01-01")
    val_end     = pd.Timestamp("2024-12-31")

    # Train: 2017-2023  |  Val: 2024  |  TrainVal: 2017-2024
    train_target    = target_ts.drop_after(val_start)
    val_target      = target_ts.drop_before(train_end).drop_after(
        pd.Timestamp("2025-01-01"))
    trainval_target = target_ts.drop_after(pd.Timestamp("2025-01-01"))

    train_past      = past_cov_ts.drop_after(val_start)
    trainval_past   = past_cov_ts.drop_after(pd.Timestamp("2025-01-01"))
    train_future    = fut_cov_ts.drop_after(val_start)
    trainval_future = fut_cov_ts.drop_after(pd.Timestamp("2025-01-01"))

    return {
        "train_target"   : train_target,
        "val_target"     : val_target,
        "trainval_target": trainval_target,
        "train_past"     : train_past,
        "trainval_past"  : trainval_past,
        "train_future"   : train_future,
        "trainval_future": trainval_future,
        "past_cov_ts"    : past_cov_ts,
        "fut_cov_ts"     : fut_cov_ts,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(data: dict):
    """Return an Optuna objective closure that uses pre-loaded data."""

    def objective(trial: optuna.Trial) -> float:
        # -- LightGBM hyperparameters --------------------------------------
        num_leaves        = trial.suggest_int("num_leaves", 31, 255)
        n_estimators      = trial.suggest_int("n_estimators", 300, 1500)
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        max_depth         = trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12])
        min_child_samples = trial.suggest_int("min_child_samples", 5, 40)
        subsample         = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)

        # -- Lag structure -------------------------------------------------
        base_lags = [-1, -2, -3, -7, -14, -28, -90]
        include_182 = trial.suggest_categorical("include_lag_182", [True, False])
        include_730 = trial.suggest_categorical("include_lag_730", [True, False])
        target_lags = sorted(
            base_lags
            + ([-182] if include_182 else [])
            + ([-365])
            + ([-730] if include_730 else []),
        )

        past_cov_depth = trial.suggest_categorical(
            "past_cov_depth",
            ["shallow", "medium", "deep"],
        )
        if past_cov_depth == "shallow":
            past_cov_lags = [-1, -2, -7]
        elif past_cov_depth == "medium":
            past_cov_lags = [-1, -2, -7, -14, -28]
        else:
            past_cov_lags = [-1, -2, -7, -14, -28, -90]

        fut_past_ctx = trial.suggest_categorical("fut_past_ctx", [3, 7, 14])
        future_cov_lags = (fut_past_ctx, 1)

        # -- Retraining strategy -------------------------------------------
        train_length = trial.suggest_categorical("train_length", [760, 900, 1095, 1460])
        stride       = trial.suggest_categorical("stride", [7, 14, 30])

        # -- Build and evaluate model with retrain=True --------------------
        model = LightGBMModel(
            lags=target_lags,
            lags_past_covariates=past_cov_lags,
            lags_future_covariates=future_cov_lags,
            output_chunk_length=1,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=-1,
        )

        try:
            # Use adaptive rolling retraining on val set — same regime as test
            val_preds = model.historical_forecasts(
                series=data["trainval_target"],
                past_covariates=data["trainval_past"],
                future_covariates=data["trainval_future"],
                start=data["val_target"].start_time(),
                forecast_horizon=1,
                stride=stride,
                retrain=True,
                train_length=train_length,
                verbose=False,
                last_points_only=True,
            )
            val_aligned = data["val_target"].slice_intersect(val_preds)
            rmse = float(darts_rmse(val_aligned, val_preds))
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {e}")

        return rmse

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for the LightGBM forecast model")
    parser.add_argument("--trials",  type=int,  default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume an existing study from the DB (default: False)")
    parser.add_argument("--timeout", type=int,  default=None,
                        help="Stop after N seconds regardless of trial count")
    args = parser.parse_args()

    print("=" * 65)
    print(" Optuna Hyperparameter Tuning – LightGBM Forecast Model")
    print("=" * 65)
    print(f"  Trials   : {args.trials}")
    print(f"  DB       : {STUDY_DB}")
    print(f"  Strategy : retrain=True on 2024 val set (same regime as test)")
    print(f"  Objective: Minimise validation RMSE (GWh/day)")
    print(f"  Note     : ~30-60 s/trial  →  {args.trials} trials ≈ {args.trials//2}-{args.trials} min")
    print()

    # Suppress Optuna's per-trial logs; we print our own summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Preprocessing data ...")
    data = prepare_data()
    print("Data ready.\n")

    storage = f"sqlite:///{STUDY_DB}"
    study_name = "lgbm_forecast_v3"   # v3: +Roll365_Mean +YoY_Delta
    load_if_exists = args.resume

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        storage=storage,
        load_if_exists=load_if_exists,
    )

    # Seed with current known-good config so early trials have a baseline
    known_good = {
        "num_leaves": 127, "n_estimators": 1000, "learning_rate": 0.03,
        "max_depth": 10,   "min_child_samples": 15, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 0.05,
        "include_lag_182": True, "include_lag_730": False,
        "past_cov_depth": "medium", "fut_past_ctx": 7,
        "train_length": 900, "stride": 30,
    }
    study.enqueue_trial(known_good)

    print(f"{'Trial':>6}  {'Val RMSE':>10}  {'Best RMSE':>10}  Params summary")
    print("-" * 75)

    completed = [0]   # mutable closure counter

    def callback(study: optuna.Study, trial: optuna.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        completed[0] += 1
        p = trial.params
        lag_notes = (
            f"lags={'base' + ('+182' if p.get('include_lag_182') else '') + ('+730' if p.get('include_lag_730') else '')}  "
            f"pc={p.get('past_cov_depth','?')}  fctx={p.get('fut_past_ctx','?')}  "
            f"tl={p.get('train_length','?')}  s={p.get('stride','?')}"
        )
        print(
            f"{trial.number:>6}  "
            f"{trial.value:>10.4f}  "
            f"{study.best_value:>10.4f}  "
            f"leaves={p['num_leaves']} est={p['n_estimators']} "
            f"lr={p['learning_rate']:.4f} md={p['max_depth']}  "
            f"{lag_notes}"
        )

    study.optimize(
        make_objective(data),
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[callback],
        show_progress_bar=False,
    )

    best  = study.best_trial
    bparams = best.params

    # Reconstruct structured lag lists from the categorical flags
    base_lags = [-1, -2, -3, -7, -14, -28, -90]
    best_target_lags = sorted(
        base_lags
        + ([-182] if bparams.get("include_lag_182") else [])
        + [-365]
        + ([-730] if bparams.get("include_lag_730") else []),
    )
    depth = bparams.get("past_cov_depth", "medium")
    if depth == "shallow":
        best_past_cov_lags = [-1, -2, -7]
    elif depth == "medium":
        best_past_cov_lags = [-1, -2, -7, -14, -28]
    else:
        best_past_cov_lags = [-1, -2, -7, -14, -28, -90]

    best_output = {
        "num_leaves"       : bparams["num_leaves"],
        "n_estimators"     : bparams["n_estimators"],
        "learning_rate"    : round(bparams["learning_rate"], 6),
        "max_depth"        : bparams["max_depth"],
        "min_child_samples": bparams["min_child_samples"],
        "subsample"        : round(bparams["subsample"], 4),
        "colsample_bytree" : round(bparams["colsample_bytree"], 4),
        "reg_alpha"        : round(bparams["reg_alpha"], 6),
        "reg_lambda"       : round(bparams["reg_lambda"], 6),
        "target_lags"      : best_target_lags,
        "past_cov_lags"    : best_past_cov_lags,
        "future_cov_lags"  : [bparams.get("fut_past_ctx", 7), 1],
        "train_length"     : bparams.get("train_length", 730),
        "stride"           : bparams.get("stride", 30),
        "_val_rmse"        : round(best.value, 4),
        "_n_trials"        : len(study.trials),
        "_optuna_trial_no" : best.number,
    }

    out_path = ARTIFACT_DIR / "best_hyperparams.json"
    with open(out_path, "w") as f:
        json.dump(best_output, f, indent=2)

    print()
    print("=" * 65)
    print(f" Best trial  #{best.number}  –  Val RMSE = {best.value:.4f} GWh/day")
    print("=" * 65)
    print(f"  num_leaves       : {best_output['num_leaves']}")
    print(f"  n_estimators     : {best_output['n_estimators']}")
    print(f"  learning_rate    : {best_output['learning_rate']}")
    print(f"  max_depth        : {best_output['max_depth']}")
    print(f"  min_child_samples: {best_output['min_child_samples']}")
    print(f"  subsample        : {best_output['subsample']}")
    print(f"  colsample_bytree : {best_output['colsample_bytree']}")
    print(f"  reg_alpha        : {best_output['reg_alpha']}")
    print(f"  reg_lambda       : {best_output['reg_lambda']}")
    print(f"  target_lags      : {best_output['target_lags']}")
    print(f"  past_cov_lags    : {best_output['past_cov_lags']}")
    print(f"  train_length     : {best_output['train_length']}")
    print(f"  stride           : {best_output['stride']}")
    print()
    print(f"  Saved → {out_path}")
    print()
    print("  Run forecast.py to train the final model with these parameters:")
    print("  .\\venv\\Scripts\\python.exe forecast.py")
    print("=" * 65)

    # Print top-5 for reference
    print("\nTop-5 trials:")
    top5 = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )[:5]
    for i, t in enumerate(top5, 1):
        print(f"  #{i}  Trial {t.number:>3}  RMSE={t.value:.4f}  "
              f"leaves={t.params['num_leaves']}  est={t.params['n_estimators']}  "
              f"lr={t.params['learning_rate']:.4f}")


if __name__ == "__main__":
    main()
