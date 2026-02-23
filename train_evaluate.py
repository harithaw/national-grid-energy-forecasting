"""Model training, rolling evaluation, and performance plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from darts.metrics import rmse as darts_rmse, mae as darts_mae, mape as darts_mape

from config import ARTIFACT_DIR


def train_and_evaluate(model,
                       train_target, val_target, test_target,
                       train_past, val_past, test_past,
                       train_future, val_future, test_future,
                       target_series, past_cov_ts, future_cov_ts,
                       retrain_train_length: int = 730,
                       retrain_stride: int = 30):
    print("\n" + "="*60)
    print("STEP 3 - Model Training & Evaluation")
    print("="*60)

    trainval_target = train_target.append(val_target)
    trainval_past   = train_past.append(val_past)
    trainval_future = train_future.append(val_future)

    print("\nFitting model on train+val (2017-2024) ...")
    model.fit(
        series=trainval_target,
        past_covariates=trainval_past,
        future_covariates=trainval_future,
    )
    print("Training complete OK")

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
    val_q50    = val_hist.univariate_component(1) if val_hist.n_components > 1 else val_hist
    val_aligned = val_target.slice_intersect(val_q50)
    print(f"  Val RMSE : {float(darts_rmse(val_aligned, val_q50)):.4f} GWh")
    print(f"  Val MAE  : {float(darts_mae(val_aligned, val_q50)):.4f} GWh")
    print(f"  Val MAPE : {float(darts_mape(val_aligned, val_q50)):.4f} %")

    print("\nGenerating adaptive rolling forecasts on test set ...")
    print(f"  (retrain=True, stride={retrain_stride}, train_length={retrain_train_length})")

    predictions = model.historical_forecasts(
        series=target_series,
        past_covariates=past_cov_ts,
        future_covariates=future_cov_ts,
        start=test_target.start_time(),
        forecast_horizon=1,
        stride=1,
        retrain=True,
        train_length=retrain_train_length,
        verbose=False,
        last_points_only=True,
    )
    print(f"Predictions generated : {len(predictions)} time-steps")

    pred_q50 = predictions.univariate_component(1) if predictions.n_components > 1 else predictions
    pred_q10 = predictions.univariate_component(0) if predictions.n_components > 1 else None
    pred_q90 = predictions.univariate_component(2) if predictions.n_components > 1 else None

    test_aligned = test_target.slice_intersect(pred_q50)
    y_true = test_aligned.to_series().values
    y_pred = pred_q50.to_series().values

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
        unit = " %" if k == "MAPE" else (" GWh" if k in ["RMSE", "MAE"] else "")
        print(f"  {k:<6}: {v:>9}{unit}")
    print("-"*65)

    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    ax = axes[0]
    test_aligned.to_series().plot(ax=ax, label="Actual",    color="#2196F3", lw=1.3)
    pred_q50.to_series().plot(    ax=ax, label="Predicted (median)", color="#FF5722", lw=1.3, ls="--")
    if pred_q10 is not None:
        ax.fill_between(
            test_aligned.to_series().index,
            pred_q10.to_series().values,
            pred_q90.to_series().values,
            alpha=0.18, color="#FF5722", label="80% confidence interval",
        )
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
    pred_df2   = pred_q50.to_dataframe();     pred_df2.columns   = ["Predicted"]
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

    errors      = y_pred - y_true
    abs_max_err = np.abs(errors).max()
    fig3, ax4 = plt.subplots(figsize=(7, 7))
    sc = ax4.scatter(y_true, y_pred, c=errors,
                     cmap="RdYlGn_r", vmin=-abs_max_err, vmax=abs_max_err,
                     alpha=0.7, s=20, edgecolors="none")
    xy_min_sc = min(y_true.min(), y_pred.min()) * 0.97
    xy_max_sc = max(y_true.max(), y_pred.max()) * 1.03
    ax4.plot([xy_min_sc, xy_max_sc], [xy_min_sc, xy_max_sc],
             color="#212121", lw=1.5, ls="--", label="Perfect fit (y = x)")
    ax4.set_xlim(xy_min_sc, xy_max_sc)
    ax4.set_ylim(xy_min_sc, xy_max_sc)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_title(f"Actual vs Predicted – Test Set  (R² = {r2_val:.4f})",
                  fontsize=13, fontweight="bold")
    ax4.set_xlabel("Actual Generation (GWh/day)")
    ax4.set_ylabel("Predicted Generation (GWh/day)")
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Error (Predicted − Actual, GWh)")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Actual vs Predicted scatter saved -> {ARTIFACT_DIR / 'actual_vs_predicted.png'}")

    return pred_q50, pred_q10, pred_q90, test_aligned, metrics
