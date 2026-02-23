"""SHAP-based model explainability."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.explainability import ShapExplainer

from config import ARTIFACT_DIR


def explain_with_shap(model, target_series, past_cov_ts, future_cov_ts, test_target):
    print("\n" + "="*60)
    print("STEP 4 - Explainability with SHAP")
    print("="*60)

    bg_start = pd.Timestamp("2019-12-31")
    bg_end   = pd.Timestamp("2025-01-01")
    bg_target = target_series.drop_before(bg_start).drop_after(bg_end)
    bg_past   = past_cov_ts.drop_before(bg_start).drop_after(bg_end)
    bg_future = future_cov_ts.drop_before(bg_start).drop_after(bg_end)

    print("Building ShapExplainer (2020-2024 background) ...")
    explainer = ShapExplainer(
        model=model,
        background_series=bg_target,
        background_past_covariates=bg_past,
        background_future_covariates=bg_future,
    )
    print("ShapExplainer built OK")

    fg_start  = test_target.end_time() - pd.Timedelta(days=800)
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
