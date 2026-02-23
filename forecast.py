"""Sri Lanka National Grid – daily generation forecasting pipeline."""

from __future__ import annotations

import json
import pickle

from config import DATA_FILE, ARTIFACT_DIR, FORECAST_HORIZON
from preprocess import load_and_preprocess
from model_setup import build_darts_series
from train_evaluate import train_and_evaluate
from future_forecast import generate_future_forecast
from explainability import explain_with_shap


def main():
    df = load_and_preprocess(DATA_FILE)

    (model,
     target_series, past_cov_ts, future_cov_ts,
     train_target, val_target, test_target,
     train_past, val_past, test_past,
     train_future, val_future, test_future,
     past_cov_cols, future_cov_cols,
     retrain_train_length, retrain_stride) = build_darts_series(df)

    predictions, test_aligned, metrics = train_and_evaluate(
        model,
        train_target, val_target, test_target,
        train_past,   val_past,   test_past,
        train_future, val_future, test_future,
        target_series, past_cov_ts, future_cov_ts,
        retrain_train_length=retrain_train_length,
        retrain_stride=retrain_stride,
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

    generate_future_forecast(
        model, df, target_series, past_cov_ts, future_cov_ts,
        horizon=FORECAST_HORIZON,
    )

    try:
        explain_with_shap(
            model, target_series, past_cov_ts, future_cov_ts, test_target
        )
    except Exception as _shap_err:
        print(f"\n[WARN] SHAP step skipped: {_shap_err}")
        print("       Re-run forecast.py to retry after adjusting bg window.")

    print(f"\nAll artifacts saved to  {ARTIFACT_DIR.resolve()}")
    print("\n" + "="*60)
    print("Pipeline complete OK")
    print("="*60)
    print("\nTo launch the dashboard:")
    print(r"  .\venv\Scripts\streamlit.exe run app.py")


if __name__ == "__main__":
    main()