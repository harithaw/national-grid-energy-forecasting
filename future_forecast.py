"""Future forecast generation beyond the known data range."""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel

from config import ARTIFACT_DIR, FORECAST_HORIZON


def generate_future_forecast(model: LightGBMModel,
                              df: pd.DataFrame,
                              target_series: "TimeSeries",
                              past_cov_ts: "TimeSeries",
                              future_cov_ts: "TimeSeries",
                              horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 5 - Future Forecast Generation")
    print("="*60)

    last_date   = df.index.max()
    future_idx  = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_idx.name = "Date"

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

    print(f"\nRefitting on complete dataset ({len(target_series)} days, "
          f"{target_series.start_time().date()} → {target_series.end_time().date()}) ...")
    model.fit(
        series            = target_series,
        past_covariates   = past_cov_ts,
        future_covariates = extended_future,
    )
    print("Refit complete OK")

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
