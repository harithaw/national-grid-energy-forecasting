"""Data loading and feature engineering."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


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

    df["Solar_Total"] = df["Solar"] + df["Solar_Estimated"] + df["Rooftop_Solar"]
    df["Mini_Hydro"]  = df["Mini_Hydro_Telemetered"] + df["Mini_Hydro_Estimated"]
    df["Total_Generation"] = (
        df["Solar_Total"] + df["Mini_Hydro"]
        + df["Biomass_Waste"] + df["Wind"] + df["Major_Hydro"]
        + df["Oil_IPP"] + df["Oil_CEB"] + df["Coal"]
    )

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

    gen = df["Total_Generation"]
    df["Roll7_Mean"]   = gen.rolling(7,   min_periods=1).mean()
    df["Roll30_Mean"]  = gen.rolling(30,  min_periods=1).mean()
    df["Roll90_Mean"]  = gen.rolling(90,  min_periods=1).mean()
    df["Roll365_Mean"] = gen.rolling(365, min_periods=30).mean()
    df["Roll365_Mean"].fillna(df["Roll365_Mean"].expanding().mean(), inplace=True)
    df["Roll30_Std"]   = gen.rolling(30, min_periods=2).std().fillna(0)
    df["Monthly_Avg"]  = gen.groupby(t.month).transform("mean")
    df["YoY_Delta"]    = (gen - gen.shift(365)).fillna(0.0)

    print(f"\nTarget statistics:")
    print(df["Total_Generation"].describe().round(3).to_string())
    print(f"\nYearly mean Total_Generation:")
    print(df["Total_Generation"].resample("YE").mean().round(2).to_string())
    print("\nPreprocessing complete OK")
    return df
