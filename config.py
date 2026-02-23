"""Shared constants for the National Grid forecasting pipeline."""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

DATA_FILE    = Path("data/merged-generation-profile-2017-2026.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 90
