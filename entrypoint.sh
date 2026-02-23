#!/bin/sh
set -e

# Run the full forecast pipeline if artifacts are missing or stale.
# When using a mounted volume the artifacts persist across container restarts,
# so the pipeline only runs once (or whenever you delete the volume).
if [ ! -f "artifacts/model.pkl" ] || [ ! -f "artifacts/future_forecast.csv" ]; then
    echo "================================================================"
    echo "  Artifacts not found – running forecast pipeline (first run)   "
    echo "  This takes several minutes. Subsequent starts are instant.    "
    echo "================================================================"
    python forecast.py
fi

echo "Starting Streamlit dashboard on port 8501 ..."
exec streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
