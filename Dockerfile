# ── Sri Lanka National Grid – Energy Forecasting ──────────────────────────────
# Build:  docker build -t grid-forecast .
# Run:    docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" grid-forecast
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# libgomp1  → required by LightGBM
# libgomp is the only non-pure-Python system dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer unless requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset
COPY *.py ./
COPY data/ data/

# Artifacts directory – mount a named volume here for persistence
RUN mkdir -p artifacts

EXPOSE 8501

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["./entrypoint.sh"]
