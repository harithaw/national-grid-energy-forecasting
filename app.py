"""
app.py – Sri Lanka National Grid Energy Forecast Dashboard
==========================================================
Run with:
    .\\venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sri Lanka Grid – Energy Forecast",
    page_icon="⚡",
    layout="wide",
)

ARTIFACT_DIR = Path("artifacts")

SOURCE_COLS = [
    "Solar_Total", "Mini_Hydro", "Biomass_Waste", "Wind",
    "Major_Hydro", "Oil_IPP", "Oil_CEB", "Coal",
]
SOURCE_LABELS = {
    "Solar_Total"  : "Solar (Total)",
    "Mini_Hydro"   : "Mini Hydro",
    "Biomass_Waste": "Biomass & Waste",
    "Wind"         : "Wind",
    "Major_Hydro"  : "Major Hydro",
    "Oil_IPP"      : "Oil (IPP)",
    "Oil_CEB"      : "Oil (CEB)",
    "Coal"         : "Coal",
}
COLOR_MAP = {
    "Solar_Total"  : "#FDD835",
    "Mini_Hydro"   : "#42A5F5",
    "Biomass_Waste": "#66BB6A",
    "Wind"         : "#26C6DA",
    "Major_Hydro"  : "#1565C0",
    "Oil_IPP"      : "#EF9A9A",
    "Oil_CEB"      : "#E53935",
    "Coal"         : "#757575",
}


def artifacts_ready() -> bool:
    return all(
        (ARTIFACT_DIR / f).exists()
        for f in ["model.pkl", "test_predictions.csv", "metrics.json",
                  "processed_data.csv", "shap_summary.png"]
    )


@st.cache_resource
def load_model():
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_predictions():
    return pd.read_csv(ARTIFACT_DIR / "test_predictions.csv",
                       index_col=0, parse_dates=True)


@st.cache_data
def load_metrics():
    with open(ARTIFACT_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_processed_data():
    return pd.read_csv(ARTIFACT_DIR / "processed_data.csv",
                       index_col=0, parse_dates=True)


@st.cache_data
def load_future_forecast():
    path = ARTIFACT_DIR / "future_forecast.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)



import plotly.graph_objects as go

# Header
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
                padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h1 style="color:white; margin:0; font-size:2rem;">
            ⚡ Sri Lanka National Grid – Daily Generation Forecast
        </h1>
        <p style="color:#90CAF9; margin:0.4rem 0 0; font-size:1rem;">
            LightGBM · Darts Library · SHAP Explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ready():
    st.error(
        "⚠️  Artifacts not found. "
        "Please run **`forecast.py`** first.\n\n"
        "```\n.\\venv\\Scripts\\python.exe forecast.py\n```"
    )
    st.stop()

model     = load_model()
pred_df   = load_predictions()
metrics   = load_metrics()
full_df   = load_processed_data()
future_df = load_future_forecast()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
                padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h1 style="color:white; margin:0; font-size:2rem;">
            ⚡ Sri Lanka National Grid – Daily Generation Forecast
        </h1>
        <p style="color:#90CAF9; margin:0.4rem 0 0; font-size:1rem;">
            LightGBM · Darts Library · SHAP Explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 1 – Future Generation Forecast  (FOREMOST)
# ============================================================
st.subheader("🔮 Future Generation Forecast")

if future_df is not None:
    src_available = [c for c in SOURCE_COLS if c in future_df.columns]
    max_horizon   = len(future_df)

    horizon = st.slider(
        "Forecast horizon (days ahead)",
        min_value=7, max_value=max_horizon,
        value=min(30, max_horizon), step=7,
        key="future_horizon_slider",
    )
    future_slice = future_df.head(horizon)

    chart_col, shap_col = st.columns([3, 2], gap="large")

    with chart_col:
        fig_bar = go.Figure()
        for src in src_available:
            fig_bar.add_trace(go.Bar(
                x=future_slice.index,
                y=future_slice[src],
                name=SOURCE_LABELS.get(src, src),
                marker_color=COLOR_MAP.get(src, "#9E9E9E"),
                hovertemplate="%{y:.2f} GWh<extra>" + SOURCE_LABELS.get(src, src) + "</extra>",
            ))
        fig_bar.add_trace(go.Scatter(
            x=future_slice.index,
            y=future_slice["Forecast_GWh"],
            name="Total (Forecast)",
            mode="lines",
            line=dict(color="black", width=1.8, dash="dot"),
            hovertemplate="%{y:.2f} GWh total<extra></extra>",
        ))
        fig_bar.update_layout(
            barmode="stack",
            title=(
                f"Forecast by Source – Next {horizon} Days  "
                f"({future_slice.index[0].strftime('%d %b %Y')} → "
                f"{future_slice.index[-1].strftime('%d %b %Y')})"
            ),
            xaxis_title="Date",
            yaxis_title="Generation (GWh/day)",
            yaxis=dict(rangemode="tozero"),
            height=460,
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11),
            ),
        )
        st.plotly_chart(fig_bar, width="stretch")

    with shap_col:
        st.markdown("#### 🔍 Why these values? – Feature Importance")
        st.caption(
            "Global SHAP summary over the test period. Features at the top "
            "have the largest influence on every forecast step. "
            "Red dots = high feature value, blue = low."
        )
        if (ARTIFACT_DIR / "shap_summary.png").exists():
            st.image(str(ARTIFACT_DIR / "shap_summary.png"), width="stretch")
        else:
            st.info("Run `forecast.py` to generate the SHAP plot.")

        if (ARTIFACT_DIR / "shap_local.png").exists():
            with st.expander("🔬 Most-recent prediction breakdown (local SHAP)"):
                st.caption(
                    "SHAP contributions for the **last test-set prediction**. "
                    "Red bars push the value up; blue bars push it down."
                )
                st.image(str(ARTIFACT_DIR / "shap_local.png"), width="stretch")

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    last_actual = float(full_df["Total_Generation"].iloc[-1])
    mean_fc     = future_slice["Forecast_GWh"].mean()
    delta_pct   = (mean_fc - last_actual) / last_actual * 100
    kc1.metric("📅 Horizon",      f"{horizon} days")
    kc2.metric("📈 Avg Forecast", f"{mean_fc:.2f} GWh/day")
    kc3.metric("⬇️ Min",         f"{future_slice['Forecast_GWh'].min():.2f} GWh/day")
    kc4.metric("⬆️ Max",         f"{future_slice['Forecast_GWh'].max():.2f} GWh/day")
    kc5.metric("🔄 vs Last Actual",
               f"{last_actual:.2f} GWh/day",
               delta=f"{delta_pct:+.1f}%")

    with st.expander("📅 View / download detailed forecast table"):
        col_labels = {
            "Forecast_GWh": "Total Forecast (GWh/day)",
            **{s: SOURCE_LABELS.get(s, s) + " (GWh/day)" for s in src_available},
        }
        show_df = future_slice[["Forecast_GWh"] + src_available].copy()
        show_df.index = show_df.index.strftime("%Y-%m-%d")
        show_df.rename(columns=col_labels, inplace=True)
        st.dataframe(show_df.round(3), height=300)
        st.download_button(
            "⬇️ Download Future Forecast CSV",
            future_df.to_csv().encode("utf-8"),
            file_name="future_forecast.csv",
            mime="text/csv",
        )
else:
    st.info(
        "⚠️ Future forecast not found. "
        "Re-run **`forecast.py`** to generate forward-looking predictions."
    )

st.markdown("---")

# ============================================================
# SECTION 2 – Model Performance
# ============================================================
st.subheader("📊 Model Performance")

mk1, mk2, mk3, mk4, mk5 = st.columns(5)
for col, (key, label, unit) in zip(
    [mk1, mk2, mk3, mk4, mk5],
    [
        ("RMSE", "Root Mean Sq. Error", "GWh/day"),
        ("MAE",  "Mean Absolute Error", "GWh/day"),
        ("MAPE", "Mean Abs % Error",    "%"),
        ("MSE",  "Mean Squared Error",  "GWh²"),
        ("R2",   "R² Score",            ""),
    ],
):
    val = metrics.get(key, "N/A")
    col.metric(label, f"{val} {unit}".strip() if isinstance(val, (int, float)) else str(val))

tab_daily, tab_monthly, tab_resid = st.tabs(
    ["📉 Daily Forecast vs Actual", "📅 Monthly Aggregation", "📐 Residuals"]
)

with tab_daily:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Actual"],
        name="Actual", line=dict(color="#2196F3", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Predicted"],
        name="Predicted", line=dict(color="#FF5722", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="Daily Total Generation – Actual vs Predicted (Test 2025–2026)",
        xaxis_title="Date", yaxis_title="Generation (GWh/day)",
        yaxis=dict(rangemode="tozero"), height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

with tab_monthly:
    if (ARTIFACT_DIR / "monthly_comparison.png").exists():
        st.image(str(ARTIFACT_DIR / "monthly_comparison.png"), width="stretch")
    else:
        monthly = pred_df.resample("ME").mean()
        xlabels = [d.strftime("%b %Y") for d in monthly.index]
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=xlabels, y=monthly["Actual"],
                               name="Actual", marker_color="#2196F3"))
        fig_m.add_trace(go.Bar(x=xlabels, y=monthly["Predicted"],
                               name="Predicted", marker_color="#FF5722"))
        fig_m.update_layout(barmode="group", height=380, template="plotly_white",
                            yaxis_title="Avg GWh/day", yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig_m, width="stretch")

with tab_resid:
    residuals = pred_df["Actual"] - pred_df["Predicted"]
    colors    = np.where(residuals >= 0, "#4CAF50", "#F44336").tolist()
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=residuals.index, y=residuals,
        marker_color=colors, name="Residual",
        hovertemplate="%{y:.2f} GWh<extra></extra>",
    ))
    fig_r.add_hline(y=0, line_color="black", line_width=1)
    fig_r.update_layout(
        title="Residuals (Actual − Predicted)",
        xaxis_title="Date", yaxis_title="Residual (GWh/day)",
        height=360, template="plotly_white",
    )
    st.plotly_chart(fig_r, width="stretch")

st.markdown("---")

# ============================================================
# SECTION 3 – Historical Generation Mix
# ============================================================
st.subheader("🔋 Historical Generation Mix")

avail_src   = [c for c in SOURCE_COLS if c in full_df.columns]
monthly_mix = full_df[avail_src].resample("ME").mean()

fig_mix = go.Figure()
for col in avail_src:
    fig_mix.add_trace(go.Scatter(
        x=monthly_mix.index, y=monthly_mix[col],
        name=SOURCE_LABELS.get(col, col),
        stackgroup="one",
        fillcolor=COLOR_MAP.get(col, "#9E9E9E"),
        line=dict(width=0.5, color=COLOR_MAP.get(col, "#9E9E9E")),
        hovertemplate="%{y:.2f} GWh/day<extra>" + SOURCE_LABELS.get(col, col) + "</extra>",
    ))
fig_mix.update_layout(
    title="Monthly Average Generation by Source (GWh/day)",
    xaxis_title="Date", yaxis_title="GWh/day",
    yaxis=dict(rangemode="tozero"),
    height=420, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
)
st.plotly_chart(fig_mix, width="stretch")

st.markdown("---")

# ============================================================
# SECTION 4 – Test-Set Data Table
# ============================================================
st.subheader("📋 Test-Set Forecast Data Table")

display_df              = pred_df.copy()
display_df["Residual"]  = (display_df["Actual"] - display_df["Predicted"]).round(3)
display_df["Abs Error"] = display_df["Residual"].abs().round(3)
display_df["Error %"]   = ((display_df["Residual"] / display_df["Actual"]) * 100).round(2)
display_df.index        = display_df.index.strftime("%Y-%m-%d")
display_df              = display_df.round(3)

st.dataframe(
    display_df.style
    .background_gradient(subset=["Abs Error"], cmap="YlOrRd")
    .format({"Error %": "{:.2f}%"}),
    width="stretch",
    height=380,
)
st.download_button(
    "⬇️ Download Test-Set Forecast CSV",
    display_df.to_csv().encode("utf-8"),
    file_name="forecast_results.csv",
    mime="text/csv",
)

st.markdown(
    """
    <hr style="border:1px solid #e0e0e0; margin-top:2rem;">
    <p style="text-align:center; color:#9E9E9E; font-size:0.85rem;">
        Sri Lanka National Grid · Daily Generation Forecast ·
        LightGBM via Darts · SHAP Explainability
    </p>
    """,
    unsafe_allow_html=True,
)
