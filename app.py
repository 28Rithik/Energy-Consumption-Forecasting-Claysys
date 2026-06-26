import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# Page config & simple light/clean theme
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Home Energy Dashboard - Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f5f5f5;
    color: #222222;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dddddd;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
}
[data-testid="stMetricLabel"] {
    color: #555555 !important;
    font-size: 0.85rem !important;
}
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #dddddd;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 6px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 4px;
    color: #555555;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #000000 !important;
    color: white !important;
}
h1 { color: #222222 !important; }
h2, h3 { color: #333333 !important; }
[data-testid="stSidebar"] label {
    color: #333333 !important;
    font-weight: 500;
    font-size: 0.85rem;
}
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
[data-testid="stDownloadButton"] > button {
    background: #000000;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.4rem;
    font-weight: 500;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TODAY = date.today().strftime("%d %b %Y")

HORIZON_MAP = {
    "1 Day Ahead":   ("data/cleaned_hourly.csv", "1day",  "h", "hourly"),
    "10 Days Ahead": ("data/cleaned_daily.csv",  "10day", "D", "daily"),
    "1 Year Ahead":  ("data/cleaned_weekly.csv", "1year", "W", "weekly"),
}

MODEL_MAP = {
    "SARIMAX":       "stats",
    "Prophet":       "prophet",
    "PyTorch LSTM":  "pytorch",
}

PALETTE = {
    "hist":    "#000000",
    "fc":      "#666666",
    "anomaly": "#ff0000",
    "band":    "rgba(0,0,0,0.1)",
    "grid":    "rgba(0,0,0,0.05)",
}

PAPER_BG  = "rgba(255,255,255,0)"
PLOT_BG   = "rgba(255,255,255,0.4)"

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies the simple light theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#333333", size=16)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#555555"),
        xaxis=dict(gridcolor=PALETTE["grid"], showgrid=True),
        yaxis=dict(gridcolor=PALETTE["grid"], showgrid=True),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def load_metrics_all(h_str: str) -> pd.DataFrame:
    """Loads and combines validation metrics from SARIMAX, Prophet, and LSTM."""
    rows = []
    sources = [
        ("SARIMAX",      "data/metrics_stats.csv"),
        ("Prophet",      "data/metrics_prophet.csv"),
        ("PyTorch LSTM", "data/metrics_pytorch.csv"),
    ]
    for name, path in sources:
        try:
            mdf = pd.read_csv(path)
            row = mdf[mdf["Horizon"] == h_str].iloc[0]
            rows.append({"Model": name,
                         "MAE (kW)": round(row["MAE"], 4),
                         "RMSE (kW)": round(row["RMSE"], 4)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def load_forecast(m_str: str, h_str: str) -> pd.DataFrame:
    """Loads a single forecast CSV."""
    return pd.read_csv(f"data/forecast_{m_str}_{h_str}.csv", parse_dates=["Datetime"]).set_index("Datetime")


def compute_confidence_band(fc_vals: np.ndarray, width: float = 0.15) -> tuple:
    """Generates simple ± percentage confidence bands around forecast values."""
    upper = fc_vals * (1 + width)
    lower = fc_vals * (1 - width)
    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Historical Insights
# ─────────────────────────────────────────────────────────────────────────────

def render_historical(df: pd.DataFrame, freq: str):
    """Renders KPIs, anomaly chart, candlestick-range chart, and heatmap calendar."""

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_kwh = round(df["Global_active_power"].sum(), 2)
    peak_kw   = round(df["Global_active_power"].max(), 4)
    avg_kw    = round(df["Global_active_power"].mean(), 4)
    n_anom    = int(df["Anomaly"].sum()) if "Anomaly" in df.columns else 0
    c1.metric("⚡ Total Consumed", f"{total_kwh} kW")
    c2.metric("🔥 Peak Demand",    f"{peak_kw} kW")
    c3.metric("📊 Avg Power",      f"{avg_kw} kW")
    c4.metric("⚠️ Anomaly Events", f"{n_anom}")

    # ── Anomaly overlay line chart ────────────────────────────────────────────
    plot_df = df["Global_active_power"].tail(500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df.values,
        mode="lines", name="Active Power",
        line=dict(color=PALETTE["hist"], width=1.5)
    ))
    if "Anomaly" in df.columns:
        anom_df = df[df["Anomaly"]]["Global_active_power"].tail(500)
        fig.add_trace(go.Scatter(
            x=anom_df.index, y=anom_df.values,
            mode="markers", name="Anomaly",
            marker=dict(color=PALETTE["anomaly"], size=7, symbol="circle")
        ))
    dark_layout(fig, "Active Power History with Z-Score Anomalies")
    st.plotly_chart(fig, width="stretch")

    # ── Candlestick-style daily range chart (only for hourly data) ────────────
    if freq == "h":
        st.subheader("📈 Daily Energy Range (Candlestick View)")
        candle_df = df["Global_active_power"].resample("D").agg(
            Open="first", High="max", Low="min", Close="last"
        ).tail(90)
        fig_c = go.Figure(go.Candlestick(
            x=candle_df.index,
            open=candle_df["Open"], high=candle_df["High"],
            low=candle_df["Low"],  close=candle_df["Close"],
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            name="Daily Range"
        ))
        dark_layout(fig_c, "Daily Active Power: Open / High / Low / Close")
        st.plotly_chart(fig_c, width="stretch")

    # ── Consumption heatmap calendar ──────────────────────────────────────────
    st.subheader("🗓️ Hourly Consumption Heatmap Calendar")
    if freq == "h":
        hm_df = df["Global_active_power"].copy()
        hm_df.index = pd.to_datetime(hm_df.index)
        hm_df = hm_df[hm_df.index.year >= hm_df.index.year.max() - 1]
        pivot = hm_df.groupby([hm_df.index.date, hm_df.index.hour]).mean().unstack(fill_value=0)
        pivot.index = pd.to_datetime(pivot.index)
        fig_h = px.imshow(
            pivot.T.values,
            x=[str(d)[:10] for d in pivot.index],
            y=[f"{h:02d}:00" for h in range(24)],
            color_continuous_scale="gray",
            aspect="auto",
            labels=dict(color="kW"),
        )
        dark_layout(fig_h, "Hour-of-Day × Date Heatmap (Active Power)")
        st.plotly_chart(fig_h, width="stretch")
    else:
        hm_df = df["Global_active_power"].resample("ME").mean()
        fig_h = px.bar(
            x=hm_df.index.strftime("%b %Y"), y=hm_df.values,
            labels={"x": "Month", "y": "Avg Active Power (kW)"},
            color_discrete_sequence=["#000000"]
        )
        dark_layout(fig_h, "Monthly Average Active Power")
        st.plotly_chart(fig_h, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Sub-Metering Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def render_submetering(df: pd.DataFrame):
    """Renders stacked area, donut, and correlation heatmap."""

    st.subheader("⚡ Sub-Metering Stacked Area")
    sm_tail = df[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].tail(500)
    fig_area = go.Figure()
    colors = ["#222222", "#777777", "#bbbbbb"]
    labels = ["Kitchen (Sub 1)", "Laundry Room (Sub 2)", "AC & Water Heater (Sub 3)"]
    for col, color, label in zip(["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"], colors, labels):
        fig_area.add_trace(go.Scatter(
            x=sm_tail.index, y=sm_tail[col],
            name=label, mode="lines",
            line=dict(width=1, color=color),
            fill="tonexty" if col != "Sub_metering_1" else "tozeroy",
            fillcolor="rgba(0,0,0,0.08)"
        ))
    dark_layout(fig_area, "Sub-Metering Breakdown Over Time")
    st.plotly_chart(fig_area, width="stretch")

    # ── Donut breakdown ───────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        unm = ((df["Global_active_power"] * 1000) -
               (df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"])).clip(lower=0)
        totals = {
            "Kitchen (Sub 1)":          df["Sub_metering_1"].sum(),
            "Laundry Room (Sub 2)":     df["Sub_metering_2"].sum(),
            "AC & Water Heater (Sub 3)":df["Sub_metering_3"].sum(),
            "Unmetered (Other)":         unm.sum(),
        }
        fig_pie = go.Figure(go.Pie(
            labels=list(totals.keys()), values=list(totals.values()),
            hole=0.45, textinfo="label+percent",
            marker=dict(colors=["#222222", "#777777", "#bbbbbb", "#eeeeee"],
                        line=dict(color="#ffffff", width=2))
        ))
        fig_pie.update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color="#555555"),
            title=dict(text="Energy Contribution Breakdown", font=dict(color="#333333", size=15)),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig_pie, width="stretch")

    # ── Feature correlation heatmap ───────────────────────────────────────────
    with col_r:
        corr_cols = [c for c in ["Global_active_power", "Sub_metering_1",
                                  "Sub_metering_2", "Sub_metering_3"] if c in df.columns]
        corr = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="gray",
            zmin=-1, zmax=1,
            labels=dict(color="Corr"),
        )
        fig_corr.update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color="#555555"),
            title=dict(text="Feature Correlation Heatmap", font=dict(color="#333333", size=15)),
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar=dict(tickfont=dict(color="#555555"))
        )
        st.plotly_chart(fig_corr, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Forecast Engine
# ─────────────────────────────────────────────────────────────────────────────

def render_forecast(hist_df: pd.DataFrame, model_name: str, horizon_name: str,
                    m_str: str, h_str: str, rate_inr: float):
    """Renders confidence-band forecast chart, cost projection, metric table, download."""

    primary_fc  = load_forecast(m_str, h_str)
    recent_hist = hist_df[["Global_active_power"]].tail(100)

    fig = go.Figure()

    # Historical trace
    fig.add_trace(go.Scatter(
        x=recent_hist.index, y=recent_hist["Global_active_power"],
        mode="lines", name="Historical", line=dict(color=PALETTE["hist"], width=2)
    ))

    # Confidence band for primary model
    fc_vals = primary_fc["Global_active_power"].values
    upper, lower = compute_confidence_band(fc_vals)
    fig.add_trace(go.Scatter(
        x=list(primary_fc.index) + list(primary_fc.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor=PALETTE["band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", showlegend=True
    ))

    # Primary forecast trace
    fc_x = [recent_hist.index[-1]] + list(primary_fc.index)
    fc_y = [recent_hist["Global_active_power"].iloc[-1]] + list(fc_vals)
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y, mode="lines",
        name=f"{model_name} Forecast",
        line=dict(color=PALETTE["fc"], width=2.5, dash="dash")
    ))

    dark_layout(fig, f"Forecast Projection — {model_name} ({horizon_name})  |  Generated: {TODAY}")
    st.plotly_chart(fig, width="stretch")

    # ── Cost projection table ─────────────────────────────────────────────────
    st.subheader("💰 Forecasted Energy Cost Projection")
    cost_df = primary_fc.copy()
    cost_df["Est. kWh"] = (cost_df["Global_active_power"] / 60.0).round(4)
    cost_df[f"Cost (₹ @ ₹{rate_inr}/kWh)"] = (cost_df["Est. kWh"] * rate_inr).round(2)
    cost_df = cost_df.rename(columns={"Global_active_power": "Forecast kW"})
    st.dataframe(cost_df.style.background_gradient(
        subset=["Forecast kW"], cmap="gray"), width="stretch")

    # ── Model metrics comparison ──────────────────────────────────────────────
    st.subheader("📐 All-Model Validation Metrics Comparison")
    metrics_df = load_metrics_all(h_str)
    if not metrics_df.empty:
        # Highlight best (lowest) values in green
        styled = metrics_df.style \
            .highlight_min(subset=["MAE (kW)", "RMSE (kW)"], color="#dcfce7") \
            .highlight_max(subset=["MAE (kW)", "RMSE (kW)"], color="#fee2e2") \
            .set_properties(**{"color": "#222222", "background-color": "#ffffff"})
        st.dataframe(styled, width="stretch")

    # ── Download button ───────────────────────────────────────────────────────
    csv_bytes = cost_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast + Cost Projections",
        data=csv_bytes,
        file_name=f"forecast_{m_str}_{h_str}_with_cost.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main App Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <h1 style='font-size:2.2rem; color:#222222;'>
            ⚡ Smart Home Energy Analytics & Forecasting Dashboard
        </h1>
        <p style='color:#666666; font-size:0.95rem;'>
            Multi-Model ML Forecasting · Anomaly Detection · Live Cost Estimation
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Dashboard Controls")

    horizon = st.sidebar.selectbox(
        "🕐 Forecast Horizon",
        list(HORIZON_MAP.keys())
    )
    model = st.sidebar.selectbox(
        "🤖 Model Architecture",
        list(MODEL_MAP.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Energy Cost Settings")
    rate_inr = st.sidebar.number_input(
        "Electricity Rate (INR per kWh)",
        min_value=0.1, max_value=50.0, value=8.0, step=0.5
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Info")
    st.sidebar.caption("Source: UCI Household Power Consumption")
    st.sidebar.caption("2,075,259 minute-level records · 47 months")
    st.sidebar.caption("Models: SARIMAX · Prophet · BiLSTM+Attention")

    # ── Load data ──────────────────────────────────────────────────────────────
    h_file, h_str, freq, freq_name = HORIZON_MAP[horizon]
    m_str = MODEL_MAP[model]
    hist_df = pd.read_csv(h_file, parse_dates=["Datetime"], index_col="Datetime")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Historical Insights",
        "🔌 Sub-Metering Breakdown",
        "🔮 Future Demand Forecast Engine"
    ])

    with tab1:
        render_historical(hist_df, freq)
    with tab2:
        render_submetering(hist_df)
    with tab3:
        render_forecast(hist_df, model, horizon, m_str, h_str, rate_inr)