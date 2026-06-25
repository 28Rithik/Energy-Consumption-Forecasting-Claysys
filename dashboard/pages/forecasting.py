import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Forecasting Performance")

comparison = pd.read_csv(
    "../reports/model_comparison.csv"
)

st.dataframe(comparison)

fig = px.bar(
    comparison,
    x="Model",
    y="RMSE",
    title="Model RMSE Comparison"
)

st.plotly_chart(fig)

pred_df = pd.read_csv(
    "../reports/lstm_predictions.csv"
)

fig2 = px.line(
    pred_df.head(500),
    title="Actual vs Predicted Energy"
)

fig2.add_scatter(
    y=pred_df["Actual"].head(500),
    mode="lines",
    name="Actual"
)

fig2.add_scatter(
    y=pred_df["Predicted"].head(500),
    mode="lines",
    name="Predicted"
)

st.plotly_chart(fig2)

forecast_days = st.slider(
    "Forecast Horizon (Days)",
    1,
    30,
    7
)

st.write(
    f"Showing forecast for next {forecast_days} days"
)