import streamlit as st
import pandas as pd

st.title("Peak Demand Analysis")

df = pd.read_csv(
    "../data/processed/hourly_energy.csv"
)

threshold = df["t_kWh"].quantile(0.80)

st.metric(
    "Peak Threshold",
    round(threshold,3)
)

peak = df[
    df["t_kWh"] >= threshold
]

st.write(
    f"Peak Records: {len(peak)}"
)

st.dataframe(
    peak.head(100)
)