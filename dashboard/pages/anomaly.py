import streamlit as st
import pandas as pd

st.title("Anomaly Detection")

anomalies = pd.read_csv(
    "../reports/anomalies_detected.csv"
)

st.metric(
    "Total Anomalies",
    len(anomalies)
)

st.dataframe(
    anomalies.head(50)
)