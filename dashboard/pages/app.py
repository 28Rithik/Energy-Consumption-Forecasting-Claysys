import streamlit as st

st.set_page_config(
    page_title="Energy Analytics Platform",
    layout="wide"
)

st.title("⚡ Smart Energy Analytics Platform")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Smart Meters",
        "84"
    )

with col2:
    st.metric(
        "Records",
        "21.4M"
    )

with col3:
    st.metric(
        "Best RMSE",
        "0.074"
    )

st.markdown("---")

st.write("""
Advanced Energy Consumption Forecasting,
Anomaly Detection,
and Peak Demand Analytics.
""")