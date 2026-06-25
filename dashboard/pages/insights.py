import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Data Insights")

df = pd.read_csv(
    "../data/processed/hourly_energy.csv"
)

hourly = (
    df.groupby("hour")["t_kWh"]
    .mean()
    .reset_index()
)

fig = px.line(
    hourly,
    x="hour",
    y="t_kWh",
    title="Average Hourly Consumption"
)

st.plotly_chart(fig)

city_consumption = (
    df.groupby("city")["t_kWh"]
    .mean()
    .reset_index()
)

fig = px.bar(
    city_consumption,
    x="city",
    y="t_kWh",
    title="Average Consumption by City"
)

st.plotly_chart(fig)