# -*- coding: utf-8 -*-
"""UrbanPulse NYC 311 Streamlit dashboard.

Run with:
    streamlit run aiml_urbanpulse_nyc.py

This dashboard loads the NYC 311 complaint dataset, cleans it, and presents
key complaint and resolution insights in an interactive web app.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import streamlit as st
except ImportError:  # pragma: no cover - only relevant when streamlit is not installed
    st = None

if st is not None:
    cache_data = st.cache_data
else:
    def cache_data(func):
        return func

DATA_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


@cache_data
def load_dataset(limit: int = 10000) -> pd.DataFrame:
    """Load the NYC 311 dataset with a limited row count."""
    url = f"{DATA_URL}?$limit={limit}"
    df = pd.read_csv(url)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard cleaning steps for the NYC 311 dataset."""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()

    df_clean["borough"] = df_clean["borough"].fillna("UNKNOWN")
    df_clean["descriptor"] = df_clean["descriptor"].fillna("Not Provided")
    df_clean["status"] = df_clean["status"].fillna("Unknown")
    df_clean["location_type"] = df_clean.get(
        "location_type", pd.Series([np.nan] * len(df_clean))
    ).fillna("Unknown")

    df_clean["created_date"] = pd.to_datetime(df_clean["created_date"], errors="coerce")
    df_clean["closed_date"] = pd.to_datetime(df_clean["closed_date"], errors="coerce")

    if "resolution_hours" not in df_clean.columns:
        df_clean["resolution_hours"] = (
            df_clean["closed_date"] - df_clean["created_date"]
        ).dt.total_seconds() / 3600

    df_clean.loc[df_clean["resolution_hours"] < 0, "resolution_hours"] = np.nan

    df_clean["date"] = df_clean["created_date"].dt.date
    df_clean["day_name"] = df_clean["created_date"].dt.day_name()
    df_clean["hour"] = df_clean["created_date"].dt.hour
    df_clean["year_month"] = df_clean["created_date"].dt.to_period("M")

    return df_clean


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dashboard filter selections from the sidebar."""
    borough = st.sidebar.selectbox("Select borough", ["All"] + sorted(df["borough"].dropna().unique().tolist()))
    status = st.sidebar.selectbox("Select status", ["All"] + sorted(df["status"].dropna().unique().tolist()))
    complaint_type = st.sidebar.selectbox(
        "Select complaint type",
        ["All"] + sorted(df["complaint_type"].dropna().unique().tolist()),
    )
    year = st.sidebar.selectbox("Select year", ["All"] + sorted(df["created_date"].dt.year.dropna().unique().tolist()))

    filtered = df.copy()
    if borough != "All":
        filtered = filtered[filtered["borough"] == borough]
    if status != "All":
        filtered = filtered[filtered["status"] == status]
    if complaint_type != "All":
        filtered = filtered[filtered["complaint_type"] == complaint_type]
    if year != "All":
        filtered = filtered[filtered["created_date"].dt.year == int(year)]

    return filtered


def render_kpis(df: pd.DataFrame) -> None:
    """Render the dashboard KPI cards."""
    total_complaints = len(df)
    closed_complaints = int((df["status"] == "Closed").sum())
    avg_resolution = df["resolution_hours"].mean() if df["resolution_hours"].notna().any() else 0
    top_borough = df["borough"].value_counts().idxmax() if not df.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total complaints", f"{total_complaints:,}")
    col2.metric("Closed", f"{closed_complaints:,}")
    col3.metric("Avg resolution", f"{avg_resolution:.1f} hrs")
    col4.metric("Top borough", top_borough)


def plot_top_complaints(df: pd.DataFrame) -> None:
    """Render top complaint types chart."""
    top10 = df["complaint_type"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top10.values, y=top10.index, ax=ax, palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, padding=5)
    ax.set_title("Top 10 Complaint Types")
    ax.set_xlabel("Number of Complaints")
    ax.set_ylabel("Complaint Type")
    st.pyplot(fig)


def plot_borough_counts(df: pd.DataFrame) -> None:
    """Render complaint counts by borough."""
    borough_counts = df["borough"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=borough_counts.index, y=borough_counts.values, ax=ax, palette="magma")
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    ax.set_title("Complaints by Borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Complaint Count")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)


def plot_status_counts(df: pd.DataFrame) -> None:
    """Render complaint status distribution."""
    status_counts = df["status"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette="deep")
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Complaint Status Distribution")
    ax.set_xlabel("Status")
    ax.set_ylabel("Complaint Count")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)


def plot_resolution_by_borough(df: pd.DataFrame) -> None:
    """Render average resolution time by borough for closed complaints."""
    resolved_df = df[(df["status"] == "Closed") & (df["resolution_hours"].notna())].copy()
    if resolved_df.empty:
        st.info("No valid closed complaint resolution data available for this selection.")
        return

    borough_resolution = resolved_df.groupby("borough")["resolution_hours"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=borough_resolution.index, y=borough_resolution.values, ax=ax, palette="coolwarm")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)
    ax.set_title("Average Resolution Time by Borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Average Resolution Time (Hours)")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)


def plot_daily_trend(df: pd.DataFrame) -> None:
    """Render a daily complaint trend chart."""
    if df.empty or df["created_date"].notna().sum() == 0:
        st.info("No valid created-date data available for the selected filter.")
        return

    daily_counts = df.groupby("date").size().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    daily_counts.plot(kind="line", marker="o", ax=ax, color="steelblue")
    ax.set_title("Daily Complaint Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Complaints")
    ax.grid(True)
    st.pyplot(fig)


def export_dashboard_csv(df: pd.DataFrame) -> None:
    """Enable CSV export for the filtered data."""
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download filtered CSV",
        data=csv,
        file_name="urbanpulse_nyc_filtered.csv",
        mime="text/csv",
    )


def show_data_table(df: pd.DataFrame) -> None:
    """Render a compact table of the filtered dataset."""
    st.subheader("Filtered data preview")
    preview_columns = [
        "created_date",
        "complaint_type",
        "descriptor",
        "borough",
        "status",
        "agency_name",
        "resolution_hours",
    ]
    columns_to_show = [col for col in preview_columns if col in df.columns]
    st.dataframe(df[columns_to_show].head(200), use_container_width=True)


def main_dashboard() -> None:
    """Apply the Streamlit dashboard layout and components."""
    st.set_page_config(page_title="UrbanPulse NYC", page_icon="🌆", layout="wide")
    st.title("UrbanPulse NYC 311 Dashboard")
    st.caption("Interactive analysis of NYC 311 complaint activity and resolution performance")

    df_raw = load_dataset(limit=15000)
    df_clean = clean_dataset(df_raw)
    filtered_df = filter_dataframe(df_clean)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    render_kpis(filtered_df)
    export_dashboard_csv(filtered_df)

    st.subheader("Visual insights")
    col_a, col_b = st.columns(2)
    with col_a:
        plot_top_complaints(filtered_df)
    with col_b:
        plot_borough_counts(filtered_df)

    col_c, col_d = st.columns(2)
    with col_c:
        plot_status_counts(filtered_df)
    with col_d:
        plot_resolution_by_borough(filtered_df)

    st.write("")
    plot_daily_trend(filtered_df)
    show_data_table(filtered_df)


if __name__ == "__main__":
    if st is None:
        print("Streamlit is not installed. Install it with: pip install streamlit")
    else:
        main_dashboard()