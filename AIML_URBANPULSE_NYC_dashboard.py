import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="UrbanPulse NYC", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
    df = pd.read_csv(url + "?$limit=10000")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    for col in ["borough", "descriptor", "status", "location_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    if "closed_date" in df.columns:
        df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

    if {"created_date", "closed_date"}.issubset(df.columns):
        df["resolution_hours"] = (
            df["closed_date"] - df["created_date"]
        ).dt.total_seconds() / 3600
        df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan

    if "complaint_type" in df.columns:
        df["complaint_type"] = df["complaint_type"].fillna("Unknown")

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    selected_boroughs = st.session_state.get("selected_boroughs", [])
    selected_statuses = st.session_state.get("selected_statuses", [])
    selected_types = st.session_state.get("selected_types", [])

    filtered = df.copy()

    if selected_boroughs:
        filtered = filtered[filtered["borough"].isin(selected_boroughs)]
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_types:
        filtered = filtered[filtered["complaint_type"].isin(selected_types)]

    return filtered


st.title("UrbanPulse NYC 311 Dashboard")
st.caption("Interactive insight dashboard for NYC 311 complaint data")

raw_df = load_data()
clean_df = clean_data(raw_df)

if "selected_boroughs" not in st.session_state:
    st.session_state.selected_boroughs = sorted(clean_df["borough"].dropna().unique().tolist())
if "selected_statuses" not in st.session_state:
    st.session_state.selected_statuses = sorted(clean_df["status"].dropna().unique().tolist())
if "selected_types" not in st.session_state:
    top_types = clean_df["complaint_type"].value_counts().head(10).index.tolist()
    st.session_state.selected_types = top_types

with st.sidebar:
    st.header("Filters")
    borough_options = sorted(clean_df["borough"].dropna().unique().tolist())
    status_options = sorted(clean_df["status"].dropna().unique().tolist())
    type_options = sorted(clean_df["complaint_type"].dropna().unique().tolist())

    st.session_state.selected_boroughs = st.multiselect(
        "Borough",
        borough_options,
        default=st.session_state.selected_boroughs,
    )
    st.session_state.selected_statuses = st.multiselect(
        "Status",
        status_options,
        default=st.session_state.selected_statuses,
    )
    st.session_state.selected_types = st.multiselect(
        "Complaint Type",
        type_options,
        default=st.session_state.selected_types,
    )

filtered_df = filter_data(clean_df)

if filtered_df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# KPI cards
resolved = filtered_df[filtered_df["resolution_hours"].notna()]
closed_count = int((filtered_df["status"] == "Closed").sum())
avg_resolution = resolved["resolution_hours"].mean() if not resolved.empty else 0
max_resolution = resolved["resolution_hours"].max() if not resolved.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total complaints", f"{len(filtered_df):,}")
col2.metric("Closed complaints", f"{closed_count:,}")
col3.metric("Avg resolution", f"{avg_resolution:.1f} hrs")
col4.metric("Max resolution", f"{max_resolution:.1f} hrs")

# Top complaint types
complaint_counts = (
    filtered_df["complaint_type"].value_counts().head(10).reset_index()
)
complaint_counts.columns = ["Complaint Type", "Count"]

# Borough distribution
borough_counts = filtered_df["borough"].value_counts().reset_index()
borough_counts.columns = ["Borough", "Count"]

# Status distribution
status_counts = filtered_df["status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]

# Daily trend
if "created_date" in filtered_df.columns:
    daily_counts = (
        filtered_df.assign(date=filtered_df["created_date"].dt.date)
        .groupby("date")
        .size()
        .reset_index(name="Complaints")
    )
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

left, right = st.columns(2)
left.subheader("Top complaint types")
left.plotly_chart(
    px.bar(
        complaint_counts,
        x="Count",
        y="Complaint Type",
        orientation="h",
        color="Complaint Type",
        title="Top 10 Complaint Types",
    ),
    use_container_width=True,
)

right.subheader("Complaints by borough")
right.plotly_chart(
    px.bar(
        borough_counts,
        x="Borough",
        y="Count",
        color="Borough",
        title="Complaints by Borough",
    ),
    use_container_width=True,
)

st.subheader("Status breakdown")
st.plotly_chart(
    px.pie(status_counts, values="Count", names="Status", title="Complaint Status Distribution"),
    use_container_width=True,
)

if "created_date" in filtered_df.columns and not daily_counts.empty:
    st.subheader("Daily complaint trend")
    st.plotly_chart(
        px.line(
            daily_counts,
            x="date",
            y="Complaints",
            title="Daily 311 Complaint Trend",
        ),
        use_container_width=True,
    )

st.subheader("Resolution time by borough")
if "resolution_hours" in filtered_df.columns:
    borough_resolution = (
        filtered_df[filtered_df["resolution_hours"].notna()]
        .groupby("borough")["resolution_hours"]
        .mean()
        .reset_index(name="avg_resolution_hours")
        .sort_values("avg_resolution_hours", ascending=False)
    )
    st.plotly_chart(
        px.bar(
            borough_resolution,
            x="borough",
            y="avg_resolution_hours",
            color="borough",
            title="Average Resolution Time by Borough",
        ),
        use_container_width=True,
    )

st.subheader("Data preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

st.text(f"Loaded rows: {len(filtered_df):,}")
