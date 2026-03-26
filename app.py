import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Client KPI Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Client KPI Dashboard")
st.caption("Interactive KPI and chart dashboard")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Ensure numeric fields are numeric
    numeric_cols = ["Engagement Score", "Meetings Count", "Revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean text fields
    text_cols = ["Industry", "Region", "Retention Status", "Client ID"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

# Change this file name if needed
DATA_FILE = "client_data.csv"

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"File not found: {DATA_FILE}")
    st.info("Upload your CSV in the same folder and name it 'client_data.csv'")
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

industry_filter = st.sidebar.multiselect(
    "Select Industry",
    options=sorted(df["Industry"].dropna().unique()),
    default=sorted(df["Industry"].dropna().unique())
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique())
)

retention_filter = st.sidebar.multiselect(
    "Select Retention Status",
    options=sorted(df["Retention Status"].dropna().unique()),
    default=sorted(df["Retention Status"].dropna().unique())
)

filtered_df = df[
    df["Industry"].isin(industry_filter) &
    df["Region"].isin(region_filter) &
    df["Retention Status"].isin(retention_filter)
].copy()

# ----------------------------
# KPI calculations
# ----------------------------
total_clients = filtered_df["Client ID"].nunique()
total_revenue = filtered_df["Revenue"].sum()
avg_engagement = filtered_df["Engagement Score"].mean()
avg_meetings = filtered_df["Meetings Count"].mean()

retained_count = filtered_df[
    filtered_df["Retention Status"].str.lower() == "retained"
].shape[0]

retention_rate = (
    (retained_count / len(filtered_df)) * 100
    if len(filtered_df) > 0 else 0
)

# ----------------------------
# KPI section
# ----------------------------
st.subheader("Key KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Clients", f"{total_clients:,}")
col2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col3.metric("Avg Engagement", f"{avg_engagement:.2f}")
col4.metric("Avg Meetings", f"{avg_meetings:.2f}")
col5.metric("Retention Rate", f"{retention_rate:.1f}%")

st.markdown("---")

# ----------------------------
# Charts
# ----------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    revenue_by_industry = (
        filtered_df.groupby("Industry", as_index=False)["Revenue"]
        .sum()
        .sort_values(by="Revenue", ascending=False)
    )

    fig_revenue_industry = px.bar(
        revenue_by_industry,
        x="Industry",
        y="Revenue",
        title="Revenue by Industry",
        text_auto=True
    )
    st.plotly_chart(fig_revenue_industry, use_container_width=True)

with chart_col2:
    revenue_by_region = (
        filtered_df.groupby("Region", as_index=False)["Revenue"]
        .sum()
        .sort_values(by="Revenue", ascending=False)
    )

    fig_revenue_region = px.pie(
        revenue_by_region,
        names="Region",
        values="Revenue",
        title="Revenue Distribution by Region"
    )
    st.plotly_chart(fig_revenue_region, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    retention_counts = (
        filtered_df.groupby("Retention Status", as_index=False)["Client ID"]
        .count()
        .rename(columns={"Client ID": "Count"})
    )

    fig_retention = px.bar(
        retention_counts,
        x="Retention Status",
        y="Count",
        title="Retention Status Count",
        text_auto=True
    )
    st.plotly_chart(fig_retention, use_container_width=True)

with chart_col4:
    fig_scatter = px.scatter(
        filtered_df,
        x="Meetings Count",
        y="Revenue",
        color="Retention Status",
        size="Engagement Score",
        hover_data=["Client ID", "Industry", "Region"],
        title="Meetings vs Revenue"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------
# Additional chart
# ----------------------------
engagement_by_region = (
    filtered_df.groupby("Region", as_index=False)["Engagement Score"]
    .mean()
    .sort_values(by="Engagement Score", ascending=False)
)

fig_engagement_region = px.line(
    engagement_by_region,
    x="Region",
    y="Engagement Score",
    markers=True,
    title="Average Engagement Score by Region"
)
st.plotly_chart(fig_engagement_region, use_container_width=True)

# ----------------------------
# Data preview
# ----------------------------
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df, use_container_width=True)
