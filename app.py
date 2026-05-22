# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px

from recommendation import generate_recommendation

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Agriculture Intelligence",
    page_icon="🌾",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================

clustering_model = joblib.load(
    "models/clustering_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

prediction_model = joblib.load(
    "models/prediction_model.pkl"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/agriculture_data.csv")

# ==========================================
# TITLE
# ==========================================

st.title("🌾 AI Agriculture Market Intelligence System")

st.markdown("""
### Smart Farmer Analytics Dashboard

This AI platform helps:
- Segment farmers
- Predict water usage
- Generate AI recommendations
- Analyze agriculture trends
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("🌱 Farmer Information")

farm_area = st.sidebar.slider(
    "Farm Area (Acres)",
    1.0,
    500.0,
    50.0
)

water_usage = st.sidebar.slider(
    "Water Usage (Cubic Meters)",
    1000.0,
    100000.0,
    20000.0
)

fertilizer = st.sidebar.slider(
    "Fertilizer Used (Tons)",
    1.0,
    100.0,
    20.0
)

pesticide = st.sidebar.slider(
    "Pesticide Used (KG)",
    1.0,
    500.0,
    50.0
)

crop_type = st.sidebar.selectbox(
    "Crop Type",
    df["Crop_Type"].unique()
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    df["Soil_Type"].unique()
)

season = st.sidebar.selectbox(
    "Season",
    df["Season"].unique()
)

# ==========================================
# ENCODE INPUT VALUES
# ==========================================

crop_encoded = list(df["Crop_Type"].unique()).index(crop_type)

soil_encoded = list(df["Soil_Type"].unique()).index(soil_type)

season_encoded = list(df["Season"].unique()).index(season)

# ==========================================
# CLUSTER INPUT
# ==========================================

cluster_input = np.array([[
    farm_area,
    water_usage,
    crop_encoded,
    soil_encoded,
    season_encoded
]])

scaled_data = scaler.transform(cluster_input)

# ==========================================
# PREDICT CLUSTER
# ==========================================

cluster = clustering_model.predict(
    scaled_data
)[0]

# ==========================================
# PREDICTION MODEL INPUT
# ==========================================

prediction_input = np.array([[
    farm_area,
    crop_encoded,
    soil_encoded
]])

predicted_water = prediction_model.predict(
    prediction_input
)[0]

# ==========================================
# KPI CARDS
# ==========================================

st.subheader("📊 Agriculture AI Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Farmer Cluster",
        int(cluster)
    )

with col2:
    st.metric(
        "Predicted Water Usage",
        round(predicted_water, 2)
    )

with col3:
    st.metric(
        "Farm Area",
        farm_area
    )

# ==========================================
# AI RECOMMENDATIONS
# ==========================================

st.subheader("🤖 AI Recommendations")

recommendation = generate_recommendation(cluster)

st.success(recommendation)

# ==========================================
# SCATTER PLOT
# ==========================================

st.subheader("📈 Agriculture Analytics")

fig1 = px.scatter(
    df,
    x="Farm_Area(acres)",
    y="Yield(tons)",
    color="Crop_Type",
    size="Water_Usage(cubic meters)",
    hover_data=["Soil_Type", "Season"],
    title="Farm Area vs Crop Yield"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ==========================================
# BAR CHART
# ==========================================

crop_count = df["Crop_Type"].value_counts()

fig2 = px.bar(
    x=crop_count.index,
    y=crop_count.values,
    title="Crop Distribution",
    labels={
        "x": "Crop Type",
        "y": "Count"
    }
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================================
# PIE CHART
# ==========================================

season_count = df["Season"].value_counts()

fig3 = px.pie(
    names=season_count.index,
    values=season_count.values,
    title="Season Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==========================================
# HISTOGRAM
# ==========================================

fig4 = px.histogram(
    df,
    x="Yield(tons)",
    nbins=20,
    title="Crop Yield Distribution"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==========================================
# DATASET PREVIEW
# ==========================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head(20))

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
### 🚀 Developed Using AI & Machine Learning

Technologies Used:
- Python
- Streamlit
- Plotly
- Scikit-Learn
- Machine Learning
""")