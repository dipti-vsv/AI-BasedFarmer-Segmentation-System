import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# LOAD DATASET
# =====================================

print("Loading Dataset...")

# Read CSV

df = pd.read_csv("data/agriculture_data.csv")

print(df.head())
# HANDLE CATEGORICAL DATA
# =====================================

encoder = LabelEncoder()

categorical_columns = [
    "Crop_Type",
    "Soil_Type",
    "Season"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
# FEATURES FOR CLUSTERING
# =====================================

X_cluster = df[[
   "Farm_Area(acres)", 
   "Water_Usage(cubic meters)", 
   "Crop_Type", 
   "Soil_Type",
   "Season"
]]
# SCALE DATA
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

# =====================================
# KMEANS CLUSTERING
# =====================================

print("Training Clustering Model...")

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

# SAVE CLUSTERING MODEL
# =====================================

joblib.dump(kmeans, "models/clustering_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Clustering Model Saved")

# PREDICTION MODEL
# =====================================

print("Training Prediction Model...")

X = df[[
    "Farm_Area(acres)",
    "Crop_Type", 
    "Soil_Type"
]]

y = df["Water_Usage(cubic meters)"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

score = rf_model.score(X_test, y_test)

print("Model Accuracy:", score)

# Save prediction model
joblib.dump(rf_model, "models/prediction_model.pkl")

print("Prediction Model Saved")

print("Training Completed Successfully")