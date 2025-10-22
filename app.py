import streamlit as st
import joblib
import pandas as pd

# Load saved objects
model = joblib.load('stacking_ensemble.pkl')
scaler = joblib.load('feature_scaler.pkl')
numeric_cols = joblib.load('numeric_columns.pkl')
le_city = joblib.load('le_city.pkl')

st.title("Air Quality Index (AQI) Prediction")

# Input form for user
city = st.selectbox("Select City", le_city.classes_)
pm25 = st.number_input("PM2.5", min_value=0.0, max_value=300.0, value=50.0)
pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=100.0)
no = st.number_input("NO", min_value=0.0, max_value=100.0, value=10.0)
no2 = st.number_input("NO2", min_value=0.0, max_value=200.0, value=20.0)
nox = st.number_input("NOx", min_value=0.0, max_value=300.0, value=30.0)
nh3 = st.number_input("NH3", min_value=0.0, max_value=200.0, value=10.0)
co = st.number_input("CO", min_value=0.0, max_value=50.0, value=5.0)
so2 = st.number_input("SO2", min_value=0.0, max_value=300.0, value=10.0)
o3 = st.number_input("O3", min_value=0.0, max_value=400.0, value=20.0)
benzene = st.number_input("Benzene", min_value=0.0, max_value=50.0, value=1.0)
toluene = st.number_input("Toluene", min_value=0.0, max_value=100.0, value=5.0)
xylene = st.number_input("Xylene", min_value=0.0, max_value=100.0, value=3.0)

# Calculate engineered features
pm_ratio = pm25 / (pm10 + 1e-6)
no_ratio = nox / (no2 + 1e-6)
day_of_week = 3  # Example: Wednesday; ideally should ask user or fetch current day

# For demo, set rolling averages = input values
pm25_3d_avg = pm25
pm10_3d_avg = pm10
no_3d_avg = no
no2_3d_avg = no2
nox_3d_avg = nox
nh3_3d_avg = nh3
co_3d_avg = co
so2_3d_avg = so2
o3_3d_avg = o3
benzene_3d_avg = benzene
toluene_3d_avg = toluene
xylene_3d_avg = xylene

# Prepare input data frame
input_df = pd.DataFrame({
    'City': [city],
    'PM2_5': [pm25],
    'PM10': [pm10],
    'NO': [no],
    'NO2': [no2],
    'NOx': [nox],
    'NH3': [nh3],
    'CO': [co],
    'SO2': [so2],
    'O3': [o3],
    'Benzene': [benzene],
    'Toluene': [toluene],
    'Xylene': [xylene],
    'PM_ratio': [pm_ratio],
    'NO_ratio': [no_ratio],
    'Day_of_week': [day_of_week],
    'PM2_5_3d_avg': [pm25_3d_avg],
    'PM10_3d_avg': [pm10_3d_avg],
    'NO_3d_avg': [no_3d_avg],
    'NO2_3d_avg': [no2_3d_avg],
    'NOx_3d_avg': [nox_3d_avg],
    'NH3_3d_avg': [nh3_3d_avg],
    'CO_3d_avg': [co_3d_avg],
    'SO2_3d_avg': [so2_3d_avg],
    'O3_3d_avg': [o3_3d_avg],
    'Benzene_3d_avg': [benzene_3d_avg],
    'Toluene_3d_avg': [toluene_3d_avg],
    'Xylene_3d_avg': [xylene_3d_avg]
})

# Encode City
input_df['City'] = le_city.transform(input_df['City'])

# Scale numeric features
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict AQI Bucket
if st.button("Predict AQI"):
    prediction = model.predict(input_df)
    st.success(f"Predicted AQI Bucket: {prediction[0]}")
