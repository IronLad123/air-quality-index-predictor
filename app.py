import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Air Quality Index Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .aqi-good { background-color: #00e400; color: white; padding: 5px; border-radius: 5px; }
    .aqi-moderate { background-color: #ffff00; color: black; padding: 5px; border-radius: 5px; }
    .aqi-poor { background-color: #ff7e00; color: white; padding: 5px; border-radius: 5px; }
    .aqi-unhealthy { background-color: #ff0000; color: white; padding: 5px; border-radius: 5px; }
    .aqi-very-unhealthy { background-color: #8f3f97; color: white; padding: 5px; border-radius: 5px; }
    .aqi-hazardous { background-color: #7e0023; color: white; padding: 5px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Load saved objects
@st.cache_resource
def load_models():
    model = joblib.load('stacking_ensemble.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    numeric_cols = joblib.load('numeric_columns.pkl')
    le_city = joblib.load('le_city.pkl')
    return model, scaler, numeric_cols, le_city

try:
    model, scaler, numeric_cols, le_city = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# AQI category mapping
aqi_categories = {
    'Good': {'range': '0-50', 'color': 'aqi-good'},
    'Moderate': {'range': '51-100', 'color': 'aqi-moderate'},
    'Poor': {'range': '101-200', 'color': 'aqi-poor'},
    'Unhealthy': {'range': '201-300', 'color': 'aqi-unhealthy'},
    'Very Unhealthy': {'range': '301-400', 'color': 'aqi-very-unhealthy'},
    'Hazardous': {'range': '401-500', 'color': 'aqi-hazardous'}
}

# Main title
st.markdown('<h1 class="main-header">🌫️ Air Quality Index Predictor</h1>', unsafe_allow_html=True)

# Sidebar for additional features
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date input for temporal context
    selected_date = st.date_input("Select Date", value=date.today())
    day_of_week = selected_date.weekday()  # Monday=0, Sunday=6
    
    # AQI Information
    st.header("📊 AQI Categories")
    for category, info in aqi_categories.items():
        st.markdown(f"<div class='{info['color']}'>{category}: {info['range']}</div>", unsafe_allow_html=True)
    
    st.header("💡 Tips")
    st.info("""
    - Lower PM2.5/PM10 values indicate better air quality
    - Monitor NO2 and SO2 levels for industrial pollution
    - Ozone (O3) is important for urban air quality
    - Benzene, Toluene, Xylene indicate chemical pollution
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🏙️ Location & Basic Parameters")
    
    # City selection with emoji
    city = st.selectbox("Select City", le_city.classes_, index=0)
    
    # Create tabs for different parameter groups
    tab1, tab2, tab3 = st.tabs(["Particulate Matter", "Gaseous Pollutants", "Chemical Compounds"])
    
    with tab1:
        st.subheader("🌫️ Particulate Matter")
        pm25 = st.slider("PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=50.0, 
                        help="Fine inhalable particles with diameter ≤ 2.5 micrometers")
        pm10 = st.slider("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0,
                        help="Inhalable particles with diameter ≤ 10 micrometers")
        
        # PM ratio visualization
        pm_ratio = pm25 / (pm10 + 1e-6)
        st.metric("PM2.5/PM10 Ratio", f"{pm_ratio:.3f}")
    
    with tab2:
        st.subheader("💨 Gaseous Pollutants")
        col1a, col1b = st.columns(2)
        with col1a:
            no = st.slider("NO (µg/m³)", min_value=0.0, max_value=100.0, value=10.0)
            no2 = st.slider("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0)
            nox = st.slider("NOx (µg/m³)", min_value=0.0, max_value=300.0, value=30.0)
        with col1b:
            so2 = st.slider("SO2 (µg/m³)", min_value=0.0, max_value=300.0, value=10.0)
            o3 = st.slider("O3 (µg/m³)", min_value=0.0, max_value=400.0, value=20.0)
            co = st.slider("CO (µg/m³)", min_value=0.0, max_value=50.0, value=5.0)
        
        nh3 = st.slider("NH3 (µg/m³)", min_value=0.0, max_value=200.0, value=10.0)
        
        # NO ratio visualization
        no_ratio = nox / (no2 + 1e-6)
        st.metric("NOx/NO2 Ratio", f"{no_ratio:.3f}")
    
    with tab3:
        st.subheader("🧪 Chemical Compounds")
        benzene = st.slider("Benzene (µg/m³)", min_value=0.0, max_value=50.0, value=1.0,
                          help="Volatile organic compound from industrial emissions")
        toluene = st.slider("Toluene (µg/m³)", min_value=0.0, max_value=100.0, value=5.0,
                          help="Industrial solvent and gasoline component")
        xylene = st.slider("Xylene (µg/m³)", min_value=0.0, max_value=100.0, value=3.0,
                         help="Industrial solvent and paint component")

with col2:
    st.header("📈 Real-time Analysis & Prediction")
    
    # Current parameters display
    st.subheader("📋 Current Parameters Summary")
    
    # Create metrics cards
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("PM2.5", f"{pm25} µg/m³")
        st.metric("PM10", f"{pm10} µg/m³")
        st.metric("NO2", f"{no2} µg/m³")
        st.metric("O3", f"{o3} µg/m³")
    
    with metrics_col2:
        st.metric("SO2", f"{so2} µg/m³")
        st.metric("CO", f"{co} µg/m³")
        st.metric("Benzene", f"{benzene} µg/m³")
        st.metric("City", city)
    
    # Pollution radar chart
    st.subheader("📊 Pollution Profile Radar")
    
    # Normalize values for radar chart
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
    values = [pm25/300, pm10/500, no2/200, so2/300, o3/400, co/50]  # Normalized
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=pollutants,
        fill='toself',
        line=dict(color='#1f77b4')
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Prediction button
    if st.button("🚀 Predict AQI Category", use_container_width=True):
        with st.spinner('Analyzing air quality data...'):
            # Prepare input data frame (same as original)
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
                'PM2_5_3d_avg': [pm25],
                'PM10_3d_avg': [pm10],
                'NO_3d_avg': [no],
                'NO2_3d_avg': [no2],
                'NOx_3d_avg': [nox],
                'NH3_3d_avg': [nh3],
                'CO_3d_avg': [co],
                'SO2_3d_avg': [so2],
                'O3_3d_avg': [o3],
                'Benzene_3d_avg': [benzene],
                'Toluene_3d_avg': [toluene],
                'Xylene_3d_avg': [xylene]
            })
            
            # Encode City
            input_df['City'] = le_city.transform(input_df['City'])
            
            # Scale numeric features
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # Predict AQI Bucket
            prediction = model.predict(input_df)
            predicted_category = prediction[0]
            
            # Display prediction with styling
            category_info = aqi_categories.get(predicted_category, {'range': 'N/A', 'color': 'aqi-moderate'})
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted AQI Category</h2>
                <h1 class="{category_info['color']}" style="font-size: 3rem; margin: 1rem 0;">{predicted_category}</h1>
                <p style="font-size: 1.2rem;">AQI Range: {category_info['range']}</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">📍 City: {city} | 📅 Date: {selected_date.strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health recommendations based on AQI category
            st.subheader("💡 Health Recommendations")
            recommendations = {
                'Good': "Air quality is satisfactory. Enjoy outdoor activities!",
                'Moderate': "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
                'Poor': "Members of sensitive groups may experience health effects. General public is less likely to be affected.",
                'Unhealthy': "Everyone may begin to experience health effects. Avoid outdoor activities.",
                'Very Unhealthy': "Health alert: everyone may experience more serious health effects.",
                'Hazardous': "Health warning of emergency conditions. The entire population is more likely to be affected."
            }
            
            st.info(recommendations.get(predicted_category, "Please take necessary precautions."))

# Footer
st.markdown("---")
st.markdown(
    "🔬 *This AQI prediction tool uses machine learning to analyze air quality parameters and provide health recommendations.*"
)
