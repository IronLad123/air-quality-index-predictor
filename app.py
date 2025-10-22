import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide"
)

# Check if model files exist
def check_model_files():
    required_files = ['stacking_ensemble.pkl', 'feature_scaler.pkl', 'numeric_columns.pkl', 'le_city.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

# Load models with proper error handling
@st.cache_resource
def load_models():
    missing_files = check_model_files()
    
    if missing_files:
        st.warning(f"⚠️ Missing model files: {', '.join(missing_files)}")
        st.info("🔧 Using demo mode with simulated predictions")
        return None, None, None, None, False
    
    try:
        # Try to load actual models
        model = joblib.load('stacking_ensemble.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        numeric_cols = joblib.load('numeric_columns.pkl')
        le_city = joblib.load('le_city.pkl')
        
        st.success("✅ Real models loaded successfully!")
        return model, scaler, numeric_cols, le_city, True
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("🔧 Falling back to demo mode")
        return None, None, None, None, False

# AQI Categories
AQI_CATEGORIES = {
    'Good': {'range': '0-50', 'color': '#00e400', 'level': 1, 'emoji': '😊'},
    'Moderate': {'range': '51-100', 'color': '#ffff00', 'level': 2, 'emoji': '😐'},
    'Poor': {'range': '101-200', 'color': '#ff7e00', 'level': 3, 'emoji': '😷'},
    'Unhealthy': {'range': '201-300', 'color': '#ff0000', 'level': 4, 'emoji': '🤢'},
    'Very Unhealthy': {'range': '301-400', 'color': '#8f3f97', 'level': 5, 'emoji': '😨'},
    'Hazardous': {'range': '401-500', 'color': '#7e0023', 'level': 6, 'emoji': '💀'}
}

# Demo prediction function
def demo_prediction(params):
    """Simulate prediction for demo purposes"""
    # Simple weighted scoring based on pollutant levels
    weighted_score = (
        params['pm25'] * 0.25 +
        params['pm10'] * 0.15 +
        params['no2'] * 0.12 +
        params['so2'] * 0.10 +
        params['o3'] * 0.10 +
        params['co'] * 0.08 +
        params['nox'] * 0.07 +
        (params['benzene'] + params['toluene'] + params['xylene']) * 0.13
    )
    
    if weighted_score <= 30:
        return 'Good'
    elif weighted_score <= 60:
        return 'Moderate'
    elif weighted_score <= 120:
        return 'Poor'
    elif weighted_score <= 200:
        return 'Unhealthy'
    elif weighted_score <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

# Real prediction function
def real_prediction(model, scaler, numeric_cols, le_city, input_df):
    """Make prediction using actual loaded models"""
    try:
        # Encode city
        input_df['City'] = le_city.transform(input_df['City'])
        
        # Scale features
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Make prediction
        prediction = model.predict(input_df)
        return prediction[0], True
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, False

# Create input dataframe
def create_input_data(city, params, le_city, use_real_model):
    """Create input dataframe with proper features"""
    # Calculate ratios
    pm_ratio = params['pm25'] / (params['pm10'] + 1e-6)
    no_ratio = params['nox'] / (params['no2'] + 1e-6)
    day_of_week = date.today().weekday()
    
    data = {
        'City': [city],
        'PM2_5': [params['pm25']],
        'PM10': [params['pm10']],
        'NO': [params['no']],
        'NO2': [params['no2']],
        'NOx': [params['nox']],
        'NH3': [params['nh3']],
        'CO': [params['co']],
        'SO2': [params['so2']],
        'O3': [params['o3']],
        'Benzene': [params['benzene']],
        'Toluene': [params['toluene']],
        'Xylene': [params['xylene']],
        'PM_ratio': [pm_ratio],
        'NO_ratio': [no_ratio],
        'Day_of_week': [day_of_week],
    }
    
    # Add rolling averages (using current values)
    for col in ['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
        data[f'{col}_3d_avg'] = [data[col][0]]
    
    return pd.DataFrame(data)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main App
st.title("🌫️ Air Quality Index Predictor")

# Load models
with st.spinner("Loading models..."):
    model, scaler, numeric_cols, le_city, models_loaded = load_models()

# Show model status
if models_loaded:
    st.success("🔧 **Using Real Trained Models**")
    st.info(f"📊 Model Features: {len(numeric_cols) if numeric_cols else 'Unknown'}")
    if le_city:
        st.info(f"🏙️ Available Cities: {', '.join(le_city.classes_[:5])}..." if len(le_city.classes_) > 5 else f"🏙️ Available Cities: {', '.join(le_city.classes_)}")
else:
    st.warning("🎮 **Demo Mode Active** - Using simulated predictions")
    st.info("To use real models, make sure these files are in your directory: stacking_ensemble.pkl, feature_scaler.pkl, numeric_columns.pkl, le_city.pkl")

# Sidebar
st.sidebar.header("Settings")

# City selection
if le_city and models_loaded:
    city_options = le_city.classes_
else:
    city_options = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Pune", "Ahmedabad"]

city = st.sidebar.selectbox("Select City", city_options)

# Main input form
st.header("Air Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Particulate Matter")
    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0)
    
    st.subheader("Nitrogen Compounds")
    no = st.slider("NO (µg/m³)", 0.0, 100.0, 10.0)
    no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 20.0)
    nox = st.slider("NOx (µg/m³)", 0.0, 300.0, 30.0)

with col2:
    st.subheader("Other Pollutants")
    so2 = st.slider("SO2 (µg/m³)", 0.0, 300.0, 10.0)
    o3 = st.slider("O3 (µg/m³)", 0.0, 400.0, 20.0)
    co = st.slider("CO (µg/m³)", 0.0, 50.0, 5.0)
    nh3 = st.slider("NH3 (µg/m³)", 0.0, 200.0, 10.0)
    
    st.subheader("Chemical Compounds")
    benzene = st.slider("Benzene (µg/m³)", 0.0, 50.0, 1.0)
    toluene = st.slider("Toluene (µg/m³)", 0.0, 100.0, 5.0)
    xylene = st.slider("Xylene (µg/m³)", 0.0, 100.0, 3.0)

# Current parameters visualization
st.header("Current Parameters Overview")

# Create gauge charts for key parameters
col1, col2, col3 = st.columns(3)

with col1:
    fig_pm25 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pm25,
        title = {'text': "PM2.5"},
        gauge = {
            'axis': {'range': [None, 300]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 300], 'color': "red"}
            ]
        }
    ))
    fig_pm25.update_layout(height=250)
    st.plotly_chart(fig_pm25, use_container_width=True)

with col2:
    fig_no2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = no2,
        title = {'text': "NO2"},
        gauge = {
            'axis': {'range': [None, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 80], 'color': "yellow"},
                {'range': [80, 200], 'color': "red"}
            ]
        }
    ))
    fig_no2.update_layout(height=250)
    st.plotly_chart(fig_no2, use_container_width=True)

with col3:
    fig_o3 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = o3,
        title = {'text': "O3"},
        gauge = {
            'axis': {'range': [None, 400]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 400], 'color': "red"}
            ]
        }
    ))
    fig_o3.update_layout(height=250)
    st.plotly_chart(fig_o3, use_container_width=True)

# Prediction button
if st.button("🎯 Predict AQI", type="primary", use_container_width=True):
    with st.spinner('Analyzing air quality data...'):
        # Collect parameters
        params = {
            'pm25': pm25, 'pm10': pm10, 'no': no, 'no2': no2, 'nox': nox,
            'nh3': nh3, 'co': co, 'so2': so2, 'o3': o3,
            'benzene': benzene, 'toluene': toluene, 'xylene': xylene
        }
        
        # Create input data
        input_df = create_input_data(city, params, le_city, models_loaded)
        
        # Make prediction
        if models_loaded and model is not None:
            # Use real model
            predicted_category, success = real_prediction(model, scaler, numeric_cols, le_city, input_df)
            prediction_source = "Real Model"
        else:
            # Use demo prediction
            predicted_category = demo_prediction(params)
            prediction_source = "Demo Simulation"
            success = True
        
        if success:
            category_info = AQI_CATEGORIES.get(predicted_category, AQI_CATEGORIES['Moderate'])
            
            # Store prediction
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'city': city,
                'category': predicted_category,
                'level': category_info['level'],
                'source': prediction_source,
                'parameters': params
            })
            
            # Display result
            st.markdown(f"""
            <div style="padding: 2.5rem; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; text-align: center; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
                <h2>🎯 Prediction Result</h2>
                <div style="font-size: 3rem; margin: 1rem 0;">
                    {category_info['emoji']}
                </div>
                <div style="font-size: 2.5rem; margin: 1rem 0; padding: 1rem; 
                          background: {category_info['color']}; border-radius: 15px;">
                    {predicted_category}
                </div>
                <p style="font-size: 1.3rem;">AQI Range: {category_info['range']}</p>
                <p style="font-size: 1rem;">Source: {prediction_source}</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">
                    📍 {city} | 🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health recommendations
            st.subheader("💡 Health Recommendations")
            recommendations = {
                'Good': "✅ Excellent air quality. Perfect for outdoor activities and exercise.",
                'Moderate': "⚠️ Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
                'Poor': "🔶 Members of sensitive groups may experience health effects. General public is less likely to be affected.",
                'Unhealthy': "🔴 Everyone may begin to experience health effects. Avoid outdoor activities.",
                'Very Unhealthy': "💀 Health alert: everyone may experience more serious health effects.",
                'Hazardous': "☠️ Health warning of emergency conditions. The entire population is more likely to be affected."
            }
            st.info(recommendations.get(predicted_category, "Please take necessary precautions."))

# Prediction History
if st.session_state.prediction_history:
    st.header("📈 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show latest predictions
    st.dataframe(display_df[['timestamp', 'city', 'category', 'source']].tail(10), use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = history_df['category'].value_counts()
        fig_pie = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            title="Prediction Category Distribution",
            color=category_counts.index,
            color_discrete_map={cat: AQI_CATEGORIES[cat]['color'] for cat in AQI_CATEGORIES}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Trend over time
        if len(history_df) > 1:
            fig_trend = px.line(
                history_df, 
                x='timestamp', 
                y='level',
                color='city',
                title="AQI Level Trend Over Time",
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# AQI Reference
st.sidebar.header("AQI Categories Reference")
for category, info in AQI_CATEGORIES.items():
    st.sidebar.markdown(
        f"<div style='background: {info['color']}; padding: 8px; border-radius: 5px; "
        f"color: {'black' if category in ['Moderate'] else 'white'}; margin: 5px 0; text-align: center;'>"
        f"{category}: {info['range']}</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("**Air Quality Index Prediction System** | Real-time monitoring and predictive analytics")

# Debug information
with st.sidebar.expander("🔧 Debug Info"):
    st.write("Model Status:", "Loaded" if models_loaded else "Demo Mode")
    st.write("Missing Files:", check_model_files())
    st.write("Prediction History Count:", len(st.session_state.prediction_history))
