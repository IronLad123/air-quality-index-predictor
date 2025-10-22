import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
import sys

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
    .error-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .debug-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        font-family: monospace;
        font-size: 12px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Safe AQI category mapping
AQI_CATEGORIES = {
    'Good': {'range': '0-50', 'color': '#00e400', 'level': 1},
    'Satisfactory': {'range': '51-100', 'color': '#ffff00', 'level': 2},
    'Moderate': {'range': '51-100', 'color': '#ffff00', 'level': 2},
    'Poor': {'range': '101-200', 'color': '#ff7e00', 'level': 3},
    'Unhealthy': {'range': '201-300', 'color': '#ff0000', 'level': 4},
    'Very Poor': {'range': '201-300', 'color': '#ff0000', 'level': 4},
    'Severe': {'range': '301-400', 'color': '#8f3f97', 'level': 5},
    'Very Unhealthy': {'range': '301-400', 'color': '#8f3f97', 'level': 5},
    'Hazardous': {'range': '401-500', 'color': '#7e0023', 'level': 6},
    'Critical': {'range': '401-500', 'color': '#7e0023', 'level': 6}
}

def safe_load_models():
    """Safely load models with comprehensive error handling"""
    models_loaded = False
    model, scaler, numeric_cols, le_city = None, None, None, None
    
    try:
        model = joblib.load('stacking_ensemble.pkl')
        st.success("✅ Model loaded successfully")
        models_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
    
    try:
        scaler = joblib.load('feature_scaler.pkl')
        st.success("✅ Scaler loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}")
    
    try:
        numeric_cols = joblib.load('numeric_columns.pkl')
        st.success("✅ Numeric columns loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load numeric columns: {e}")
    
    try:
        le_city = joblib.load('le_city.pkl')
        st.success("✅ City encoder loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load city encoder: {e}")
    
    return model, scaler, numeric_cols, le_city, models_loaded

def safe_get_category(prediction):
    """Safely extract category from prediction with multiple fallbacks"""
    try:
        # Handle different prediction types
        if hasattr(prediction, '__len__') and len(prediction) > 0:
            category = prediction[0]
        else:
            category = prediction
            
        # Convert to string and clean
        category_str = str(category).strip()
        
        # Direct match
        if category_str in AQI_CATEGORIES:
            return category_str, AQI_CATEGORIES[category_str]
        
        # Case-insensitive match
        for key in AQI_CATEGORIES.keys():
            if key.lower() == category_str.lower():
                return key, AQI_CATEGORIES[key]
        
        # Partial match
        for key in AQI_CATEGORIES.keys():
            if category_str.lower() in key.lower() or key.lower() in category_str.lower():
                return key, AQI_CATEGORIES[key]
        
        # Numeric AQI value
        try:
            aqi_value = float(category_str)
            if aqi_value <= 50:
                return 'Good', AQI_CATEGORIES['Good']
            elif aqi_value <= 100:
                return 'Moderate', AQI_CATEGORIES['Moderate']
            elif aqi_value <= 200:
                return 'Poor', AQI_CATEGORIES['Poor']
            elif aqi_value <= 300:
                return 'Unhealthy', AQI_CATEGORIES['Unhealthy']
            elif aqi_value <= 400:
                return 'Very Unhealthy', AQI_CATEGORIES['Very Unhealthy']
            else:
                return 'Hazardous', AQI_CATEGORIES['Hazardous']
        except ValueError:
            pass
            
        # Final fallback
        return 'Moderate', AQI_CATEGORIES['Moderate']
        
    except Exception as e:
        st.error(f"Error processing prediction category: {e}")
        return 'Moderate', AQI_CATEGORIES['Moderate']

def create_input_dataframe(city, params, le_city):
    """Safely create input dataframe with all required features"""
    try:
        # Calculate ratios safely
        pm_ratio = params['pm25'] / (params['pm10'] + 1e-6)
        no_ratio = params['nox'] / (params['no2'] + 1e-6)
        day_of_week = date.today().weekday()
        
        # Base features
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
        
        # Add rolling averages (using current values as fallback)
        for col in ['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
            data[f'{col}_3d_avg'] = [data[col][0]]  # Use current value as average
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error creating input data: {e}")
        return None

def safe_predict(model, input_df, scaler, numeric_cols, le_city):
    """Safely make prediction with comprehensive error handling"""
    try:
        # Encode city
        if 'City' in input_df.columns and le_city is not None:
            try:
                input_df['City'] = le_city.transform(input_df['City'])
            except ValueError:
                # If city not in encoder, use first available city
                if len(le_city.classes_) > 0:
                    input_df['City'] = le_city.transform([le_city.classes_[0]])
        
        # Scale features
        if scaler is not None and numeric_cols is not None:
            try:
                # Ensure all numeric columns exist
                available_cols = [col for col in numeric_cols if col in input_df.columns]
                input_df[available_cols] = scaler.transform(input_df[available_cols])
            except Exception as e:
                st.warning(f"Feature scaling warning: {e}")
        
        # Make prediction
        prediction = model.predict(input_df)
        return prediction, True
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, False

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Main app
st.title("🌫️ Air Quality Index Predictor")

# Load models
with st.spinner("Loading AI models..."):
    model, scaler, numeric_cols, le_city, models_loaded = safe_load_models()

# Debug toggle
st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Sidebar
st.sidebar.header("Settings")
if le_city is not None:
    city_options = le_city.classes_
else:
    city_options = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"]
    st.sidebar.warning("Using default cities - city encoder not loaded")

city = st.sidebar.selectbox("Select City", city_options)

# Main input form
st.header("Air Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=50.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0)
    no = st.number_input("NO (µg/m³)", min_value=0.0, max_value=100.0, value=10.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0)
    nox = st.number_input("NOx (µg/m³)", min_value=0.0, max_value=300.0, value=30.0)

with col2:
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, max_value=200.0, value=10.0)
    co = st.number_input("CO (µg/m³)", min_value=0.0, max_value=50.0, value=5.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=300.0, value=10.0)
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=400.0, value=20.0)
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, max_value=50.0, value=1.0)
    toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, max_value=100.0, value=5.0)
    xylene = st.number_input("Xylene (µg/m³)", min_value=0.0, max_value=100.0, value=3.0)

# Prediction button
if st.button("Predict AQI", type="primary"):
    if not models_loaded:
        st.error("Cannot make prediction - models not loaded properly")
    else:
        with st.spinner("Analyzing air quality data..."):
            try:
                # Collect parameters
                params = {
                    'pm25': pm25, 'pm10': pm10, 'no': no, 'no2': no2, 'nox': nox,
                    'nh3': nh3, 'co': co, 'so2': so2, 'o3': o3,
                    'benzene': benzene, 'toluene': toluene, 'xylene': xylene
                }
                
                # Create input data
                input_df = create_input_dataframe(city, params, le_city)
                
                if input_df is None:
                    st.error("Failed to create input data")
                else:
                    # Show debug info
                    if st.session_state.debug_mode:
                        with st.expander("Debug Information", expanded=True):
                            st.write("Input DataFrame:")
                            st.dataframe(input_df)
                            st.write("DataFrame shape:", input_df.shape)
                            st.write("Available columns:", list(input_df.columns))
                    
                    # Make prediction
                    prediction, success = safe_predict(model, input_df, scaler, numeric_cols, le_city)
                    
                    if success:
                        # Safely get category
                        category_name, category_info = safe_get_category(prediction)
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'city': city,
                            'category': category_name,
                            'level': category_info['level'],
                            'raw_prediction': str(prediction)
                        })
                        
                        # Display result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted AQI Category</h2>
                            <div style="font-size: 2.5rem; margin: 1rem 0; padding: 1rem; 
                                      background: {category_info['color']}; border-radius: 10px;
                                      color: {'black' if category_name in ['Moderate', 'Satisfactory'] else 'white'};">
                                {category_name}
                            </div>
                            <p style="font-size: 1.2rem;">AQI Range: {category_info['range']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show debug info
                        if st.session_state.debug_mode:
                            st.write(f"Raw prediction: {prediction}")
                            st.write(f"Mapped to category: {category_name}")
                        
                    else:
                        st.error("Prediction failed")
                        
            except Exception as e:
                st.error(f"Unexpected error during prediction: {e}")
                if st.session_state.debug_mode:
                    st.exception(e)

# Show prediction history
if st.session_state.prediction_history:
    st.header("Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Convert timestamp for display
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show latest predictions
    st.dataframe(display_df[['timestamp', 'city', 'category']].tail(5), use_container_width=True)

# AQI category reference
st.sidebar.header("AQI Categories Reference")
for category, info in AQI_CATEGORIES.items():
    if category in ['Good', 'Moderate', 'Poor', 'Unhealthy', 'Very Unhealthy', 'Hazardous']:
        st.sidebar.markdown(
            f"<div style='background: {info['color']}; padding: 5px; border-radius: 5px; "
            f"color: {'black' if category in ['Moderate'] else 'white'}; margin: 2px 0;'>"
            f"{category}: {info['range']}</div>",
            unsafe_allow_html=True
        )

# Model status
st.sidebar.header("Model Status")
if models_loaded:
    st.sidebar.success("✅ Models Ready")
else:
    st.sidebar.error("❌ Models Failed")

# Footer
st.markdown("---")
st.markdown("*Air Quality Index Prediction System*")
