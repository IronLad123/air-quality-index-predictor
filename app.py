import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time

# Page configuration
st.set_page_config(
    page_title="Air Quality Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    .debug-info {
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

# Enhanced AQI category mapping with all possible categories
aqi_categories = {
    'Good': {'range': '0-50', 'color': 'aqi-good', 'level': 1},
    'Satisfactory': {'range': '51-100', 'color': 'aqi-moderate', 'level': 2},
    'Moderate': {'range': '51-100', 'color': 'aqi-moderate', 'level': 2},
    'Poor': {'range': '101-200', 'color': 'aqi-poor', 'level': 3},
    'Unhealthy': {'range': '201-300', 'color': 'aqi-unhealthy', 'level': 4},
    'Very Poor': {'range': '201-300', 'color': 'aqi-unhealthy', 'level': 4},
    'Severe': {'range': '301-400', 'color': 'aqi-very-unhealthy', 'level': 5},
    'Very Unhealthy': {'range': '301-400', 'color': 'aqi-very-unhealthy', 'level': 5},
    'Hazardous': {'range': '401-500', 'color': 'aqi-hazardous', 'level': 6},
    'Critical': {'range': '401-500', 'color': 'aqi-hazardous', 'level': 6}
}

# Load saved objects with better error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('stacking_ensemble.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        numeric_cols = joblib.load('numeric_columns.pkl')
        le_city = joblib.load('le_city.pkl')
        
        st.success("✅ All models loaded successfully!")
        return model, scaler, numeric_cols, le_city
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("🔍 Please make sure all model files (stacking_ensemble.pkl, feature_scaler.pkl, numeric_columns.pkl, le_city.pkl) are in the same directory as this app.")
        return None, None, None, None

# Initialize models
model, scaler, numeric_cols, le_city = load_models()

# Debug function to see what's happening
def debug_prediction(input_df, prediction, predicted_category):
    """Display debug information for prediction"""
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("### Input Data Preview:")
        st.dataframe(input_df.head())
        
        st.write("### Prediction Details:")
        st.write(f"Raw prediction: {prediction}")
        st.write(f"Predicted category: {predicted_category}")
        st.write(f"Prediction type: {type(prediction)}")
        st.write(f"Prediction shape: {getattr(prediction, 'shape', 'No shape')}")
        
        st.write("### Available AQI Categories:")
        st.write(list(aqi_categories.keys()))
        
        st.write("### Model Information:")
        if model is not None:
            st.write(f"Model type: {type(model)}")
            st.write(f"Model features: {getattr(model, 'n_features_in_', 'Unknown')}")

# Safe category getter function
def get_category_info(predicted_category):
    """Safely get category info with fallback"""
    # Convert to string and strip any whitespace
    category_str = str(predicted_category).strip()
    
    # Try exact match first
    if category_str in aqi_categories:
        return aqi_categories[category_str], category_str
    
    # Try case-insensitive match
    for key in aqi_categories.keys():
        if key.lower() == category_str.lower():
            return aqi_categories[key], key
    
    # Try partial match
    for key in aqi_categories.keys():
        if category_str.lower() in key.lower() or key.lower() in category_str.lower():
            return aqi_categories[key], key
    
    # Fallback to Moderate
    st.warning(f"⚠️ Unknown category '{predicted_category}'. Using 'Moderate' as fallback.")
    return aqi_categories['Moderate'], 'Moderate'

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main title
st.markdown('<h1 class="main-header">🌍 Air Quality Intelligence Platform</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    if model is not None:
        st.success("✅ Model Connected")
    else:
        st.error("❌ Model Not Loaded")
    
    st.header("🏙️ Location Settings")
    if le_city is not None:
        city = st.selectbox("Select City", le_city.classes_, index=0)
    else:
        st.error("City encoder not loaded")
        city = "Delhi"  # Fallback
    
    show_debug = st.toggle("🐛 Show Debug Info", value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🌫️ Air Quality Parameters")
    
    with st.expander("🧪 Pollutant Parameters", expanded=True):
        col1a, col1b = st.columns(2)
        with col1a:
            pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
            pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0)
            no = st.slider("NO (µg/m³)", 0.0, 100.0, 10.0)
            no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 20.0)
        with col1b:
            nox = st.slider("NOx (µg/m³)", 0.0, 300.0, 30.0)
            so2 = st.slider("SO2 (µg/m³)", 0.0, 300.0, 10.0)
            o3 = st.slider("O3 (µg/m³)", 0.0, 400.0, 20.0)
            co = st.slider("CO (µg/m³)", 0.0, 50.0, 5.0)
        
        nh3 = st.slider("NH3 (µg/m³)", 0.0, 200.0, 10.0)
        benzene = st.slider("Benzene (µg/m³)", 0.0, 50.0, 1.0)
        toluene = st.slider("Toluene (µg/m³)", 0.0, 100.0, 5.0)
        xylene = st.slider("Xylene (µg/m³)", 0.0, 100.0, 3.0)

with col2:
    st.header("🚀 Prediction")
    
    if st.button("🎯 Predict AQI", use_container_width=True, type="primary"):
        if model is not None and scaler is not None and le_city is not None:
            try:
                with st.spinner('🤖 Analyzing air quality data...'):
                    # Calculate features
                    pm_ratio = pm25 / (pm10 + 1e-6)
                    no_ratio = nox / (no2 + 1e-6)
                    day_of_week = date.today().weekday()
                    
                    # Prepare input data
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
                    
                    # Debug: Show input data
                    if show_debug:
                        st.write("### Input Data Before Processing:")
                        st.dataframe(input_df)
                    
                    # Encode City
                    try:
                        input_df['City'] = le_city.transform(input_df['City'])
                    except Exception as e:
                        st.error(f"Error encoding city: {e}")
                        # Fallback: use first city in encoder
                        if len(le_city.classes_) > 0:
                            input_df['City'] = le_city.transform([le_city.classes_[0]])
                    
                    # Scale numeric features
                    try:
                        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                    except Exception as e:
                        st.error(f"Error scaling features: {e}")
                    
                    # Debug: Show processed data
                    if show_debug:
                        st.write("### Processed Data:")
                        st.dataframe(input_df)
                    
                    # Make prediction
                    try:
                        prediction = model.predict(input_df)
                        predicted_category = prediction[0]
                        
                        # Debug information
                        if show_debug:
                            debug_prediction(input_df, prediction, predicted_category)
                        
                        # Safely get category info
                        category_info, final_category = get_category_info(predicted_category)
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'city': city,
                            'category': final_category,
                            'level': category_info['level'],
                            'raw_prediction': str(predicted_category)
                        })
                        
                        # Display results
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 Prediction Result</h2>
                            <div style="font-size: 2.5rem; margin: 1rem 0; padding: 1rem; background: {category_info['color']}; border-radius: 10px;">
                                {final_category}
                            </div>
                            <p style="font-size: 1.3rem;">📊 AQI Range: {category_info['range']}</p>
                            <p style="font-size: 0.9rem; margin-top: 1rem;">
                                📍 {city} | 🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show raw prediction for debugging
                        if show_debug and str(predicted_category) != final_category:
                            st.info(f"🔍 Raw prediction was: '{predicted_category}', mapped to: '{final_category}'")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                if show_debug:
                    st.exception(e)
        else:
            st.error("❌ Models not properly loaded. Please check the model files.")
            
            # Show what's loaded
            st.write("### Loaded Components:")
            st.write(f"- Model: {'✅' if model else '❌'}")
            st.write(f"- Scaler: {'✅' if scaler else '❌'}")
            st.write(f"- Numeric Columns: {'✅' if numeric_cols else '❌'}")
            st.write(f"- City Encoder: {'✅' if le_city else '❌'}")

# Show prediction history
if st.session_state.prediction_history:
    st.header("📈 Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Convert to display format
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show recent predictions
    st.dataframe(display_df.tail(5), use_container_width=True)
    
    # Show all available categories from history
    if show_debug:
        st.write("### All Categories in History:")
        unique_categories = history_df['category'].unique()
        st.write(list(unique_categories))

# Footer
st.markdown("---")
st.markdown(
    "🔬 *This AQI prediction tool uses machine learning to analyze air quality parameters. "
    "If you encounter issues, enable debug mode for more information.*"
)
