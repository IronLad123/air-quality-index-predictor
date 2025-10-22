import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Air Quality Index Predictor")

# Debug: Check what files exist
st.sidebar.header("🔍 Debug Information")
current_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
st.sidebar.write("Found .pkl files:", current_files)

# Try to load models with detailed error handling
@st.cache_resource
def load_models():
    model_data = {}
    errors = []
    
    # List of required files
    files_to_load = {
        'model': 'stacking_ensemble.pkl',
        'scaler': 'feature_scaler.pkl', 
        'numeric_cols': 'numeric_columns.pkl',
        'le_city': 'le_city.pkl'
    }
    
    for key, filename in files_to_load.items():
        try:
            if os.path.exists(filename):
                model_data[key] = joblib.load(filename)
                st.sidebar.success(f"✅ {filename} loaded")
                
                # Show some info about loaded objects
                if key == 'le_city' and hasattr(model_data[key], 'classes_'):
                    st.sidebar.write(f"🏙️ Cities: {list(model_data[key].classes_)}")
                elif key == 'numeric_cols':
                    st.sidebar.write(f"📊 Numeric columns: {len(model_data[key])}")
                    
            else:
                errors.append(f"❌ {filename} not found")
                model_data[key] = None
                
        except Exception as e:
            errors.append(f"❌ Error loading {filename}: {str(e)}")
            model_data[key] = None
    
    return model_data, errors

# Load models
with st.spinner("Loading models..."):
    model_data, errors = load_models()

# Show loading results
if errors:
    st.error("Model Loading Issues:")
    for error in errors:
        st.write(error)

# Check if we have the minimum required components
has_model = model_data['model'] is not None
has_scaler = model_data['scaler'] is not None  
has_numeric_cols = model_data['numeric_cols'] is not None
has_le_city = model_data['le_city'] is not None

if has_model and has_scaler and has_numeric_cols and has_le_city:
    st.success("🎉 All models loaded successfully! Ready for predictions.")
else:
    st.warning("⚠️ Some components missing. Check debug info above.")

# Simple input form
st.header("Enter Air Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    # Get available cities from encoder or use defaults
    if has_le_city:
        city_options = list(model_data['le_city'].classes_)
    else:
        city_options = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"]
    
    city = st.selectbox("City", city_options)
    
    pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=100.0)
    no = st.number_input("NO", min_value=0.0, max_value=100.0, value=10.0)
    no2 = st.number_input("NO2", min_value=0.0, max_value=200.0, value=20.0)

with col2:
    nox = st.number_input("NOx", min_value=0.0, max_value=300.0, value=30.0)
    nh3 = st.number_input("NH3", min_value=0.0, max_value=200.0, value=10.0)
    co = st.number_input("CO", min_value=0.0, max_value=50.0, value=5.0)
    so2 = st.number_input("SO2", min_value=0.0, max_value=300.0, value=10.0)
    o3 = st.number_input("O3", min_value=0.0, max_value=400.0, value=20.0)
    benzene = st.number_input("Benzene", min_value=0.0, max_value=50.0, value=1.0)
    toluene = st.number_input("Toluene", min_value=0.0, max_value=100.0, value=5.0)
    xylene = st.number_input("Xylene", min_value=0.0, max_value=100.0, value=3.0)

# Prediction function
def make_prediction():
    try:
        # Calculate engineered features
        pm_ratio = pm25 / (pm10 + 1e-6)
        no_ratio = nox / (no2 + 1e-6)
        day_of_week = 3  # Wednesday as example
        
        # Create input dataframe - match exactly what your model expects
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
            # Rolling averages - using current values as fallback
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
        
        st.write("📊 Input data preview:")
        st.dataframe(input_df)
        
        # Transform data
        # 1. Encode city
        if has_le_city:
            input_df['City'] = model_data['le_city'].transform(input_df['City'])
            st.write("✅ City encoded")
        
        # 2. Scale numeric features
        if has_scaler and has_numeric_cols:
            # Check which numeric columns actually exist in our input
            available_numeric_cols = [col for col in model_data['numeric_cols'] if col in input_df.columns]
            st.write(f"📈 Scaling {len(available_numeric_cols)} numeric columns")
            
            input_df[available_numeric_cols] = model_data['scaler'].transform(input_df[available_numeric_cols])
            st.write("✅ Features scaled")
        
        st.write("📊 Processed data preview:")
        st.dataframe(input_df)
        
        # Make prediction
        if has_model:
            st.write("🤖 Making prediction...")
            prediction = model_data['model'].predict(input_df)
            st.write(f"🎯 Raw prediction: {prediction}")
            return prediction[0]
        else:
            st.error("Model not available for prediction")
            return None
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write("Full error details:")
        st.exception(e)
        return None

# Prediction button
if st.button("Predict AQI", type="primary"):
    if not all([has_model, has_scaler, has_numeric_cols, has_le_city]):
        st.error("Cannot make prediction - required models not loaded")
    else:
        with st.spinner("Processing..."):
            result = make_prediction()
            
            if result is not None:
                st.success(f"✅ Prediction completed: {result}")
                
                # Simple AQI interpretation
                aqi_categories = {
                    'Good': '😊 Good (0-50) - Enjoy outdoor activities!',
                    'Moderate': '😐 Moderate (51-100) - Acceptable air quality',
                    'Poor': '😷 Poor (101-200) - Sensitive groups affected', 
                    'Unhealthy': '🤢 Unhealthy (201-300) - Everyone may be affected',
                    'Very Unhealthy': '😨 Very Unhealthy (301-400) - Health alert',
                    'Hazardous': '💀 Hazardous (401-500) - Emergency conditions'
                }
                
                # Try to map the prediction to a category
                if hasattr(result, 'lower'):
                    predicted_category = result
                else:
                    # If it's numeric, map to categories
                    try:
                        aqi_value = float(result)
                        if aqi_value <= 50:
                            predicted_category = 'Good'
                        elif aqi_value <= 100:
                            predicted_category = 'Moderate'
                        elif aqi_value <= 200:
                            predicted_category = 'Poor'
                        elif aqi_value <= 300:
                            predicted_category = 'Unhealthy'
                        elif aqi_value <= 400:
                            predicted_category = 'Very Unhealthy'
                        else:
                            predicted_category = 'Hazardous'
                    except:
                        predicted_category = str(result)
                
                st.info(f"**Interpretation:** {aqi_categories.get(predicted_category, f'Category: {predicted_category}')}")

# Model information
with st.expander("🔧 Model Details"):
    if has_model:
        st.write("**Model Type:**", type(model_data['model']))
        st.write("**Model Features:**", getattr(model_data['model'], 'n_features_in_', 'Unknown'))
    
    if has_scaler:
        st.write("**Scaler Type:**", type(model_data['scaler']))
    
    if has_le_city:
        st.write("**Number of Cities:**", len(model_data['le_city'].classes_))
    
    if has_numeric_cols:
        st.write("**Numeric Columns:**", model_data['numeric_cols'])

# Instructions
with st.expander("📋 Setup Instructions"):
    st.markdown("""
    **To fix model loading issues:**
    
    1. **Check file locations** - All .pkl files should be in the same directory as this script
    2. **Verify file names** - They should be exactly:
       - `stacking_ensemble.pkl`
       - `feature_scaler.pkl` 
       - `numeric_columns.pkl`
       - `le_city.pkl`
    3. **Check file permissions** - Make sure the files are readable
    4. **Verify model compatibility** - Ensure models were created with compatible library versions
    
    **If models still don't load:**
    - Check the debug information in the sidebar
    - Look for any error messages in red
    - Make sure all required files are present
    """)
