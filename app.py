import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import traceback

# Page configuration
st.set_page_config(
    page_title="AQI Predictor - Fixed",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Air Quality Index Predictor - Fixed Version")

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
                loaded_obj = joblib.load(filename)
                model_data[key] = loaded_obj
                st.sidebar.success(f"✅ {filename} loaded")
                
                # Show some info about loaded objects
                if key == 'le_city' and hasattr(loaded_obj, 'classes_'):
                    st.sidebar.write(f"🏙️ Cities: {list(loaded_obj.classes_)}")
                elif key == 'numeric_cols':
                    st.sidebar.write(f"📊 Numeric columns: {len(loaded_obj)}")
                elif key == 'model':
                    st.sidebar.write(f"🤖 Model type: {type(loaded_obj)}")
                    if hasattr(loaded_obj, 'n_features_in_'):
                        st.sidebar.write(f"📈 Model expects {loaded_obj.n_features_in_} features")
                    
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
has_model = model_data.get('model') is not None
has_scaler = model_data.get('scaler') is not None  
has_numeric_cols = model_data.get('numeric_cols') is not None
has_le_city = model_data.get('le_city') is not None

if has_model and has_scaler and has_numeric_cols and has_le_city:
    st.success("🎉 All models loaded successfully! Ready for predictions.")
else:
    st.warning("⚠️ Some components missing. Check debug info above.")

# Get expected features from model
expected_features = []
if has_model and hasattr(model_data['model'], 'feature_names_in_'):
    expected_features = list(model_data['model'].feature_names_in_)
    st.info(f"📋 Model expects these features: {expected_features}")

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
    nox = st.number_input("NOx", min_value=0.0, max_value=300.0, value=30.0)

with col2:
    nh3 = st.number_input("NH3", min_value=0.0, max_value=200.0, value=10.0)
    co = st.number_input("CO", min_value=0.0, max_value=50.0, value=5.0)
    so2 = st.number_input("SO2", min_value=0.0, max_value=300.0, value=10.0)
    o3 = st.number_input("O3", min_value=0.0, max_value=400.0, value=20.0)
    benzene = st.number_input("Benzene", min_value=0.0, max_value=50.0, value=1.0)
    toluene = st.number_input("Toluene", min_value=0.0, max_value=100.0, value=5.0)
    xylene = st.number_input("Xylene", min_value=0.0, max_value=100.0, value=3.0)

# FIXED: Create input data that matches exactly what the model was trained on
def create_input_dataframe():
    try:
        # Calculate engineered features
        pm_ratio = pm25 / (pm10 + 1e-6)
        no_ratio = nox / (no2 + 1e-6)
        day_of_week = 3  # Wednesday as example
        
        # Create base data
        data = {
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
        }
        
        # Add rolling averages - using current values
        rolling_features = [
            'PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 
            'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
        ]
        
        for feature in rolling_features:
            data[f'{feature}_3d_avg'] = [data[feature][0]]
        
        df = pd.DataFrame(data)
        
        # FIXED: Ensure correct column order if we know what model expects
        if expected_features:
            # Add any missing columns with default values
            for feature in expected_features:
                if feature not in df.columns:
                    st.warning(f"⚠️ Adding missing feature: {feature} with default value 0")
                    df[feature] = 0.0
            
            # Reorder columns to match model expectations
            df = df[expected_features]
        
        return df
        
    except Exception as e:
        st.error(f"Error creating input data: {e}")
        return None

# FIXED: Prediction function with proper error handling
def make_prediction(input_df):
    try:
        # Step 1: Create a copy to avoid modifying original
        processed_df = input_df.copy()
        
        # Step 2: Encode city
        if has_le_city:
            try:
                processed_df['City'] = model_data['le_city'].transform(processed_df['City'])
                st.write("✅ City encoded successfully")
            except ValueError as e:
                st.error(f"City encoding error: {e}")
                # Try to use the first available city as fallback
                if len(model_data['le_city'].classes_) > 0:
                    fallback_city = model_data['le_city'].classes_[0]
                    processed_df['City'] = model_data['le_city'].transform([fallback_city])
                    st.warning(f"Using fallback city: {fallback_city}")
        
        # Step 3: Scale numeric features
        if has_scaler and has_numeric_cols:
            try:
                # Ensure all numeric columns exist in the dataframe
                available_cols = [col for col in model_data['numeric_cols'] if col in processed_df.columns]
                missing_cols = [col for col in model_data['numeric_cols'] if col not in processed_df.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns for scaling: {missing_cols}")
                    # Add missing columns with 0 values
                    for col in missing_cols:
                        processed_df[col] = 0.0
                    available_cols = model_data['numeric_cols']
                
                st.write(f"📈 Scaling {len(available_cols)} numeric columns")
                processed_df[available_cols] = model_data['scaler'].transform(processed_df[available_cols])
                st.write("✅ Features scaled successfully")
                
            except Exception as e:
                st.error(f"Scaling error: {e}")
                return None
        
        # Step 4: Make prediction
        if has_model:
            st.write("🤖 Making prediction...")
            
            # FIXED: Ensure the dataframe has exactly the features the model expects
            if hasattr(model_data['model'], 'feature_names_in_'):
                model_expected_features = list(model_data['model'].feature_names_in_)
                missing_model_features = [f for f in model_expected_features if f not in processed_df.columns]
                extra_features = [f for f in processed_df.columns if f not in model_expected_features]
                
                if missing_model_features:
                    st.error(f"❌ Missing features for model: {missing_model_features}")
                    return None
                
                if extra_features:
                    st.warning(f"⚠️ Removing extra features: {extra_features}")
                    processed_df = processed_df[model_expected_features]
            
            # Final check before prediction
            st.write(f"📊 Final input shape: {processed_df.shape}")
            
            # Make the prediction
            prediction = model_data['model'].predict(processed_df)
            st.write(f"🎯 Raw prediction result: {prediction}")
            
            return prediction[0] if hasattr(prediction, '__len__') else prediction
            
        else:
            st.error("Model not available for prediction")
            return None
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write("🔍 Full error traceback:")
        st.code(traceback.format_exc())
        return None

# Prediction button
if st.button("Predict AQI", type="primary", use_container_width=True):
    if not all([has_model, has_scaler, has_numeric_cols, has_le_city]):
        st.error("Cannot make prediction - required models not loaded")
        st.info("Please ensure all .pkl files are in the same directory as this script")
    else:
        with st.spinner("Processing..."):
            # Step 1: Create input data
            st.write("### Step 1: Creating input data...")
            input_df = create_input_dataframe()
            
            if input_df is not None:
                st.write("📊 Input data created successfully:")
                st.dataframe(input_df)
                st.write(f"Input shape: {input_df.shape}")
                
                # Step 2: Make prediction
                st.write("### Step 2: Making prediction...")
                result = make_prediction(input_df)
                
                if result is not None:
                    st.success("✅ Prediction completed successfully!")
                    
                    # Display results
                    st.write("### 🎯 Prediction Result")
                    
                    # AQI categories mapping
                    aqi_categories = {
                        'Good': {'range': '0-50', 'emoji': '😊', 'color': 'green'},
                        'Moderate': {'range': '51-100', 'emoji': '😐', 'color': 'yellow'},
                        'Poor': {'range': '101-200', 'emoji': '😷', 'color': 'orange'},
                        'Unhealthy': {'range': '201-300', 'emoji': '🤢', 'color': 'red'},
                        'Very Unhealthy': {'range': '301-400', 'emoji': '😨', 'color': 'purple'},
                        'Hazardous': {'range': '401-500', 'emoji': '💀', 'color': 'maroon'}
                    }
                    
                    # Determine category
                    if isinstance(result, str):
                        predicted_category = result
                    else:
                        # If numeric, map to categories
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
                    
                    category_info = aqi_categories.get(predicted_category, 
                                                     {'range': 'Unknown', 'emoji': '❓', 'color': 'gray'})
                    
                    # Display result with styling
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; text-align: center; margin: 1rem 0; border: 3px solid {category_info['color']};">
                        <h2>Predicted AQI Category</h2>
                        <div style="font-size: 4rem; margin: 1rem 0;">{category_info['emoji']}</div>
                        <div style="font-size: 2.5rem; font-weight: bold; margin: 1rem 0;">{predicted_category}</div>
                        <div style="font-size: 1.5rem;">AQI Range: {category_info['range']}</div>
                        <div style="font-size: 1rem; margin-top: 1rem;">Raw output: {result}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Health recommendations
                    st.write("### 💡 Health Recommendations")
                    recommendations = {
                        'Good': "✅ **Excellent air quality** - Perfect for outdoor activities and exercise.",
                        'Moderate': "⚠️ **Acceptable air quality** - Unusually sensitive people should consider reducing prolonged outdoor exertion.",
                        'Poor': "🔶 **Poor air quality** - Members of sensitive groups may experience health effects. General public is less likely to be affected.",
                        'Unhealthy': "🔴 **Unhealthy air quality** - Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activities.",
                        'Very Unhealthy': "💀 **Very unhealthy air quality** - Health alert: everyone may experience more serious health effects.",
                        'Hazardous': "☠️ **Hazardous air quality** - Health warning of emergency conditions. The entire population is affected."
                    }
                    
                    st.info(recommendations.get(predicted_category, "Please take necessary precautions based on local health advisories."))

# Model information section
with st.expander("🔧 Technical Details"):
    st.write("### Model Information")
    
    if has_model:
        model = model_data['model']
        st.write(f"**Model Type:** {type(model)}")
        st.write(f"**Model Class:** {model.__class__.__name__}")
        
        if hasattr(model, 'n_features_in_'):
            st.write(f"**Expected Features:** {model.n_features_in_}")
        
        if hasattr(model, 'feature_names_in_'):
            st.write(f"**Feature Names:** {list(model.feature_names_in_)}")
    
    if has_scaler:
        st.write(f"**Scaler Type:** {type(model_data['scaler'])}")
    
    if has_le_city:
        st.write(f"**Number of Cities:** {len(model_data['le_city'].classes_)}")
        st.write(f"**Available Cities:** {list(model_data['le_city'].classes_)}")
    
    if has_numeric_cols:
        st.write(f"**Numeric Columns to Scale:** {model_data['numeric_cols']}")

# Troubleshooting guide
with st.expander("🚨 Troubleshooting Guide"):
    st.markdown("""
    **Common Issues and Solutions:**
    
    🔹 **Issue: Model files not found**
    - Ensure all .pkl files are in the same directory as this script
    - Check file names are exact: `stacking_ensemble.pkl`, `feature_scaler.pkl`, `numeric_columns.pkl`, `le_city.pkl`
    
    🔹 **Issue: Feature mismatch errors**
    - The app now automatically checks what features your model expects
    - It will warn you about missing or extra features
    
    🔹 **Issue: City encoding errors**
    - Make sure the city you select exists in the encoder's classes
    - The app will use a fallback city if there's an issue
    
    🔹 **Issue: Prediction fails**
    - Check the debug information above each step
    - Look for any error messages in red
    - The app shows exactly what's happening at each stage
    """)

# Footer
st.markdown("---")
st.markdown("**Air Quality Index Prediction System** | Built with Streamlit")
