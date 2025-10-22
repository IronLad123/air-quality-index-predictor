import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import traceback

st.set_page_config(page_title="AQI Predictor - Model Check", page_icon="🌫️")
st.title("🔍 AQI Predictor - Model Verification")

# First, let's check what's REALLY in your directory
st.header("📁 File System Check")
current_dir = os.listdir('.')
st.write("All files in current directory:", current_dir)

pkl_files = [f for f in current_dir if f.endswith('.pkl')]
st.write("Found .pkl files:", pkl_files)

# Try to load each file individually and see what's inside
st.header("🔧 Model Loading Debug")

def inspect_file(filename):
    try:
        if os.path.exists(filename):
            obj = joblib.load(filename)
            st.success(f"✅ {filename} loaded successfully")
            
            # Show what type of object it is
            st.write(f"   - Type: {type(obj)}")
            st.write(f"   - Class: {obj.__class__.__name__}")
            
            # Show specific attributes based on object type
            if hasattr(obj, 'classes_'):
                st.write(f"   - Cities: {list(obj.classes_)}")
            elif hasattr(obj, 'n_features_in_'):
                st.write(f"   - Expected features: {obj.n_features_in_}")
            elif hasattr(obj, 'feature_names_in_'):
                st.write(f"   - Feature names: {list(obj.feature_names_in_)}")
            elif isinstance(obj, list):
                st.write(f"   - Length: {len(obj)}")
                st.write(f"   - Content: {obj}")
            else:
                st.write(f"   - Available methods: {[m for m in dir(obj) if not m.startswith('_')][:10]}")
            
            return obj
        else:
            st.error(f"❌ {filename} not found")
            return None
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {str(e)}")
        return None

# Load and inspect each file
st.subheader("Loading stacking_ensemble.pkl")
model = inspect_file('stacking_ensemble.pkl')

st.subheader("Loading feature_scaler.pkl")  
scaler = inspect_file('feature_scaler.pkl')

st.subheader("Loading numeric_columns.pkl")
numeric_cols = inspect_file('numeric_columns.pkl')

st.subheader("Loading le_city.pkl")
le_city = inspect_file('le_city.pkl')

# Check if we can actually use the model
st.header("🤖 Model Compatibility Check")

if model is not None:
    st.success("🎉 Model object loaded!")
    
    # Try to understand the model structure
    st.subheader("Model Analysis:")
    
    # Check model type and capabilities
    model_type = type(model).__name__
    st.write(f"**Model Type:** {model_type}")
    
    # Check for common ensemble attributes
    if hasattr(model, 'estimators_'):
        st.write(f"**Number of base estimators:** {len(model.estimators_)}")
        st.write(f"**Base estimators:** {[type(est).__name__ for est in model.estimators_]}")
    
    if hasattr(model, 'final_estimator_'):
        st.write(f"**Final estimator:** {type(model.final_estimator_).__name__}")
    
    # Check feature requirements
    if hasattr(model, 'n_features_in_'):
        st.write(f"**Expected number of features:** {model.n_features_in_}")
    
    if hasattr(model, 'feature_names_in_'):
        st.write(f"**Expected feature names:** {list(model.feature_names_in_)}")
        expected_features = list(model.feature_names_in_)
    else:
        # If we don't have feature names, let's check what features the model might expect
        st.warning("⚠️ Model doesn't have feature_names_in_ attribute")
        expected_features = None

else:
    st.error("❌ Cannot proceed - model not loaded")
    expected_features = None

# Now let's create the exact input the model expects
st.header("🎯 Create Prediction Input")

if model is not None and expected_features:
    st.info(f"Model expects these {len(expected_features)} features: {expected_features}")
    
    # Show which features we need to provide
    st.subheader("Required Input Features:")
    
    # Group features by type
    basic_features = [f for f in expected_features if not f.endswith('_3d_avg') and f not in ['PM_ratio', 'NO_ratio', 'Day_of_week', 'City']]
    engineered_features = [f for f in expected_features if f in ['PM_ratio', 'NO_ratio', 'Day_of_week']]
    rolling_features = [f for f in expected_features if f.endswith('_3d_avg')]
    city_feature = [f for f in expected_features if f == 'City']
    
    st.write("**Basic Pollutants:**", basic_features)
    st.write("**Engineered Features:**", engineered_features) 
    st.write("**Rolling Averages:**", rolling_features)
    st.write("**City Feature:**", city_feature)

# Simple input form
st.header("📝 Enter Parameters")

col1, col2 = st.columns(2)

with col1:
    # City selection
    if le_city is not None:
        city = st.selectbox("City", le_city.classes_)
    else:
        city = st.selectbox("City", ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"])
    
    pm25 = st.number_input("PM2.5", value=50.0)
    pm10 = st.number_input("PM10", value=100.0)
    no = st.number_input("NO", value=10.0)
    no2 = st.number_input("NO2", value=20.0)
    nox = st.number_input("NOx", value=30.0)

with col2:
    nh3 = st.number_input("NH3", value=10.0)
    co = st.number_input("CO", value=5.0)
    so2 = st.number_input("SO2", value=10.0)
    o3 = st.number_input("O3", value=20.0)
    benzene = st.number_input("Benzene", value=1.0)
    toluene = st.number_input("Toluene", value=5.0)
    xylene = st.number_input("Xylene", value=3.0)

# Create input data based on what the model actually expects
def create_exact_input():
    # Calculate engineered features
    pm_ratio = pm25 / (pm10 + 1e-6)
    no_ratio = nox / (no2 + 1e-6)
    day_of_week = 3  # Example: Wednesday
    
    # Start with all possible features
    all_possible_features = {
        'City': city,
        'PM2_5': pm25,
        'PM10': pm10,
        'NO': no,
        'NO2': no2,
        'NOx': nox,
        'NH3': nh3,
        'CO': co,
        'SO2': so2,
        'O3': o3,
        'Benzene': benzene,
        'Toluene': toluene,
        'Xylene': xylene,
        'PM_ratio': pm_ratio,
        'NO_ratio': no_ratio,
        'Day_of_week': day_of_week,
        # Rolling averages (using current values)
        'PM2_5_3d_avg': pm25,
        'PM10_3d_avg': pm10,
        'NO_3d_avg': no,
        'NO2_3d_avg': no2,
        'NOx_3d_avg': nox,
        'NH3_3d_avg': nh3,
        'CO_3d_avg': co,
        'SO2_3d_avg': so2,
        'O3_3d_avg': o3,
        'Benzene_3d_avg': benzene,
        'Toluene_3d_avg': toluene,
        'Xylene_3d_avg': xylene
    }
    
    # If we know what features model expects, use only those
    if expected_features:
        input_data = {feature: all_possible_features.get(feature, 0.0) for feature in expected_features}
    else:
        # Otherwise use all features we have
        input_data = all_possible_features
    
    return pd.DataFrame([input_data])

# Prediction function
def make_prediction_with_model(input_df):
    try:
        st.write("### Step 1: Input Data")
        st.dataframe(input_df)
        st.write(f"Input shape: {input_df.shape}")
        
        # Encode city
        if 'City' in input_df.columns and le_city is not None:
            try:
                input_df['City'] = le_city.transform(input_df['City'])
                st.success("✅ City encoded")
            except Exception as e:
                st.error(f"❌ City encoding failed: {e}")
                return None
        
        # Scale features
        if scaler is not None and numeric_cols is not None:
            try:
                # Only scale columns that exist in both
                cols_to_scale = [col for col in numeric_cols if col in input_df.columns]
                input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
                st.success(f"✅ Scaled {len(cols_to_scale)} features")
            except Exception as e:
                st.error(f"❌ Feature scaling failed: {e}")
                return None
        
        st.write("### Step 2: Processed Data")
        st.dataframe(input_df)
        
        # Make prediction
        st.write("### Step 3: Making Prediction")
        prediction = model.predict(input_df)
        st.success("✅ Prediction completed")
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.code(traceback.format_exc())
        return None

# Prediction button
if st.button("🚀 Test Prediction", type="primary"):
    if model is None:
        st.error("Cannot test - model not loaded")
    else:
        with st.spinner("Testing prediction..."):
            input_df = create_exact_input()
            result = make_prediction_with_model(input_df)
            
            if result is not None:
                st.success(f"🎯 **Prediction Result:** {result}")
                
                # Try to interpret the result
                if hasattr(result, '__len__') and len(result) == 1:
                    final_result = result[0]
                else:
                    final_result = result
                
                st.write(f"**Final Output:** {final_result}")
                st.write(f"**Output Type:** {type(final_result)}")

# Show what we learned about the model
st.header("📊 Model Summary")

if model is not None:
    st.success("✅ Model is loaded and ready")
    
    # Show model capabilities
    capabilities = []
    if hasattr(model, 'predict'): capabilities.append("predict()")
    if hasattr(model, 'predict_proba'): capabilities.append("predict_proba()")
    if hasattr(model, 'score'): capabilities.append("score()")
    
    st.write(f"**Model Capabilities:** {capabilities}")
    
    # Show if it's a classifier or regressor
    if hasattr(model, 'classes_'):
        st.write("**Type:** Classifier")
        st.write(f"**Classes:** {model.classes_}")
    else:
        st.write("**Type:** Regressor (predicts numeric AQI values)")
        
else:
    st.error("❌ Model failed to load")

st.header("🔧 Next Steps")

if model is None:
    st.error("""
    **Critical Issue: Model file cannot be loaded.**
    
    Possible solutions:
    1. Check if `stacking_ensemble.pkl` exists in the same directory
    2. Verify the file is not corrupted
    3. Check if joblib version is compatible
    4. Try loading the model in a separate Python script to test
    """)
elif scaler is None or numeric_cols is None or le_city is None:
    st.warning("""
    **Partial Loading: Some support files missing.**
    
    The main model loaded but some preprocessing components failed.
    We can try to work around this with default values.
    """)
else:
    st.success("""
    **All systems go!** 
    
    Your model and all support files are loaded successfully.
    You can now make predictions using the form above.
    """)
