import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Air Quality Index Predictor",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Air Quality Index Predictor")

# Load all models
@st.cache_resource
def load_models():
    model = joblib.load('stacking_ensemble.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    numeric_cols = joblib.load('numeric_columns.pkl')
    le_city = joblib.load('le_city.pkl')
    le_aqi_bucket = joblib.load('le_aqi_bucket.pkl')
    return model, scaler, numeric_cols, le_city, le_aqi_bucket

try:
    model, scaler, numeric_cols, le_city, le_aqi_bucket = load_models()
    st.success("✅ All models loaded successfully!")
    
    # Show what AQI categories we have
    st.sidebar.info(f"🎯 AQI Categories: {list(le_aqi_bucket.classes_)}")
    
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# AQI category styling - map numeric predictions to actual categories
aqi_styles = {
    'Good': {'color': '#00e400', 'emoji': '😊', 'description': 'Excellent air quality'},
    'Satisfactory': {'color': '#87e887', 'emoji': '🙂', 'description': 'Good air quality'},
    'Moderate': {'color': '#ffff00', 'emoji': '😐', 'description': 'Acceptable air quality'},
    'Poor': {'color': '#ff7e00', 'emoji': '😷', 'description': 'Unhealthy for sensitive groups'},
    'Very Poor': {'color': '#ff0000', 'emoji': '🤢', 'description': 'Unhealthy for everyone'},
    'Severe': {'color': '#8f3f97', 'emoji': '😨', 'description': 'Very unhealthy'}
}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📍 Location & Parameters")
    
    city = st.selectbox("Select City", le_city.classes_)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Particulate Matter", "Gaseous Pollutants", "Chemical Compounds"])
    
    with tab1:
        st.subheader("🌫️ Particulate Matter")
        col1a, col1b = st.columns(2)
        with col1a:
            pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
            pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0)
        
        with col1b:
            # Real-time gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pm25,
                title = {'text': "PM2.5 Level"},
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
            fig_gauge.update_layout(height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        st.subheader("💨 Gaseous Pollutants")
        col2a, col2b = st.columns(2)
        with col2a:
            no = st.slider("NO (µg/m³)", 0.0, 100.0, 10.0)
            no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 20.0)
            nox = st.slider("NOx (µg/m³)", 0.0, 300.0, 30.0)
        with col2b:
            so2 = st.slider("SO2 (µg/m³)", 0.0, 300.0, 10.0)
            o3 = st.slider("O3 (µg/m³)", 0.0, 400.0, 20.0)
            co = st.slider("CO (µg/m³)", 0.0, 50.0, 5.0)
        
        nh3 = st.slider("NH3 (µg/m³)", 0.0, 200.0, 10.0)
    
    with tab3:
        st.subheader("🧪 Chemical Compounds")
        benzene = st.slider("Benzene (µg/m³)", 0.0, 50.0, 1.0)
        toluene = st.slider("Toluene (µg/m³)", 0.0, 100.0, 5.0)
        xylene = st.slider("Xylene (µg/m³)", 0.0, 100.0, 3.0)

with col2:
    st.header("🚀 Prediction")
    
    # Current parameters summary
    st.subheader("📋 Current Parameters")
    params = {
        'PM2.5': f"{pm25} µg/m³",
        'PM10': f"{pm10} µg/m³", 
        'NO2': f"{no2} µg/m³",
        'O3': f"{o3} µg/m³",
        'SO2': f"{so2} µg/m³",
        'CO': f"{co} µg/m³"
    }
    
    for param, value in params.items():
        st.write(f"**{param}:** {value}")
    
    # Prediction button
    if st.button("🎯 Predict AQI", type="primary", use_container_width=True):
        with st.spinner('Analyzing air quality data...'):
            # Calculate engineered features
            pm_ratio = pm25 / (pm10 + 1e-6)
            no_ratio = nox / (no2 + 1e-6)
            day_of_week = datetime.now().weekday()  # Current day
            
            # Create input dataframe exactly as model expects
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
                # Rolling averages (using current values)
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
            
            # Transform data
            input_df['City'] = le_city.transform(input_df['City'])
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # Make prediction
            prediction_numeric = model.predict(input_df)[0]
            prediction_category = le_aqi_bucket.inverse_transform([prediction_numeric])[0]
            
            # Store in history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'city': city,
                'category': prediction_category,
                'numeric_prediction': prediction_numeric
            })
            
            # Display result
            category_style = aqi_styles.get(prediction_category, aqi_styles['Moderate'])
            
            st.markdown(f"""
            <div style="padding: 2.5rem; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; text-align: center; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                        border: 4px solid {category_style['color']};">
                <h2>🎯 Prediction Result</h2>
                <div style="font-size: 4rem; margin: 1rem 0;">{category_style['emoji']}</div>
                <div style="font-size: 2.5rem; margin: 1rem 0; padding: 1rem; 
                          background: {category_style['color']}; border-radius: 15px; font-weight: bold;">
                    {prediction_category}
                </div>
                <p style="font-size: 1.3rem;">{category_style['description']}</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">
                    📍 {city} | 🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction probabilities if available
            try:
                probabilities = model.predict_proba(input_df)[0]
                st.subheader("📊 Prediction Confidence")
                
                prob_df = pd.DataFrame({
                    'Category': le_aqi_bucket.classes_,
                    'Probability': probabilities
                })
                
                fig = px.bar(prob_df, x='Category', y='Probability', 
                           color='Category', color_discrete_map={
                               cat: aqi_styles.get(cat, {}).get('color', '#666666') 
                               for cat in le_aqi_bucket.classes_
                           })
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("Confidence scores not available for this model")

# Prediction History
if st.session_state.prediction_history:
    st.header("📈 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show recent predictions
    st.dataframe(display_df[['timestamp', 'city', 'category']].tail(5), use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = history_df['category'].value_counts()
        fig_pie = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            title="Prediction Distribution",
            color=category_counts.index,
            color_discrete_map={cat: aqi_styles.get(cat, {}).get('color', '#666666') 
                              for cat in category_counts.index}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Trend chart
        if len(history_df) > 1:
            # Convert categories to numeric for trend line
            category_to_num = {cat: i for i, cat in enumerate(le_aqi_bucket.classes_)}
            history_df['level'] = history_df['category'].map(category_to_num)
            
            fig_trend = px.line(history_df, x='timestamp', y='level', color='city',
                              title="AQI Trend Over Time", markers=True)
            fig_trend.update_yaxes(tickvals=list(range(len(le_aqi_bucket.classes_))),
                                 ticktext=list(le_aqi_bucket.classes_))
            st.plotly_chart(fig_trend, use_container_width=True)

# AQI Guide
st.sidebar.header("📚 AQI Categories Guide")
for category in le_aqi_bucket.classes_:
    style = aqi_styles.get(category, {'color': '#666666', 'emoji': '❓', 'description': 'Unknown'})
    st.sidebar.markdown(
        f"<div style='background: {style['color']}; padding: 10px; border-radius: 8px; "
        f"color: {'black' if category in ['Moderate'] else 'white'}; margin: 5px 0; text-align: center;'>"
        f"<strong>{style['emoji']} {category}</strong><br>"
        f"<small>{style['description']}</small>"
        f"</div>", 
        unsafe_allow_html=True
    )

# Model Info
st.sidebar.header("🤖 Model Info")
st.sidebar.write(f"**Model:** Stacking Ensemble")
st.sidebar.write(f"**Base Models:** 5")
st.sidebar.write(f"**Cities:** {len(le_city.classes_)}")
st.sidebar.write(f"**Features:** {len(numeric_cols)}")

# Footer
st.markdown("---")
st.markdown("**Air Quality Index Prediction System** | Powered by Machine Learning")
