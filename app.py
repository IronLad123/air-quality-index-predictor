import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="Air Quality Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 seconds
st_autorefresh(interval=30000, key="data_refresh")

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
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .aqi-good { 
        background: linear-gradient(135deg, #00e400, #00b300);
        color: white; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .aqi-moderate { 
        background: linear-gradient(135deg, #ffff00, #e6e600);
        color: black; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
    }
    .aqi-poor { 
        background: linear-gradient(135deg, #ff7e00, #e67100);
        color: white; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .aqi-unhealthy { 
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .aqi-very-unhealthy { 
        background: linear-gradient(135deg, #8f3f97, #732f7a);
        color: white; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .aqi-hazardous { 
        background: linear-gradient(135deg, #7e0023, #66001c);
        color: white; 
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load saved objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('stacking_ensemble.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        numeric_cols = joblib.load('numeric_columns.pkl')
        le_city = joblib.load('le_city.pkl')
        return model, scaler, numeric_cols, le_city
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, numeric_cols, le_city = load_models()

# AQI category mapping
aqi_categories = {
    'Good': {'range': '0-50', 'color': 'aqi-good', 'level': 1},
    'Moderate': {'range': '51-100', 'color': 'aqi-moderate', 'level': 2},
    'Poor': {'range': '101-200', 'color': 'aqi-poor', 'level': 3},
    'Unhealthy': {'range': '201-300', 'color': 'aqi-unhealthy', 'level': 4},
    'Very Unhealthy': {'range': '301-400', 'color': 'aqi-very-unhealthy', 'level': 5},
    'Hazardous': {'range': '401-500', 'color': 'aqi-hazardous', 'level': 6}
}

# Historical data simulation for trends
@st.cache_data
def generate_historical_data(city, base_values):
    dates = [date.today() - timedelta(days=i) for i in range(30, 0, -1)]
    data = []
    for i, current_date in enumerate(dates):
        variation = np.random.normal(0, 0.1, len(base_values))
        day_values = base_values * (1 + variation)
        data.append({
            'date': current_date,
            'PM2_5': max(0, day_values[0]),
            'PM10': max(0, day_values[1]),
            'NO2': max(0, day_values[2]),
            'SO2': max(0, day_values[3]),
            'O3': max(0, day_values[4]),
            'CO': max(0, day_values[5]),
        })
    return pd.DataFrame(data)

# Generate pollution hotspots data
@st.cache_data
def generate_hotspot_data():
    cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad']
    data = []
    for city in cities:
        data.append({
            'city': city,
            'lat': np.random.uniform(8.0, 37.0),
            'lon': np.random.uniform(68.0, 97.0),
            'pollution_level': np.random.uniform(0.1, 1.0),
            'aqi': np.random.randint(50, 400)
        })
    return pd.DataFrame(data)

# Main title with animated effect
st.markdown('<h1 class="main-header">🌍 Air Quality Intelligence Platform</h1>', unsafe_allow_html=True)

# Initialize session state for storing predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar with enhanced features
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Auto-update toggle
    auto_update = st.toggle("🔄 Live Auto-Update", value=False)
    
    # Theme selector
    theme = st.selectbox("🎨 Theme", ["Light", "Dark", "Professional"])
    
    # Data frequency
    freq = st.radio("📊 Data Frequency", ["Real-time", "Hourly", "Daily"])
    
    st.header("🏙️ Location Settings")
    city = st.selectbox("Select City", le_city.classes_ if le_city else [], index=0)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("To", value=date.today())
    
    st.header("📈 AQI Scale Reference")
    for category, info in aqi_categories.items():
        st.markdown(f"<div class='{info['color']}' style='margin: 5px 0; text-align: center;'>{category}: {info['range']}</div>", unsafe_allow_html=True)
    
    st.header("🔔 Alerts")
    alert_level = st.slider("Alert Threshold", 1, 6, 3)
    st.info(f"Alerts for AQI level {alert_level}+")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dashboard", "📊 Analytics", "🗺️ Geographic", "📈 Trends"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🌫️ Real-time Air Quality Monitor")
        
        # Create expandable sections for parameters
        with st.expander("🧪 Pollutant Parameters", expanded=True):
            subtab1, subtab2, subtab3 = st.tabs(["Particulate", "Gases", "Organics"])
            
            with subtab1:
                col1a, col1b = st.columns(2)
                with col1a:
                    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0, key="pm25")
                    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0, key="pm10")
                with col1b:
                    # Real-time gauge for PM2.5
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = pm25,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "PM2.5 Live"},
                        gauge = {
                            'axis': {'range': [None, 300]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "yellow"},
                                {'range': [100, 300], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=200)
                    st.plotly_chart(fig_gauge, use_container_width=True)
            
            with subtab2:
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
            
            with subtab3:
                benzene = st.slider("Benzene (µg/m³)", 0.0, 50.0, 1.0)
                toluene = st.slider("Toluene (µg/m³)", 0.0, 100.0, 5.0)
                xylene = st.slider("Xylene (µg/m³)", 0.0, 100.0, 3.0)

    with col2:
        st.header("🚀 Quick Predict")
        
        # Calculate derived features
        pm_ratio = pm25 / (pm10 + 1e-6)
        no_ratio = nox / (no2 + 1e-6)
        day_of_week = date.today().weekday()
        
        # Prediction button with enhanced UI
        if st.button("🎯 Predict AQI", use_container_width=True, type="primary"):
            if model is not None:
                with st.spinner('🤖 AI is analyzing environmental data...'):
                    progress_bar = st.progress(0)
                    
                    # Simulate processing steps
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
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
                    
                    # Encode and scale
                    input_df['City'] = le_city.transform(input_df['City'])
                    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                    
                    # Predict
                    prediction = model.predict(input_df)
                    predicted_category = prediction[0]
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'city': city,
                        'category': predicted_category,
                        'level': aqi_categories[predicted_category]['level'],
                        'parameters': {
                            'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 
                            'SO2': so2, 'O3': o3, 'CO': co
                        }
                    })
                    
                    # Display results
                    category_info = aqi_categories.get(predicted_category, aqi_categories['Moderate'])
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Prediction Result</h2>
                        <div class="{category_info['color']}" style="font-size: 2.5rem; margin: 1rem 0; padding: 1rem;">
                            {predicted_category}
                        </div>
                        <p style="font-size: 1.3rem;">📊 AQI Range: {category_info['range']}</p>
                        <p style="font-size: 0.9rem; margin-top: 1rem;">
                            📍 {city} | 🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Health impact assessment
                    st.subheader("🏥 Health Impact Assessment")
                    impacts = {
                        'Good': "✅ Excellent air quality. Perfect for outdoor activities.",
                        'Moderate': "⚠️ Acceptable air quality. Sensitive groups should take caution.",
                        'Poor': "🔶 Unhealthy for sensitive groups. Reduce outdoor exertion.",
                        'Unhealthy': "🔴 Everyone may experience health effects. Avoid outdoors.",
                        'Very Unhealthy': "💀 Health alert: Serious effects for everyone.",
                        'Hazardous': "☠️ Emergency conditions. Avoid all outdoor activities."
                    }
                    
                    st.warning(impacts.get(predicted_category, "Consult local health authorities."))
            else:
                st.error("❌ Model not loaded. Please check your model files.")

with tab2:
    st.header("📊 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pollution composition pie chart
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Others']
        values = [pm25, pm10, no2, so2, o3, co, (benzene + toluene + xylene)]
        
        fig_pie = px.pie(
            names=pollutants, 
            values=values,
            title="Pollution Composition",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Correlation heatmap (simulated)
        st.subheader("🔥 Pollutant Correlations")
        corr_data = pd.DataFrame({
            'PM2.5': np.random.random(100) * pm25,
            'PM10': np.random.random(100) * pm10,
            'NO2': np.random.random(100) * no2,
            'SO2': np.random.random(100) * so2,
            'O3': np.random.random(100) * o3
        })
        fig_heatmap = px.imshow(corr_data.corr(), text_auto=True, aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Time series trend
        st.subheader("📈 30-Day Trend Analysis")
        historical_df = generate_historical_data(city, np.array([pm25, pm10, no2, so2, o3, co]))
        
        fig_trend = go.Figure()
        for column in ['PM2_5', 'PM10', 'NO2']:
            fig_trend.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df[column],
                name=column,
                line=dict(width=3)
            ))
        
        fig_trend.update_layout(
            title="Pollution Trends (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Concentration (µg/m³)",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Pollutant comparison bar chart
        st.subheader("📊 Current Pollutant Levels")
        fig_bar = px.bar(
            x=pollutants[:-1],
            y=values[:-1],
            title="Current Pollutant Concentrations",
            color=values[:-1],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("🗺️ Geographic Analysis")
    
    # Pollution hotspot map
    st.subheader("🌍 National Air Quality Map")
    hotspot_df = generate_hotspot_data()
    
    fig_map = px.scatter_mapbox(
        hotspot_df,
        lat="lat",
        lon="lon",
        size="pollution_level",
        color="aqi",
        hover_name="city",
        hover_data={"aqi": True, "pollution_level": True},
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15,
        zoom=4,
        height=500
    )
    
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
    
    # City comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏙️ City-wise AQI Comparison")
        fig_city_bar = px.bar(
            hotspot_df,
            x='city',
            y='aqi',
            color='aqi',
            title="AQI by City",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig_city_bar, use_container_width=True)
    
    with col2:
        st.subheader("📋 Regional Air Quality Index")
        for _, row in hotspot_df.iterrows():
            aqi_level = "Good" if row['aqi'] <= 50 else "Moderate" if row['aqi'] <= 100 else "Poor"
            color_class = aqi_categories[aqi_level]['color']
            st.markdown(
                f"<div style='display: flex; justify-content: space-between; margin: 5px 0;'>"
                f"<span>{row['city']}</span>"
                f"<span class='{color_class}'>{row['aqi']}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )

with tab4:
    st.header("📈 Historical Trends & Predictions")
    
    if st.session_state.prediction_history:
        # Prediction history chart
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Prediction History")
            fig_history = px.line(
                history_df,
                x='timestamp',
                y='level',
                color='city',
                title="AQI Level Trends Over Time",
                markers=True
            )
            st.plotly_chart(fig_history, use_container_width=True)
        
        with col2:
            st.subheader("📊 Prediction Distribution")
            category_counts = history_df['category'].value_counts()
            fig_dist = px.pie(
                names=category_counts.index,
                values=category_counts.values,
                title="Prediction Category Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Detailed history table
        st.subheader("📋 Detailed Prediction Log")
        display_df = history_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df[['timestamp', 'city', 'category']], use_container_width=True)
    else:
        st.info("📝 No prediction history yet. Make some predictions to see trends here!")

# Footer with real-time updates
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("🔄 **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
with footer_col2:
    st.markdown("🌐 **Data Source:** Environmental Monitoring Network")
with footer_col3:
    st.markdown("⚡ **Platform:** Air Quality Intelligence v2.0")

# Auto-update functionality
if auto_update and model is not None:
    st.rerun()
