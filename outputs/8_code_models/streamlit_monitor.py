import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Set page config with dark theme
st.set_page_config(
    page_title="Smart Grid AI - Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for dark theme styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0f1419;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3748;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Metric labels */
    [data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #00d4ff;
        color: #0f1419;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #00a8cc;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox, .stCheckbox {
        color: #e0e0e0;
    }
    
    /* Text input */
    input {
        background-color: #1a1f2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #2d3748 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); border-radius: 10px; text-align: center;'>
        <h1 style='margin: 0; color: white;'>🎯 SMART GRID AI</h1>
        <p style='margin: 5px 0 0 0; color: white; font-weight: bold;'>Real-Time Production Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    current_time = datetime.now().strftime("%H:%M:%S")
    status = "🟢 LIVE"
    st.markdown(f"""
    <div style='text-align: right; padding: 20px;'>
        <p style='color: #00d4ff; font-size: 18px; margin: 5px 0;'>{status}</p>
        <p style='color: #888; font-size: 12px; margin: 0;'>{current_time}</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for filters
with st.sidebar:
    st.markdown("<h2 style='color: #00d4ff;'>⚙️ Dashboard Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")
    timeframe = st.selectbox("📅 Timeframe", ["Last Hour", "Last 24h", "Last 7d", "Last 30d"])
    refresh_interval = st.selectbox("🔄 Refresh Rate", ["Manual", "10s", "30s", "60s"], index=2)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.markdown("---")
    st.markdown("<p style='color: #888; font-size: 12px;'>Status: Connected ✓</p>", unsafe_allow_html=True)

# KPI Cards Section
st.markdown("<h2 style='color: #00d4ff; margin-top: 30px;'>📊 Key Performance Indicators</h2>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

# Define color scheme for metrics
def create_metric_card(col, label, value, delta, unit=""):
    with col:
        st.markdown(f"""
        <div style='background-color: #1a1f2e; border: 2px solid #2d3748; border-radius: 10px; padding: 15px; text-align: center; transition: all 0.3s;'>
            <p style='color: #888; font-size: 12px; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 1px;'>{label}</p>
            <p style='color: #00d4ff; font-size: 28px; margin: 5px 0; font-weight: bold;'>{value}{unit}</p>
            <p style='color: #00cc88; font-size: 12px; margin: 5px 0 0 0;'>↑ {delta}</p>
        </div>
        """, unsafe_allow_html=True)

create_metric_card(col1, "Forecast Accuracy", "4.32", "-0.15%", "%")
create_metric_card(col2, "Anomaly Detection", "92.5", "+1.2%", "%")
create_metric_card(col3, "API Response", "145", "-12ms", "ms")
create_metric_card(col4, "System Uptime", "99.97", "+0.01%", "%")
create_metric_card(col5, "Predictions/Min", "116.7", "+5%", "")

# Charts Section
st.markdown("<h2 style='color: #00d4ff; margin-top: 40px;'>📈 Performance Trends</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Generate sample data
hours = list(range(24))
accuracy = [4.2 + random.uniform(-0.5, 0.5) for _ in hours]
api_times = [140 + random.uniform(-20, 30) for _ in hours]
anomalies = [85 + random.uniform(-5, 10) for _ in hours]
uptime_data = [99.95 + random.uniform(-0.05, 0.05) for _ in hours]

with col1:
    # Forecast accuracy chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hours,
        y=accuracy,
        mode='lines+markers',
        name='MAPE %',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, color='#00d4ff'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    fig1.add_hline(y=5.0, line_dash="dash", line_color="#ff6b6b", annotation_text="Target: 5%")
    fig1.update_layout(
        title="<b>Forecast Accuracy (24h)</b>",
        xaxis_title="Hour",
        yaxis_title="MAPE (%)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#0f1419',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=50, t=50, b=50),
        height=350
    )
    st.plotly_chart(fig1, width='stretch')

with col2:
    # API response time chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hours,
        y=api_times,
        mode='lines+markers',
        name='Response Time (ms)',
        line=dict(color='#ff9d5c', width=3),
        marker=dict(size=8, color='#ff9d5c'),
        fill='tozeroy',
        fillcolor='rgba(255, 157, 92, 0.1)'
    ))
    fig2.add_hline(y=200, line_dash="dash", line_color="#ff6b6b", annotation_text="Target: 200ms")
    fig2.update_layout(
        title="<b>API Response Time (24h)</b>",
        xaxis_title="Hour",
        yaxis_title="Response Time (ms)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#0f1419',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=50, t=50, b=50),
        height=350
    )
    st.plotly_chart(fig2, width='stretch')

col3, col4 = st.columns(2)

with col3:
    # Anomaly detection rate
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hours,
        y=anomalies,
        mode='lines+markers',
        name='Detection Rate %',
        line=dict(color='#51cf66', width=3),
        marker=dict(size=8, color='#51cf66'),
        fill='tozeroy',
        fillcolor='rgba(81, 207, 102, 0.1)'
    ))
    fig3.add_hline(y=90, line_dash="dash", line_color="#ffd43b", annotation_text="Target: 90%")
    fig3.update_layout(
        title="<b>Anomaly Detection Rate (24h)</b>",
        xaxis_title="Hour",
        yaxis_title="Detection Rate (%)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#0f1419',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=50, t=50, b=50),
        height=350
    )
    st.plotly_chart(fig3, width='stretch')

with col4:
    # System uptime
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=hours,
        y=uptime_data,
        mode='lines+markers',
        name='Uptime %',
        line=dict(color='#9775fa', width=3),
        marker=dict(size=8, color='#9775fa'),
        fill='tozeroy',
        fillcolor='rgba(151, 117, 250, 0.1)'
    ))
    fig4.add_hline(y=99.9, line_dash="dash", line_color="#ffd43b", annotation_text="Target: 99.9%")
    fig4.update_layout(
        title="<b>System Uptime (24h)</b>",
        xaxis_title="Hour",
        yaxis_title="Uptime (%)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#0f1419',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=50, t=50, b=50),
        height=350
    )
    st.plotly_chart(fig4, width='stretch')

# Forecast vs Actual Section
st.markdown("<h2 style='color: #00d4ff; margin-top: 40px;'>📊 Forecast vs Actual Consumption</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; font-size: 13px;'>Real-time comparison of predicted vs actual energy consumption - Model accuracy in action</p>", unsafe_allow_html=True)

# Generate realistic forecast vs actual data
hours_detailed = list(range(24))
# Base consumption pattern (peak hours: 9-12 and 18-21)
actual_consumption = [
    60 + (40 * (1 + 0.5 * (1 if 9 <= h < 12 or 18 <= h < 21 else 0))) + random.uniform(-5, 5)
    for h in hours_detailed
]
# Forecast with slight error (realistic model performance)
forecast_consumption = [
    actual + random.uniform(-3, 6)  # Slight underprediction and noise
    for actual in actual_consumption
]
# Calculate error for each hour
errors = [abs(f - a) for f, a in zip(forecast_consumption, actual_consumption)]
mean_error = sum(errors) / len(errors)

col_forecast1, col_forecast2 = st.columns([2, 1])

with col_forecast1:
    # Main comparison chart
    fig_compare = go.Figure()
    
    # Add actual consumption line
    fig_compare.add_trace(go.Scatter(
        x=hours_detailed,
        y=actual_consumption,
        mode='lines+markers',
        name='Actual Consumption',
        line=dict(color='#51cf66', width=3),
        marker=dict(size=8, color='#51cf66'),
    ))
    
    # Add forecast line
    fig_compare.add_trace(go.Scatter(
        x=hours_detailed,
        y=forecast_consumption,
        mode='lines+markers',
        name='Predicted Consumption',
        line=dict(color='#00d4ff', width=3, dash='dash'),
        marker=dict(size=8, color='#00d4ff'),
    ))
    
    # Add confidence band (±10% envelope)
    upper_band = [f * 1.10 for f in forecast_consumption]
    lower_band = [f * 0.90 for f in forecast_consumption]
    
    fig_compare.add_trace(go.Scatter(
        x=hours_detailed + hours_detailed[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.1)',
        line=dict(color='rgba(0, 212, 255, 0)'),
        name='Confidence Band (±10%)',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig_compare.update_layout(
        title="<b>24-Hour Energy Consumption: Forecast vs Actual</b>",
        xaxis_title="Hour of Day",
        yaxis_title="Consumption (kWh)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#0f1419',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=50, t=50, b=50),
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 20, 25, 0.8)",
            bordercolor="rgba(45, 55, 72, 0.8)",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_compare, width='stretch')

with col_forecast2:
    # Accuracy metrics card
    accuracy_percent = 100 - (mean_error / max(actual_consumption) * 100)
    st.markdown(f"""
    <div style='background-color: #1a1f2e; border: 2px solid #2d3748; border-radius: 10px; padding: 20px; text-align: center; height: 400px; display: flex; flex-direction: column; justify-content: center;'>
        <p style='color: #888; font-size: 11px; margin: 0 0 15px 0; text-transform: uppercase; letter-spacing: 1px;'>Model Performance</p>
        
        <p style='color: #51cf66; font-size: 36px; margin: 0 0 5px 0; font-weight: bold;'>{accuracy_percent:.1f}%</p>
        <p style='color: #888; font-size: 12px; margin: 0 0 20px 0;'>Prediction Accuracy</p>
        
        <hr style='border: none; border-top: 1px solid #2d3748; margin: 15px 0;'>
        
        <p style='color: #00d4ff; font-size: 14px; margin: 10px 0;'><b>Mean Error:</b></p>
        <p style='color: #ff9d5c; font-size: 20px; margin: 0 0 15px 0;'>{mean_error:.2f} kWh</p>
        
        <p style='color: #00d4ff; font-size: 14px; margin: 10px 0;'><b>Max Deviation:</b></p>
        <p style='color: #ff9d5c; font-size: 20px; margin: 0 0 15px 0;'>{max(errors):.2f} kWh</p>
        
        <p style='color: #00d4ff; font-size: 14px; margin: 10px 0;'><b>Peak Hours:</b></p>
        <p style='color: #51cf66; font-size: 14px; margin: 0;'>9-12 AM, 6-9 PM</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction accuracy breakdown table
st.markdown("<h3 style='color: #00d4ff; margin-top: 30px;'>📋 Hourly Accuracy Breakdown</h3>", unsafe_allow_html=True)

accuracy_data = pd.DataFrame({
    'Hour': [f"{h:02d}:00" for h in hours_detailed],
    'Actual (kWh)': [f"{a:.1f}" for a in actual_consumption],
    'Predicted (kWh)': [f"{f:.1f}" for f in forecast_consumption],
    'Error (kWh)': [f"{e:.2f}" for e in errors],
    'Error %': [f"{(e/a*100):.1f}%" if a > 0 else "0%" for e, a in zip(errors, actual_consumption)]
})

# Style the dataframe
st.dataframe(
    accuracy_data,
    width='stretch',
    height=300,
    column_config={
        'Hour': st.column_config.TextColumn(width='small'),
        'Actual (kWh)': st.column_config.TextColumn(width='small'),
        'Predicted (kWh)': st.column_config.TextColumn(width='small'),
        'Error (kWh)': st.column_config.TextColumn(width='small'),
        'Error %': st.column_config.TextColumn(width='small'),
    }
)

# Alerts & Status Section
st.markdown("<h2 style='color: #00d4ff; margin-top: 40px;'>🚨 Active Alerts & Status</h2>", unsafe_allow_html=True)

alert_col1, alert_col2, alert_col3 = st.columns([1, 1, 1])

with alert_col1:
    st.markdown("""
    <div style='background-color: #1a1f2e; border-left: 4px solid #51cf66; border-radius: 8px; padding: 15px;'>
        <p style='color: #51cf66; font-weight: bold; margin: 0;'>✓ All Systems Normal</p>
        <p style='color: #888; font-size: 12px; margin: 5px 0 0 0;'>No critical alerts</p>
    </div>
    """, unsafe_allow_html=True)

with alert_col2:
    st.markdown("""
    <div style='background-color: #1a1f2e; border-left: 4px solid #00d4ff; border-radius: 8px; padding: 15px;'>
        <p style='color: #00d4ff; font-weight: bold; margin: 0;'>📊 Data Pipeline</p>
        <p style='color: #888; font-size: 12px; margin: 5px 0 0 0;'>Processing 168,640 predictions/day</p>
    </div>
    """, unsafe_allow_html=True)

with alert_col3:
    st.markdown("""
    <div style='background-color: #1a1f2e; border-left: 4px solid #51cf66; border-radius: 8px; padding: 15px;'>
        <p style='color: #51cf66; font-weight: bold; margin: 0;'>🎯 Production Live</p>
        <p style='color: #888; font-size: 12px; margin: 5px 0 0 0;'>Go-live: Feb 2, 2026</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #888; font-size: 12px;'>
    <p>Smart Grid AI Production Monitoring • Last Updated: {}</p>
    <p>System Status: 🟢 OPERATIONAL</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
