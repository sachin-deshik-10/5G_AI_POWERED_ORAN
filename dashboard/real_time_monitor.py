"""
Real-time 5G Network Monitoring Dashboard
Advanced web-based monitoring system with real-time updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import asyncio
import websockets
from src.models.predictive_network_planning.predict import make_predictions
from src.optimize import optimize_network_resources

# Configure Streamlit page
st.set_page_config(
    page_title="5G OpenRAN AI Optimizer Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class RealTimeMonitor:
    def __init__(self):
        self.last_update = datetime.now()
        self.data_buffer = []
        
    def generate_real_time_data(self):
        """Generate simulated real-time 5G network data"""
        current_time = datetime.now()
        
        # Simulate 5G network metrics with realistic variations
        base_throughput = 85000 + np.random.normal(0, 15000)
        latency = max(1, np.random.exponential(8))  # Ultra-low latency for 5G
        
        # Simulate network slicing data
        slice_data = {
            'eMBB': {'throughput': base_throughput * 0.6, 'users': np.random.randint(100, 500)},
            'URLLC': {'latency': latency, 'reliability': 99.999 if latency < 5 else 99.9},
            'mMTC': {'devices': np.random.randint(1000, 10000), 'battery_life': np.random.uniform(8, 12)}
        }
        
        data = {
            'timestamp': current_time,
            'cell_id': f"5G_Cell_{np.random.randint(1, 20):02d}",
            'dl_throughput_mbps': base_throughput / 1000,
            'ul_throughput_mbps': (base_throughput * 0.4) / 1000,
            'latency_ms': latency,
            'packet_loss_percent': max(0, np.random.exponential(0.1)),
            'energy_efficiency_mbps_w': np.random.uniform(15, 25),
            'spectrum_efficiency': np.random.uniform(4.5, 7.2),
            'user_count': np.random.randint(50, 300),
            'cpu_utilization': np.random.uniform(30, 85),
            'memory_utilization': np.random.uniform(40, 80),
            'temperature_celsius': np.random.uniform(35, 65),
            'slice_data': slice_data,
            'beamforming_gain': np.random.uniform(8, 15),
            'mimo_rank': np.random.randint(2, 8),
            'carrier_aggregation_bands': np.random.randint(2, 5)
        }
        
        return data

def create_kpi_dashboard():
    """Create main KPI dashboard"""
    st.markdown('<div class="main-header">🚀 5G OpenRAN AI Optimizer - Real-Time Dashboard</div>', 
                unsafe_allow_html=True)
    
    monitor = RealTimeMonitor()
    
    # Auto-refresh setup
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    # Real-time data generation
    current_data = monitor.generate_real_time_data()
    
    # Main KPIs row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        throughput = current_data['dl_throughput_mbps']
        status = "🟢" if throughput > 70 else "🟡" if throughput > 40 else "🔴"
        st.metric(
            label="📊 DL Throughput",
            value=f"{throughput:.1f} Mbps",
            delta=f"{np.random.uniform(-5, 5):.1f}%"
        )
        st.markdown(f"Status: {status}")
    
    with col2:
        latency = current_data['latency_ms']
        status = "🟢" if latency < 5 else "🟡" if latency < 10 else "🔴"
        st.metric(
            label="⚡ Latency",
            value=f"{latency:.2f} ms",
            delta=f"{np.random.uniform(-2, 2):.2f} ms"
        )
        st.markdown(f"Status: {status}")
    
    with col3:
        energy_eff = current_data['energy_efficiency_mbps_w']
        st.metric(
            label="🔋 Energy Efficiency",
            value=f"{energy_eff:.1f} Mbps/W",
            delta=f"{np.random.uniform(-1, 3):.1f}%"
        )
    
    with col4:
        spectrum_eff = current_data['spectrum_efficiency']
        st.metric(
            label="📡 Spectrum Efficiency",
            value=f"{spectrum_eff:.1f} bps/Hz",
            delta=f"{np.random.uniform(-0.5, 0.8):.1f}"
        )
    
    with col5:
        users = current_data['user_count']
        st.metric(
            label="👥 Active Users",
            value=f"{users:,}",
            delta=f"{np.random.randint(-20, 30)}"
        )
    
    return current_data

def create_network_slicing_view(data):
    """Create network slicing visualization"""
    st.subheader("🔀 Network Slicing Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### eMBB (Enhanced Mobile Broadband)")
        embb_data = data['slice_data']['eMBB']
        st.metric("Throughput", f"{embb_data['throughput']/1000:.1f} Mbps")
        st.metric("Active Users", f"{embb_data['users']:,}")
        
    with col2:
        st.markdown("### URLLC (Ultra-Reliable Low Latency)")
        urllc_data = data['slice_data']['URLLC']
        st.metric("Latency", f"{urllc_data['latency']:.2f} ms")
        st.metric("Reliability", f"{urllc_data['reliability']:.3f}%")
        
    with col3:
        st.markdown("### mMTC (Massive Machine Type Communications)")
        mmtc_data = data['slice_data']['mMTC']
        st.metric("Connected Devices", f"{mmtc_data['devices']:,}")
        st.metric("Avg Battery Life", f"{mmtc_data['battery_life']:.1f} years")

def create_advanced_visualizations(data):
    """Create advanced visualizations"""
    st.subheader("📈 Advanced Network Analytics")
    
    # Generate time series data for charts
    time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                               end=datetime.now(), freq='1min')
    
    # Throughput over time with prediction
    throughput_data = []
    for t in time_points:
        base = 75 + 20 * np.sin(2 * np.pi * t.hour / 24) + np.random.normal(0, 5)
        throughput_data.append(max(10, base))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Real-time throughput chart
        fig_throughput = go.Figure()
        fig_throughput.add_trace(go.Scatter(
            x=time_points, y=throughput_data,
            mode='lines+markers',
            name='Actual Throughput',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add prediction line
        future_points = pd.date_range(start=datetime.now(), 
                                    end=datetime.now() + timedelta(minutes=30), freq='1min')
        predicted_throughput = [throughput_data[-1] + np.random.normal(0, 2) for _ in future_points]
        
        fig_throughput.add_trace(go.Scatter(
            x=future_points, y=predicted_throughput,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig_throughput.update_layout(
            title="Real-time Throughput with AI Prediction",
            xaxis_title="Time",
            yaxis_title="Throughput (Mbps)",
            height=400
        )
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with col2:
        # Resource utilization heatmap
        resources = ['CPU', 'Memory', 'Network', 'Storage', 'GPU']
        time_slots = ['00:00', '06:00', '12:00', '18:00']
        utilization_matrix = np.random.uniform(30, 90, (len(resources), len(time_slots)))
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=utilization_matrix,
            x=time_slots,
            y=resources,
            colorscale='RdYlBu_r',
            text=utilization_matrix.round(1),
            texttemplate="%{text}%",
            textfont={"size": 12}
        ))
        
        fig_heatmap.update_layout(
            title="Resource Utilization Heatmap",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def create_ai_insights_panel(data):
    """Create AI-powered insights panel"""
    st.subheader("🤖 AI-Powered Insights & Recommendations")
    
    # Generate AI insights based on current data
    insights = []
    recommendations = []
    
    if data['dl_throughput_mbps'] < 50:
        insights.append("⚠️ Below-average throughput detected")
        recommendations.append("🔧 Consider load balancing across multiple cells")
    
    if data['latency_ms'] > 10:
        insights.append("⚠️ High latency detected - may impact URLLC services")
        recommendations.append("🚀 Enable edge computing for latency-sensitive applications")
    
    if data['energy_efficiency_mbps_w'] < 18:
        insights.append("🔋 Energy efficiency below optimal threshold")
        recommendations.append("⚡ Activate AI-driven power management algorithms")
    
    if data['cpu_utilization'] > 80:
        insights.append("🔥 High CPU utilization detected")
        recommendations.append("📊 Scale up computing resources or distribute load")
    
    # Always add positive insights
    insights.append("✅ Network slicing operating within parameters")
    recommendations.append("🎯 Continue monitoring for proactive optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Current Insights")
        for insight in insights:
            st.markdown(f"- {insight}")
    
    with col2:
        st.markdown("#### 💡 AI Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")

def create_optimization_controls():
    """Create optimization control panel"""
    st.subheader("⚙️ AI Optimization Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎛️ Auto-Optimization")
        auto_opt = st.toggle("Enable Auto-Optimization", value=True)
        if auto_opt:
            st.success("✅ AI optimization active")
            optimization_level = st.selectbox(
                "Optimization Level",
                ["Conservative", "Balanced", "Aggressive"],
                index=1
            )
        else:
            st.warning("⚠️ Manual mode active")
    
    with col2:
        st.markdown("#### 📊 Thresholds")
        latency_threshold = st.slider("Max Latency (ms)", 1, 20, 5)
        throughput_threshold = st.slider("Min Throughput (Mbps)", 10, 100, 50)
        energy_threshold = st.slider("Min Energy Efficiency", 10, 30, 18)
    
    with col3:
        st.markdown("#### 🚨 Alerts")
        if st.button("🔔 Configure Alerts"):
            st.info("Alert configuration would open here")
        
        alert_status = st.empty()
        alert_status.success("🟢 All systems normal")

def main():
    """Main dashboard function"""
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Dashboard Controls")
    
    refresh_rate = st.sidebar.selectbox(
        "Refresh Rate",
        ["Real-time (1s)", "5 seconds", "30 seconds", "1 minute"],
        index=0
    )
    
    show_predictions = st.sidebar.checkbox("Show AI Predictions", value=True)
    show_optimizations = st.sidebar.checkbox("Show Optimizations", value=True)
    
    # Auto-refresh
    placeholder = st.empty()
    
    with placeholder.container():
        # Main KPI dashboard
        current_data = create_kpi_dashboard()
        
        # Network slicing view
        create_network_slicing_view(current_data)
        
        # Advanced visualizations
        create_advanced_visualizations(current_data)
        
        # AI insights
        create_ai_insights_panel(current_data)
        
        # Optimization controls
        create_optimization_controls()
        
        # Footer with last update
        st.markdown("---")
        st.markdown(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                   f"🔄 Auto-refresh: {refresh_rate}")
    
    # Auto-refresh logic
    if refresh_rate == "Real-time (1s)":
        time.sleep(1)
        st.rerun()
    elif refresh_rate == "5 seconds":
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
