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
# Import new visualization modules
from src.visualization.geospatial.integrated_visualizer import IntegratedNetworkVisualizer
from src.visualization.geospatial.geojson_generator import GeoJSONNetworkGenerator
from src.visualization.geospatial.topojson_converter import TopoJSONConverter
from src.visualization.geospatial.stl_3d_generator import STL3DNetworkGenerator
import redis
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import traceback

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

@dataclass
class AlertConfig:
    """Configuration for real-time alerts."""
    throughput_threshold: float = 50000  # kbps
    latency_threshold: float = 10  # ms
    cpu_threshold: float = 80  # %
    energy_threshold: float = 75  # W
    alert_cooldown: int = 300  # seconds

class RealTimeDataManager:
    """Manages real-time data connections and caching."""
    
    def __init__(self):
        self.redis_client = None
        self.websocket_url = "ws://localhost:8000/ws"
        self.cache_ttl = 30  # seconds
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
        except (redis.ConnectionError, redis.ResponseError):
            logging.warning("Redis not available, using memory cache")
            
    def get_cached_data(self, key: str) -> Optional[Dict]:
        """Get data from cache."""
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            except Exception:
                return None
        return None
        
    def set_cached_data(self, key: str, data: Dict, ttl: int = None):
        """Set data in cache."""
        if self.redis_client:
            try:
                ttl = ttl or self.cache_ttl
                self.redis_client.setex(key, ttl, json.dumps(data))
            except Exception:
                pass

class AlertManager:
    """Manages real-time alerts and notifications."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.last_alerts = {}
        
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        current_time = datetime.now()
        
        # Throughput alert
        if metrics.get('dl_throughput_mbps', 0) * 1000 < self.config.throughput_threshold:
            if self._should_alert('throughput', current_time):
                alerts.append({
                    'type': 'warning',
                    'metric': 'throughput',
                    'value': metrics['dl_throughput_mbps'] * 1000,
                    'threshold': self.config.throughput_threshold,
                    'message': f"Low throughput detected: {metrics['dl_throughput_mbps']:.1f} Mbps"
                })
                
        # Latency alert
        if metrics.get('latency_ms', 0) > self.config.latency_threshold:
            if self._should_alert('latency', current_time):
                alerts.append({
                    'type': 'error',
                    'metric': 'latency',
                    'value': metrics['latency_ms'],
                    'threshold': self.config.latency_threshold,
                    'message': f"High latency detected: {metrics['latency_ms']:.1f} ms"
                })
                
        # CPU alert
        if metrics.get('cpu_utilization', 0) > self.config.cpu_threshold:
            if self._should_alert('cpu', current_time):
                alerts.append({
                    'type': 'warning',
                    'metric': 'cpu_utilization',
                    'value': metrics['cpu_utilization'],
                    'threshold': self.config.cpu_threshold,
                    'message': f"High CPU usage: {metrics['cpu_utilization']:.1f}%"
                })
                
        return alerts
        
    def _should_alert(self, metric: str, current_time: datetime) -> bool:
        """Check if enough time has passed since last alert."""
        last_alert_time = self.last_alerts.get(metric)
        if not last_alert_time:
            self.last_alerts[metric] = current_time
            return True
            
        time_diff = (current_time - last_alert_time).total_seconds()
        if time_diff >= self.config.alert_cooldown:
            self.last_alerts[metric] = current_time
            return True
            
        return False

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

def create_advanced_ai_panel():
    """Create advanced AI features panel."""
    st.subheader("🧠 Advanced AI Features")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecasting", "🔍 Anomaly Detection", "🎯 Multi-Objective Optimization", "🌐 Federated Learning"])
    
    with tab1:
        st.markdown("#### 📈 AI-Powered Network Forecasting")
        
        # Forecasting controls
        col1, col2 = st.columns(2)
        with col1:
            forecast_horizon = st.selectbox("Forecast Horizon", ["15 minutes", "1 hour", "6 hours", "24 hours"])
            forecast_metrics = st.multiselect("Metrics to Forecast", 
                                           ["Throughput", "Latency", "Energy Usage", "User Count"],
                                           default=["Throughput", "Latency"])
        
        with col2:
            model_type = st.selectbox("AI Model", ["Transformer", "LSTM", "Prophet", "Ensemble"])
            confidence_interval = st.slider("Confidence Interval", 80, 99, 95)
        
        # Generate sample forecast
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Running AI forecast models..."):
                time.sleep(2)  # Simulate processing
                
                # Create forecast visualization
                future_time = pd.date_range(start=datetime.now(), periods=30, freq='15min')
                forecast_data = 80 + 10 * np.sin(np.arange(30) * 0.2) + np.random.normal(0, 3, 30)
                upper_bound = forecast_data + 5
                lower_bound = forecast_data - 5
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=future_time, y=forecast_data, name='Forecast', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=future_time, y=upper_bound, fill=None, mode='lines', line_color='rgba(0,100,80,0)', showlegend=False))
                fig.add_trace(go.Scatter(x=future_time, y=lower_bound, fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)', name='Confidence Interval'))
                
                fig.update_layout(title="Network Throughput Forecast", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### 🚨 Real-time Anomaly Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            sensitivity = st.slider("Detection Sensitivity", 1, 10, 7)
            detection_window = st.selectbox("Detection Window", ["1 minute", "5 minutes", "15 minutes"])
        
        with col2:
            anomaly_types = st.multiselect("Anomaly Types", 
                                         ["Performance", "Security", "Resource", "Network"],
                                         default=["Performance", "Security"])
        
        # Anomaly detection results
        st.markdown("##### Recent Anomalies Detected:")
        anomalies = [
            {"timestamp": "2025-07-03 14:23:15", "type": "Performance", "severity": "Medium", "description": "Unusual latency spike in Cell_07"},
            {"timestamp": "2025-07-03 14:18:42", "type": "Resource", "severity": "Low", "description": "Memory usage pattern deviation"},
            {"timestamp": "2025-07-03 14:12:08", "type": "Network", "severity": "High", "description": "Suspicious traffic pattern detected"}
        ]
        
        for anomaly in anomalies:
            severity_color = {"Low": "🟡", "Medium": "🟠", "High": "🔴"}
            st.markdown(f"{severity_color[anomaly['severity']]} **{anomaly['type']}** | {anomaly['timestamp']} | {anomaly['description']}")
    
    with tab3:
        st.markdown("#### ⚖️ Multi-Objective Optimization")
        
        st.markdown("Configure optimization objectives and their weights:")
        
        col1, col2 = st.columns(2)
        with col1:
            throughput_weight = st.slider("Throughput Maximization", 0.0, 1.0, 0.3)
            latency_weight = st.slider("Latency Minimization", 0.0, 1.0, 0.3)
            energy_weight = st.slider("Energy Efficiency", 0.0, 1.0, 0.2)
        
        with col2:
            reliability_weight = st.slider("Reliability", 0.0, 1.0, 0.1)
            user_satisfaction_weight = st.slider("User Satisfaction", 0.0, 1.0, 0.1)
            
        # Pareto front visualization
        if st.button("🎯 Run Multi-Objective Optimization"):
            with st.spinner("Computing Pareto-optimal solutions..."):
                time.sleep(3)
                
                # Generate sample Pareto front
                n_solutions = 50
                throughput_vals = np.random.uniform(50, 100, n_solutions)
                energy_vals = 120 - throughput_vals + np.random.normal(0, 5, n_solutions)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=throughput_vals, y=energy_vals, mode='markers',
                                       marker=dict(size=8, color='red'), name='Pareto Solutions'))
                fig.update_layout(title="Pareto Front: Throughput vs Energy Efficiency",
                                xaxis_title="Throughput (Mbps)", yaxis_title="Energy Efficiency",
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### 🌐 Federated Learning Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Connected Sites", "12", "↑2")
            st.metric("Global Model Version", "v2.3.1", "Updated 2h ago")
            st.metric("Training Round", "847", "↑1")
        
        with col2:
            st.metric("Model Accuracy", "94.2%", "↑0.3%")
            st.metric("Convergence Status", "Stable", "")
            st.metric("Data Privacy", "✅ Secured", "")
        
        # Federated learning participants
        participants = pd.DataFrame({
            'Site': [f'Site_{i:02d}' for i in range(1, 13)],
            'Data Samples': np.random.randint(1000, 5000, 12),
            'Model Accuracy': np.random.uniform(90, 96, 12),
            'Last Update': [f"{np.random.randint(1, 24)}h ago" for _ in range(12)]
        })
        
        st.markdown("##### Federated Learning Participants:")
        st.dataframe(participants, use_container_width=True)

def create_security_panel():
    """Create security monitoring panel."""
    st.subheader("🔒 Security & Compliance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🛡️ Security Status")
        st.metric("Security Score", "98.7%", "↑0.2%")
        st.markdown("- ✅ All endpoints secured")
        st.markdown("- ✅ Encryption active")
        st.markdown("- ✅ Access control enforced")
        st.markdown("- ⚠️ 2 pending updates")
    
    with col2:
        st.markdown("#### 📋 Compliance")
        st.metric("Compliance Score", "100%", "")
        st.markdown("- ✅ 3GPP Standards")
        st.markdown("- ✅ GDPR Compliant")
        st.markdown("- ✅ ISO 27001")
        st.markdown("- ✅ NIST Framework")
    
    with col3:
        st.markdown("#### 🚨 Recent Security Events")
        events = [
            "✅ Successful authentication: Admin",
            "⚠️ Failed login attempt blocked",
            "✅ Certificate renewed: api.5g-oran.com",
            "🔍 Security scan completed"
        ]
        for event in events:
            st.markdown(f"- {event}")

def create_geospatial_3d_visualization(data):
    """Create geospatial and 3D network visualization section"""
    st.subheader("🌍 Geospatial & 3D Network Visualization")
    
    # Create tabs for different visualization types
    geo_tab1, geo_tab2, geo_tab3, geo_tab4 = st.tabs(["📍 Network Map", "🌐 3D Topology", "📊 Interactive GeoJSON", "🎯 Coverage Analysis"])
    
    with geo_tab1:
        st.markdown("### Real-time Network Coverage Map")
        
        # Initialize the integrated visualizer
        try:
            visualizer = IntegratedNetworkVisualizer()
            
            # Create sample network data from current dashboard data
            network_data = {
                'base_stations': [
                    {
                        'id': f'BS_{i}',
                        'lat': 40.7589 + np.random.uniform(-0.1, 0.1),
                        'lon': -73.9851 + np.random.uniform(-0.1, 0.1),
                        'type': np.random.choice(['macro', 'micro', 'pico']),
                        'throughput': data.get('dl_throughput_mbps', 100) + np.random.uniform(-20, 20),
                        'coverage_radius': np.random.uniform(0.5, 3.0),
                        'status': np.random.choice(['active', 'maintenance', 'overloaded'])
                    }
                    for i in range(15)
                ],
                'edges': [],
                'coverage_areas': []
            }
            
            # Generate edges between nearby base stations
            for i, bs1 in enumerate(network_data['base_stations']):
                for j, bs2 in enumerate(network_data['base_stations'][i+1:], i+1):
                    if np.random.random() < 0.3:  # 30% chance of connection
                        network_data['edges'].append({
                            'source': bs1['id'],
                            'target': bs2['id'],
                            'bandwidth': np.random.uniform(100, 1000),
                            'latency': np.random.uniform(1, 15)
                        })
            
            # Create GeoJSON visualization
            geojson_gen = GeoJSONNetworkGenerator()
            geojson_data = geojson_gen.create_network_geojson(
                network_data['base_stations'],
                network_data['edges']
            )
            
            # Display the map using Plotly
            import plotly.express as px
            
            # Create a simple scatter plot on map
            df_bs = pd.DataFrame(network_data['base_stations'])
            
            fig = px.scatter_mapbox(
                df_bs,
                lat="lat",
                lon="lon",
                color="type",
                size="throughput",
                hover_data=["id", "status", "coverage_radius"],
                color_discrete_map={
                    "macro": "red",
                    "micro": "blue", 
                    "pico": "green"
                },
                zoom=10,
                height=500,
                title="5G Network Base Stations"
            )
            
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show GeoJSON data
            with st.expander("📋 View GeoJSON Data"):
                st.json(geojson_data)
                
        except Exception as e:
            st.error(f"Error creating network map: {str(e)}")
            st.info("Make sure all visualization dependencies are installed.")
    
    with geo_tab2:
        st.markdown("### 3D Network Topology")
        
        try:
            # Create 3D visualization
            fig_3d = go.Figure()
            
            # Add base stations as 3D scatter points
            df_bs = pd.DataFrame(network_data['base_stations'])
            
            fig_3d.add_trace(go.Scatter3d(
                x=df_bs['lon'],
                y=df_bs['lat'],
                z=[np.random.uniform(10, 100) for _ in range(len(df_bs))],  # Height/elevation
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=df_bs['throughput'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Throughput (Mbps)")
                ),
                text=df_bs['id'],
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>" +
                             "Lat: %{y}<br>" +
                             "Lon: %{x}<br>" +
                             "Height: %{z} m<br>" +
                             "<extra></extra>",
                name="Base Stations"
            ))
            
            # Add connections as lines
            for edge in network_data['edges'][:10]:  # Limit to first 10 for performance
                source_bs = next(bs for bs in network_data['base_stations'] if bs['id'] == edge['source'])
                target_bs = next(bs for bs in network_data['base_stations'] if bs['id'] == edge['target'])
                
                fig_3d.add_trace(go.Scatter3d(
                    x=[source_bs['lon'], target_bs['lon']],
                    y=[source_bs['lat'], target_bs['lat']],
                    z=[50, 50],  # Fixed height for connections
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.6)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_3d.update_layout(
                title="3D Network Topology View",
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude", 
                    zaxis_title="Height (m)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # STL generation option
            if st.button("🔧 Generate 3D STL Model"):
                try:
                    stl_gen = STL3DNetworkGenerator()
                    stl_file = stl_gen.create_network_stl(network_data['base_stations'])
                    st.success(f"STL model generated: {stl_file}")
                    st.info("STL file can be used for 3D printing or CAD applications")
                except Exception as e:
                    st.error(f"Error generating STL: {str(e)}")
            
        except Exception as e:
            st.error(f"Error creating 3D visualization: {str(e)}")
    
    with geo_tab3:
        st.markdown("### Interactive GeoJSON Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Network Statistics")
            if 'network_data' in locals():
                st.metric("Total Base Stations", len(network_data['base_stations']))
                st.metric("Network Connections", len(network_data['edges']))
                
                # Base station type distribution
                type_counts = pd.Series([bs['type'] for bs in network_data['base_stations']]).value_counts()
                
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Base Station Types"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### Coverage Analysis")
            if 'network_data' in locals():
                # Throughput distribution
                throughputs = [bs['throughput'] for bs in network_data['base_stations']]
                
                fig_hist = px.histogram(
                    x=throughputs,
                    nbins=10,
                    title="Throughput Distribution",
                    labels={'x': 'Throughput (Mbps)', 'y': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Status overview
                status_counts = pd.Series([bs['status'] for bs in network_data['base_stations']]).value_counts()
                st.bar_chart(status_counts)
    
    with geo_tab4:
        st.markdown("### Coverage Analysis & Optimization")
        
        # Coverage heatmap
        try:
            # Create a grid for coverage analysis
            lat_range = np.linspace(40.65, 40.85, 20)
            lon_range = np.linspace(-74.1, -73.85, 20)
            
            coverage_grid = []
            for lat in lat_range:
                for lon in lon_range:
                    # Calculate coverage score based on distance to nearest base station
                    min_distance = float('inf')
                    for bs in network_data['base_stations']:
                        distance = np.sqrt((lat - bs['lat'])**2 + (lon - bs['lon'])**2)
                        min_distance = min(min_distance, distance)
                    
                    # Convert distance to coverage score (closer = better coverage)
                    coverage_score = max(0, 100 * (1 - min_distance / 0.1))
                    coverage_grid.append({'lat': lat, 'lon': lon, 'coverage': coverage_score})
            
            df_coverage = pd.DataFrame(coverage_grid)
            
            fig_heatmap = px.density_mapbox(
                df_coverage,
                lat='lat',
                lon='lon',
                z='coverage',
                radius=10,
                center=dict(lat=40.75, lon=-73.95),
                zoom=10,
                mapbox_style="open-street-map",
                title="Network Coverage Heatmap",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Coverage optimization suggestions
            st.markdown("#### 🎯 Coverage Optimization Recommendations")
            
            low_coverage_areas = df_coverage[df_coverage['coverage'] < 30]
            if len(low_coverage_areas) > 0:
                st.warning(f"Found {len(low_coverage_areas)} areas with low coverage (<30%)")
                
                # Suggest new base station locations
                if st.button("🔍 Suggest Optimization"):
                    # Find centroid of low coverage areas
                    centroid_lat = low_coverage_areas['lat'].mean()
                    centroid_lon = low_coverage_areas['lon'].mean()
                    
                    st.success(f"Recommended new base station location:")
                    st.write(f"📍 Latitude: {centroid_lat:.4f}")
                    st.write(f"📍 Longitude: {centroid_lon:.4f}")
                    st.write(f"🎯 Expected coverage improvement: +25%")
            else:
                st.success("✅ Good coverage across the entire network area!")
                
        except Exception as e:
            st.error(f"Error in coverage analysis: {str(e)}")

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
    show_geospatial = st.sidebar.checkbox("Show Geospatial Visualization", value=True)
    
    # Auto-refresh
    placeholder = st.empty()
    
    with placeholder.container():
        # Main KPI dashboard
        current_data = create_kpi_dashboard()
        
        # Network slicing view
        create_network_slicing_view(current_data)
        
        # Advanced visualizations
        create_advanced_visualizations(current_data)
        
        # Geospatial and 3D visualization
        if show_geospatial:
            create_geospatial_3d_visualization(current_data)
        
        # AI insights
        create_ai_insights_panel(current_data)
        
        # Optimization controls
        create_optimization_controls()
        
        # Advanced AI panel
        create_advanced_ai_panel()
        
        # Security panel
        create_security_panel()
        
        # Geospatial and 3D visualizations
        create_geospatial_3d_visualization(current_data)
        
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
