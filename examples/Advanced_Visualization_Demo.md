# Advanced 5G Network Visualization Examples

This notebook demonstrates the advanced geospatial and 3D visualization capabilities of the AI-Powered 5G Open RAN Optimizer.

## 📋 Table of Contents

1. [Setup and Dependencies](#setup)
2. [GeoJSON Network Visualization](#geojson)
3. [TopoJSON Topology Analysis](#topojson)
4. [3D STL Model Generation](#stl)
5. [Integrated Visualization Pipeline](#integrated)
6. [Real-world Use Cases](#use-cases)

## 🚀 Setup and Dependencies {#setup}

First, let's install and import the required dependencies:

```python
# Core dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Our custom visualization modules
from src.visualization.geospatial.geojson_generator import GeoJSONNetworkGenerator
from src.visualization.geospatial.topojson_converter import TopoJSONConverter
from src.visualization.geospatial.stl_3d_generator import STL3DNetworkGenerator
from src.visualization.geospatial.integrated_visualizer import IntegratedNetworkVisualizer

# Set up plotting
plt.style.use('seaborn-v0_8')
%matplotlib inline
```

## 🌍 GeoJSON Network Visualization {#geojson}

### Creating a Sample 5G Network

Let's start by creating a realistic 5G network topology for a city area:

```python
# Create sample network data for New York City area
np.random.seed(42)  # For reproducible results

# Base station locations (roughly Manhattan area)
base_stations = []
for i in range(25):
    bs = {
        'id': f'gNB_{i:03d}',
        'lat': 40.7589 + np.random.uniform(-0.05, 0.05),
        'lon': -73.9851 + np.random.uniform(-0.05, 0.05),
        'type': np.random.choice(['macro', 'micro', 'pico'], p=[0.3, 0.5, 0.2]),
        'frequency_band': np.random.choice(['n77', 'n78', 'n79', 'n41']),
        'max_power': np.random.uniform(10, 46),  # dBm
        'antenna_height': np.random.uniform(10, 150),  # meters
        'coverage_radius': np.random.uniform(0.1, 2.0),  # km
        'throughput_mbps': np.random.uniform(50, 1000),
        'connected_users': np.random.randint(10, 500),
        'cpu_utilization': np.random.uniform(20, 95),
        'status': np.random.choice(['active', 'maintenance', 'overloaded'], p=[0.8, 0.1, 0.1])
    }
    base_stations.append(bs)

print(f"Generated {len(base_stations)} base stations")
print(f"Network types: {set(bs['type'] for bs in base_stations)}")
```

### Generating GeoJSON

```python
# Initialize the GeoJSON generator
geojson_gen = GeoJSONNetworkGenerator()

# Create network edges (connections between base stations)
edges = []
for i, bs1 in enumerate(base_stations):
    for j, bs2 in enumerate(base_stations[i+1:], i+1):
        # Calculate distance
        distance = geojson_gen._calculate_distance(
            bs1['lat'], bs1['lon'], bs2['lat'], bs2['lon']
        )
        
        # Connect nearby base stations (within 1.5 km)
        if distance < 1.5 and np.random.random() < 0.4:
            edges.append({
                'source': bs1['id'],
                'target': bs2['id'],
                'distance_km': distance,
                'bandwidth_gbps': np.random.uniform(1, 10),
                'latency_ms': distance * 0.1 + np.random.uniform(0.5, 2.0),
                'utilization': np.random.uniform(10, 90)
            })

print(f"Generated {len(edges)} network connections")

# Generate the GeoJSON
geojson_data = geojson_gen.create_network_geojson(base_stations, edges)

# Save to file
with open('examples/visualization_data/generated_network.geojson', 'w') as f:
    json.dump(geojson_data, f, indent=2)

print("GeoJSON saved to examples/visualization_data/generated_network.geojson")
```

### Visualizing with Plotly

```python
# Create an interactive map visualization
df_bs = pd.DataFrame(base_stations)

# Color mapping for base station types
color_map = {'macro': 'red', 'micro': 'blue', 'pico': 'green'}

fig = px.scatter_mapbox(
    df_bs,
    lat="lat",
    lon="lon",
    color="type",
    size="throughput_mbps",
    hover_data=["id", "frequency_band", "connected_users", "status"],
    color_discrete_map=color_map,
    size_max=20,
    zoom=12,
    height=600,
    title="5G Network Base Stations - Manhattan Area",
    mapbox_style="open-street-map"
)

# Add network connections
for edge in edges[:20]:  # Show first 20 connections for clarity
    source_bs = next(bs for bs in base_stations if bs['id'] == edge['source'])
    target_bs = next(bs for bs in base_stations if bs['id'] == edge['target'])
    
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[source_bs['lon'], target_bs['lon']],
        lat=[source_bs['lat'], target_bs['lat']],
        line=dict(width=2, color='rgba(255,255,255,0.6)'),
        showlegend=False,
        hoverinfo='skip'
    ))

fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.show()
```

## 🌐 TopoJSON Topology Analysis {#topojson}

### Converting to TopoJSON

```python
# Initialize the TopoJSON converter
topojson_conv = TopoJSONConverter()

# Convert our GeoJSON to TopoJSON for more efficient storage and topology analysis
topojson_data = topojson_conv.geojson_to_topojson(geojson_data, quantization=1000)

# Save TopoJSON
with open('examples/visualization_data/generated_topology.topojson', 'w') as f:
    json.dump(topojson_data, f, indent=2)

print("TopoJSON saved to examples/visualization_data/generated_topology.topojson")

# Calculate topology metrics
topology_metrics = topojson_conv.calculate_topology_metrics(base_stations, edges)

print("\n📊 Network Topology Metrics:")
for metric, value in topology_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")
```

### Network Graph Analysis

```python
# Create a network graph analysis
import networkx as nx

# Build NetworkX graph
G = nx.Graph()

# Add nodes
for bs in base_stations:
    G.add_node(bs['id'], **bs)

# Add edges
for edge in edges:
    G.add_edge(edge['source'], edge['target'], **edge)

# Calculate centrality measures
centrality_measures = {
    'betweenness': nx.betweenness_centrality(G),
    'closeness': nx.closeness_centrality(G),
    'degree': nx.degree_centrality(G),
    'eigenvector': nx.eigenvector_centrality(G)
}

# Create centrality visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, (measure, values) in enumerate(centrality_measures.items()):
    ax = axes[i]
    
    # Get positions for consistent layout
    pos = {bs['id']: (bs['lon'], bs['lat']) for bs in base_stations}
    
    # Draw network
    nx.draw(G, pos, ax=ax, 
            node_color=list(values.values()), 
            node_size=100,
            cmap='viridis',
            with_labels=False)
    
    ax.set_title(f'{measure.capitalize()} Centrality')
    ax.set_aspect('equal')

plt.tight_layout()
plt.suptitle('Network Centrality Analysis', y=1.02, fontsize=16)
plt.show()

# Print top nodes by centrality
print("\n🎯 Most Important Nodes (by betweenness centrality):")
top_nodes = sorted(centrality_measures['betweenness'].items(), 
                  key=lambda x: x[1], reverse=True)[:5]
for node, centrality in top_nodes:
    bs_info = next(bs for bs in base_stations if bs['id'] == node)
    print(f"  {node}: {centrality:.3f} (Type: {bs_info['type']}, "
          f"Users: {bs_info['connected_users']})")
```

## 🔧 3D STL Model Generation {#stl}

### Creating 3D Network Models

```python
# Initialize the STL generator
stl_gen = STL3DNetworkGenerator()

# Generate 3D STL model
stl_filename = stl_gen.create_network_stl(
    base_stations,
    output_file="examples/visualization_data/network_3d_model.stl",
    scale_factor=1000  # Scale for better 3D printing
)

print(f"3D STL model saved to: {stl_filename}")

# Create 3D visualization
fig_3d = go.Figure()

# Add base stations as 3D points
df_bs = pd.DataFrame(base_stations)

# Normalize coordinates for 3D display
lon_norm = (df_bs['lon'] - df_bs['lon'].min()) * 1000
lat_norm = (df_bs['lat'] - df_bs['lat'].min()) * 1000
height = df_bs['antenna_height']

fig_3d.add_trace(go.Scatter3d(
    x=lon_norm,
    y=lat_norm,
    z=height,
    mode='markers+text',
    marker=dict(
        size=8,
        color=df_bs['throughput_mbps'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Throughput (Mbps)")
    ),
    text=df_bs['id'],
    textposition="top center",
    hovertemplate="<b>%{text}</b><br>" +
                 "Throughput: %{marker.color:.1f} Mbps<br>" +
                 "Height: %{z:.1f} m<br>" +
                 "Users: " + df_bs['connected_users'].astype(str) + "<br>" +
                 "<extra></extra>",
    name="Base Stations"
))

# Add coverage volumes as wireframes
for i, bs in enumerate(base_stations[:10]):  # Show first 10 for performance
    if bs['type'] == 'macro':  # Only show macro cells
        # Create sphere for coverage
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        radius = bs['coverage_radius'] * 100  # Scale coverage radius
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + lon_norm.iloc[i]
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + lat_norm.iloc[i]
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + height.iloc[i]
        
        fig_3d.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.2,
            showscale=False,
            colorscale='Reds',
            name=f"Coverage {bs['id']}"
        ))

fig_3d.update_layout(
    title="3D Network Topology with Coverage Volumes",
    scene=dict(
        xaxis_title="Longitude (scaled)",
        yaxis_title="Latitude (scaled)", 
        zaxis_title="Height (m)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    height=700
)

fig_3d.show()
```

## 🔄 Integrated Visualization Pipeline {#integrated}

### Using the Integrated Visualizer

```python
# Initialize the integrated visualizer
visualizer = IntegratedNetworkVisualizer()

# Create a complete visualization pipeline
results = visualizer.create_complete_visualization(
    base_stations, 
    edges,
    output_dir="examples/visualization_data/",
    include_3d=True
)

print("📁 Generated Files:")
for format_type, filename in results.items():
    print(f"  {format_type}: {filename}")

# Generate optimization heatmap
optimization_data = visualizer.generate_optimization_heatmap(
    base_stations,
    grid_size=20,
    region_bounds={
        'min_lat': min(bs['lat'] for bs in base_stations) - 0.01,
        'max_lat': max(bs['lat'] for bs in base_stations) + 0.01,
        'min_lon': min(bs['lon'] for bs in base_stations) - 0.01,
        'max_lon': max(bs['lon'] for bs in base_stations) + 0.01
    }
)

# Visualize coverage optimization
fig_heatmap = px.density_mapbox(
    optimization_data,
    lat='lat',
    lon='lon',
    z='coverage_score',
    radius=15,
    center=dict(lat=np.mean([bs['lat'] for bs in base_stations]),
                lon=np.mean([bs['lon'] for bs in base_stations])),
    zoom=11,
    mapbox_style="open-street-map",
    title="Network Coverage Optimization Heatmap",
    color_continuous_scale="RdYlGn"
)

# Add base stations overlay
df_bs = pd.DataFrame(base_stations)
fig_heatmap.add_trace(go.Scattermapbox(
    lat=df_bs['lat'],
    lon=df_bs['lon'],
    mode='markers',
    marker=dict(size=8, color='black', symbol='circle'),
    text=df_bs['id'],
    name="Base Stations"
))

fig_heatmap.show()
```

## 🎯 Real-world Use Cases {#use-cases}

### Network Performance Analysis

```python
# Analyze network performance metrics
performance_analysis = {
    'total_throughput': sum(bs['throughput_mbps'] for bs in base_stations),
    'avg_utilization': np.mean([bs['cpu_utilization'] for bs in base_stations]),
    'coverage_efficiency': len([bs for bs in base_stations if bs['status'] == 'active']) / len(base_stations),
    'user_density': sum(bs['connected_users'] for bs in base_stations) / len(base_stations)
}

print("📊 Network Performance Metrics:")
for metric, value in performance_analysis.items():
    print(f"  {metric}: {value:.2f}")

# Identify optimization opportunities
overloaded_stations = [bs for bs in base_stations if bs['cpu_utilization'] > 80]
underutilized_stations = [bs for bs in base_stations if bs['cpu_utilization'] < 30]

print(f"\n⚠️  Overloaded stations: {len(overloaded_stations)}")
print(f"🔋 Underutilized stations: {len(underutilized_stations)}")

# Capacity planning
if overloaded_stations:
    print("\n📈 Capacity Planning Recommendations:")
    for bs in overloaded_stations[:3]:  # Show top 3
        print(f"  - {bs['id']}: Add micro/pico cells in coverage area")
        print(f"    Current load: {bs['cpu_utilization']:.1f}%, Users: {bs['connected_users']}")
```

### Energy Efficiency Analysis

```python
# Energy consumption estimation
energy_consumption = []
for bs in base_stations:
    # Simplified energy model based on type and utilization
    base_power = {'macro': 500, 'micro': 100, 'pico': 50}[bs['type']]  # Watts
    load_factor = bs['cpu_utilization'] / 100
    total_power = base_power * (0.3 + 0.7 * load_factor)  # Linear model
    
    energy_consumption.append({
        'id': bs['id'],
        'type': bs['type'],
        'power_watts': total_power,
        'daily_kwh': total_power * 24 / 1000,
        'utilization': bs['cpu_utilization']
    })

df_energy = pd.DataFrame(energy_consumption)

# Energy efficiency visualization
fig_energy = px.scatter(
    df_energy,
    x='utilization',
    y='power_watts',
    color='type',
    size='daily_kwh',
    title="Energy Consumption vs. Utilization",
    labels={'utilization': 'CPU Utilization (%)', 'power_watts': 'Power Consumption (W)'}
)

fig_energy.show()

# Energy savings opportunities
total_daily_energy = df_energy['daily_kwh'].sum()
potential_savings = df_energy[df_energy['utilization'] < 30]['daily_kwh'].sum() * 0.3

print(f"\n⚡ Energy Analysis:")
print(f"  Total daily consumption: {total_daily_energy:.1f} kWh")
print(f"  Potential daily savings: {potential_savings:.1f} kWh ({potential_savings/total_daily_energy*100:.1f}%)")
print(f"  Annual CO2 reduction potential: {potential_savings * 365 * 0.4:.1f} kg CO2")
```

### Real-time Monitoring Dashboard

```python
# Simulate real-time data updates
import time
from IPython.display import display, clear_output

def simulate_real_time_monitoring(duration_seconds=30):
    """Simulate real-time network monitoring"""
    
    for t in range(duration_seconds):
        clear_output(wait=True)
        
        # Update base station metrics with some randomness
        for bs in base_stations:
            bs['cpu_utilization'] += np.random.uniform(-5, 5)
            bs['cpu_utilization'] = np.clip(bs['cpu_utilization'], 0, 100)
            bs['throughput_mbps'] += np.random.uniform(-10, 10)
            bs['throughput_mbps'] = np.clip(bs['throughput_mbps'], 10, 1000)
            bs['connected_users'] += np.random.randint(-20, 20)
            bs['connected_users'] = np.clip(bs['connected_users'], 0, 800)
        
        # Calculate current KPIs
        current_metrics = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'total_users': sum(bs['connected_users'] for bs in base_stations),
            'avg_throughput': np.mean([bs['throughput_mbps'] for bs in base_stations]),
            'avg_utilization': np.mean([bs['cpu_utilization'] for bs in base_stations]),
            'active_stations': len([bs for bs in base_stations if bs['status'] == 'active'])
        }
        
        print(f"🕐 Real-time Network Status - {current_metrics['timestamp']}")
        print("=" * 50)
        print(f"📊 Total Connected Users: {current_metrics['total_users']:,}")
        print(f"🚀 Average Throughput: {current_metrics['avg_throughput']:.1f} Mbps")
        print(f"⚡ Average CPU Usage: {current_metrics['avg_utilization']:.1f}%")
        print(f"📡 Active Stations: {current_metrics['active_stations']}/{len(base_stations)}")
        
        # Alert conditions
        alerts = []
        if current_metrics['avg_utilization'] > 85:
            alerts.append("⚠️  HIGH CPU UTILIZATION")
        if current_metrics['avg_throughput'] < 100:
            alerts.append("⚠️  LOW NETWORK THROUGHPUT")
        
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print("\n✅ All systems operating normally")
        
        time.sleep(1)

# Uncomment to run real-time simulation
# simulate_real_time_monitoring(30)
```

## 📝 Summary

This notebook demonstrated the comprehensive visualization capabilities of the AI-Powered 5G Open RAN Optimizer:

1. **GeoJSON Visualization**: Interactive maps with network topology
2. **TopoJSON Analysis**: Efficient topology storage and analysis
3. **3D STL Models**: Physical 3D models for network planning
4. **Integrated Pipeline**: Complete workflow automation
5. **Real-world Applications**: Performance analysis, energy optimization, and monitoring

### Key Benefits

- 🌍 **Spatial Awareness**: Understand geographic distribution of network resources
- 📊 **Performance Insights**: Identify bottlenecks and optimization opportunities  
- 🔋 **Energy Efficiency**: Optimize power consumption across the network
- 🎯 **Capacity Planning**: Data-driven decisions for network expansion
- 📈 **Real-time Monitoring**: Live network status and alerting

### Next Steps

1. Integrate with live network data sources
2. Implement machine learning for predictive analytics
3. Add support for additional visualization formats
4. Develop automated optimization algorithms
5. Create deployment pipelines for production use

---

*For more information, see the [Geospatial & 3D Visualization Guide](docs/visualization/GEOSPATIAL_3D_GUIDE.md)*
