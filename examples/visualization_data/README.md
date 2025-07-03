# Advanced Geospatial and 3D Visualization Capabilities

This directory contains the advanced visualization modules for the AI-Powered 5G Open RAN Optimizer, providing comprehensive geospatial mapping, topology analysis, and 3D model generation capabilities.

## 🌟 Features

### GeoJSON Network Mapping

- **Comprehensive 5G Infrastructure Mapping**: Generate detailed GeoJSON representations of base stations, cell towers, antennas, and coverage areas
- **Interactive Visualization**: Create interactive web maps with Leaflet integration
- **Real-time Data Integration**: Support for live network data feeds
- **Multi-operator Support**: Handle networks from multiple operators with proper attribution

### TopoJSON Topology Analysis

- **Network Topology Conversion**: Convert GeoJSON to efficient TopoJSON format
- **Topology Analysis**: Analyze network connectivity, density, and structure
- **Performance Optimization**: Reduce file sizes through quantization and simplification
- **Graph Analytics**: Network centrality, connectivity analysis

### STL 3D Model Generation

- **Realistic 3D Models**: Generate accurate 3D models of cell towers, antennas, and coverage volumes
- **Multiple Tower Types**: Support for lattice, monopole, and stealth tower designs
- **Coverage Visualization**: 3D representation of signal coverage and interference patterns
- **Building Integration**: Model buildings and obstacles for RF propagation analysis

### Integrated Visualization Pipeline

- **Unified Interface**: Single API for all visualization formats
- **Batch Processing**: Generate visualizations for multiple networks simultaneously
- **Performance Optimization**: Automatic optimization for web deployment
- **Dashboard Integration**: Seamless integration with monitoring dashboards

## 📁 Directory Structure

```
src/visualization/geospatial/
├── __init__.py                    # Package initialization
├── geojson_generator.py          # GeoJSON generation and mapping
├── topojson_converter.py         # TopoJSON conversion and topology analysis
├── stl_3d_generator.py           # STL 3D model generation
└── integrated_visualizer.py      # Unified visualization interface

examples/visualization_data/
├── sample_5g_network.geojson     # Sample GeoJSON network data
├── sample_network_topology.topojson  # Sample TopoJSON topology
└── README.md                     # This documentation

demo_advanced_visualization.py    # Comprehensive demonstration script
```

## 🚀 Quick Start

### 1. Basic GeoJSON Generation

```python
from src.visualization.geospatial.geojson_generator import GeoJSON5GGenerator, BaseStation, CellTower

# Initialize generator
generator = GeoJSON5GGenerator()

# Create a base station
base_station = BaseStation(
    id="bs_001",
    location=[-74.0059, 40.7128],  # NYC coordinates
    cell_towers=[],
    properties={
        "operator": "VerizonWireless",
        "technology": "5G NR",
        "bands": ["n77", "n78"]
    }
)

# Add to generator and export
generator.add_base_station(base_station)
geojson_data = generator.generate_geojson()
generator.export_geojson(geojson_data, "network.geojson")
```

### 2. TopoJSON Conversion

```python
from src.visualization.geospatial.topojson_converter import TopoJSONConverter

# Initialize converter
converter = TopoJSONConverter(quantization=1e6, simplification=0.001)

# Convert GeoJSON to TopoJSON
topojson_data = converter.convert_geojson_to_topojson(geojson_data)

# Analyze network topology
features = geojson_data.get('features', [])
topology = converter._build_network_topology(features)
analysis = converter.analyze_topology(topology)

# Export TopoJSON
converter.export_topojson(topojson_data, "network_topology.topojson")
```

### 3. STL 3D Model Generation

```python
from src.visualization.geospatial.stl_3d_generator import STL3DGenerator, Vector3D

# Initialize generator
generator = STL3DGenerator(detail_level="medium")

# Create cell tower
tower_position = Vector3D(0, 0, 0)
tower_mesh = generator.generate_cell_tower(
    position=tower_position,
    height=50,
    tower_type="lattice"
)

# Create antenna array
antenna_position = Vector3D(0, 0, 45)
antenna_mesh = generator.generate_antenna_array(
    position=antenna_position,
    array_type="panel",
    count=3,
    orientation=120
)

# Export STL models
generator.export_stl_ascii(tower_mesh, "cell_tower.stl")
generator.export_stl_ascii(antenna_mesh, "antenna_array.stl")
```

### 4. Integrated Visualization

```python
from src.visualization.geospatial.integrated_visualizer import (
    IntegratedNetworkVisualizer, VisualizationConfig
)

# Configure visualization
config = VisualizationConfig(
    output_formats=["geojson", "topojson", "stl"],
    detail_level="medium",
    enable_3d=True,
    enable_topology=True,
    enable_coverage=True
)

# Initialize visualizer
visualizer = IntegratedNetworkVisualizer(config)

# Network data structure
network_data = {
    "name": "sample_network",
    "base_stations": [...],  # Base station definitions
    "connections": [...],    # Network connections
    "buildings": [...]       # Building obstacles
}

# Generate complete visualization
result = visualizer.generate_complete_visualization(
    network_data, 
    "output_directory"
)

# Create dashboard configuration
dashboard_data = visualizer.create_interactive_dashboard_data(
    result, 
    "dashboard_config.json"
)
```

## 🎯 Use Cases

### Network Planning and Optimization

- **Coverage Analysis**: Visualize signal coverage and identify gaps
- **Interference Analysis**: Model interference patterns between cells
- **Capacity Planning**: Analyze network capacity and user distribution
- **Site Selection**: Evaluate potential cell site locations

### Regulatory Compliance

- **RF Exposure Modeling**: Visualize electromagnetic field patterns
- **Zoning Compliance**: Check compliance with local zoning regulations
- **Environmental Impact**: Assess visual and environmental impact

### Operational Monitoring

- **Real-time Network Status**: Live visualization of network performance
- **Fault Localization**: Quickly identify and locate network issues
- **Performance Analytics**: Analyze network KPIs spatially

### Research and Development

- **Algorithm Validation**: Visualize results of optimization algorithms
- **Simulation Integration**: Integrate with network simulation tools
- **Academic Research**: Support for research publications and presentations

## 🛠️ Configuration Options

### Detail Levels

- **Low**: Optimized for performance, reduced geometry detail
- **Medium**: Balanced detail and performance (recommended)
- **High**: Maximum detail for high-quality visualizations

### Output Formats

- **GeoJSON**: Standard geospatial format for web mapping
- **TopoJSON**: Compressed topology format for efficient transmission
- **STL**: 3D model format for CAD and 3D printing

### Visualization Features

- **3D Models**: Enable/disable 3D STL model generation
- **Topology Analysis**: Enable/disable network topology analysis
- **Coverage Visualization**: Enable/disable coverage area rendering

## 📊 Performance Considerations

### Memory Usage

- **Large Networks**: Use batch processing for networks with >1000 base stations
- **Detail Level**: Reduce detail level for better performance
- **Format Selection**: Choose appropriate output formats based on use case

### Processing Time

- **GeoJSON**: Fast generation, suitable for real-time applications
- **TopoJSON**: Moderate processing time, good for web deployment
- **STL**: Intensive processing, best for high-quality visualizations

### File Sizes

- **GeoJSON**: Larger files, good compression with gzip
- **TopoJSON**: Smaller files, efficient for web transmission
- **STL**: Variable size based on detail level and model complexity

## 🔧 Advanced Features

### Custom Styling

```python
# Custom GeoJSON styling
style_config = {
    "base_stations": {
        "marker_color": "#FF0000",
        "marker_size": 10,
        "popup_template": "Base Station: {name}"
    },
    "coverage_areas": {
        "fill_color": "#00FF00",
        "fill_opacity": 0.3,
        "stroke_color": "#0000FF"
    }
}

generator.set_style_config(style_config)
```

### Performance Optimization

```python
# Web optimization
optimized_result = visualizer.optimize_for_web(original_result)

# Batch processing
network_datasets = [network1, network2, network3]
results = visualizer.generate_batch_visualizations(
    network_datasets, 
    "batch_output"
)
```

### Integration with External Systems

```python
# Real-time data integration
def update_network_data(new_data):
    # Update visualization with new network data
    updated_result = visualizer.update_visualization(new_data)
    return updated_result

# Dashboard integration
dashboard_config = visualizer.create_dashboard_config(
    result,
    include_controls=True,
    enable_real_time=True
)
```

## 📝 Example Data Formats

### Network Configuration Input

```json
{
  "name": "example_network",
  "base_stations": [
    {
      "id": "bs_001",
      "position": [-74.0059, 40.7128],
      "properties": {
        "operator": "VerizonWireless",
        "technology": "5G NR"
      },
      "towers": [
        {
          "id": "tower_001",
          "height": 50,
          "type": "lattice",
          "antennas": [
            {
              "id": "ant_001",
              "type": "panel",
              "azimuth": 0,
              "range": 1000
            }
          ]
        }
      ]
    }
  ]
}
```

### Generated GeoJSON Output

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-74.0059, 40.7128]
      },
      "properties": {
        "id": "bs_001",
        "type": "base_station",
        "operator": "VerizonWireless"
      }
    }
  ]
}
```

## 🧪 Running the Demo

Execute the comprehensive demonstration:

```bash
# Full demo with all features
python demo_advanced_visualization.py

# Skip 3D generation for faster demo
python demo_advanced_visualization.py --skip-3d

# Custom output directory
python demo_advanced_visualization.py --output-dir custom_output

# Skip resource-intensive features
python demo_advanced_visualization.py --skip-3d --skip-batch --skip-performance
```

## 📚 API Reference

### GeoJSON5GGenerator

- `add_base_station(base_station)`: Add base station to the network
- `add_connection(source, target, type, properties)`: Add network connection
- `add_coverage_area(center, radius, properties)`: Add coverage area
- `generate_geojson()`: Generate complete GeoJSON representation
- `create_interactive_map(geojson, output_path)`: Create interactive web map

### TopoJSONConverter

- `convert_geojson_to_topojson(geojson)`: Convert GeoJSON to TopoJSON
- `analyze_topology(topology)`: Analyze network topology characteristics
- `simplify_topology(topojson, tolerance)`: Simplify topology for optimization

### STL3DGenerator

- `generate_cell_tower(position, height, type)`: Generate 3D tower model
- `generate_antenna_array(position, type, orientation)`: Generate antenna model
- `generate_coverage_volume(position, range, azimuth, elevation)`: Generate coverage volume
- `generate_building(position, width, length, height)`: Generate building obstacle

### IntegratedNetworkVisualizer

- `generate_complete_visualization(network_data, output_dir)`: Generate all formats
- `generate_batch_visualizations(datasets, output_dir)`: Batch processing
- `optimize_for_web(visualization_data)`: Optimize for web deployment
- `create_interactive_dashboard_data(data, output_path)`: Create dashboard config

## 🤝 Contributing

Contributions to the visualization modules are welcome! Please refer to the main project's CONTRIBUTING.md for guidelines.

### Development Setup

1. Install required dependencies: `pip install -r requirements.txt`
2. Run tests: `python -m pytest tests/visualization/`
3. Run demo: `python demo_advanced_visualization.py`

### Adding New Features

- Follow the existing code structure and documentation standards
- Add comprehensive tests for new functionality
- Update this README with new features and examples
- Ensure compatibility with existing visualization pipeline

## 📄 License

This visualization module is part of the AI-Powered 5G Open RAN Optimizer project and is licensed under the MIT License. See the main project LICENSE file for details.

---

*For more information about the AI-Powered 5G Open RAN Optimizer project, see the main README.md file.*
