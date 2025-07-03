"""
🌍 Advanced GeoJSON Network Visualization Module

This module provides comprehensive geospatial visualization capabilities for 5G network infrastructure,
including cell tower mapping, coverage areas, and network slice boundaries using GeoJSON format.

Author: AI-Powered 5G Open RAN Optimizer Team
License: Apache 2.0
Version: 2.0.0
"""

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import folium
from folium import plugins


@dataclass
class CellTower:
    """5G Cell Tower Data Structure"""
    cell_id: str
    latitude: float
    longitude: float
    elevation: float
    technology: str  # "5G_NR", "LTE", "5G_SA"
    frequency_bands: List[str]  # ["n78", "n41", "n1"]
    max_power_dbm: float
    antenna_type: str  # "massive_mimo", "traditional", "beam_forming"
    azimuth: float  # degrees
    tilt: float  # degrees
    coverage_radius_m: float
    network_slice: str  # "eMBB", "URLLC", "mMTC"
    operator: str
    site_type: str  # "macro", "micro", "pico", "femto"
    installation_date: str
    status: str  # "active", "planned", "maintenance", "decommissioned"


@dataclass
class CoverageArea:
    """Network Coverage Area Data Structure"""
    area_id: str
    geometry: Polygon
    signal_strength_dbm: float
    technology: str
    frequency_band: str
    network_slice: str
    qos_metrics: Dict[str, float]  # {"latency_ms": 1.5, "throughput_mbps": 1000}


@dataclass
class NetworkSlice:
    """5G Network Slice Boundary Data Structure"""
    slice_id: str
    slice_type: str  # "eMBB", "URLLC", "mMTC"
    boundary: Polygon
    sla_requirements: Dict[str, float]
    allocated_resources: Dict[str, float]
    traffic_characteristics: Dict[str, Any]


class NetworkGeoVisualizer:
    """
    Advanced GeoJSON visualization for 5G network infrastructure.
    
    This class provides comprehensive geospatial visualization capabilities including:
    - 5G cell tower mapping with detailed metadata
    - Coverage area visualization with signal strength
    - Network slice boundary representation
    - Real-time network performance overlay
    - Interactive web-based visualizations
    """
    
    def __init__(self):
        """Initialize the Network GeoJSON Visualizer."""
        self.geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        self.coverage_features = []
        self.slice_features = []
        
    def create_cell_tower_geojson(self, cell_towers: List[CellTower]) -> Dict[str, Any]:
        """
        Create GeoJSON representation of 5G cell towers.
        
        Args:
            cell_towers: List of CellTower objects
            
        Returns:
            GeoJSON FeatureCollection with cell tower data
        """
        features = []
        
        for tower in cell_towers:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [tower.longitude, tower.latitude, tower.elevation]
                },
                "properties": {
                    "cell_id": tower.cell_id,
                    "technology": tower.technology,
                    "frequency_bands": tower.frequency_bands,
                    "max_power_dbm": tower.max_power_dbm,
                    "antenna_type": tower.antenna_type,
                    "azimuth": tower.azimuth,
                    "tilt": tower.tilt,
                    "coverage_radius_m": tower.coverage_radius_m,
                    "network_slice": tower.network_slice,
                    "operator": tower.operator,
                    "site_type": tower.site_type,
                    "installation_date": tower.installation_date,
                    "status": tower.status,
                    "marker_color": self._get_tower_color(tower.technology, tower.status),
                    "marker_size": self._get_tower_size(tower.site_type),
                    "marker_symbol": self._get_tower_symbol(tower.antenna_type)
                }
            }
            features.append(feature)
            
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_towers": len(features),
                "generation_timestamp": datetime.now().isoformat(),
                "coordinate_system": "WGS84",
                "elevation_unit": "meters"
            }
        }
    
    def create_coverage_geojson(self, coverage_areas: List[CoverageArea]) -> Dict[str, Any]:
        """
        Create GeoJSON representation of network coverage areas.
        
        Args:
            coverage_areas: List of CoverageArea objects
            
        Returns:
            GeoJSON FeatureCollection with coverage area data
        """
        features = []
        
        for area in coverage_areas:
            # Convert Shapely Polygon to GeoJSON geometry
            geometry = {
                "type": "Polygon",
                "coordinates": [list(area.geometry.exterior.coords)]
            }
            
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "area_id": area.area_id,
                    "signal_strength_dbm": area.signal_strength_dbm,
                    "technology": area.technology,
                    "frequency_band": area.frequency_band,
                    "network_slice": area.network_slice,
                    "qos_metrics": area.qos_metrics,
                    "fill_color": self._get_signal_color(area.signal_strength_dbm),
                    "fill_opacity": self._get_signal_opacity(area.signal_strength_dbm),
                    "stroke_color": self._get_slice_color(area.network_slice),
                    "stroke_width": 2
                }
            }
            features.append(feature)
            
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_areas": len(features),
                "signal_strength_range": self._get_signal_range(coverage_areas),
                "generation_timestamp": datetime.now().isoformat()
            }
        }
    
    def create_network_slices_geojson(self, network_slices: List[NetworkSlice]) -> Dict[str, Any]:
        """
        Create GeoJSON representation of network slice boundaries.
        
        Args:
            network_slices: List of NetworkSlice objects
            
        Returns:
            GeoJSON FeatureCollection with network slice data
        """
        features = []
        
        for slice_obj in network_slices:
            geometry = {
                "type": "Polygon",
                "coordinates": [list(slice_obj.boundary.exterior.coords)]
            }
            
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "slice_id": slice_obj.slice_id,
                    "slice_type": slice_obj.slice_type,
                    "sla_requirements": slice_obj.sla_requirements,
                    "allocated_resources": slice_obj.allocated_resources,
                    "traffic_characteristics": slice_obj.traffic_characteristics,
                    "fill_color": self._get_slice_color(slice_obj.slice_type),
                    "fill_opacity": 0.3,
                    "stroke_color": self._get_slice_color(slice_obj.slice_type),
                    "stroke_width": 3,
                    "stroke_dasharray": self._get_slice_dash_pattern(slice_obj.slice_type)
                }
            }
            features.append(feature)
            
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_slices": len(features),
                "slice_types": list(set(s.slice_type for s in network_slices)),
                "generation_timestamp": datetime.now().isoformat()
            }
        }
    
    def create_comprehensive_network_geojson(
        self,
        cell_towers: List[CellTower],
        coverage_areas: List[CoverageArea],
        network_slices: List[NetworkSlice]
    ) -> Dict[str, Any]:
        """
        Create comprehensive GeoJSON with all network components.
        
        Args:
            cell_towers: List of cell tower data
            coverage_areas: List of coverage area data
            network_slices: List of network slice data
            
        Returns:
            Comprehensive GeoJSON with all network infrastructure
        """
        all_features = []
        
        # Add cell towers
        tower_geojson = self.create_cell_tower_geojson(cell_towers)
        for feature in tower_geojson["features"]:
            feature["layer_type"] = "cell_tower"
            all_features.append(feature)
        
        # Add coverage areas
        coverage_geojson = self.create_coverage_geojson(coverage_areas)
        for feature in coverage_geojson["features"]:
            feature["layer_type"] = "coverage_area"
            all_features.append(feature)
        
        # Add network slices
        slice_geojson = self.create_network_slices_geojson(network_slices)
        for feature in slice_geojson["features"]:
            feature["layer_type"] = "network_slice"
            all_features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": all_features,
            "metadata": {
                "total_features": len(all_features),
                "layers": ["cell_tower", "coverage_area", "network_slice"],
                "towers_count": len(cell_towers),
                "coverage_areas_count": len(coverage_areas),
                "network_slices_count": len(network_slices),
                "generation_timestamp": datetime.now().isoformat(),
                "coordinate_system": "WGS84",
                "version": "2.0.0"
            }
        }
    
    def create_interactive_map(
        self,
        geojson_data: Dict[str, Any],
        center_coordinates: Tuple[float, float] = None,
        zoom_level: int = 10
    ) -> folium.Map:
        """
        Create interactive Folium map with network data.
        
        Args:
            geojson_data: GeoJSON data to visualize
            center_coordinates: Map center (lat, lon)
            zoom_level: Initial zoom level
            
        Returns:
            Folium map object
        """
        # Calculate center if not provided
        if center_coordinates is None:
            center_coordinates = self._calculate_center(geojson_data)
        
        # Create base map
        m = folium.Map(
            location=center_coordinates,
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        
        # Add multiple tile layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter').add_to(m)
        
        # Separate features by layer type
        tower_features = []
        coverage_features = []
        slice_features = []
        
        for feature in geojson_data["features"]:
            layer_type = feature.get("layer_type", "unknown")
            if layer_type == "cell_tower":
                tower_features.append(feature)
            elif layer_type == "coverage_area":
                coverage_features.append(feature)
            elif layer_type == "network_slice":
                slice_features.append(feature)
        
        # Add network slices (bottom layer)
        if slice_features:
            slice_layer = folium.FeatureGroup(name="Network Slices")
            for feature in slice_features:
                self._add_slice_to_map(slice_layer, feature)
            slice_layer.add_to(m)
        
        # Add coverage areas (middle layer)
        if coverage_features:
            coverage_layer = folium.FeatureGroup(name="Coverage Areas")
            for feature in coverage_features:
                self._add_coverage_to_map(coverage_layer, feature)
            coverage_layer.add_to(m)
        
        # Add cell towers (top layer)
        if tower_features:
            tower_layer = folium.FeatureGroup(name="Cell Towers")
            for feature in tower_features:
                self._add_tower_to_map(tower_layer, feature)
            tower_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add plugins for enhanced functionality
        plugins.MeasureControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MiniMap().add_to(m)
        
        # Add legend
        self._add_legend(m)
        
        return m
    
    def export_geojson(self, geojson_data: Dict[str, Any], filename: str) -> None:
        """
        Export GeoJSON data to file.
        
        Args:
            geojson_data: GeoJSON data to export
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
    
    def validate_geojson(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GeoJSON data structure and content.
        
        Args:
            geojson_data: GeoJSON data to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Basic structure validation
            if geojson_data.get("type") != "FeatureCollection":
                validation_report["errors"].append("Invalid root type, expected 'FeatureCollection'")
                validation_report["valid"] = False
            
            features = geojson_data.get("features", [])
            if not isinstance(features, list):
                validation_report["errors"].append("Features must be a list")
                validation_report["valid"] = False
            
            # Feature validation
            geometry_types = set()
            property_keys = set()
            
            for i, feature in enumerate(features):
                if feature.get("type") != "Feature":
                    validation_report["errors"].append(f"Feature {i}: Invalid type")
                
                geometry = feature.get("geometry", {})
                geometry_type = geometry.get("type")
                if geometry_type:
                    geometry_types.add(geometry_type)
                
                properties = feature.get("properties", {})
                property_keys.update(properties.keys())
            
            # Statistics
            validation_report["statistics"] = {
                "total_features": len(features),
                "geometry_types": list(geometry_types),
                "property_keys": list(property_keys),
                "file_size_kb": len(json.dumps(geojson_data)) / 1024
            }
            
        except Exception as e:
            validation_report["errors"].append(f"Validation error: {str(e)}")
            validation_report["valid"] = False
        
        return validation_report
    
    # Helper methods for styling and visualization
    
    def _get_tower_color(self, technology: str, status: str) -> str:
        """Get color for cell tower based on technology and status."""
        if status != "active":
            return "#808080"  # Gray for inactive
        
        color_map = {
            "5G_NR": "#FF0000",      # Red for 5G NR
            "5G_SA": "#FF4500",      # Orange Red for 5G SA
            "LTE": "#0000FF",        # Blue for LTE
            "NR_NSA": "#FF8C00",     # Dark Orange for NR NSA
        }
        return color_map.get(technology, "#000000")
    
    def _get_tower_size(self, site_type: str) -> int:
        """Get marker size based on site type."""
        size_map = {
            "macro": 12,
            "micro": 8,
            "pico": 6,
            "femto": 4
        }
        return size_map.get(site_type, 8)
    
    def _get_tower_symbol(self, antenna_type: str) -> str:
        """Get marker symbol based on antenna type."""
        symbol_map = {
            "massive_mimo": "antenna",
            "beam_forming": "triangle",
            "traditional": "circle"
        }
        return symbol_map.get(antenna_type, "circle")
    
    def _get_signal_color(self, signal_strength_dbm: float) -> str:
        """Get color based on signal strength."""
        if signal_strength_dbm >= -60:
            return "#00FF00"      # Excellent (Green)
        elif signal_strength_dbm >= -70:
            return "#FFFF00"      # Good (Yellow)
        elif signal_strength_dbm >= -80:
            return "#FFA500"      # Fair (Orange)
        elif signal_strength_dbm >= -90:
            return "#FF4500"      # Poor (Orange Red)
        else:
            return "#FF0000"      # Very Poor (Red)
    
    def _get_signal_opacity(self, signal_strength_dbm: float) -> float:
        """Get opacity based on signal strength."""
        # Normalize signal strength to opacity (0.3 to 0.8)
        min_signal, max_signal = -100, -50
        normalized = (signal_strength_dbm - min_signal) / (max_signal - min_signal)
        return max(0.3, min(0.8, 0.3 + normalized * 0.5))
    
    def _get_slice_color(self, slice_type: str) -> str:
        """Get color for network slice type."""
        color_map = {
            "eMBB": "#4169E1",       # Royal Blue
            "URLLC": "#FF1493",      # Deep Pink
            "mMTC": "#32CD32"        # Lime Green
        }
        return color_map.get(slice_type, "#808080")
    
    def _get_slice_dash_pattern(self, slice_type: str) -> str:
        """Get dash pattern for network slice type."""
        pattern_map = {
            "eMBB": "5,5",           # Dashed
            "URLLC": "10,5,5,5",     # Dash-dot
            "mMTC": "20,5"           # Long dash
        }
        return pattern_map.get(slice_type, "")
    
    def _get_signal_range(self, coverage_areas: List[CoverageArea]) -> Dict[str, float]:
        """Get signal strength range from coverage areas."""
        if not coverage_areas:
            return {"min": -100, "max": -50}
        
        strengths = [area.signal_strength_dbm for area in coverage_areas]
        return {"min": min(strengths), "max": max(strengths)}
    
    def _calculate_center(self, geojson_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate the center point of all features."""
        coordinates = []
        
        for feature in geojson_data.get("features", []):
            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type")
            coords = geometry.get("coordinates", [])
            
            if geom_type == "Point":
                coordinates.append(coords[:2])  # [lon, lat]
            elif geom_type == "Polygon":
                for point in coords[0]:  # Exterior ring
                    coordinates.append(point[:2])
        
        if coordinates:
            avg_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
            avg_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
            return (avg_lat, avg_lon)
        
        return (0.0, 0.0)  # Default center
    
    def _add_tower_to_map(self, layer: folium.FeatureGroup, feature: Dict[str, Any]) -> None:
        """Add cell tower feature to map layer."""
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        
        # Create popup content
        popup_content = f"""
        <div style="width: 300px;">
        <h4>🗼 Cell Tower: {props['cell_id']}</h4>
        <table style="font-size: 12px;">
        <tr><td><b>Technology:</b></td><td>{props['technology']}</td></tr>
        <tr><td><b>Bands:</b></td><td>{', '.join(props['frequency_bands'])}</td></tr>
        <tr><td><b>Power:</b></td><td>{props['max_power_dbm']} dBm</td></tr>
        <tr><td><b>Antenna:</b></td><td>{props['antenna_type']}</td></tr>
        <tr><td><b>Coverage:</b></td><td>{props['coverage_radius_m']} m</td></tr>
        <tr><td><b>Operator:</b></td><td>{props['operator']}</td></tr>
        <tr><td><b>Status:</b></td><td>{props['status']}</td></tr>
        </table>
        </div>
        """
        
        folium.Marker(
            location=[coords[1], coords[0]],  # [lat, lon]
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"Tower: {props['cell_id']}",
            icon=folium.Icon(
                color='red' if props['technology'].startswith('5G') else 'blue',
                icon='tower-cell',
                prefix='fa'
            )
        ).add_to(layer)
    
    def _add_coverage_to_map(self, layer: folium.FeatureGroup, feature: Dict[str, Any]) -> None:
        """Add coverage area feature to map layer."""
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"][0]
        
        popup_content = f"""
        <div style="width: 250px;">
        <h4>📶 Coverage Area: {props['area_id']}</h4>
        <p><b>Signal:</b> {props['signal_strength_dbm']} dBm</p>
        <p><b>Technology:</b> {props['technology']}</p>
        <p><b>Band:</b> {props['frequency_band']}</p>
        <p><b>Slice:</b> {props['network_slice']}</p>
        </div>
        """
        
        folium.Polygon(
            locations=[[point[1], point[0]] for point in coords],
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=f"Coverage: {props['area_id']}",
            color=props['stroke_color'],
            weight=props['stroke_width'],
            fillColor=props['fill_color'],
            fillOpacity=props['fill_opacity']
        ).add_to(layer)
    
    def _add_slice_to_map(self, layer: folium.FeatureGroup, feature: Dict[str, Any]) -> None:
        """Add network slice feature to map layer."""
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"][0]
        
        popup_content = f"""
        <div style="width: 280px;">
        <h4>🍕 Network Slice: {props['slice_id']}</h4>
        <p><b>Type:</b> {props['slice_type']}</p>
        <p><b>SLA Requirements:</b></p>
        <ul style="font-size: 11px; margin: 5px 0;">
        """
        
        for key, value in props['sla_requirements'].items():
            popup_content += f"<li>{key}: {value}</li>"
        
        popup_content += "</ul></div>"
        
        folium.Polygon(
            locations=[[point[1], point[0]] for point in coords],
            popup=folium.Popup(popup_content, max_width=280),
            tooltip=f"Slice: {props['slice_type']}",
            color=props['stroke_color'],
            weight=props['stroke_width'],
            fillColor=props['fill_color'],
            fillOpacity=props['fill_opacity'],
            dashArray=props.get('stroke_dasharray', '')
        ).add_to(layer)
    
    def _add_legend(self, map_obj: folium.Map) -> None:
        """Add legend to the map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Network Legend</b></p>
        <p><i class="fa fa-tower-cell" style="color:red"></i> 5G Towers</p>
        <p><i class="fa fa-tower-cell" style="color:blue"></i> LTE Towers</p>
        <p><span style="color:green">■</span> Excellent Signal</p>
        <p><span style="color:yellow">■</span> Good Signal</p>
        <p><span style="color:red">■</span> Poor Signal</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))


# Example usage and demonstration
def create_sample_network_data():
    """Create sample network data for demonstration."""
    
    # Sample cell towers around a city center
    sample_towers = [
        CellTower(
            cell_id="TOWER_001",
            latitude=40.7128,
            longitude=-74.0060,
            elevation=50.0,
            technology="5G_NR",
            frequency_bands=["n78", "n41"],
            max_power_dbm=43.0,
            antenna_type="massive_mimo",
            azimuth=120.0,
            tilt=6.0,
            coverage_radius_m=1000.0,
            network_slice="eMBB",
            operator="Operator_A",
            site_type="macro",
            installation_date="2024-01-15",
            status="active"
        ),
        CellTower(
            cell_id="TOWER_002",
            latitude=40.7589,
            longitude=-73.9851,
            elevation=45.0,
            technology="5G_SA",
            frequency_bands=["n78", "n1"],
            max_power_dbm=40.0,
            antenna_type="beam_forming",
            azimuth=240.0,
            tilt=4.0,
            coverage_radius_m=800.0,
            network_slice="URLLC",
            operator="Operator_B",
            site_type="micro",
            installation_date="2024-02-20",
            status="active"
        )
    ]
    
    # Sample coverage areas
    from shapely.geometry import Point as ShapelyPoint
    
    sample_coverage = [
        CoverageArea(
            area_id="COV_001",
            geometry=ShapelyPoint(40.7128, -74.0060).buffer(0.01),
            signal_strength_dbm=-65.0,
            technology="5G_NR",
            frequency_band="n78",
            network_slice="eMBB",
            qos_metrics={"latency_ms": 1.5, "throughput_mbps": 1000}
        )
    ]
    
    # Sample network slices
    sample_slices = [
        NetworkSlice(
            slice_id="SLICE_eMBB_001",
            slice_type="eMBB",
            boundary=ShapelyPoint(40.7128, -74.0060).buffer(0.02),
            sla_requirements={"latency_ms": 10, "throughput_mbps": 1000},
            allocated_resources={"spectrum_mhz": 100, "cpu_cores": 16},
            traffic_characteristics={"peak_users": 10000, "avg_session_mb": 50}
        )
    ]
    
    return sample_towers, sample_coverage, sample_slices


if __name__ == "__main__":
    # Demonstration of the GeoJSON visualization capabilities
    visualizer = NetworkGeoVisualizer()
    
    # Create sample data
    towers, coverage, slices = create_sample_network_data()
    
    # Generate comprehensive GeoJSON
    network_geojson = visualizer.create_comprehensive_network_geojson(
        cell_towers=towers,
        coverage_areas=coverage,
        network_slices=slices
    )
    
    # Export to file
    visualizer.export_geojson(network_geojson, "network_infrastructure.geojson")
    
    # Validate the generated GeoJSON
    validation_report = visualizer.validate_geojson(network_geojson)
    print("GeoJSON Validation Report:")
    print(json.dumps(validation_report, indent=2))
    
    # Create interactive map
    interactive_map = visualizer.create_interactive_map(network_geojson)
    interactive_map.save("network_visualization.html")
    
    print("✅ GeoJSON network visualization generated successfully!")
    print("📁 Files created:")
    print("   - network_infrastructure.geojson")
    print("   - network_visualization.html")
