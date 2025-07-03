"""
TopoJSON Converter for 5G Network Topology Visualization

This module provides functionality to convert GeoJSON network data to TopoJSON format
for efficient network topology visualization and analysis.

Features:
- GeoJSON to TopoJSON conversion with topology preservation
- Quantization for reduced file size
- Simplification for performance optimization
- Network topology analysis and validation
- Interactive topology visualization

Author: AI-Powered 5G Open RAN Optimizer Team
License: MIT
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopologyNode:
    """Represents a network node in the topology."""
    id: str
    type: str
    coordinates: Tuple[float, float]
    properties: Dict[str, Any]
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []


@dataclass
class TopologyEdge:
    """Represents a connection between network nodes."""
    source: str
    target: str
    type: str
    properties: Dict[str, Any]
    geometry: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.geometry is None:
            self.geometry = []


@dataclass
class NetworkTopology:
    """Complete network topology representation."""
    nodes: List[TopologyNode]
    edges: List[TopologyEdge]
    metadata: Dict[str, Any]


class TopoJSONConverter:
    """
    Converts GeoJSON network data to TopoJSON format for efficient topology visualization.
    
    TopoJSON is a format that encodes geographic data structures into JSON,
    eliminating redundancy and enabling topology-preserving transformations.
    """
    
    def __init__(self, quantization: int = 1e4, simplification: float = 0.0001):
        """
        Initialize TopoJSON converter.
        
        Args:
            quantization: Quantization parameter for coordinate precision
            simplification: Simplification tolerance for geometry reduction
        """
        self.quantization = quantization
        self.simplification = simplification
        self.topology_graph = nx.Graph()
        
    def convert_geojson_to_topojson(self, geojson_data: Dict) -> Dict:
        """
        Convert GeoJSON network data to TopoJSON format.
        
        Args:
            geojson_data: Input GeoJSON FeatureCollection
            
        Returns:
            TopoJSON representation of the network
        """
        try:
            logger.info("Converting GeoJSON to TopoJSON...")
            
            # Extract features and build topology
            features = geojson_data.get('features', [])
            topology = self._build_network_topology(features)
            
            # Create TopoJSON structure
            topojson = {
                "type": "Topology",
                "bbox": self._calculate_bbox(features),
                "transform": self._create_transform(),
                "objects": {
                    "network": {
                        "type": "GeometryCollection",
                        "geometries": []
                    }
                },
                "arcs": []
            }
            
            # Convert geometries to TopoJSON format
            self._convert_geometries(topology, topojson)
            
            logger.info(f"TopoJSON conversion completed. Objects: {len(topojson['objects']['network']['geometries'])}")
            return topojson
            
        except Exception as e:
            logger.error(f"Error converting GeoJSON to TopoJSON: {e}")
            raise
    
    def _build_network_topology(self, features: List[Dict]) -> NetworkTopology:
        """Build network topology from GeoJSON features."""
        nodes = []
        edges = []
        node_positions = {}
        
        # Extract nodes and their positions
        for feature in features:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if geometry.get('type') == 'Point':
                node_id = properties.get('id', f"node_{len(nodes)}")
                coords = geometry['coordinates']
                
                node = TopologyNode(
                    id=node_id,
                    type=properties.get('type', 'unknown'),
                    coordinates=(coords[0], coords[1]),
                    properties=properties
                )
                nodes.append(node)
                node_positions[node_id] = coords
                self.topology_graph.add_node(node_id, **properties)
        
        # Extract edges and connections
        for feature in features:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if geometry.get('type') == 'LineString':
                coords = geometry['coordinates']
                source_id = properties.get('source')
                target_id = properties.get('target')
                
                if source_id and target_id:
                    edge = TopologyEdge(
                        source=source_id,
                        target=target_id,
                        type=properties.get('type', 'connection'),
                        properties=properties,
                        geometry=coords
                    )
                    edges.append(edge)
                    self.topology_graph.add_edge(source_id, target_id, **properties)
        
        # Build adjacency relationships
        for node in nodes:
            node.connections = list(self.topology_graph.neighbors(node.id))
        
        metadata = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_density": nx.density(self.topology_graph),
            "connected_components": nx.number_connected_components(self.topology_graph)
        }
        
        return NetworkTopology(nodes=nodes, edges=edges, metadata=metadata)
    
    def _calculate_bbox(self, features: List[Dict]) -> List[float]:
        """Calculate bounding box for the network."""
        coords = []
        
        for feature in features:
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Point':
                coords.append(geometry['coordinates'])
            elif geometry.get('type') == 'LineString':
                coords.extend(geometry['coordinates'])
        
        if not coords:
            return [0, 0, 0, 0]
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        return [min(lons), min(lats), max(lons), max(lats)]
    
    def _create_transform(self) -> Dict:
        """Create quantization transform for coordinate precision."""
        return {
            "scale": [1.0 / self.quantization, 1.0 / self.quantization],
            "translate": [0, 0]
        }
    
    def _convert_geometries(self, topology: NetworkTopology, topojson: Dict):
        """Convert network topology to TopoJSON geometries."""
        arc_index = 0
        
        # Convert nodes
        for node in topology.nodes:
            geometry = {
                "type": "Point",
                "coordinates": self._quantize_coordinates([node.coordinates])[0],
                "properties": node.properties
            }
            topojson["objects"]["network"]["geometries"].append(geometry)
        
        # Convert edges as arcs
        for edge in topology.edges:
            if edge.geometry:
                arc = self._quantize_coordinates(edge.geometry)
                topojson["arcs"].append(arc)
                
                geometry = {
                    "type": "LineString",
                    "arcs": [arc_index],
                    "properties": edge.properties
                }
                topojson["objects"]["network"]["geometries"].append(geometry)
                arc_index += 1
    
    def _quantize_coordinates(self, coordinates: List[Tuple[float, float]]) -> List[List[int]]:
        """Quantize coordinates for reduced precision and file size."""
        quantized = []
        for coord in coordinates:
            x = int(coord[0] * self.quantization)
            y = int(coord[1] * self.quantization)
            quantized.append([x, y])
        return quantized
    
    def analyze_topology(self, topology: NetworkTopology) -> Dict[str, Any]:
        """
        Analyze network topology characteristics.
        
        Args:
            topology: Network topology to analyze
            
        Returns:
            Topology analysis results
        """
        try:
            analysis = {
                "basic_metrics": {
                    "node_count": len(topology.nodes),
                    "edge_count": len(topology.edges),
                    "density": nx.density(self.topology_graph),
                    "diameter": nx.diameter(self.topology_graph) if nx.is_connected(self.topology_graph) else "Disconnected"
                },
                "connectivity": {
                    "connected": nx.is_connected(self.topology_graph),
                    "components": nx.number_connected_components(self.topology_graph),
                    "largest_component_size": len(max(nx.connected_components(self.topology_graph), key=len))
                },
                "centrality": {
                    "degree_centrality": nx.degree_centrality(self.topology_graph),
                    "betweenness_centrality": nx.betweenness_centrality(self.topology_graph),
                    "closeness_centrality": nx.closeness_centrality(self.topology_graph)
                },
                "node_types": {},
                "edge_types": {}
            }
            
            # Analyze node types
            for node in topology.nodes:
                node_type = node.type
                if node_type not in analysis["node_types"]:
                    analysis["node_types"][node_type] = 0
                analysis["node_types"][node_type] += 1
            
            # Analyze edge types
            for edge in topology.edges:
                edge_type = edge.type
                if edge_type not in analysis["edge_types"]:
                    analysis["edge_types"][edge_type] = 0
                analysis["edge_types"][edge_type] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing topology: {e}")
            return {"error": str(e)}
    
    def simplify_topology(self, topojson_data: Dict, tolerance: float = None) -> Dict:
        """
        Simplify TopoJSON topology for performance optimization.
        
        Args:
            topojson_data: Input TopoJSON data
            tolerance: Simplification tolerance
            
        Returns:
            Simplified TopoJSON data
        """
        if tolerance is None:
            tolerance = self.simplification
        
        try:
            simplified = topojson_data.copy()
            
            # Simplify arcs
            simplified_arcs = []
            for arc in topojson_data.get("arcs", []):
                if len(arc) > 2:  # Only simplify if more than 2 points
                    simplified_arc = self._douglas_peucker_simplify(arc, tolerance)
                    simplified_arcs.append(simplified_arc)
                else:
                    simplified_arcs.append(arc)
            
            simplified["arcs"] = simplified_arcs
            
            logger.info(f"Topology simplified. Original arcs: {len(topojson_data.get('arcs', []))}, "
                       f"Simplified arcs: {len(simplified_arcs)}")
            
            return simplified
            
        except Exception as e:
            logger.error(f"Error simplifying topology: {e}")
            return topojson_data
    
    def _douglas_peucker_simplify(self, points: List[List[int]], tolerance: float) -> List[List[int]]:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than tolerance, recursively simplify
        if dmax > tolerance:
            results1 = self._douglas_peucker_simplify(points[:index+1], tolerance)
            results2 = self._douglas_peucker_simplify(points[index:], tolerance)
            
            return results1[:-1] + results2
        else:
            return [points[0], points[end]]
    
    def _perpendicular_distance(self, point: List[int], line_start: List[int], line_end: List[int]) -> float:
        """Calculate perpendicular distance from point to line."""
        if line_start == line_end:
            return np.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
        
        # Calculate distance using cross product
        A = line_end[0] - line_start[0]
        B = line_end[1] - line_start[1]
        C = point[0] - line_start[0]
        D = point[1] - line_start[1]
        
        dot = A * C + B * D
        len_sq = A * A + B * B
        param = dot / len_sq if len_sq != 0 else -1
        
        if param < 0:
            xx = line_start[0]
            yy = line_start[1]
        elif param > 1:
            xx = line_end[0]
            yy = line_end[1]
        else:
            xx = line_start[0] + param * A
            yy = line_start[1] + param * B
        
        dx = point[0] - xx
        dy = point[1] - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def export_topojson(self, topojson_data: Dict, output_path: Union[str, Path]):
        """
        Export TopoJSON data to file.
        
        Args:
            topojson_data: TopoJSON data to export
            output_path: Output file path
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(topojson_data, f, indent=2)
            
            logger.info(f"TopoJSON exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting TopoJSON: {e}")
            raise


def create_sample_topojson():
    """Create sample TopoJSON network data for demonstration."""
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-74.0059, 40.7128]
                },
                "properties": {
                    "id": "cell_tower_001",
                    "type": "gNodeB",
                    "band": "n78",
                    "power_dbm": 43,
                    "coverage_radius_m": 1000
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-74.0159, 40.7228]
                },
                "properties": {
                    "id": "cell_tower_002",
                    "type": "gNodeB",
                    "band": "n78",
                    "power_dbm": 43,
                    "coverage_radius_m": 1000
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-74.0059, 40.7128],
                        [-74.0109, 40.7178],
                        [-74.0159, 40.7228]
                    ]
                },
                "properties": {
                    "source": "cell_tower_001",
                    "target": "cell_tower_002",
                    "type": "backhaul",
                    "capacity_gbps": 10,
                    "latency_ms": 2
                }
            }
        ]
    }
    
    converter = TopoJSONConverter()
    topojson_data = converter.convert_geojson_to_topojson(sample_geojson)
    
    return topojson_data


# Example usage
if __name__ == "__main__":
    # Create sample TopoJSON
    sample_topojson = create_sample_topojson()
    
    # Initialize converter
    converter = TopoJSONConverter(quantization=1e6, simplification=0.001)
    
    # Export sample data
    converter.export_topojson(sample_topojson, "sample_network_topology.topojson")
    
    print("TopoJSON converter demonstration completed!")
    print(f"Sample TopoJSON structure: {json.dumps(sample_topojson, indent=2)[:500]}...")
