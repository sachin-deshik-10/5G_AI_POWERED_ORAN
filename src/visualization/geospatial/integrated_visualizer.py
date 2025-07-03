"""
Integrated Visualization Module for 5G Network Infrastructure

This module provides a unified interface for all geospatial and 3D visualization
capabilities, integrating GeoJSON, TopoJSON, and STL 3D model generation.

Features:
- Unified visualization pipeline
- Multi-format export (GeoJSON, TopoJSON, STL)
- Interactive dashboard integration
- Real-time network visualization
- Performance optimization
- Batch processing capabilities

Author: AI-Powered 5G Open RAN Optimizer Team
License: MIT
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .geojson_generator import GeoJSON5GGenerator, CellTower, AntennaArray, BaseStation
from .topojson_converter import TopoJSONConverter, NetworkTopology
from .stl_3d_generator import STL3DGenerator, Vector3D, STLMesh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_formats: List[str] = None  # ["geojson", "topojson", "stl"]
    detail_level: str = "medium"  # "low", "medium", "high"
    enable_3d: bool = True
    enable_topology: bool = True
    enable_coverage: bool = True
    coordinate_system: str = "WGS84"
    units: str = "meters"
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["geojson", "topojson", "stl"]


@dataclass
class NetworkVisualizationData:
    """Complete network visualization dataset."""
    geojson_data: Optional[Dict] = None
    topojson_data: Optional[Dict] = None
    stl_meshes: Optional[List[STLMesh]] = None
    metadata: Optional[Dict] = None
    generation_time: float = 0.0


class IntegratedNetworkVisualizer:
    """
    Integrated visualization system for 5G network infrastructure.
    
    Provides unified access to GeoJSON, TopoJSON, and STL 3D visualization
    capabilities with optimized performance and batch processing.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize integrated visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize component generators
        self.geojson_generator = GeoJSON5GGenerator()
        self.topojson_converter = TopoJSONConverter()
        self.stl_generator = STL3DGenerator(detail_level=self.config.detail_level)
        
        # Performance tracking
        self.performance_metrics = {
            "generation_times": [],
            "memory_usage": [],
            "output_sizes": {}
        }
        
        logger.info(f"Integrated visualizer initialized with config: {asdict(self.config)}")
    
    def generate_complete_visualization(self, network_data: Dict, 
                                      output_directory: Union[str, Path]) -> NetworkVisualizationData:
        """
        Generate complete visualization suite for network data.
        
        Args:
            network_data: Input network configuration data
            output_directory: Directory for output files
            
        Returns:
            Complete visualization dataset
        """
        start_time = time.time()
        
        try:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Starting complete visualization generation...")
            
            result = NetworkVisualizationData()
            
            # Generate GeoJSON
            if "geojson" in self.config.output_formats:
                result.geojson_data = self._generate_geojson_visualization(network_data, output_dir)
            
            # Generate TopoJSON
            if "topojson" in self.config.output_formats and result.geojson_data:
                result.topojson_data = self._generate_topojson_visualization(result.geojson_data, output_dir)
            
            # Generate STL 3D models
            if "stl" in self.config.output_formats:
                result.stl_meshes = self._generate_stl_visualization(network_data, output_dir)
            
            # Generate metadata
            result.metadata = self._generate_metadata(network_data, result)
            result.generation_time = time.time() - start_time
            
            # Save combined metadata
            self._save_metadata(result.metadata, output_dir / "visualization_metadata.json")
            
            logger.info(f"Complete visualization generated in {result.generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating complete visualization: {e}")
            raise
    
    def _generate_geojson_visualization(self, network_data: Dict, output_dir: Path) -> Dict:
        """Generate GeoJSON visualization."""
        logger.info("Generating GeoJSON visualization...")
        
        # Extract network components
        base_stations = network_data.get('base_stations', [])
        connections = network_data.get('connections', [])
        coverage_areas = network_data.get('coverage_areas', [])
        
        # Create base stations
        for bs_data in base_stations:
            position = bs_data.get('position', [0, 0])
            
            base_station = BaseStation(
                id=bs_data.get('id', 'unknown'),
                location=position,
                cell_towers=[],
                properties=bs_data.get('properties', {})
            )
            
            # Add cell towers
            for tower_data in bs_data.get('towers', []):
                tower = CellTower(
                    id=tower_data.get('id', 'unknown'),
                    location=position,
                    height=tower_data.get('height', 30),
                    antenna_arrays=[],
                    properties=tower_data.get('properties', {})
                )
                
                # Add antennas
                for antenna_data in tower_data.get('antennas', []):
                    antenna = AntennaArray(
                        id=antenna_data.get('id', 'unknown'),
                        location=position,
                        azimuth=antenna_data.get('azimuth', 0),
                        tilt=antenna_data.get('tilt', 0),
                        properties=antenna_data.get('properties', {})
                    )
                    tower.antenna_arrays.append(antenna)
                
                base_station.cell_towers.append(tower)
            
            self.geojson_generator.add_base_station(base_station)
        
        # Add connections
        for conn_data in connections:
            self.geojson_generator.add_connection(
                source_id=conn_data.get('source'),
                target_id=conn_data.get('target'),
                connection_type=conn_data.get('type', 'backhaul'),
                properties=conn_data.get('properties', {})
            )
        
        # Add coverage areas
        if self.config.enable_coverage:
            for coverage_data in coverage_areas:
                self.geojson_generator.add_coverage_area(
                    center=coverage_data.get('center'),
                    radius=coverage_data.get('radius', 1000),
                    properties=coverage_data.get('properties', {})
                )
        
        # Generate GeoJSON
        geojson_data = self.geojson_generator.generate_geojson()
        
        # Export to file
        output_path = output_dir / "network_visualization.geojson"
        self.geojson_generator.export_geojson(geojson_data, output_path)
        
        return geojson_data
    
    def _generate_topojson_visualization(self, geojson_data: Dict, output_dir: Path) -> Dict:
        """Generate TopoJSON visualization from GeoJSON data."""
        logger.info("Generating TopoJSON visualization...")
        
        if not self.config.enable_topology:
            return None
        
        # Convert GeoJSON to TopoJSON
        topojson_data = self.topojson_converter.convert_geojson_to_topojson(geojson_data)
        
        # Simplify if needed
        if self.config.detail_level == "low":
            topojson_data = self.topojson_converter.simplify_topology(topojson_data, tolerance=0.01)
        
        # Export to file
        output_path = output_dir / "network_topology.topojson"
        self.topojson_converter.export_topojson(topojson_data, output_path)
        
        return topojson_data
    
    def _generate_stl_visualization(self, network_data: Dict, output_dir: Path) -> List[STLMesh]:
        """Generate STL 3D visualization."""
        logger.info("Generating STL 3D visualization...")
        
        if not self.config.enable_3d:
            return []
        
        meshes = []
        
        # Generate base station models
        for bs_data in network_data.get('base_stations', []):
            position_2d = bs_data.get('position', [0, 0])
            position_3d = Vector3D(position_2d[0], position_2d[1], 0)
            
            # Generate towers
            for tower_data in bs_data.get('towers', []):
                tower_mesh = self.stl_generator.generate_cell_tower(
                    position=position_3d,
                    height=tower_data.get('height', 30),
                    tower_type=tower_data.get('type', 'lattice')
                )
                meshes.append(tower_mesh)
                
                # Generate antennas
                antenna_height = tower_data.get('height', 30) * 0.9
                antenna_position = Vector3D(position_2d[0], position_2d[1], antenna_height)
                
                for antenna_data in tower_data.get('antennas', []):
                    antenna_mesh = self.stl_generator.generate_antenna_array(
                        position=antenna_position,
                        array_type=antenna_data.get('type', 'panel'),
                        orientation=antenna_data.get('azimuth', 0)
                    )
                    meshes.append(antenna_mesh)
                    
                    # Generate coverage volume if enabled
                    if self.config.enable_coverage:
                        coverage_mesh = self.stl_generator.generate_coverage_volume(
                            position=antenna_position,
                            range_m=antenna_data.get('range', 1000),
                            azimuth_deg=antenna_data.get('beamwidth_az', 120),
                            elevation_deg=antenna_data.get('beamwidth_el', 90)
                        )
                        meshes.append(coverage_mesh)
        
        # Generate buildings if provided
        for building_data in network_data.get('buildings', []):
            position_2d = building_data.get('position', [0, 0])
            position_3d = Vector3D(position_2d[0], position_2d[1], 0)
            
            building_mesh = self.stl_generator.generate_building(
                position=position_3d,
                width=building_data.get('width', 50),
                length=building_data.get('length', 50),
                height=building_data.get('height', 20),
                building_type=building_data.get('type', 'rectangular')
            )
            meshes.append(building_mesh)
        
        # Export individual meshes
        for i, mesh in enumerate(meshes):
            output_path = output_dir / f"model_{mesh.name}_{i:03d}.stl"
            self.stl_generator.export_stl_ascii(mesh, output_path)
        
        # Create combined scene
        if meshes:
            combined_mesh = self.stl_generator.combine_meshes(meshes)
            combined_path = output_dir / "complete_network_scene.stl"
            self.stl_generator.export_stl_ascii(combined_mesh, combined_path)
            
            # Also export binary version for smaller file size
            binary_path = output_dir / "complete_network_scene_binary.stl"
            self.stl_generator.export_stl_binary(combined_mesh, binary_path)
        
        return meshes
    
    def _generate_metadata(self, network_data: Dict, result: NetworkVisualizationData) -> Dict:
        """Generate comprehensive metadata for the visualization."""
        metadata = {
            "generation_info": {
                "timestamp": time.time(),
                "config": asdict(self.config),
                "generation_time": result.generation_time,
                "formats_generated": []
            },
            "network_statistics": {
                "base_stations": len(network_data.get('base_stations', [])),
                "total_towers": sum(len(bs.get('towers', [])) for bs in network_data.get('base_stations', [])),
                "total_antennas": sum(
                    sum(len(tower.get('antennas', [])) for tower in bs.get('towers', []))
                    for bs in network_data.get('base_stations', [])
                ),
                "connections": len(network_data.get('connections', [])),
                "buildings": len(network_data.get('buildings', []))
            },
            "output_files": {},
            "performance_metrics": {}
        }
        
        # Add format-specific metadata
        if result.geojson_data:
            metadata["generation_info"]["formats_generated"].append("geojson")
            metadata["geojson_metadata"] = {
                "feature_count": len(result.geojson_data.get('features', [])),
                "bbox": result.geojson_data.get('bbox'),
                "coordinate_system": self.config.coordinate_system
            }
        
        if result.topojson_data:
            metadata["generation_info"]["formats_generated"].append("topojson")
            metadata["topojson_metadata"] = {
                "arc_count": len(result.topojson_data.get('arcs', [])),
                "object_count": len(result.topojson_data.get('objects', {}).get('network', {}).get('geometries', [])),
                "quantization": result.topojson_data.get('transform', {}).get('scale', [])
            }
        
        if result.stl_meshes:
            metadata["generation_info"]["formats_generated"].append("stl")
            metadata["stl_metadata"] = {
                "mesh_count": len(result.stl_meshes),
                "total_triangles": sum(len(mesh.triangles) for mesh in result.stl_meshes),
                "detail_level": self.config.detail_level,
                "individual_meshes": [
                    {
                        "name": mesh.name,
                        "triangles": len(mesh.triangles),
                        "type": mesh.metadata.get('type', 'unknown')
                    }
                    for mesh in result.stl_meshes
                ]
            }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict, output_path: Path):
        """Save metadata to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def generate_batch_visualizations(self, network_datasets: List[Dict], 
                                    output_base_directory: Union[str, Path]) -> List[NetworkVisualizationData]:
        """
        Generate visualizations for multiple network datasets in batch.
        
        Args:
            network_datasets: List of network configuration datasets
            output_base_directory: Base directory for outputs
            
        Returns:
            List of visualization results
        """
        logger.info(f"Starting batch visualization generation for {len(network_datasets)} datasets...")
        
        output_base = Path(output_base_directory)
        results = []
        
        for i, network_data in enumerate(network_datasets):
            try:
                dataset_name = network_data.get('name', f'dataset_{i:03d}')
                output_dir = output_base / dataset_name
                
                logger.info(f"Processing dataset {i+1}/{len(network_datasets)}: {dataset_name}")
                
                result = self.generate_complete_visualization(network_data, output_dir)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing dataset {i}: {e}")
                continue
        
        # Generate batch summary
        self._generate_batch_summary(results, output_base)
        
        logger.info(f"Batch processing completed. {len(results)} datasets processed successfully.")
        return results
    
    def _generate_batch_summary(self, results: List[NetworkVisualizationData], output_dir: Path):
        """Generate summary report for batch processing."""
        summary = {
            "batch_info": {
                "total_datasets": len(results),
                "successful_generations": len([r for r in results if r.geojson_data or r.stl_meshes]),
                "total_processing_time": sum(r.generation_time for r in results),
                "average_processing_time": np.mean([r.generation_time for r in results]) if results else 0
            },
            "aggregate_statistics": {
                "total_features": sum(
                    len(r.geojson_data.get('features', [])) if r.geojson_data else 0
                    for r in results
                ),
                "total_meshes": sum(
                    len(r.stl_meshes) if r.stl_meshes else 0
                    for r in results
                ),
                "total_triangles": sum(
                    sum(len(mesh.triangles) for mesh in r.stl_meshes) if r.stl_meshes else 0
                    for r in results
                )
            },
            "performance_metrics": {
                "generation_times": [r.generation_time for r in results],
                "formats_distribution": {}
            }
        }
        
        # Count format usage
        for result in results:
            if result.geojson_data:
                summary["performance_metrics"]["formats_distribution"]["geojson"] = \
                    summary["performance_metrics"]["formats_distribution"].get("geojson", 0) + 1
            if result.topojson_data:
                summary["performance_metrics"]["formats_distribution"]["topojson"] = \
                    summary["performance_metrics"]["formats_distribution"].get("topojson", 0) + 1
            if result.stl_meshes:
                summary["performance_metrics"]["formats_distribution"]["stl"] = \
                    summary["performance_metrics"]["formats_distribution"].get("stl", 0) + 1
        
        # Save summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Batch summary saved to: {summary_path}")
    
    def create_interactive_dashboard_data(self, visualization_data: NetworkVisualizationData,
                                        output_path: Union[str, Path]) -> Dict:
        """
        Create optimized data package for interactive dashboard visualization.
        
        Args:
            visualization_data: Generated visualization data
            output_path: Output path for dashboard data
            
        Returns:
            Dashboard data configuration
        """
        logger.info("Creating interactive dashboard data...")
        
        dashboard_data = {
            "metadata": visualization_data.metadata,
            "config": {
                "initial_view": {
                    "center": [0, 0],
                    "zoom": 10,
                    "bearing": 0,
                    "pitch": 45
                },
                "layers": [],
                "controls": {
                    "navigation": True,
                    "fullscreen": True,
                    "scale": True,
                    "layer_switcher": True
                }
            },
            "data_sources": {}
        }
        
        # Add GeoJSON layer
        if visualization_data.geojson_data:
            dashboard_data["config"]["layers"].append({
                "id": "network_geojson",
                "type": "geojson",
                "source": "network_geojson",
                "visible": True,
                "interactive": True
            })
            dashboard_data["data_sources"]["network_geojson"] = {
                "type": "geojson",
                "data": visualization_data.geojson_data
            }
        
        # Add TopoJSON layer
        if visualization_data.topojson_data:
            dashboard_data["config"]["layers"].append({
                "id": "network_topology",
                "type": "topojson",
                "source": "network_topology",
                "visible": False,
                "interactive": True
            })
            dashboard_data["data_sources"]["network_topology"] = {
                "type": "topojson",
                "data": visualization_data.topojson_data
            }
        
        # Add 3D model references
        if visualization_data.stl_meshes:
            dashboard_data["config"]["layers"].append({
                "id": "network_3d",
                "type": "3d_models",
                "source": "stl_models",
                "visible": False,
                "interactive": True
            })
            dashboard_data["data_sources"]["stl_models"] = {
                "type": "stl_collection",
                "models": [
                    {
                        "name": mesh.name,
                        "metadata": mesh.metadata,
                        "triangle_count": len(mesh.triangles)
                    }
                    for mesh in visualization_data.stl_meshes
                ]
            }
        
        # Save dashboard configuration
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Dashboard data created: {output_path}")
        return dashboard_data
    
    def optimize_for_web(self, visualization_data: NetworkVisualizationData) -> NetworkVisualizationData:
        """
        Optimize visualization data for web deployment.
        
        Args:
            visualization_data: Original visualization data
            
        Returns:
            Optimized visualization data
        """
        logger.info("Optimizing visualization data for web deployment...")
        
        optimized = NetworkVisualizationData()
        
        # Optimize GeoJSON (reduce precision, remove unnecessary properties)
        if visualization_data.geojson_data:
            optimized.geojson_data = self._optimize_geojson(visualization_data.geojson_data)
        
        # Optimize TopoJSON (increase simplification)
        if visualization_data.topojson_data:
            optimized.topojson_data = self.topojson_converter.simplify_topology(
                visualization_data.topojson_data, tolerance=0.001
            )
        
        # For STL, create lower detail versions
        if visualization_data.stl_meshes:
            low_detail_generator = STL3DGenerator(detail_level="low")
            # Note: In practice, you'd regenerate with lower detail settings
            optimized.stl_meshes = visualization_data.stl_meshes
        
        optimized.metadata = visualization_data.metadata.copy()
        optimized.metadata["optimization"] = {
            "optimized_for": "web",
            "original_sizes": self._calculate_data_sizes(visualization_data),
            "optimized_sizes": self._calculate_data_sizes(optimized)
        }
        
        return optimized
    
    def _optimize_geojson(self, geojson_data: Dict) -> Dict:
        """Optimize GeoJSON for web use."""
        optimized = geojson_data.copy()
        
        # Reduce coordinate precision
        for feature in optimized.get('features', []):
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Point':
                coords = geometry['coordinates']
                geometry['coordinates'] = [round(coords[0], 6), round(coords[1], 6)]
            elif geometry.get('type') == 'LineString':
                coords = geometry['coordinates']
                geometry['coordinates'] = [[round(c[0], 6), round(c[1], 6)] for c in coords]
        
        return optimized
    
    def _calculate_data_sizes(self, visualization_data: NetworkVisualizationData) -> Dict:
        """Calculate approximate data sizes."""
        sizes = {}
        
        if visualization_data.geojson_data:
            sizes['geojson'] = len(json.dumps(visualization_data.geojson_data))
        
        if visualization_data.topojson_data:
            sizes['topojson'] = len(json.dumps(visualization_data.topojson_data))
        
        if visualization_data.stl_meshes:
            sizes['stl'] = sum(len(mesh.triangles) * 50 for mesh in visualization_data.stl_meshes)  # Rough estimate
        
        return sizes


def create_sample_network_dataset():
    """Create sample network dataset for demonstration."""
    return {
        "name": "sample_5g_network",
        "description": "Sample 5G network for visualization testing",
        "base_stations": [
            {
                "id": "bs_001",
                "position": [-74.0059, 40.7128],  # NYC coordinates
                "properties": {
                    "operator": "TestCorp",
                    "technology": "5G NR",
                    "band": "n78"
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
                                "tilt": 10,
                                "range": 1000,
                                "beamwidth_az": 120,
                                "beamwidth_el": 90
                            },
                            {
                                "id": "ant_002",
                                "type": "panel",
                                "azimuth": 120,
                                "tilt": 10,
                                "range": 1000,
                                "beamwidth_az": 120,
                                "beamwidth_el": 90
                            }
                        ]
                    }
                ]
            },
            {
                "id": "bs_002",
                "position": [-74.0159, 40.7228],
                "properties": {
                    "operator": "TestCorp",
                    "technology": "5G NR",
                    "band": "n78"
                },
                "towers": [
                    {
                        "id": "tower_002",
                        "height": 45,
                        "type": "monopole",
                        "antennas": [
                            {
                                "id": "ant_003",
                                "type": "panel",
                                "azimuth": 180,
                                "tilt": 8,
                                "range": 800,
                                "beamwidth_az": 120,
                                "beamwidth_el": 90
                            }
                        ]
                    }
                ]
            }
        ],
        "connections": [
            {
                "source": "bs_001",
                "target": "bs_002",
                "type": "backhaul",
                "properties": {
                    "capacity_gbps": 10,
                    "latency_ms": 2,
                    "technology": "fiber"
                }
            }
        ],
        "buildings": [
            {
                "position": [-74.0109, 40.7178],
                "width": 50,
                "length": 30,
                "height": 20,
                "type": "rectangular"
            }
        ]
    }


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = VisualizationConfig(
        output_formats=["geojson", "topojson", "stl"],
        detail_level="medium",
        enable_3d=True,
        enable_topology=True,
        enable_coverage=True
    )
    
    # Initialize visualizer
    visualizer = IntegratedNetworkVisualizer(config)
    
    # Create sample network data
    network_data = create_sample_network_dataset()
    
    # Generate complete visualization
    result = visualizer.generate_complete_visualization(network_data, "output/sample_network")
    
    # Create dashboard data
    dashboard_data = visualizer.create_interactive_dashboard_data(result, "output/sample_network/dashboard_config.json")
    
    # Optimize for web
    optimized_result = visualizer.optimize_for_web(result)
    
    print("Integrated visualization generation completed!")
    print(f"Generated formats: {result.metadata['generation_info']['formats_generated']}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Total features: {result.metadata.get('geojson_metadata', {}).get('feature_count', 0)}")
    print(f"Total meshes: {result.metadata.get('stl_metadata', {}).get('mesh_count', 0)}")
