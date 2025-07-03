"""
Advanced Geospatial and 3D Visualization Demo

This script demonstrates the advanced visualization capabilities of the 5G Open RAN
optimizer, including GeoJSON network mapping, TopoJSON topology visualization,
and STL 3D model generation.

Usage:
    python demo_advanced_visualization.py

Features Demonstrated:
- GeoJSON network infrastructure mapping
- TopoJSON topology conversion and analysis
- STL 3D model generation for cell towers, antennas, and coverage
- Integrated visualization pipeline
- Interactive dashboard data preparation
- Batch processing capabilities
- Performance optimization

Author: AI-Powered 5G Open RAN Optimizer Team
License: MIT
"""

import json
import time
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List

# Import visualization modules
from src.visualization.geospatial.geojson_generator import (
    GeoJSON5GGenerator, CellTower, AntennaArray, BaseStation
)
from src.visualization.geospatial.topojson_converter import TopoJSONConverter
from src.visualization.geospatial.stl_3d_generator import STL3DGenerator, Vector3D
from src.visualization.geospatial.integrated_visualizer import (
    IntegratedNetworkVisualizer, VisualizationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_comprehensive_network_dataset():
    """Create a comprehensive 5G network dataset for demonstration."""
    logger.info("Creating comprehensive network dataset...")
    
    # Major US cities for demonstration
    cities = [
        {"name": "New York", "coords": [-74.0059, 40.7128]},
        {"name": "Los Angeles", "coords": [-118.2437, 34.0522]},
        {"name": "Chicago", "coords": [-87.6298, 41.8781]},
        {"name": "Houston", "coords": [-95.3698, 29.7604]},
        {"name": "Phoenix", "coords": [-112.0740, 33.4484]}
    ]
    
    network_data = {
        "name": "comprehensive_5g_network",
        "description": "Comprehensive 5G network spanning major US cities",
        "base_stations": [],
        "connections": [],
        "buildings": [],
        "coverage_areas": []
    }
    
    # Generate base stations for each city
    for i, city in enumerate(cities):
        # Multiple base stations per city
        for j in range(3):
            # Add some random offset for multiple base stations
            lat_offset = (np.random.random() - 0.5) * 0.02
            lon_offset = (np.random.random() - 0.5) * 0.02
            
            bs_position = [
                city["coords"][0] + lon_offset,
                city["coords"][1] + lat_offset
            ]
            
            base_station = {
                "id": f"bs_{city['name'].lower()}_{j:02d}",
                "position": bs_position,
                "properties": {
                    "city": city["name"],
                    "operator": f"Operator_{(i + j) % 3 + 1}",
                    "technology": "5G NR",
                    "bands": ["n78", "n79", "n41"],
                    "deployment_date": "2024-01-01"
                },
                "towers": []
            }
            
            # Add towers to base station
            tower_types = ["lattice", "monopole", "stealth"]
            tower_heights = [30, 45, 60]
            
            for k in range(2):  # 2 towers per base station
                tower = {
                    "id": f"tower_{base_station['id']}_{k}",
                    "height": tower_heights[k % len(tower_heights)],
                    "type": tower_types[k % len(tower_types)],
                    "properties": {
                        "installation_date": "2024-01-01",
                        "maintenance_status": "operational"
                    },
                    "antennas": []
                }
                
                # Add antennas to tower
                antenna_configs = [
                    {"azimuth": 0, "type": "panel", "range": 1200},
                    {"azimuth": 120, "type": "panel", "range": 1000},
                    {"azimuth": 240, "type": "panel", "range": 1100}
                ]
                
                for l, ant_config in enumerate(antenna_configs):
                    antenna = {
                        "id": f"ant_{tower['id']}_{l}",
                        "type": ant_config["type"],
                        "azimuth": ant_config["azimuth"],
                        "tilt": 8 + np.random.randint(-3, 4),
                        "range": ant_config["range"],
                        "beamwidth_az": 120,
                        "beamwidth_el": 90,
                        "properties": {
                            "frequency_band": "n78",
                            "power_dbm": 43,
                            "mimo_config": "8T8R"
                        }
                    }
                    tower["antennas"].append(antenna)
                
                base_station["towers"].append(tower)
            
            network_data["base_stations"].append(base_station)
    
    # Generate inter-city connections (backhaul)
    for i in range(len(cities) - 1):
        for j in range(3):  # 3 base stations per city
            connection = {
                "source": f"bs_{cities[i]['name'].lower()}_{j:02d}",
                "target": f"bs_{cities[i+1]['name'].lower()}_{j:02d}",
                "type": "backhaul",
                "properties": {
                    "capacity_gbps": 25 + np.random.randint(-5, 11),
                    "latency_ms": 1 + np.random.random() * 3,
                    "technology": "fiber",
                    "redundancy": "yes"
                }
            }
            network_data["connections"].append(connection)
    
    # Generate buildings for obstruction analysis
    for city in cities:
        # Add several buildings around each city
        for i in range(5):
            building_offset = (np.random.random(2) - 0.5) * 0.01
            building_position = [
                city["coords"][0] + building_offset[0],
                city["coords"][1] + building_offset[1]
            ]
            
            building = {
                "position": building_position,
                "width": 30 + np.random.randint(20, 71),
                "length": 25 + np.random.randint(15, 61),
                "height": 15 + np.random.randint(5, 86),
                "type": "rectangular",
                "properties": {
                    "building_type": np.random.choice(["office", "residential", "commercial"]),
                    "construction": "concrete"
                }
            }
            network_data["buildings"].append(building)
    
    # Generate coverage areas
    for bs in network_data["base_stations"]:
        for tower in bs["towers"]:
            for antenna in tower["antennas"]:
                coverage_area = {
                    "center": bs["position"],
                    "radius": antenna["range"],
                    "properties": {
                        "antenna_id": antenna["id"],
                        "azimuth": antenna["azimuth"],
                        "beamwidth": antenna["beamwidth_az"],
                        "signal_strength_dbm": -70 + np.random.randint(-20, 21)
                    }
                }
                network_data["coverage_areas"].append(coverage_area)
    
    logger.info(f"Created network dataset with:")
    logger.info(f"  - {len(network_data['base_stations'])} base stations")
    logger.info(f"  - {sum(len(bs['towers']) for bs in network_data['base_stations'])} towers")
    logger.info(f"  - {sum(sum(len(t['antennas']) for t in bs['towers']) for bs in network_data['base_stations'])} antennas")
    logger.info(f"  - {len(network_data['connections'])} connections")
    logger.info(f"  - {len(network_data['buildings'])} buildings")
    
    return network_data


def demo_geojson_generation(network_data: Dict, output_dir: Path):
    """Demonstrate GeoJSON generation capabilities."""
    logger.info("=== GeoJSON Generation Demo ===")
    
    generator = GeoJSON5GGenerator()
    
    # Add base stations to generator
    for bs_data in network_data["base_stations"]:
        base_station = BaseStation(
            id=bs_data["id"],
            location=bs_data["position"],
            cell_towers=[],
            properties=bs_data["properties"]
        )
        
        # Add cell towers
        for tower_data in bs_data["towers"]:
            tower = CellTower(
                id=tower_data["id"],
                location=bs_data["position"],
                height=tower_data["height"],
                antenna_arrays=[],
                properties=tower_data["properties"]
            )
            
            # Add antennas
            for antenna_data in tower_data["antennas"]:
                antenna = AntennaArray(
                    id=antenna_data["id"],
                    location=bs_data["position"],
                    azimuth=antenna_data["azimuth"],
                    tilt=antenna_data["tilt"],
                    properties=antenna_data["properties"]
                )
                tower.antenna_arrays.append(antenna)
            
            base_station.cell_towers.append(tower)
        
        generator.add_base_station(base_station)
    
    # Add connections
    for conn_data in network_data["connections"]:
        generator.add_connection(
            source_id=conn_data["source"],
            target_id=conn_data["target"],
            connection_type=conn_data["type"],
            properties=conn_data["properties"]
        )
    
    # Add coverage areas
    for coverage_data in network_data["coverage_areas"]:
        generator.add_coverage_area(
            center=coverage_data["center"],
            radius=coverage_data["radius"],
            properties=coverage_data["properties"]
        )
    
    # Generate and export GeoJSON
    geojson_data = generator.generate_geojson()
    output_path = output_dir / "comprehensive_network.geojson"
    generator.export_geojson(geojson_data, output_path)
    
    # Generate interactive map
    map_path = output_dir / "interactive_network_map.html"
    generator.create_interactive_map(geojson_data, map_path)
    
    logger.info(f"GeoJSON exported to: {output_path}")
    logger.info(f"Interactive map created: {map_path}")
    logger.info(f"Features generated: {len(geojson_data['features'])}")
    
    return geojson_data


def demo_topojson_conversion(geojson_data: Dict, output_dir: Path):
    """Demonstrate TopoJSON conversion and analysis."""
    logger.info("=== TopoJSON Conversion Demo ===")
    
    converter = TopoJSONConverter(quantization=1e6, simplification=0.001)
    
    # Convert GeoJSON to TopoJSON
    topojson_data = converter.convert_geojson_to_topojson(geojson_data)
    
    # Analyze topology
    features = geojson_data.get('features', [])
    topology = converter._build_network_topology(features)
    analysis = converter.analyze_topology(topology)
    
    # Simplify topology
    simplified_topojson = converter.simplify_topology(topojson_data, tolerance=0.01)
    
    # Export files
    topojson_path = output_dir / "network_topology.topojson"
    simplified_path = output_dir / "network_topology_simplified.topojson"
    analysis_path = output_dir / "topology_analysis.json"
    
    converter.export_topojson(topojson_data, topojson_path)
    converter.export_topojson(simplified_topojson, simplified_path)
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"TopoJSON exported to: {topojson_path}")
    logger.info(f"Simplified TopoJSON exported to: {simplified_path}")
    logger.info(f"Topology analysis saved to: {analysis_path}")
    logger.info(f"Network topology - Nodes: {analysis['basic_metrics']['node_count']}, "
               f"Edges: {analysis['basic_metrics']['edge_count']}")
    
    return topojson_data


def demo_stl_3d_generation(network_data: Dict, output_dir: Path):
    """Demonstrate STL 3D model generation."""
    logger.info("=== STL 3D Model Generation Demo ===")
    
    generator = STL3DGenerator(scale_factor=1.0, detail_level="medium")
    all_meshes = []
    
    # Generate cell tower models
    for bs_data in network_data["base_stations"][:3]:  # Limit to first 3 for demo
        position_2d = bs_data["position"]
        # Convert to local coordinates (simplified)
        position_3d = Vector3D(position_2d[0] * 100000, position_2d[1] * 100000, 0)
        
        for tower_data in bs_data["towers"]:
            # Generate tower
            tower_mesh = generator.generate_cell_tower(
                position=position_3d,
                height=tower_data["height"],
                tower_type=tower_data["type"]
            )
            all_meshes.append(tower_mesh)
            
            # Generate antennas
            antenna_height = tower_data["height"] * 0.9
            antenna_position = Vector3D(position_3d.x, position_3d.y, antenna_height)
            
            for antenna_data in tower_data["antennas"][:2]:  # Limit antennas for demo
                antenna_mesh = generator.generate_antenna_array(
                    position=antenna_position,
                    array_type=antenna_data["type"],
                    orientation=antenna_data["azimuth"]
                )
                all_meshes.append(antenna_mesh)
                
                # Generate coverage volume
                coverage_mesh = generator.generate_coverage_volume(
                    position=antenna_position,
                    range_m=antenna_data["range"] / 10,  # Scale down for visualization
                    azimuth_deg=antenna_data["beamwidth_az"],
                    elevation_deg=antenna_data["beamwidth_el"]
                )
                all_meshes.append(coverage_mesh)
    
    # Generate building models
    for building_data in network_data["buildings"][:5]:  # Limit to first 5 for demo
        position_2d = building_data["position"]
        position_3d = Vector3D(position_2d[0] * 100000, position_2d[1] * 100000, 0)
        
        building_mesh = generator.generate_building(
            position=position_3d,
            width=building_data["width"],
            length=building_data["length"],
            height=building_data["height"],
            building_type=building_data["type"]
        )
        all_meshes.append(building_mesh)
    
    # Export individual meshes
    individual_dir = output_dir / "individual_models"
    individual_dir.mkdir(exist_ok=True)
    
    for i, mesh in enumerate(all_meshes):
        mesh_path = individual_dir / f"{mesh.name}_{i:03d}.stl"
        generator.export_stl_ascii(mesh, mesh_path)
    
    # Create combined scene
    if all_meshes:
        combined_mesh = generator.combine_meshes(all_meshes)
        
        # Export combined scene in both formats
        combined_ascii_path = output_dir / "complete_5g_network_scene.stl"
        combined_binary_path = output_dir / "complete_5g_network_scene_binary.stl"
        
        generator.export_stl_ascii(combined_mesh, combined_ascii_path)
        generator.export_stl_binary(combined_mesh, combined_binary_path)
        
        logger.info(f"Individual models exported to: {individual_dir}")
        logger.info(f"Combined scene (ASCII) exported to: {combined_ascii_path}")
        logger.info(f"Combined scene (Binary) exported to: {combined_binary_path}")
        logger.info(f"Total meshes: {len(all_meshes)}, Total triangles: {len(combined_mesh.triangles)}")
    
    return all_meshes


def demo_integrated_visualization(network_data: Dict, output_dir: Path):
    """Demonstrate integrated visualization pipeline."""
    logger.info("=== Integrated Visualization Demo ===")
    
    # Configure visualization
    config = VisualizationConfig(
        output_formats=["geojson", "topojson", "stl"],
        detail_level="medium",
        enable_3d=True,
        enable_topology=True,
        enable_coverage=True
    )
    
    # Initialize integrated visualizer
    visualizer = IntegratedNetworkVisualizer(config)
    
    # Generate complete visualization
    integrated_dir = output_dir / "integrated"
    result = visualizer.generate_complete_visualization(network_data, integrated_dir)
    
    # Create dashboard data
    dashboard_path = integrated_dir / "dashboard_config.json"
    dashboard_data = visualizer.create_interactive_dashboard_data(result, dashboard_path)
    
    # Optimize for web
    optimized_result = visualizer.optimize_for_web(result)
    
    # Export optimized version
    optimized_dir = output_dir / "optimized"
    optimized_dir.mkdir(exist_ok=True)
    
    if optimized_result.geojson_data:
        with open(optimized_dir / "optimized_network.geojson", 'w') as f:
            json.dump(optimized_result.geojson_data, f)
    
    if optimized_result.topojson_data:
        with open(optimized_dir / "optimized_topology.topojson", 'w') as f:
            json.dump(optimized_result.topojson_data, f)
    
    logger.info(f"Integrated visualization completed in {result.generation_time:.2f}s")
    logger.info(f"Generated formats: {result.metadata['generation_info']['formats_generated']}")
    logger.info(f"Dashboard configuration: {dashboard_path}")
    logger.info(f"Optimized files: {optimized_dir}")
    
    return result


def demo_batch_processing(output_dir: Path):
    """Demonstrate batch processing capabilities."""
    logger.info("=== Batch Processing Demo ===")
    
    # Create multiple network datasets
    datasets = []
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    
    for region in regions:
        # Create simplified dataset for each region
        dataset = {
            "name": f"5g_network_{region.lower()}",
            "description": f"5G network deployment in {region} region",
            "base_stations": [],
            "connections": [],
            "buildings": []
        }
        
        # Add a few base stations per region
        for i in range(2):
            bs = {
                "id": f"bs_{region.lower()}_{i:02d}",
                "position": [
                    -120 + np.random.random() * 40,  # Random US longitude
                    25 + np.random.random() * 25     # Random US latitude
                ],
                "properties": {"region": region},
                "towers": [{
                    "id": f"tower_{region.lower()}_{i}",
                    "height": 30 + np.random.randint(20),
                    "type": "lattice",
                    "antennas": [{
                        "id": f"ant_{region.lower()}_{i}",
                        "type": "panel",
                        "azimuth": 0,
                        "range": 1000
                    }]
                }]
            }
            dataset["base_stations"].append(bs)
        
        datasets.append(dataset)
    
    # Configure for batch processing
    config = VisualizationConfig(
        output_formats=["geojson", "topojson"],  # Skip STL for faster batch processing
        detail_level="low",
        enable_3d=False,
        enable_topology=True,
        enable_coverage=False
    )
    
    visualizer = IntegratedNetworkVisualizer(config)
    
    # Process batch
    batch_dir = output_dir / "batch_processing"
    results = visualizer.generate_batch_visualizations(datasets, batch_dir)
    
    logger.info(f"Batch processing completed:")
    logger.info(f"  - Processed {len(results)} datasets")
    logger.info(f"  - Total processing time: {sum(r.generation_time for r in results):.2f}s")
    logger.info(f"  - Average time per dataset: {np.mean([r.generation_time for r in results]):.2f}s")
    logger.info(f"  - Batch summary: {batch_dir / 'batch_summary.json'}")
    
    return results


def demo_performance_analysis(network_data: Dict, output_dir: Path):
    """Demonstrate performance analysis and optimization."""
    logger.info("=== Performance Analysis Demo ===")
    
    performance_results = {
        "detail_levels": {},
        "format_comparisons": {},
        "optimization_impact": {}
    }
    
    # Test different detail levels
    detail_levels = ["low", "medium", "high"]
    
    for detail_level in detail_levels:
        logger.info(f"Testing detail level: {detail_level}")
        
        config = VisualizationConfig(
            output_formats=["geojson", "stl"],
            detail_level=detail_level,
            enable_3d=True,
            enable_topology=False,
            enable_coverage=False
        )
        
        visualizer = IntegratedNetworkVisualizer(config)
        
        # Use subset of data for performance testing
        test_data = {
            "name": f"perf_test_{detail_level}",
            "base_stations": network_data["base_stations"][:2],  # Limit data
            "connections": network_data["connections"][:2],
            "buildings": network_data["buildings"][:3]
        }
        
        start_time = time.time()
        test_dir = output_dir / f"performance_{detail_level}"
        result = visualizer.generate_complete_visualization(test_data, test_dir)
        
        performance_results["detail_levels"][detail_level] = {
            "generation_time": result.generation_time,
            "triangle_count": result.metadata.get("stl_metadata", {}).get("total_triangles", 0),
            "feature_count": result.metadata.get("geojson_metadata", {}).get("feature_count", 0)
        }
    
    # Test format-specific performance
    formats = ["geojson", "topojson", "stl"]
    
    for format_type in formats:
        logger.info(f"Testing format: {format_type}")
        
        config = VisualizationConfig(
            output_formats=[format_type],
            detail_level="medium"
        )
        
        visualizer = IntegratedNetworkVisualizer(config)
        test_data = {
            "name": f"format_test_{format_type}",
            "base_stations": network_data["base_stations"][:2],
            "connections": network_data["connections"][:2],
            "buildings": network_data["buildings"][:3]
        }
        
        start_time = time.time()
        test_dir = output_dir / f"format_{format_type}"
        result = visualizer.generate_complete_visualization(test_data, test_dir)
        
        performance_results["format_comparisons"][format_type] = {
            "generation_time": result.generation_time,
            "memory_usage": "N/A"  # Would need memory profiling for actual measurement
        }
    
    # Test optimization impact
    logger.info("Testing optimization impact...")
    
    config = VisualizationConfig(output_formats=["geojson", "topojson"])
    visualizer = IntegratedNetworkVisualizer(config)
    
    test_data = {
        "name": "optimization_test",
        "base_stations": network_data["base_stations"][:3],
        "connections": network_data["connections"][:3],
        "buildings": network_data["buildings"][:5]
    }
    
    # Generate original
    orig_start = time.time()
    orig_dir = output_dir / "optimization_original"
    original_result = visualizer.generate_complete_visualization(test_data, orig_dir)
    
    # Generate optimized
    opt_start = time.time()
    optimized_result = visualizer.optimize_for_web(original_result)
    opt_time = time.time() - opt_start
    
    performance_results["optimization_impact"] = {
        "original_generation_time": original_result.generation_time,
        "optimization_time": opt_time,
        "size_reduction": optimized_result.metadata.get("optimization", {})
    }
    
    # Save performance results
    perf_path = output_dir / "performance_analysis.json"
    with open(perf_path, 'w') as f:
        json.dump(performance_results, f, indent=2)
    
    logger.info(f"Performance analysis completed: {perf_path}")
    logger.info("Detail level performance:")
    for level, metrics in performance_results["detail_levels"].items():
        logger.info(f"  {level}: {metrics['generation_time']:.2f}s, "
                   f"{metrics['triangle_count']} triangles")
    
    return performance_results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Advanced 5G Network Visualization Demo")
    parser.add_argument("--output-dir", default="demo_output",
                      help="Output directory for generated files")
    parser.add_argument("--skip-3d", action="store_true",
                      help="Skip 3D STL generation for faster demo")
    parser.add_argument("--skip-batch", action="store_true",
                      help="Skip batch processing demo")
    parser.add_argument("--skip-performance", action="store_true",
                      help="Skip performance analysis")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting Advanced 5G Network Visualization Demo")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Create comprehensive network dataset
    network_data = create_comprehensive_network_dataset()
    
    # Save the dataset for reference
    dataset_path = output_dir / "network_dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(network_data, f, indent=2)
    logger.info(f"Network dataset saved: {dataset_path}")
    
    # Run demonstrations
    demo_results = {}
    
    try:
        # 1. GeoJSON Generation Demo
        geojson_data = demo_geojson_generation(network_data, output_dir / "geojson")
        demo_results["geojson"] = "✅ Completed"
        
        # 2. TopoJSON Conversion Demo
        topojson_data = demo_topojson_conversion(geojson_data, output_dir / "topojson")
        demo_results["topojson"] = "✅ Completed"
        
        # 3. STL 3D Generation Demo (optional)
        if not args.skip_3d:
            stl_meshes = demo_stl_3d_generation(network_data, output_dir / "stl")
            demo_results["stl_3d"] = "✅ Completed"
        else:
            demo_results["stl_3d"] = "⏩ Skipped"
        
        # 4. Integrated Visualization Demo
        integrated_result = demo_integrated_visualization(network_data, output_dir / "integrated")
        demo_results["integrated"] = "✅ Completed"
        
        # 5. Batch Processing Demo (optional)
        if not args.skip_batch:
            batch_results = demo_batch_processing(output_dir)
            demo_results["batch_processing"] = "✅ Completed"
        else:
            demo_results["batch_processing"] = "⏩ Skipped"
        
        # 6. Performance Analysis (optional)
        if not args.skip_performance:
            performance_results = demo_performance_analysis(network_data, output_dir / "performance")
            demo_results["performance_analysis"] = "✅ Completed"
        else:
            demo_results["performance_analysis"] = "⏩ Skipped"
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    
    # Generate summary report
    summary = {
        "demo_completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_directory": str(output_dir.absolute()),
        "network_statistics": {
            "base_stations": len(network_data["base_stations"]),
            "towers": sum(len(bs["towers"]) for bs in network_data["base_stations"]),
            "antennas": sum(sum(len(t["antennas"]) for t in bs["towers"]) 
                          for bs in network_data["base_stations"]),
            "connections": len(network_data["connections"]),
            "buildings": len(network_data["buildings"])
        },
        "demo_results": demo_results,
        "generated_files": {
            "geojson": "network_visualization.geojson",
            "topojson": "network_topology.topojson",
            "stl_models": "complete_5g_network_scene.stl",
            "interactive_map": "interactive_network_map.html",
            "dashboard_config": "integrated/dashboard_config.json"
        },
        "next_steps": [
            "Open the interactive map in a web browser",
            "Load STL models in a 3D viewer (Blender, MeshLab, etc.)",
            "Use the dashboard configuration for web deployment",
            "Integrate with existing network management systems",
            "Customize visualization parameters for specific use cases"
        ]
    }
    
    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("🎉 ADVANCED 5G NETWORK VISUALIZATION DEMO COMPLETED")
    print("="*80)
    print(f"📁 Output Directory: {output_dir.absolute()}")
    print("\n📊 Demo Results:")
    for demo_name, status in demo_results.items():
        print(f"  • {demo_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 Network Scale:")
    print(f"  • Base Stations: {summary['network_statistics']['base_stations']}")
    print(f"  • Cell Towers: {summary['network_statistics']['towers']}")
    print(f"  • Antennas: {summary['network_statistics']['antennas']}")
    print(f"  • Connections: {summary['network_statistics']['connections']}")
    print(f"  • Buildings: {summary['network_statistics']['buildings']}")
    
    print(f"\n🗂️  Key Generated Files:")
    for file_type, filename in summary["generated_files"].items():
        print(f"  • {file_type.upper()}: {filename}")
    
    print(f"\n📋 Summary Report: {summary_path}")
    print("\n🎯 Next Steps:")
    for i, step in enumerate(summary["next_steps"], 1):
        print(f"  {i}. {step}")
    
    print("\n🚀 The advanced visualization capabilities are now ready for production use!")
    print("="*80)


if __name__ == "__main__":
    main()
