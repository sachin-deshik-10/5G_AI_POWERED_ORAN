"""
STL 3D Model Generator for 5G Network Infrastructure Visualization

This module provides functionality to generate 3D STL models of 5G network infrastructure
for immersive visualization, simulation, and analysis.

Features:
- 3D cell tower and antenna models
- Network coverage volume visualization
- Terrain-aware placement
- Building and obstacle modeling
- Interference pattern visualization
- Export to STL format for 3D printing and CAD

Author: AI-Powered 5G Open RAN Optimizer Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Vector3D:
    """3D vector representation."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def normalize(self) -> 'Vector3D':
        magnitude = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if magnitude == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / magnitude, self.y / magnitude, self.z / magnitude)


@dataclass
class Triangle:
    """3D triangle for STL mesh."""
    vertices: List[Vector3D]
    normal: Vector3D
    
    def calculate_normal(self):
        """Calculate triangle normal vector."""
        v1 = self.vertices[1] + Vector3D(-self.vertices[0].x, -self.vertices[0].y, -self.vertices[0].z)
        v2 = self.vertices[2] + Vector3D(-self.vertices[0].x, -self.vertices[0].y, -self.vertices[0].z)
        
        # Cross product
        normal = Vector3D(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x
        )
        
        self.normal = normal.normalize()


@dataclass
class STLMesh:
    """STL mesh representation."""
    triangles: List[Triangle]
    name: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class STL3DGenerator:
    """
    Generates 3D STL models for 5G network infrastructure visualization.
    
    Creates realistic 3D models of cell towers, antennas, coverage areas,
    and network infrastructure for visualization and analysis.
    """
    
    def __init__(self, scale_factor: float = 1.0, detail_level: str = "medium"):
        """
        Initialize STL 3D generator.
        
        Args:
            scale_factor: Scaling factor for all models
            detail_level: Level of detail ("low", "medium", "high")
        """
        self.scale_factor = scale_factor
        self.detail_level = detail_level
        self.meshes = []
        
        # Detail level settings
        self.detail_settings = {
            "low": {"segments": 8, "height_segments": 4},
            "medium": {"segments": 16, "height_segments": 8},
            "high": {"segments": 32, "height_segments": 16}
        }
    
    def generate_cell_tower(self, position: Vector3D, height: float = 50.0, 
                          base_width: float = 2.0, tower_type: str = "lattice") -> STLMesh:
        """
        Generate a 3D cell tower model.
        
        Args:
            position: Tower base position
            height: Tower height in meters
            base_width: Base width in meters
            tower_type: Type of tower ("lattice", "monopole", "stealth")
            
        Returns:
            STL mesh of the cell tower
        """
        try:
            triangles = []
            
            if tower_type == "lattice":
                triangles = self._create_lattice_tower(position, height, base_width)
            elif tower_type == "monopole":
                triangles = self._create_monopole_tower(position, height, base_width)
            elif tower_type == "stealth":
                triangles = self._create_stealth_tower(position, height, base_width)
            else:
                triangles = self._create_lattice_tower(position, height, base_width)
            
            mesh = STLMesh(
                triangles=triangles,
                name=f"cell_tower_{tower_type}",
                metadata={
                    "type": "cell_tower",
                    "tower_type": tower_type,
                    "height": height,
                    "base_width": base_width,
                    "position": asdict(position)
                }
            )
            
            logger.info(f"Generated {tower_type} cell tower with {len(triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error generating cell tower: {e}")
            raise
    
    def _create_lattice_tower(self, position: Vector3D, height: float, base_width: float) -> List[Triangle]:
        """Create triangular lattice tower structure."""
        triangles = []
        segments = self.detail_settings[self.detail_level]["height_segments"]
        
        # Tower legs (3 vertical beams)
        leg_positions = [
            Vector3D(base_width/2, 0, 0),
            Vector3D(-base_width/4, base_width * np.sqrt(3)/4, 0),
            Vector3D(-base_width/4, -base_width * np.sqrt(3)/4, 0)
        ]
        
        beam_radius = base_width * 0.02
        
        for leg_pos in leg_positions:
            leg_triangles = self._create_cylinder(
                Vector3D(position.x + leg_pos.x, position.y + leg_pos.y, position.z),
                Vector3D(position.x + leg_pos.x * 0.3, position.y + leg_pos.y * 0.3, position.z + height),
                beam_radius
            )
            triangles.extend(leg_triangles)
        
        # Horizontal bracing
        for i in range(segments):
            z = height * i / segments
            radius_at_height = base_width * (1 - 0.7 * i / segments)
            
            # Create horizontal ring
            ring_triangles = self._create_horizontal_bracing(position, z, radius_at_height, beam_radius * 0.5)
            triangles.extend(ring_triangles)
        
        # Diagonal bracing
        for i in range(segments - 1):
            z1 = height * i / segments
            z2 = height * (i + 1) / segments
            diagonal_triangles = self._create_diagonal_bracing(position, z1, z2, base_width, beam_radius * 0.3)
            triangles.extend(diagonal_triangles)
        
        return triangles
    
    def _create_monopole_tower(self, position: Vector3D, height: float, base_width: float) -> List[Triangle]:
        """Create monopole tower structure."""
        radius = base_width / 2
        top_radius = radius * 0.3
        
        return self._create_tapered_cylinder(
            Vector3D(position.x, position.y, position.z),
            Vector3D(position.x, position.y, position.z + height),
            radius, top_radius
        )
    
    def _create_stealth_tower(self, position: Vector3D, height: float, base_width: float) -> List[Triangle]:
        """Create stealth (tree-like) tower structure."""
        triangles = []
        
        # Main trunk
        trunk_radius = base_width * 0.3
        trunk_triangles = self._create_cylinder(
            position,
            Vector3D(position.x, position.y, position.z + height),
            trunk_radius
        )
        triangles.extend(trunk_triangles)
        
        # Branches (antenna arrays disguised as branches)
        branch_count = 6
        for i in range(branch_count):
            branch_height = height * (0.3 + 0.6 * i / branch_count)
            angle = 2 * np.pi * i / branch_count
            
            branch_start = Vector3D(
                position.x,
                position.y,
                position.z + branch_height
            )
            
            branch_end = Vector3D(
                position.x + trunk_radius * 2 * np.cos(angle),
                position.y + trunk_radius * 2 * np.sin(angle),
                position.z + branch_height + trunk_radius * 0.5
            )
            
            branch_triangles = self._create_cylinder(branch_start, branch_end, trunk_radius * 0.2)
            triangles.extend(branch_triangles)
        
        return triangles
    
    def generate_antenna_array(self, position: Vector3D, array_type: str = "panel",
                             count: int = 3, orientation: float = 0.0) -> STLMesh:
        """
        Generate 3D antenna array model.
        
        Args:
            position: Antenna position
            array_type: Type of antenna ("panel", "dish", "omni")
            count: Number of antenna elements
            orientation: Azimuth orientation in degrees
            
        Returns:
            STL mesh of the antenna array
        """
        try:
            triangles = []
            
            if array_type == "panel":
                triangles = self._create_panel_antenna(position, count, orientation)
            elif array_type == "dish":
                triangles = self._create_dish_antenna(position, orientation)
            elif array_type == "omni":
                triangles = self._create_omni_antenna(position)
            
            mesh = STLMesh(
                triangles=triangles,
                name=f"antenna_{array_type}",
                metadata={
                    "type": "antenna",
                    "array_type": array_type,
                    "count": count,
                    "orientation": orientation,
                    "position": asdict(position)
                }
            )
            
            logger.info(f"Generated {array_type} antenna with {len(triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error generating antenna array: {e}")
            raise
    
    def _create_panel_antenna(self, position: Vector3D, count: int, orientation: float) -> List[Triangle]:
        """Create panel antenna array."""
        triangles = []
        panel_width = 0.4
        panel_height = 0.8
        panel_depth = 0.1
        
        for i in range(count):
            panel_pos = Vector3D(
                position.x,
                position.y,
                position.z + i * (panel_height + 0.1)
            )
            
            # Create rectangular panel
            panel_triangles = self._create_box(panel_pos, panel_width, panel_depth, panel_height)
            triangles.extend(panel_triangles)
        
        return self._rotate_triangles(triangles, position, orientation)
    
    def _create_dish_antenna(self, position: Vector3D, orientation: float) -> List[Triangle]:
        """Create dish antenna model."""
        radius = 1.0
        depth = 0.3
        
        # Create parabolic dish
        triangles = self._create_parabolic_dish(position, radius, depth)
        
        # Add support structure
        support_triangles = self._create_cylinder(
            Vector3D(position.x, position.y, position.z - 0.5),
            position,
            0.05
        )
        triangles.extend(support_triangles)
        
        return self._rotate_triangles(triangles, position, orientation)
    
    def _create_omni_antenna(self, position: Vector3D) -> List[Triangle]:
        """Create omnidirectional antenna model."""
        return self._create_cylinder(position, Vector3D(position.x, position.y, position.z + 2.0), 0.02)
    
    def generate_coverage_volume(self, position: Vector3D, range_m: float, 
                               azimuth_deg: float = 360, elevation_deg: float = 90) -> STLMesh:
        """
        Generate 3D coverage volume visualization.
        
        Args:
            position: Antenna position
            range_m: Coverage range in meters
            azimuth_deg: Azimuth beamwidth in degrees
            elevation_deg: Elevation beamwidth in degrees
            
        Returns:
            STL mesh of the coverage volume
        """
        try:
            triangles = []
            
            if azimuth_deg >= 360:
                # Omnidirectional coverage
                triangles = self._create_sphere_sector(position, range_m, 0, 360, 0, elevation_deg)
            else:
                # Directional coverage
                triangles = self._create_sphere_sector(position, range_m, 0, azimuth_deg, 0, elevation_deg)
            
            mesh = STLMesh(
                triangles=triangles,
                name="coverage_volume",
                metadata={
                    "type": "coverage",
                    "range_m": range_m,
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "position": asdict(position)
                }
            )
            
            logger.info(f"Generated coverage volume with {len(triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error generating coverage volume: {e}")
            raise
    
    def generate_building(self, position: Vector3D, width: float, length: float, 
                         height: float, building_type: str = "rectangular") -> STLMesh:
        """
        Generate 3D building model for obstruction analysis.
        
        Args:
            position: Building position
            width: Building width
            length: Building length
            height: Building height
            building_type: Type of building ("rectangular", "l_shaped", "complex")
            
        Returns:
            STL mesh of the building
        """
        try:
            if building_type == "rectangular":
                triangles = self._create_box(position, width, length, height)
            elif building_type == "l_shaped":
                triangles = self._create_l_shaped_building(position, width, length, height)
            else:
                triangles = self._create_box(position, width, length, height)
            
            mesh = STLMesh(
                triangles=triangles,
                name=f"building_{building_type}",
                metadata={
                    "type": "building",
                    "building_type": building_type,
                    "dimensions": {"width": width, "length": length, "height": height},
                    "position": asdict(position)
                }
            )
            
            logger.info(f"Generated {building_type} building with {len(triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error generating building: {e}")
            raise
    
    def _create_cylinder(self, start: Vector3D, end: Vector3D, radius: float) -> List[Triangle]:
        """Create cylinder between two points."""
        triangles = []
        segments = self.detail_settings[self.detail_level]["segments"]
        
        # Calculate direction vector
        direction = Vector3D(end.x - start.x, end.y - start.y, end.z - start.z)
        length = np.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        
        if length == 0:
            return triangles
        
        direction = direction * (1.0 / length)
        
        # Create perpendicular vectors
        if abs(direction.z) < 0.9:
            perp1 = Vector3D(direction.y, -direction.x, 0).normalize()
        else:
            perp1 = Vector3D(1, 0, 0)
        
        perp2 = Vector3D(
            direction.y * perp1.z - direction.z * perp1.y,
            direction.z * perp1.x - direction.x * perp1.z,
            direction.x * perp1.y - direction.y * perp1.x
        ).normalize()
        
        # Generate cylinder vertices
        start_circle = []
        end_circle = []
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            offset = Vector3D(
                radius * (cos_a * perp1.x + sin_a * perp2.x),
                radius * (cos_a * perp1.y + sin_a * perp2.y),
                radius * (cos_a * perp1.z + sin_a * perp2.z)
            )
            
            start_circle.append(start + offset)
            end_circle.append(end + offset)
        
        # Create side triangles
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Two triangles per face
            triangle1 = Triangle([start_circle[i], end_circle[i], start_circle[next_i]], Vector3D(0, 0, 1))
            triangle1.calculate_normal()
            triangles.append(triangle1)
            
            triangle2 = Triangle([start_circle[next_i], end_circle[i], end_circle[next_i]], Vector3D(0, 0, 1))
            triangle2.calculate_normal()
            triangles.append(triangle2)
        
        # Create end caps
        for i in range(1, segments - 1):
            # Start cap
            triangle_start = Triangle([start, start_circle[0], start_circle[i]], Vector3D(0, 0, -1))
            triangle_start.calculate_normal()
            triangles.append(triangle_start)
            
            # End cap
            triangle_end = Triangle([end, end_circle[i], end_circle[0]], Vector3D(0, 0, 1))
            triangle_end.calculate_normal()
            triangles.append(triangle_end)
        
        return triangles
    
    def _create_box(self, position: Vector3D, width: float, length: float, height: float) -> List[Triangle]:
        """Create rectangular box."""
        triangles = []
        
        # Define box vertices
        vertices = [
            Vector3D(position.x - width/2, position.y - length/2, position.z),
            Vector3D(position.x + width/2, position.y - length/2, position.z),
            Vector3D(position.x + width/2, position.y + length/2, position.z),
            Vector3D(position.x - width/2, position.y + length/2, position.z),
            Vector3D(position.x - width/2, position.y - length/2, position.z + height),
            Vector3D(position.x + width/2, position.y - length/2, position.z + height),
            Vector3D(position.x + width/2, position.y + length/2, position.z + height),
            Vector3D(position.x - width/2, position.y + length/2, position.z + height)
        ]
        
        # Define faces (each face has 2 triangles)
        faces = [
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face
            [4, 6, 5], [4, 7, 6],
            # Front face
            [0, 4, 5], [0, 5, 1],
            # Back face
            [2, 6, 7], [2, 7, 3],
            # Left face
            [0, 3, 7], [0, 7, 4],
            # Right face
            [1, 5, 6], [1, 6, 2]
        ]
        
        for face in faces:
            triangle = Triangle([vertices[face[0]], vertices[face[1]], vertices[face[2]]], Vector3D(0, 0, 1))
            triangle.calculate_normal()
            triangles.append(triangle)
        
        return triangles
    
    def _create_sphere_sector(self, center: Vector3D, radius: float, 
                            azimuth_start: float, azimuth_end: float,
                            elevation_start: float, elevation_end: float) -> List[Triangle]:
        """Create spherical sector for coverage visualization."""
        triangles = []
        segments = self.detail_settings[self.detail_level]["segments"]
        
        azimuth_range = np.radians(azimuth_end - azimuth_start)
        elevation_range = np.radians(elevation_end - elevation_start)
        
        azimuth_steps = max(4, int(segments * azimuth_range / (2 * np.pi)))
        elevation_steps = max(4, int(segments * elevation_range / np.pi))
        
        vertices = []
        
        # Generate vertices
        for i in range(elevation_steps + 1):
            elevation = np.radians(elevation_start) + elevation_range * i / elevation_steps
            
            for j in range(azimuth_steps + 1):
                azimuth = np.radians(azimuth_start) + azimuth_range * j / azimuth_steps
                
                x = center.x + radius * np.sin(elevation) * np.cos(azimuth)
                y = center.y + radius * np.sin(elevation) * np.sin(azimuth)
                z = center.z + radius * np.cos(elevation)
                
                vertices.append(Vector3D(x, y, z))
        
        # Create triangles
        for i in range(elevation_steps):
            for j in range(azimuth_steps):
                # Current quad vertices
                v1 = vertices[i * (azimuth_steps + 1) + j]
                v2 = vertices[i * (azimuth_steps + 1) + j + 1]
                v3 = vertices[(i + 1) * (azimuth_steps + 1) + j]
                v4 = vertices[(i + 1) * (azimuth_steps + 1) + j + 1]
                
                # Two triangles per quad
                triangle1 = Triangle([v1, v2, v3], Vector3D(0, 0, 1))
                triangle1.calculate_normal()
                triangles.append(triangle1)
                
                triangle2 = Triangle([v2, v4, v3], Vector3D(0, 0, 1))
                triangle2.calculate_normal()
                triangles.append(triangle2)
        
        return triangles
    
    def _create_tapered_cylinder(self, start: Vector3D, end: Vector3D, 
                               start_radius: float, end_radius: float) -> List[Triangle]:
        """Create tapered cylinder (cone-like shape)."""
        triangles = []
        segments = self.detail_settings[self.detail_level]["segments"]
        
        # Similar to cylinder but with different radii at each end
        direction = Vector3D(end.x - start.x, end.y - start.y, end.z - start.z)
        length = np.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        
        if length == 0:
            return triangles
        
        direction = direction * (1.0 / length)
        
        # Create perpendicular vectors
        if abs(direction.z) < 0.9:
            perp1 = Vector3D(direction.y, -direction.x, 0).normalize()
        else:
            perp1 = Vector3D(1, 0, 0)
        
        perp2 = Vector3D(
            direction.y * perp1.z - direction.z * perp1.y,
            direction.z * perp1.x - direction.x * perp1.z,
            direction.x * perp1.y - direction.y * perp1.x
        ).normalize()
        
        # Generate vertices
        start_circle = []
        end_circle = []
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            start_offset = Vector3D(
                start_radius * (cos_a * perp1.x + sin_a * perp2.x),
                start_radius * (cos_a * perp1.y + sin_a * perp2.y),
                start_radius * (cos_a * perp1.z + sin_a * perp2.z)
            )
            
            end_offset = Vector3D(
                end_radius * (cos_a * perp1.x + sin_a * perp2.x),
                end_radius * (cos_a * perp1.y + sin_a * perp2.y),
                end_radius * (cos_a * perp1.z + sin_a * perp2.z)
            )
            
            start_circle.append(start + start_offset)
            end_circle.append(end + end_offset)
        
        # Create side triangles
        for i in range(segments):
            next_i = (i + 1) % segments
            
            triangle1 = Triangle([start_circle[i], end_circle[i], start_circle[next_i]], Vector3D(0, 0, 1))
            triangle1.calculate_normal()
            triangles.append(triangle1)
            
            triangle2 = Triangle([start_circle[next_i], end_circle[i], end_circle[next_i]], Vector3D(0, 0, 1))
            triangle2.calculate_normal()
            triangles.append(triangle2)
        
        return triangles
    
    def _create_horizontal_bracing(self, center: Vector3D, height: float, 
                                 radius: float, beam_radius: float) -> List[Triangle]:
        """Create horizontal bracing for lattice tower."""
        triangles = []
        segments = 3  # Triangular bracing
        
        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments
            
            start = Vector3D(
                center.x + radius * np.cos(angle1),
                center.y + radius * np.sin(angle1),
                center.z + height
            )
            
            end = Vector3D(
                center.x + radius * np.cos(angle2),
                center.y + radius * np.sin(angle2),
                center.z + height
            )
            
            beam_triangles = self._create_cylinder(start, end, beam_radius)
            triangles.extend(beam_triangles)
        
        return triangles
    
    def _create_diagonal_bracing(self, center: Vector3D, z1: float, z2: float, 
                               base_width: float, beam_radius: float) -> List[Triangle]:
        """Create diagonal bracing between tower levels."""
        triangles = []
        
        # Simplified diagonal bracing
        radius1 = base_width * (1 - 0.7 * z1 / 50)  # Assuming 50m height
        radius2 = base_width * (1 - 0.7 * z2 / 50)
        
        for i in range(3):
            angle = 2 * np.pi * i / 3
            
            start = Vector3D(
                center.x + radius1 * np.cos(angle),
                center.y + radius1 * np.sin(angle),
                center.z + z1
            )
            
            end = Vector3D(
                center.x + radius2 * np.cos(angle + np.pi/3),
                center.y + radius2 * np.sin(angle + np.pi/3),
                center.z + z2
            )
            
            beam_triangles = self._create_cylinder(start, end, beam_radius)
            triangles.extend(beam_triangles)
        
        return triangles
    
    def _create_parabolic_dish(self, center: Vector3D, radius: float, depth: float) -> List[Triangle]:
        """Create parabolic dish antenna."""
        triangles = []
        segments = self.detail_settings[self.detail_level]["segments"]
        rings = self.detail_settings[self.detail_level]["height_segments"]
        
        vertices = []
        
        # Generate parabolic surface vertices
        for ring in range(rings + 1):
            ring_radius = radius * ring / rings
            ring_z = center.z - depth * (ring_radius / radius) ** 2
            
            for seg in range(segments):
                angle = 2 * np.pi * seg / segments
                x = center.x + ring_radius * np.cos(angle)
                y = center.y + ring_radius * np.sin(angle)
                vertices.append(Vector3D(x, y, ring_z))
        
        # Create triangles
        for ring in range(rings):
            for seg in range(segments):
                next_seg = (seg + 1) % segments
                
                # Current ring vertices
                v1 = vertices[ring * segments + seg]
                v2 = vertices[ring * segments + next_seg]
                
                # Next ring vertices
                v3 = vertices[(ring + 1) * segments + seg]
                v4 = vertices[(ring + 1) * segments + next_seg]
                
                # Two triangles per quad
                triangle1 = Triangle([v1, v2, v3], Vector3D(0, 0, 1))
                triangle1.calculate_normal()
                triangles.append(triangle1)
                
                triangle2 = Triangle([v2, v4, v3], Vector3D(0, 0, 1))
                triangle2.calculate_normal()
                triangles.append(triangle2)
        
        return triangles
    
    def _create_l_shaped_building(self, position: Vector3D, width: float, 
                                length: float, height: float) -> List[Triangle]:
        """Create L-shaped building."""
        triangles = []
        
        # Create two rectangular sections for L-shape
        section1_triangles = self._create_box(
            Vector3D(position.x - width/4, position.y, position.z),
            width/2, length, height
        )
        
        section2_triangles = self._create_box(
            Vector3D(position.x + width/4, position.y - length/4, position.z),
            width/2, length/2, height
        )
        
        triangles.extend(section1_triangles)
        triangles.extend(section2_triangles)
        
        return triangles
    
    def _rotate_triangles(self, triangles: List[Triangle], center: Vector3D, 
                        angle_degrees: float) -> List[Triangle]:
        """Rotate triangles around Z-axis."""
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotated_triangles = []
        
        for triangle in triangles:
            rotated_vertices = []
            
            for vertex in triangle.vertices:
                # Translate to origin
                x = vertex.x - center.x
                y = vertex.y - center.y
                
                # Rotate
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                
                # Translate back
                rotated_vertex = Vector3D(new_x + center.x, new_y + center.y, vertex.z)
                rotated_vertices.append(rotated_vertex)
            
            rotated_triangle = Triangle(rotated_vertices, triangle.normal)
            rotated_triangle.calculate_normal()
            rotated_triangles.append(rotated_triangle)
        
        return rotated_triangles
    
    def combine_meshes(self, meshes: List[STLMesh]) -> STLMesh:
        """Combine multiple STL meshes into one."""
        all_triangles = []
        combined_metadata = {"combined_meshes": []}
        
        for mesh in meshes:
            all_triangles.extend(mesh.triangles)
            combined_metadata["combined_meshes"].append({
                "name": mesh.name,
                "triangle_count": len(mesh.triangles),
                "metadata": mesh.metadata
            })
        
        return STLMesh(
            triangles=all_triangles,
            name="combined_network_model",
            metadata=combined_metadata
        )
    
    def export_stl_ascii(self, mesh: STLMesh, output_path: Union[str, Path]):
        """Export STL mesh in ASCII format."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"solid {mesh.name}\n")
                
                for triangle in mesh.triangles:
                    f.write(f"  facet normal {triangle.normal.x:.6f} {triangle.normal.y:.6f} {triangle.normal.z:.6f}\n")
                    f.write(f"    outer loop\n")
                    
                    for vertex in triangle.vertices:
                        f.write(f"      vertex {vertex.x:.6f} {vertex.y:.6f} {vertex.z:.6f}\n")
                    
                    f.write(f"    endloop\n")
                    f.write(f"  endfacet\n")
                
                f.write(f"endsolid {mesh.name}\n")
            
            logger.info(f"STL exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting STL: {e}")
            raise
    
    def export_stl_binary(self, mesh: STLMesh, output_path: Union[str, Path]):
        """Export STL mesh in binary format."""
        try:
            import struct
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                # Write header (80 bytes)
                header = f"{mesh.name}".ljust(80)[:80].encode('ascii')
                f.write(header)
                
                # Write triangle count
                f.write(struct.pack('<I', len(mesh.triangles)))
                
                # Write triangles
                for triangle in mesh.triangles:
                    # Normal vector
                    f.write(struct.pack('<fff', triangle.normal.x, triangle.normal.y, triangle.normal.z))
                    
                    # Vertices
                    for vertex in triangle.vertices:
                        f.write(struct.pack('<fff', vertex.x, vertex.y, vertex.z))
                    
                    # Attribute byte count (usually 0)
                    f.write(struct.pack('<H', 0))
            
            logger.info(f"Binary STL exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting binary STL: {e}")
            raise


def create_sample_network_scene():
    """Create a sample 5G network scene with multiple components."""
    generator = STL3DGenerator(scale_factor=1.0, detail_level="medium")
    
    # Create cell tower
    tower_position = Vector3D(0, 0, 0)
    tower_mesh = generator.generate_cell_tower(tower_position, height=50, tower_type="lattice")
    
    # Create antennas
    antenna_position = Vector3D(0, 0, 45)
    antenna_mesh = generator.generate_antenna_array(antenna_position, array_type="panel", count=3)
    
    # Create coverage volume
    coverage_mesh = generator.generate_coverage_volume(
        antenna_position, range_m=1000, azimuth_deg=120, elevation_deg=90
    )
    
    # Create buildings
    building1 = generator.generate_building(Vector3D(200, 200, 0), 50, 30, 20)
    building2 = generator.generate_building(Vector3D(-150, 100, 0), 40, 40, 15)
    
    # Combine all meshes
    scene_mesh = generator.combine_meshes([tower_mesh, antenna_mesh, coverage_mesh, building1, building2])
    
    return scene_mesh, generator


# Example usage
if __name__ == "__main__":
    # Create sample network scene
    scene_mesh, generator = create_sample_network_scene()
    
    # Export to STL files
    generator.export_stl_ascii(scene_mesh, "sample_5g_network_scene.stl")
    generator.export_stl_binary(scene_mesh, "sample_5g_network_scene_binary.stl")
    
    print(f"Sample 5G network scene created with {len(scene_mesh.triangles)} triangles")
    print("STL files exported successfully!")
