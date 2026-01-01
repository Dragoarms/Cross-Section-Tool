# geotools\auto_labeler.py
"""
Auto-labeling module for geological polygon classification.

This module provides automatic suggestion of unit assignments based on:
1. User-assigned polygons (seed assignments)
2. Stratigraphic column order
3. Spatial relationships (above/below)
4. Fault boundaries (don't auto-label across faults)

The labeler does NOT overwrite existing user assignments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points
import logging

logger = logging.getLogger(__name__)


class AutoLabeler:
    """
    Automatic polygon labeling based on stratigraphic relationships.
    
    Works by:
    1. Finding polygons adjacent to user-assigned polygons
    2. Determining their relative position (above/below)
    3. Suggesting the next unit in stratigraphic order
    4. Respecting fault boundaries
    """
    
    def __init__(self, strat_column):
        """
        Initialize the auto-labeler.
        
        Args:
            strat_column: StratColumnV2 instance with unit definitions
        """
        self.strat_column = strat_column
    
    def find_adjacent_polygons(
        self,
        target_polygon_vertices: List[float],
        all_polygons: Dict[str, Dict],
        tolerance: float = 5.0,
    ) -> Dict[str, str]:
        """
        Find polygons adjacent to the target and their relative positions.
        
        Args:
            target_polygon_vertices: Flat list of vertices [e1, rl1, e2, rl2, ...]
            all_polygons: Dict of all polygons {name: {vertices: [...], ...}}
            tolerance: Distance tolerance for adjacency detection
            
        Returns:
            Dict mapping adjacent polygon names to position ("above" or "below")
        """
        adjacent = {}
        
        # Build target polygon
        target_coords = self._vertices_to_coords(target_polygon_vertices)
        if len(target_coords) < 3:
            return adjacent
        
        try:
            target_poly = Polygon(target_coords)
            if not target_poly.is_valid:
                target_poly = target_poly.buffer(0)
            target_centroid = target_poly.centroid
            target_boundary = target_poly.boundary
        except Exception as e:
            logger.warning(f"Invalid target polygon: {e}")
            return adjacent
        
        # Check each other polygon
        for name, poly_data in all_polygons.items():
            other_vertices = poly_data.get("vertices", [])
            other_coords = self._vertices_to_coords(other_vertices)
            
            if len(other_coords) < 3:
                continue
            
            try:
                other_poly = Polygon(other_coords)
                if not other_poly.is_valid:
                    other_poly = other_poly.buffer(0)
                other_boundary = other_poly.boundary
                
                # Check if boundaries are close (adjacent)
                distance = target_boundary.distance(other_boundary)
                
                if distance <= tolerance:
                    # Determine relative position
                    other_centroid = other_poly.centroid
                    position = self._determine_relative_position(
                        target_centroid, other_centroid
                    )
                    adjacent[name] = position
                    
            except Exception as e:
                logger.debug(f"Error checking polygon {name}: {e}")
                continue
        
        return adjacent
    
    def _vertices_to_coords(self, vertices: List[float]) -> List[Tuple[float, float]]:
        """Convert flat vertex list to coordinate tuples."""
        coords = []
        for i in range(0, len(vertices), 2):
            if i + 1 < len(vertices):
                coords.append((vertices[i], vertices[i + 1]))
        return coords
    
    def _determine_relative_position(
        self, 
        reference_centroid: Point, 
        other_centroid: Point
    ) -> str:
        """
        Determine if other polygon is above or below reference.
        
        In geological cross-sections:
        - Higher RL (y value) = "above" (younger in normal stratigraphy)
        - Lower RL (y value) = "below" (older in normal stratigraphy)
        """
        if other_centroid.y > reference_centroid.y:
            return "above"
        else:
            return "below"
    
    def find_polygons_separated_by_fault(
        self,
        target_polygon_vertices: List[float],
        all_polygons: Dict[str, Dict],
        fault_lines: List[Dict],
        tolerance: float = 5.0,
    ) -> Set[str]:
        """
        Find polygons that are separated from target by a fault line.
        
        Args:
            target_polygon_vertices: Vertices of the assigned polygon
            all_polygons: All polygons in the section
            fault_lines: List of fault line dicts with 'vertices' key
            tolerance: Adjacency tolerance
            
        Returns:
            Set of polygon names separated by fault
        """
        separated = set()
        
        if not fault_lines:
            return separated
        
        target_coords = self._vertices_to_coords(target_polygon_vertices)
        if len(target_coords) < 3:
            return separated
        
        try:
            target_poly = Polygon(target_coords)
            if not target_poly.is_valid:
                target_poly = target_poly.buffer(0)
            target_centroid = target_poly.centroid
        except Exception:
            return separated
        
        # Build fault LineStrings
        fault_geometries = []
        for fault in fault_lines:
            fault_vertices = fault.get("vertices", [])
            fault_coords = self._vertices_to_coords(fault_vertices)
            if len(fault_coords) >= 2:
                try:
                    fault_line = LineString(fault_coords)
                    # Buffer the fault slightly for intersection testing
                    fault_geometries.append(fault_line.buffer(tolerance / 2))
                except Exception:
                    continue
        
        if not fault_geometries:
            return separated
        
        # Check each adjacent polygon
        for name, poly_data in all_polygons.items():
            other_vertices = poly_data.get("vertices", [])
            other_coords = self._vertices_to_coords(other_vertices)
            
            if len(other_coords) < 3:
                continue
            
            try:
                other_poly = Polygon(other_coords)
                if not other_poly.is_valid:
                    other_poly = other_poly.buffer(0)
                other_centroid = other_poly.centroid
                
                # Create line between centroids
                connection_line = LineString([
                    (target_centroid.x, target_centroid.y),
                    (other_centroid.x, other_centroid.y)
                ])
                
                # Check if any fault intersects this connection
                for fault_buffer in fault_geometries:
                    if connection_line.intersects(fault_buffer):
                        separated.add(name)
                        break
                        
            except Exception:
                continue
        
        return separated
    
    def suggest_labels(
        self,
        assigned_polygon_name: str,
        assigned_unit_name: str,
        section_polygons: Dict[str, Dict],
        section_faults: List[Dict],
        user_assignments: Dict[str, str],  # {polygon_name: unit_name}
    ) -> Dict[str, str]:
        """
        Suggest labels for polygons adjacent to a newly assigned polygon.
        
        This is the main entry point for auto-labeling. It:
        1. Finds adjacent polygons
        2. Excludes those separated by faults
        3. Excludes those already assigned by user
        4. Suggests units based on stratigraphic order
        
        Args:
            assigned_polygon_name: Name of the polygon just assigned
            assigned_unit_name: Unit name it was assigned to
            section_polygons: All polygons in the section
            section_faults: All fault lines in the section
            user_assignments: Existing user assignments to preserve
            
        Returns:
            Dict mapping polygon names to suggested unit names
        """
        suggestions = {}
        
        # Get the assigned polygon's vertices
        assigned_polygon = section_polygons.get(assigned_polygon_name)
        if not assigned_polygon:
            logger.warning(f"Polygon '{assigned_polygon_name}' not found")
            return suggestions
        
        assigned_vertices = assigned_polygon.get("vertices", [])
        
        # Find adjacent polygons (excluding the assigned one)
        other_polygons = {
            name: data for name, data in section_polygons.items()
            if name != assigned_polygon_name
        }
        
        adjacent = self.find_adjacent_polygons(
            assigned_vertices, other_polygons
        )
        
        if not adjacent:
            logger.debug(f"No adjacent polygons found for '{assigned_polygon_name}'")
            return suggestions
        
        # Find polygons separated by fault
        fault_separated = self.find_polygons_separated_by_fault(
            assigned_vertices, other_polygons, section_faults
        )
        
        # Get already assigned polygon names
        already_assigned = set(user_assignments.keys())
        
        # Use strat column to suggest assignments
        suggestions = self.strat_column.suggest_adjacent_assignments(
            assigned_unit_name=assigned_unit_name,
            adjacent_polygon_positions=adjacent,
            already_assigned=already_assigned,
            fault_separated=fault_separated,
        )
        
        logger.info(f"Auto-label suggestions from '{assigned_polygon_name}' ({assigned_unit_name}):")
        for poly_name, suggested_unit in suggestions.items():
            logger.info(f"  - {poly_name} -> {suggested_unit}")
        
        return suggestions
    
    def propagate_labels(
        self,
        section_polygons: Dict[str, Dict],
        section_faults: List[Dict],
        user_assignments: Dict[str, str],
        max_iterations: int = 10,
    ) -> Dict[str, str]:
        """
        Propagate labels from user assignments to all reachable polygons.
        
        This iteratively applies suggest_labels until no new suggestions
        are generated, effectively flood-filling the stratigraphy.
        
        Args:
            section_polygons: All polygons in the section
            section_faults: All fault lines in the section
            user_assignments: User assignments (these are preserved)
            max_iterations: Maximum propagation iterations
            
        Returns:
            Dict of all assignments (user + auto-suggested)
        """
        all_assignments = dict(user_assignments)
        
        for iteration in range(max_iterations):
            new_suggestions = {}
            
            # For each assigned polygon, suggest labels for neighbors
            for poly_name, unit_name in all_assignments.items():
                suggestions = self.suggest_labels(
                    assigned_polygon_name=poly_name,
                    assigned_unit_name=unit_name,
                    section_polygons=section_polygons,
                    section_faults=section_faults,
                    user_assignments=all_assignments,  # Include all current assignments
                )
                
                # Only add suggestions for polygons not yet assigned
                for suggested_poly, suggested_unit in suggestions.items():
                    if suggested_poly not in all_assignments:
                        new_suggestions[suggested_poly] = suggested_unit
            
            if not new_suggestions:
                logger.info(f"Propagation complete after {iteration + 1} iterations")
                break
            
            # Add new suggestions to assignments
            all_assignments.update(new_suggestions)
            logger.debug(f"Iteration {iteration + 1}: added {len(new_suggestions)} suggestions")
        
        # Return only auto-generated suggestions (not user assignments)
        auto_suggestions = {
            name: unit for name, unit in all_assignments.items()
            if name not in user_assignments
        }
        
        return auto_suggestions


def create_auto_labeler(strat_column) -> AutoLabeler:
    """Factory function to create an AutoLabeler."""
    return AutoLabeler(strat_column)
