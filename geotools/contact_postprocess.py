# geotools\contact_postprocess.py
"""
Contact post-processing utilities for geological cross-section analysis.

This module provides tools for:
1. Cleaning short terminal segments from extracted contacts
2. Resampling contact nodes to ensure consistent density between tie lines
3. Fitting splines to contacts for smoother representation
4. Exporting contacts as DXF SPLINE entities

Author: Claude / George's Geological Tools
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplineControlPoint:
    """A control point for a B-spline curve."""
    x: float
    y: float
    z: float
    weight: float = 1.0


def clean_terminal_segments(
    coords: List[Tuple[float, float]],
    min_segment_ratio: float = 0.15,
    max_angle_change: float = 60.0
) -> List[Tuple[float, float]]:
    """
    Remove short segments at the start and end of a contact line.
    
    Short terminal segments often occur when the boundary-walking algorithm
    picks up spurious points as the boundary curves away from the contact zone.
    
    Args:
        coords: List of (x, y) coordinate tuples
        min_segment_ratio: Minimum segment length as ratio of median segment length
        max_angle_change: Maximum angle change (degrees) to keep a short segment
    
    Returns:
        Cleaned coordinate list
    """
    if len(coords) < 4:
        return coords
    
    # Calculate all segment lengths
    lengths = []
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        lengths.append(np.sqrt(dx**2 + dy**2))
    
    if not lengths:
        return coords
    
    median_length = np.median(lengths)
    min_length = median_length * min_segment_ratio
    
    def get_angle_change(i: int) -> float:
        """Get angle change at vertex i (in degrees)."""
        if i <= 0 or i >= len(coords) - 1:
            return 0
        
        v1 = np.array([coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1]])
        v2 = np.array([coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 180  # Degenerate case - treat as sharp turn
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    # Clean from start
    start_idx = 0
    while start_idx < len(coords) - 2:
        if lengths[start_idx] < min_length:
            angle = get_angle_change(start_idx + 1)
            if angle > max_angle_change:
                start_idx += 1
                continue
        break
    
    # Clean from end
    end_idx = len(coords)
    while end_idx > start_idx + 2:
        seg_idx = end_idx - 2
        if seg_idx >= 0 and seg_idx < len(lengths) and lengths[seg_idx] < min_length:
            angle = get_angle_change(end_idx - 2)
            if angle > max_angle_change:
                end_idx -= 1
                continue
        break
    
    if end_idx - start_idx < 2:
        return coords
    
    return coords[start_idx:end_idx]


def resample_contact_between_ties(
    coords: List[Tuple[float, float]],
    tie_indices: List[int],
    target_points_per_segment: int = 10
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Resample a contact line to have consistent node density between tie lines.
    
    This ensures that when building 3D surfaces, corresponding contacts have
    the same number of vertices between tie line connections.
    
    Args:
        coords: Original contact coordinates [(x, y), ...]
        tie_indices: Indices of vertices that are tie line anchor points
        target_points_per_segment: Number of points between each tie
    
    Returns:
        (new_coords, new_tie_indices) - Resampled coordinates and updated tie indices
    """
    if len(coords) < 2:
        return coords, tie_indices
    
    if not tie_indices:
        # No tie lines - just resample uniformly
        return resample_uniform(coords, target_points_per_segment * 2)
    
    # Sort tie indices
    tie_indices = sorted(set(tie_indices))
    
    # Ensure we have start and end
    if 0 not in tie_indices:
        tie_indices = [0] + tie_indices
    if len(coords) - 1 not in tie_indices:
        tie_indices = tie_indices + [len(coords) - 1]
    
    new_coords = []
    new_tie_indices = []
    
    for seg_idx in range(len(tie_indices) - 1):
        start_idx = tie_indices[seg_idx]
        end_idx = tie_indices[seg_idx + 1]
        
        # Extract segment
        segment = coords[start_idx:end_idx + 1]
        
        # Resample segment
        if len(segment) >= 2:
            resampled, _ = resample_uniform(segment, target_points_per_segment + 1)
            
            # Record tie index at start of segment
            new_tie_indices.append(len(new_coords))
            
            # Add resampled points (skip last if not the final segment)
            if seg_idx < len(tie_indices) - 2:
                new_coords.extend(resampled[:-1])
            else:
                new_coords.extend(resampled)
                new_tie_indices.append(len(new_coords) - 1)
    
    return new_coords, new_tie_indices


def resample_uniform(
    coords: List[Tuple[float, float]],
    num_points: int
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Resample a polyline to have uniformly spaced points.
    
    Args:
        coords: Original coordinates
        num_points: Target number of points
    
    Returns:
        (resampled_coords, [0, num_points-1]) - Resampled coordinates and endpoint indices
    """
    if len(coords) < 2 or num_points < 2:
        return coords, [0, len(coords) - 1] if coords else []
    
    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i-1][0]
        dy = coords[i][1] - coords[i-1][1]
        distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))
    
    total_length = distances[-1]
    if total_length < 1e-10:
        return coords, [0, len(coords) - 1]
    
    # Generate uniformly spaced sample points
    sample_distances = np.linspace(0, total_length, num_points)
    
    new_coords = []
    for d in sample_distances:
        # Find segment containing this distance
        for i in range(1, len(distances)):
            if distances[i] >= d:
                # Interpolate
                t = (d - distances[i-1]) / (distances[i] - distances[i-1]) if distances[i] > distances[i-1] else 0
                x = coords[i-1][0] + t * (coords[i][0] - coords[i-1][0])
                y = coords[i-1][1] + t * (coords[i][1] - coords[i-1][1])
                new_coords.append((x, y))
                break
    
    return new_coords, [0, len(new_coords) - 1]


def fit_cubic_bspline(
    coords: List[Tuple[float, float]],
    smoothing: float = 0.5
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Fit a cubic B-spline to the given coordinates.
    
    Args:
        coords: Input coordinates
        smoothing: Smoothing factor (0 = interpolate all points, 1 = maximum smoothing)
    
    Returns:
        (control_points, knots) for the B-spline
    """
    if len(coords) < 4:
        # Not enough points for cubic spline
        return coords, []
    
    try:
        from scipy import interpolate
        
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        
        # Parameterize by cumulative chord length
        t = np.zeros(len(coords))
        for i in range(1, len(coords)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            t[i] = t[i-1] + np.sqrt(dx**2 + dy**2)
        
        if t[-1] > 0:
            t /= t[-1]  # Normalize to [0, 1]
        
        # Fit splines with smoothing
        s = smoothing * len(coords)
        tck_x, _ = interpolate.splprep([xs], u=t, k=3, s=s)
        tck_y, _ = interpolate.splprep([ys], u=t, k=3, s=s)
        
        # Extract control points
        # Note: This is a simplified extraction - full B-spline export would need
        # proper knot vector and control point handling
        control_xs = tck_x[1][0]
        control_ys = tck_y[1][0]
        knots = list(tck_x[0])
        
        control_points = list(zip(control_xs, control_ys))
        
        return control_points, knots
        
    except ImportError:
        logger.warning("scipy not available - returning original points")
        return coords, []


def find_nearest_vertex(
    coords: List[Tuple[float, float]],
    point: Tuple[float, float],
    max_distance: float = float('inf')
) -> Tuple[int, float]:
    """
    Find the nearest vertex to a given point.
    
    Args:
        coords: List of coordinates
        point: Point to find nearest vertex to
        max_distance: Maximum distance to consider
    
    Returns:
        (index, distance) of nearest vertex, or (-1, inf) if none found
    """
    if not coords:
        return -1, float('inf')
    
    min_dist = float('inf')
    min_idx = -1
    
    for i, (x, y) in enumerate(coords):
        dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            min_idx = i
    
    return min_idx, min_dist


def write_dxf_spline(
    f,
    coords: List[Tuple[float, float, float]],
    layer_name: str = "0",
    degree: int = 3,
    fit_points: bool = True
):
    """
    Write a DXF SPLINE entity.
    
    Args:
        f: File handle
        coords: List of (x, y, z) coordinates
        layer_name: DXF layer name
        degree: Spline degree (3 = cubic)
        fit_points: If True, treat coords as fit points; if False, as control points
    """
    if len(coords) < 2:
        return
    
    # Ensure we have enough points for the degree
    if len(coords) <= degree:
        # Fall back to polyline
        write_dxf_3d_polyline(f, coords, layer_name)
        return
    
    f.write("0\nSPLINE\n")
    f.write(f"8\n{layer_name}\n")
    f.write("100\nAcDbEntity\n")
    f.write("100\nAcDbSpline\n")
    
    # Spline flags: 8 = planar, 1 = closed (we don't close)
    f.write("70\n8\n")
    
    # Degree
    f.write(f"71\n{degree}\n")
    
    if fit_points:
        # Using fit points - DXF will compute control points
        num_fit = len(coords)
        f.write(f"74\n{num_fit}\n")  # Number of fit points
        
        # Fit point tolerance
        f.write("44\n0.0\n")
        
        # Write fit points (group codes 11, 21, 31)
        for x, y, z in coords:
            f.write(f"11\n{x:.6f}\n")
            f.write(f"21\n{y:.6f}\n")
            f.write(f"31\n{z:.6f}\n")
    else:
        # Using control points - need to compute knot vector
        num_control = len(coords)
        num_knots = num_control + degree + 1
        
        f.write(f"73\n{num_control}\n")  # Number of control points
        f.write(f"72\n{num_knots}\n")    # Number of knots
        
        # Generate clamped knot vector
        knots = []
        for _ in range(degree + 1):
            knots.append(0.0)
        for i in range(1, num_control - degree):
            knots.append(i / (num_control - degree))
        for _ in range(degree + 1):
            knots.append(1.0)
        
        # Write knot values (group code 40)
        for k in knots:
            f.write(f"40\n{k:.6f}\n")
        
        # Write control points (group codes 10, 20, 30)
        for x, y, z in coords:
            f.write(f"10\n{x:.6f}\n")
            f.write(f"20\n{y:.6f}\n")
            f.write(f"30\n{z:.6f}\n")


def write_dxf_3d_polyline(
    f,
    coords: List[Tuple[float, float, float]],
    layer_name: str = "0",
    spline_fit: bool = False
):
    """
    Write a DXF 3D POLYLINE entity.
    
    Args:
        f: File handle
        coords: List of (x, y, z) coordinates
        layer_name: DXF layer name
        spline_fit: If True, use spline-fit polyline (flag 70=8 with frame vertices)
    """
    if len(coords) < 2:
        return
    
    f.write("0\nPOLYLINE\n")
    f.write(f"8\n{layer_name}\n")
    f.write("66\n1\n")  # Vertices follow
    
    if spline_fit:
        # Spline-fit polyline
        f.write("70\n8\n")  # 3D polyline + spline fit
        vertex_flag = 32  # Spline frame control point
    else:
        # Regular 3D polyline
        f.write("70\n8\n")  # 3D polyline
        vertex_flag = 32  # 3D vertex
    
    for x, y, z in coords:
        f.write("0\nVERTEX\n")
        f.write(f"8\n{layer_name}\n")
        f.write(f"10\n{x:.6f}\n")
        f.write(f"20\n{y:.6f}\n")
        f.write(f"30\n{z:.6f}\n")
        f.write(f"70\n{vertex_flag}\n")
    
    f.write("0\nSEQEND\n")


def export_contacts_with_options(
    grouped_contacts: Dict,
    filepath: str,
    use_splines: bool = False,
    clean_endpoints: bool = True,
    resample: bool = False,
    target_points: int = 20
):
    """
    Export contacts to DXF with post-processing options.
    
    Args:
        grouped_contacts: Dictionary of GroupedContact objects
        filepath: Output DXF file path
        use_splines: Export as SPLINE entities instead of POLYLINEs
        clean_endpoints: Remove short terminal segments
        resample: Resample contacts to uniform density
        target_points: Target number of points when resampling
    """
    with open(filepath, "w") as f:
        # DXF header
        f.write("0\nSECTION\n2\nENTITIES\n")
        
        contacts_exported = 0
        
        for group_name, group in grouped_contacts.items():
            layer_name = group_name.replace("-", "_").replace(" ", "_").replace("/", "_")[:31]
            
            for polyline in group.polylines:
                vertices = polyline.vertices
                northing = polyline.northing
                
                if northing is None or len(vertices) < 4:
                    continue
                
                # Convert to coordinate list
                coords_2d = []
                for i in range(0, len(vertices), 2):
                    if i + 1 < len(vertices):
                        coords_2d.append((float(vertices[i]), float(vertices[i + 1])))
                
                if len(coords_2d) < 2:
                    continue
                
                # Apply post-processing
                if clean_endpoints:
                    coords_2d = clean_terminal_segments(coords_2d)
                
                if resample:
                    # Get tie line indices for this polyline
                    tie_indices = []  # TODO: Extract from group.tie_lines
                    if tie_indices:
                        coords_2d, _ = resample_contact_between_ties(
                            coords_2d, tie_indices, target_points // max(1, len(tie_indices) - 1)
                        )
                    else:
                        coords_2d, _ = resample_uniform(coords_2d, target_points)
                
                if len(coords_2d) < 2:
                    continue
                
                # Build 3D coordinates
                coords_3d = [(x, float(northing), y) for x, y in coords_2d]
                
                # Export
                if use_splines and len(coords_3d) >= 4:
                    write_dxf_spline(f, coords_3d, layer_name, fit_points=True)
                else:
                    write_dxf_3d_polyline(f, coords_3d, layer_name)
                
                contacts_exported += 1
        
        # DXF footer
        f.write("0\nENDSEC\n0\nEOF\n")
    
    logger.info(f"Exported {contacts_exported} contacts to {filepath}")
    return contacts_exported


# Convenience function for cleaning contacts in-place
def clean_all_contacts(grouped_contacts: Dict):
    """
    Clean terminal segments from all contacts in-place.
    
    Args:
        grouped_contacts: Dictionary of GroupedContact objects
    """
    cleaned_count = 0
    
    for group_name, group in grouped_contacts.items():
        for polyline in group.polylines:
            vertices = polyline.vertices
            
            if len(vertices) < 8:
                continue
            
            # Convert to coordinate list
            coords = []
            for i in range(0, len(vertices), 2):
                if i + 1 < len(vertices):
                    coords.append((vertices[i], vertices[i + 1]))
            
            original_len = len(coords)
            coords = clean_terminal_segments(coords)
            
            if len(coords) < original_len:
                # Flatten back to vertex list
                new_vertices = []
                for x, y in coords:
                    new_vertices.extend([x, y])
                polyline.vertices = new_vertices
                cleaned_count += 1
    
    logger.info(f"Cleaned {cleaned_count} contact polylines")
    return cleaned_count
