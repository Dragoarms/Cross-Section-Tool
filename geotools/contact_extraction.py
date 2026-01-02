# geotools\contact_extraction.py
"""
Contact extraction module using buffer-based point sampling.

This module extracts geological contact POINTS between adjacent polygon units.
Points are simpler and more robust than polylines:
- No ordering/merging artifacts
- No terminal hooks (cleaned by algorithm)
- Faults naturally break contacts by excluding nearby points
- Leapfrog can create surfaces from point clouds

The approach:
1. Sample boundary points densely along each polygon
2. Find where boundaries from different units are within buffer distance
3. Only create contacts where there's EXACTLY ONE other unit nearby (no ambiguity)
4. Compute centerline midpoints between matched boundary points
5. Sort, clean hooks/backtracking, and simplify the contact polyline
6. Exclude points near fault lines
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, NamedTuple
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import linemerge, unary_union
from collections import defaultdict
from dataclasses import dataclass, field
import logging
import re
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ContactPoint:
    """A single contact point."""
    easting: float
    rl: float
    northing: float
    section_key: Tuple  # (pdf_path, page_num)
    source_unit1: str
    source_unit2: str


@dataclass
class ContactPolyline:
    """
    A contact represented as a list of points from one section.
    
    Note: Despite the name "polyline", this is really just an ordered list of points.
    The points are ordered by easting for consistent export, but there's no guarantee
    of connectivity - they're just sample points along the contact.
    """
    vertices: List[float]  # Flat list [e1, rl1, e2, rl2, ...] in real-world coords
    northing: float
    section_key: Tuple  # (pdf_path, page_num)
    source_unit1: str
    source_unit2: str
    pdf_vertices: List[float] = None  # Optional PDF coords
    
    def get_coords(self) -> List[Tuple[float, float]]:
        """Return vertices as list of (easting, rl) tuples."""
        coords = []
        for i in range(0, len(self.vertices), 2):
            if i + 1 < len(self.vertices):
                coords.append((self.vertices[i], self.vertices[i + 1]))
        return coords
    
    def get_points(self) -> List[ContactPoint]:
        """Return as list of ContactPoint objects."""
        points = []
        for e, rl in self.get_coords():
            points.append(ContactPoint(
                easting=e,
                rl=rl,
                northing=self.northing,
                section_key=self.section_key,
                source_unit1=self.source_unit1,
                source_unit2=self.source_unit2
            ))
        return points
    
    def get_start_point(self) -> Optional[Tuple[float, float]]:
        """Return the start point (westernmost)."""
        if len(self.vertices) >= 2:
            return (self.vertices[0], self.vertices[1])
        return None
    
    def get_end_point(self) -> Optional[Tuple[float, float]]:
        """Return the end point (easternmost)."""
        if len(self.vertices) >= 2:
            return (self.vertices[-2], self.vertices[-1])
        return None
    
    def get_midpoint(self) -> Optional[Tuple[float, float]]:
        """Return the middle point."""
        coords = self.get_coords()
        if not coords:
            return None
        mid_idx = len(coords) // 2
        return coords[mid_idx]
    
    def trim_terminal_hooks(self, min_length: float = 5.0, max_angle: float = 60.0) -> 'ContactPolyline':
        """
        No-op for point-based contacts - hooks are removed during extraction.
        Kept for API compatibility.
        """
        return self


@dataclass
class GroupedContact:
    """A contact grouped by formation pair, containing points from multiple sections."""
    formation1: str
    formation2: str
    polylines: List[ContactPolyline] = field(default_factory=list)
    tie_lines: List[Dict] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        """Return canonical name for this contact group."""
        names = sorted([self.formation1, self.formation2])
        return f"{names[0]}-{names[1]}"
    
    def get_sections(self) -> List[float]:
        """Return sorted list of northings where this contact appears."""
        return sorted(set(p.northing for p in self.polylines), reverse=True)
    
    def get_polylines_for_section(self, northing: float, tolerance: float = 0.1) -> List[ContactPolyline]:
        """Get all polylines for a specific section."""
        return [p for p in self.polylines if abs(p.northing - northing) < tolerance]
    
    def get_all_points(self) -> List[ContactPoint]:
        """Get all points across all sections."""
        points = []
        for polyline in self.polylines:
            points.extend(polyline.get_points())
        return points
    
    def add_tie_line(self, from_northing: float, from_point: Tuple[float, float],
                     to_northing: float, to_point: Tuple[float, float]):
        """Add a tie line between two sections."""
        self.tie_lines.append({
            "from_northing": from_northing,
            "from_point": from_point,
            "to_northing": to_northing,
            "to_point": to_point
        })


class BoundaryPoint(NamedTuple):
    """A sampled point along a polygon boundary."""
    x: float  # Easting
    y: float  # RL (elevation)
    unit_name: str
    poly_index: int


class ContactMidpoint(NamedTuple):
    """A contact midpoint with metadata."""
    x: float  # Easting (midpoint)
    y: float  # RL (midpoint)
    unit_a: str
    unit_b: str
    distance: float  # Distance between the two boundary points


def _get_exterior_rings(geom) -> List[LineString]:
    """
    Get exterior ring(s) from a geometry, handling both Polygon and MultiPolygon.

    Args:
        geom: A Shapely Polygon or MultiPolygon

    Returns:
        List of exterior rings as LineStrings
    """
    from shapely.geometry import Polygon, MultiPolygon

    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        if geom.exterior:
            return [geom.exterior]
        return []
    elif isinstance(geom, MultiPolygon):
        exteriors = []
        for poly in geom.geoms:
            if poly.exterior:
                exteriors.append(poly.exterior)
        return exteriors
    else:
        # Try to handle other geometry types gracefully
        if hasattr(geom, 'exterior'):
            return [geom.exterior]
        return []


def _is_polygon_like(geom) -> bool:
    """Check if geometry is a Polygon or MultiPolygon."""
    from shapely.geometry import Polygon, MultiPolygon
    return isinstance(geom, (Polygon, MultiPolygon))


# =============================================================================
# Geometry Utility Functions
# =============================================================================

def vertices_to_coords(vertices: List[float]) -> List[Tuple[float, float]]:
    """Convert flat vertex list [x1, y1, x2, y2, ...] to coordinate tuples."""
    coords = []
    for i in range(0, len(vertices), 2):
        if i + 1 < len(vertices):
            coords.append((vertices[i], vertices[i + 1]))
    return coords


def coords_to_vertices(coords: List[Tuple[float, float]]) -> List[float]:
    """Convert coordinate tuples to flat vertex list."""
    vertices = []
    for x, y in coords:
        vertices.extend([x, y])
    return vertices


def _point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _vector_subtract(p2: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    """Create vector from p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1])


def _vector_length(v: Tuple[float, float]) -> float:
    """Length of a vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def _vector_normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize vector to unit length."""
    length = _vector_length(v)
    if length < 1e-10:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def _vector_dot(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def _angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Angle between two vectors in degrees (0-180)."""
    len1 = _vector_length(v1)
    len2 = _vector_length(v2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    
    dot = _vector_dot(v1, v2) / (len1 * len2)
    dot = max(-1.0, min(1.0, dot))
    
    return math.degrees(math.acos(dot))


# =============================================================================
# Boundary Sampling Functions
# =============================================================================

def _sample_polygon_boundary(
    polygon,
    unit_name: str,
    poly_index: int,
    sample_distance: float = 2.0
) -> List[BoundaryPoint]:
    """
    Sample points along a polygon boundary at regular intervals.

    Args:
        polygon: Shapely Polygon or MultiPolygon
        unit_name: Name of the geological unit
        poly_index: Index of this polygon within the unit
        sample_distance: Distance between sample points in map units

    Returns:
        List of BoundaryPoint objects
    """
    from shapely.geometry import MultiPolygon

    points = []

    if polygon is None or polygon.is_empty:
        return points

    # Get all exterior rings (handles both Polygon and MultiPolygon)
    exterior_rings = _get_exterior_rings(polygon)

    if not exterior_rings:
        return points

    for exterior in exterior_rings:
        coords = list(exterior.coords)
        n = len(coords)

        if n < 3:
            continue

        for i in range(n - 1):  # -1 because last point duplicates first in closed ring
            p1 = coords[i]
            p2 = coords[i + 1]

            seg_len = _point_distance(p1, p2)
            if seg_len < 0.001:
                continue

            num_samples = max(1, int(math.ceil(seg_len / sample_distance)))

            for j in range(num_samples):
                t = j / num_samples
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t

                points.append(BoundaryPoint(
                    x=x,
                    y=y,
                    unit_name=unit_name,
                    poly_index=poly_index
                ))

    return points


def _get_all_boundary_points(
    polygons: Dict[str, Polygon],
    sample_distance: float = 2.0
) -> Dict[str, List[BoundaryPoint]]:
    """
    Get all boundary sample points for all units.
    
    Args:
        polygons: Dictionary {unit_name: Polygon}
        sample_distance: Distance between sample points
    
    Returns:
        Dictionary {unit_name: [BoundaryPoint, ...]}
    """
    all_points = {}
    
    for unit_name, polygon in polygons.items():
        unit_points = _sample_polygon_boundary(polygon, unit_name, 0, sample_distance)
        all_points[unit_name] = unit_points
        logger.debug(f"Unit {unit_name}: {len(unit_points)} boundary points")
    
    return all_points


# =============================================================================
# Unambiguous Contact Detection
# =============================================================================

def _find_nearest_point_in_unit(
    point: Tuple[float, float],
    unit_points: List[BoundaryPoint]
) -> Tuple[Optional[BoundaryPoint], float]:
    """
    Find the nearest boundary point in a unit to a given point.
    
    Returns:
        Tuple of (nearest_point, distance) or (None, inf) if no points
    """
    if not unit_points:
        return None, float('inf')
    
    min_dist = float('inf')
    nearest = None
    
    for bp in unit_points:
        d = _point_distance(point, (bp.x, bp.y))
        if d < min_dist:
            min_dist = d
            nearest = bp
    
    return nearest, min_dist


def _find_unambiguous_contacts(
    all_boundary_points: Dict[str, List[BoundaryPoint]],
    buffer_distance: float = 10.0,
    ambiguity_factor: float = 1.5
) -> Dict[str, List[ContactMidpoint]]:
    """
    Find contact points where a boundary point from one unit is:
    - Within buffer_distance of EXACTLY ONE other unit
    - Not within (buffer_distance * ambiguity_factor) of any third unit
    
    This ensures we only create confident, unambiguous contacts.
    
    Args:
        all_boundary_points: Dictionary {unit_name: [BoundaryPoint, ...]}
        buffer_distance: Maximum distance to consider as "in contact"
        ambiguity_factor: Multiplier for detecting nearby third units
    
    Returns:
        Dictionary {contact_name: [ContactMidpoint, ...]}
    """
    unit_names = list(all_boundary_points.keys())
    contacts = defaultdict(list)
    
    stats = {
        'total_points': 0,
        'no_neighbor': 0,
        'one_neighbor': 0,
        'ambiguous': 0
    }
    
    # For each unit, check each boundary point
    for unit_a in unit_names:
        points_a = all_boundary_points[unit_a]
        other_units = [u for u in unit_names if u != unit_a]
        
        if not other_units:
            continue
        
        for pt_a in points_a:
            stats['total_points'] += 1
            pt_a_tuple = (pt_a.x, pt_a.y)
            
            # Find distance to nearest point in each other unit
            distances = {}
            nearest_points = {}
            
            for unit_b in other_units:
                nearest, dist = _find_nearest_point_in_unit(pt_a_tuple, all_boundary_points[unit_b])
                distances[unit_b] = dist
                nearest_points[unit_b] = nearest
            
            # Count how many units are "close" (within buffer) and "nearby" (within ambiguity zone)
            close_units = [u for u, d in distances.items() if d <= buffer_distance]
            nearby_units = [u for u, d in distances.items() if d <= buffer_distance * ambiguity_factor]
            
            if len(close_units) == 0:
                # No contact here
                stats['no_neighbor'] += 1
                continue
            elif len(close_units) == 1 and len(nearby_units) == 1:
                # Perfect: exactly one close neighbor, no ambiguity
                stats['one_neighbor'] += 1
                unit_b = close_units[0]
                pt_b = nearest_points[unit_b]
                dist = distances[unit_b]
                
                # Compute midpoint
                mid_x = (pt_a.x + pt_b.x) / 2
                mid_y = (pt_a.y + pt_b.y) / 2
                
                contact_midpoint = ContactMidpoint(
                    x=mid_x,
                    y=mid_y,
                    unit_a=unit_a,
                    unit_b=unit_b,
                    distance=dist
                )
                
                # Use canonical name (alphabetically sorted)
                contact_name = "-".join(sorted([unit_a, unit_b]))
                contacts[contact_name].append(contact_midpoint)
            else:
                # Ambiguous: multiple units nearby
                stats['ambiguous'] += 1
                logger.debug(f"Ambiguous point at ({pt_a.x:.1f}, {pt_a.y:.1f}): "
                           f"close={close_units}, nearby={nearby_units}")
    
    logger.debug(f"Contact detection stats: total={stats['total_points']}, "
                f"isolated={stats['no_neighbor']}, confident={stats['one_neighbor']}, "
                f"ambiguous={stats['ambiguous']}")
    
    return dict(contacts)


# =============================================================================
# Polyline Sorting and Cleaning
# =============================================================================

def _sort_points_nearest_neighbor(
    points: List[Tuple[float, float]],
    max_gap: float = 50.0
) -> List[Tuple[float, float]]:
    """
    Sort points into an ordered polyline using nearest-neighbor.
    
    Args:
        points: Unsorted points as (x, y) tuples
        max_gap: Maximum distance to bridge between points
    
    Returns:
        Ordered list of (x, y) coordinates
    """
    if len(points) < 2:
        return list(points)
    
    # Start with the leftmost point (smallest x)
    remaining = list(points)
    remaining.sort(key=lambda p: p[0])
    
    result = [remaining.pop(0)]
    
    while remaining:
        last = result[-1]
        
        # Find nearest remaining point
        min_dist = float('inf')
        min_idx = -1
        
        for i, pt in enumerate(remaining):
            d = _point_distance(last, pt)
            if d < min_dist:
                min_dist = d
                min_idx = i
        
        if min_dist > max_gap:
            logger.debug(f"Gap too large ({min_dist:.1f}m > {max_gap}m), stopping polyline")
            break
        
        result.append(remaining.pop(min_idx))
    
    return result


def _is_near_boundary_edge(
    point: Tuple[float, float],
    polyline: List[Tuple[float, float]],
    edge_threshold_ratio: float = 0.05
) -> bool:
    """
    Check if a point is near the boundary edge of the polyline's extent.

    A point at the "edge" is within edge_threshold_ratio of the min/max extent.
    This helps identify if a hook is at a natural termination point.
    """
    if len(polyline) < 2:
        return True

    x_coords = [p[0] for p in polyline]
    y_coords = [p[1] for p in polyline]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Check if point is near any edge
    x_threshold = x_range * edge_threshold_ratio
    y_threshold = y_range * edge_threshold_ratio

    near_x_edge = (point[0] < x_min + x_threshold) or (point[0] > x_max - x_threshold)
    near_y_edge = (point[1] < y_min + y_threshold) or (point[1] > y_max - y_threshold)

    return near_x_edge or near_y_edge


def _get_continuation_length(
    polyline: List[Tuple[float, float]],
    from_index: int,
    direction: str = "forward"
) -> float:
    """
    Calculate the total length of polyline continuing from a given index.

    Args:
        polyline: List of points
        from_index: Index to start measuring from
        direction: "forward" (toward end) or "backward" (toward start)

    Returns:
        Total length of the continuation
    """
    total_length = 0.0

    if direction == "forward":
        for i in range(from_index, len(polyline) - 1):
            total_length += _point_distance(polyline[i], polyline[i + 1])
    else:  # backward
        for i in range(from_index, 0, -1):
            total_length += _point_distance(polyline[i], polyline[i - 1])

    return total_length


def _detect_and_trim_hooks(
    polyline: List[Tuple[float, float]],
    reversal_angle: float = 140.0,
    max_hook_length: float = 20.0,
    min_points_remaining: int = 5,
    model_boundary=None
) -> List[Tuple[float, float]]:
    """
    Detect and remove true hooks at both ends of a polyline.

    A hook is defined as a SHORT terminal segment that:
    1. REVERSES direction sharply (angle > reversal_angle)
    2. Is near a natural termination point (boundary edge)
    3. Does NOT have significant polyline continuation after it

    For folded contacts, even if there's a sharp reversal, if there's
    significant continuation after the reversal point, it's preserved
    as it's likely a fold limb, not an artifact.

    Args:
        polyline: List of (x, y) coordinates
        reversal_angle: Angle (degrees) between segments that indicates reversal
        max_hook_length: Maximum length of segment to consider as a hook
        min_points_remaining: Minimum points to keep in polyline
        model_boundary: Optional boundary for termination point checking

    Returns:
        Cleaned polyline with hooks removed
    """
    if len(polyline) < min_points_remaining + 2:
        return polyline

    result = list(polyline)

    # Calculate total polyline length for comparisons
    total_length = sum(
        _point_distance(polyline[i], polyline[i+1])
        for i in range(len(polyline) - 1)
    )

    # Very short polylines - don't trim at all
    if total_length < max_hook_length * 3:
        return polyline

    # Check start for hooks
    start_trimmed = 0
    max_start_trim = min(3, len(result) - min_points_remaining)

    while start_trimmed < max_start_trim and len(result) > min_points_remaining:
        if len(result) < 4:
            break

        # First segment
        seg1 = _vector_subtract(result[1], result[0])
        seg1_len = _vector_length(seg1)

        # Second segment
        seg2 = _vector_subtract(result[2], result[1])

        angle = _angle_between_vectors(seg1, seg2)

        # Check if this looks like a hook
        if angle > reversal_angle and seg1_len < max_hook_length:
            # Calculate continuation length after the reversal
            continuation = _get_continuation_length(result, 1, "forward")

            # Check if start point is near boundary edge
            near_edge = _is_near_boundary_edge(result[0], polyline, 0.1)

            # Only trim if:
            # 1. Near edge (natural termination point), OR
            # 2. Very short hook with substantial continuation
            # But NEVER trim if continuation after reversal is significant
            # (indicates a fold limb, not an artifact)

            hook_to_continuation_ratio = seg1_len / continuation if continuation > 0 else float('inf')

            # If the "hook" segment is less than 5% of the continuation,
            # and near edge, it's probably an artifact
            if near_edge and hook_to_continuation_ratio < 0.05:
                logger.debug(f"Trimming start hook: angle={angle:.1f}°, len={seg1_len:.1f}m, ratio={hook_to_continuation_ratio:.3f}")
                result.pop(0)
                start_trimmed += 1
            else:
                # Not a hook - it's either a fold limb or has significant geometry
                break
        else:
            break

    # Check end for hooks
    end_trimmed = 0
    max_end_trim = min(3, len(result) - min_points_remaining)

    while end_trimmed < max_end_trim and len(result) > min_points_remaining:
        if len(result) < 4:
            break

        n = len(result)

        # Last segment
        seg_last = _vector_subtract(result[n-1], result[n-2])
        seg_last_len = _vector_length(seg_last)

        # Second-to-last segment
        seg_prev = _vector_subtract(result[n-2], result[n-3])

        angle = _angle_between_vectors(seg_prev, seg_last)

        if angle > reversal_angle and seg_last_len < max_hook_length:
            # Calculate continuation length before the reversal
            continuation = _get_continuation_length(result, n-2, "backward")

            # Check if end point is near boundary edge
            near_edge = _is_near_boundary_edge(result[-1], polyline, 0.1)

            hook_to_continuation_ratio = seg_last_len / continuation if continuation > 0 else float('inf')

            if near_edge and hook_to_continuation_ratio < 0.05:
                logger.debug(f"Trimming end hook: angle={angle:.1f}°, len={seg_last_len:.1f}m, ratio={hook_to_continuation_ratio:.3f}")
                result.pop()
                end_trimmed += 1
            else:
                break
        else:
            break

    if start_trimmed > 0 or end_trimmed > 0:
        logger.debug(f"Hook trimming: removed {start_trimmed} from start, {end_trimmed} from end")

    return result


def _remove_backtracking(
    polyline: List[Tuple[float, float]],
    angle_threshold: float = 120.0
) -> List[Tuple[float, float]]:
    """
    Remove points that cause the polyline to backtrack (sharp reversals).
    
    Args:
        polyline: List of (x, y) coordinates
        angle_threshold: Angle (degrees) above which is considered backtracking
    
    Returns:
        Cleaned polyline with backtracking removed
    """
    if len(polyline) < 3:
        return polyline
    
    result = [polyline[0], polyline[1]]
    removed_count = 0
    
    for i in range(2, len(polyline)):
        # Check angle between last two segments
        prev_vec = _vector_subtract(result[-1], result[-2])
        next_vec = _vector_subtract(polyline[i], result[-1])
        
        angle = _angle_between_vectors(prev_vec, next_vec)
        
        if angle > angle_threshold:
            # This would be backtracking - replace last point with current
            logger.debug(f"Removing backtrack at index {len(result)-1}: angle = {angle:.1f}°")
            result[-1] = polyline[i]
            removed_count += 1
        else:
            result.append(polyline[i])
    
    if removed_count > 0:
        logger.debug(f"Removed {removed_count} backtracking points")

    return result


def _decluster_points(
    polyline: List[Tuple[float, float]],
    min_distance: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Merge points that are too close together into single averaged points.

    This helps reduce artifacts from dense sampling where many points cluster
    together. Points within min_distance of each other are merged by averaging
    their positions.

    Args:
        polyline: List of (x, y) tuples
        min_distance: Minimum distance between points (default 5m)

    Returns:
        Declustered polyline with merged points
    """
    if len(polyline) < 2:
        return list(polyline)

    result = []
    cluster = [polyline[0]]

    for i in range(1, len(polyline)):
        current = polyline[i]
        # Check distance from cluster centroid
        centroid_x = sum(p[0] for p in cluster) / len(cluster)
        centroid_y = sum(p[1] for p in cluster) / len(cluster)
        dist = math.sqrt((current[0] - centroid_x)**2 + (current[1] - centroid_y)**2)

        if dist < min_distance:
            # Add to current cluster
            cluster.append(current)
        else:
            # Save cluster centroid and start new cluster
            result.append((centroid_x, centroid_y))
            cluster = [current]

    # Don't forget the last cluster
    if cluster:
        centroid_x = sum(p[0] for p in cluster) / len(cluster)
        centroid_y = sum(p[1] for p in cluster) / len(cluster)
        result.append((centroid_x, centroid_y))

    original_count = len(polyline)
    new_count = len(result)
    if original_count != new_count:
        logger.debug(f"Declustered {original_count} points to {new_count} points (min_distance={min_distance}m)")

    return result


def _simplify_polyline_douglas_peucker(
    polyline: List[Tuple[float, float]],
    tolerance: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Simplify a polyline using Douglas-Peucker algorithm.
    
    Args:
        polyline: List of (x, y) coordinates
        tolerance: Maximum perpendicular distance for point removal
    
    Returns:
        Simplified polyline
    """
    if len(polyline) < 3:
        return polyline
    
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line."""
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        line_len = math.sqrt(dx * dx + dy * dy)
        if line_len < 1e-10:
            return _point_distance(point, line_start)
        
        # Cross product gives area of parallelogram, divide by base for height
        cross = abs((point[0] - line_start[0]) * dy - (point[1] - line_start[1]) * dx)
        return cross / line_len
    
    def dp_recursive(points, start, end, tolerance):
        """Recursive Douglas-Peucker implementation."""
        if end - start < 2:
            return [points[start]]
        
        # Find point with maximum distance
        max_dist = 0
        max_idx = start
        
        for i in range(start + 1, end):
            d = perpendicular_distance(points[i], points[start], points[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            left = dp_recursive(points, start, max_idx, tolerance)
            right = dp_recursive(points, max_idx, end, tolerance)
            return left + right
        else:
            return [points[start]]
    
    result = dp_recursive(polyline, 0, len(polyline) - 1, tolerance)
    result.append(polyline[-1])
    
    logger.debug(f"Simplified polyline from {len(polyline)} to {len(result)} points")
    return result


def _remove_duplicate_points(
    points: List[Tuple[float, float]],
    tolerance: float
) -> List[Tuple[float, float]]:
    """Remove points that are within tolerance of each other."""
    if not points:
        return []
    
    result = [points[0]]
    for pt in points[1:]:
        is_dup = False
        for existing in result:
            dist = _point_distance(pt, existing)
            if dist < tolerance:
                is_dup = True
                break
        if not is_dup:
            result.append(pt)
    
    return result


def _clean_polyline(
    polyline: List[Tuple[float, float]],
    simplify_tolerance: float = 2.0,
    hook_reversal_angle: float = 140.0,  # More conservative - only remove clear reversals
    hook_max_length: float = 15.0,  # Shorter max hook length
    backtrack_angle: float = 150.0,  # More conservative backtrack detection
    decluster_distance: float = 5.0  # Merge points closer than this
) -> List[Tuple[float, float]]:
    """
    Full polyline cleaning pipeline.

    Args:
        polyline: Raw sorted contact points
        simplify_tolerance: Douglas-Peucker tolerance
        hook_reversal_angle: Angle indicating direction reversal for hooks
        hook_max_length: Maximum length of a hook segment
        backtrack_angle: Angle threshold for backtracking detection
        decluster_distance: Merge points within this distance

    Returns:
        Cleaned and simplified polyline
    """
    if len(polyline) < 3:
        return polyline

    logger.debug(f"Cleaning polyline with {len(polyline)} points")

    # Step 1: Decluster first to merge very close points
    if decluster_distance > 0:
        result = _decluster_points(polyline, decluster_distance)
        logger.debug(f"  After declustering: {len(result)} points")
    else:
        result = list(polyline)

    # Step 2: Simplify to reduce noise (but preserve shape)
    if simplify_tolerance > 0 and len(result) > 10:
        result = _simplify_polyline_douglas_peucker(result, simplify_tolerance)
        logger.debug(f"  After simplification: {len(result)} points")

    # Step 3: Remove only very clear hooks from simplified data
    # (this is more conservative because we check against overall trend)
    result = _detect_and_trim_hooks(result, hook_reversal_angle, hook_max_length, min_points_remaining=4)
    logger.debug(f"  After hook detection: {len(result)} points")

    # Step 4: Remove obvious backtracking (very conservative threshold)
    result = _remove_backtracking(result, backtrack_angle)
    logger.debug(f"  After backtrack removal: {len(result)} points")

    return result


# =============================================================================
# Core Contact Point Extraction (Legacy API - Two Boundaries)
# =============================================================================

def extract_contact_points(
    boundary1: LineString,
    boundary2: LineString,
    fault_lines: Optional[List[LineString]] = None,
    proximity_threshold: float = 10.0,
    sample_distance: float = 5.0,
    fault_exclusion_distance: float = 15.0
) -> List[Tuple[float, float]]:
    """
    Extract contact POINTS between two polygon boundaries.
    
    This is the legacy API for extracting contacts between exactly two boundaries.
    For multi-unit aware extraction, use extract_contacts_multi_unit().
    
    Algorithm:
    1. Sample points densely along both boundaries
    2. For each point on boundary1, find nearest on boundary2
    3. Take midpoint if within proximity threshold
    4. Exclude points near fault lines
    5. Clean hooks and backtracking
    
    Args:
        boundary1: First polygon boundary (exterior ring)
        boundary2: Second polygon boundary (exterior ring)
        fault_lines: List of fault LineStrings - points near these are excluded
        proximity_threshold: Maximum distance between boundaries to consider adjacent
        sample_distance: Distance between sample points
        fault_exclusion_distance: Exclude points within this distance of faults
    
    Returns:
        List of (easting, rl) tuples - cleaned contact points
    """
    if boundary1.is_empty or boundary2.is_empty:
        return []
    
    # Sample both boundaries
    boundary1_points = []
    boundary2_points = []
    
    for line, point_list in [(boundary1, boundary1_points), (boundary2, boundary2_points)]:
        coords = list(line.coords)
        n = len(coords)
        for i in range(n - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            seg_len = _point_distance(p1, p2)
            if seg_len < 0.001:
                continue
            num_samples = max(1, int(math.ceil(seg_len / sample_distance)))
            for j in range(num_samples):
                t = j / num_samples
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                point_list.append((x, y))
    
    if not boundary1_points or not boundary2_points:
        return []
    
    points = []
    
    # Sample from boundary1 to boundary2
    for pt1 in boundary1_points:
        min_dist = float('inf')
        nearest_pt2 = None
        
        for pt2 in boundary2_points:
            d = _point_distance(pt1, pt2)
            if d < min_dist:
                min_dist = d
                nearest_pt2 = pt2
        
        if min_dist > proximity_threshold or nearest_pt2 is None:
            continue
        
        mid_x = (pt1[0] + nearest_pt2[0]) / 2
        mid_y = (pt1[1] + nearest_pt2[1]) / 2
        points.append((mid_x, mid_y))

    # Also sample from boundary2 to boundary1
    for pt2 in boundary2_points:
        min_dist = float('inf')
        nearest_pt1 = None

        for pt1 in boundary1_points:
            d = _point_distance(pt2, pt1)
            if d < min_dist:
                min_dist = d
                nearest_pt1 = pt1

        if min_dist > proximity_threshold or nearest_pt1 is None:
            continue

        mid_x = (pt2[0] + nearest_pt1[0]) / 2
        mid_y = (pt2[1] + nearest_pt1[1]) / 2
        points.append((mid_x, mid_y))

    # Remove duplicates
    if points:
        points = _remove_duplicate_points(points, tolerance=sample_distance / 3)

    # Sort using nearest-neighbor
    if len(points) > 2:
        points = _sort_points_nearest_neighbor(points)
    else:
        points.sort(key=lambda p: p[0])

    # Clean the polyline
    if len(points) > 3:
        points = _clean_polyline(points, simplify_tolerance=2.0)

    # Terminate at faults (instead of excluding points near faults)
    # This gives clean termination rather than gaps
    if fault_lines and len(points) >= 2:
        points = terminate_at_faults(points, fault_lines, tolerance=fault_exclusion_distance)

    return points


# =============================================================================
# Multi-Unit Contact Extraction
# =============================================================================

def extract_contacts_multi_unit(
    polygons: Dict[str, Polygon],
    fault_lines: Optional[List[LineString]] = None,
    buffer_distance: float = 10.0,
    sample_distance: float = 2.0,
    simplify_tolerance: float = 2.0,
    max_gap: float = 50.0,
    fault_exclusion_distance: float = 15.0,
    ambiguity_factor: float = 1.5
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract contacts between all unit pairs with ambiguity detection.
    
    This is the preferred method when you have multiple units - it only creates
    contacts where a boundary point has EXACTLY ONE other unit nearby, avoiding
    artifacts at triple junctions.
    
    Args:
        polygons: Dictionary {unit_name: Polygon}
        fault_lines: List of fault LineStrings - points near these are excluded
        buffer_distance: Maximum distance to consider as "in contact"
        sample_distance: Distance between boundary sample points
        simplify_tolerance: Douglas-Peucker simplification tolerance
        max_gap: Maximum gap when constructing polylines
        fault_exclusion_distance: Exclude points within this distance of faults
        ambiguity_factor: Multiplier for detecting nearby third units
    
    Returns:
        Dictionary {contact_name: [(easting, rl), ...]}
    """
    if len(polygons) < 2:
        return {}
    
    # Step 1: Sample all boundaries
    all_boundary_points = _get_all_boundary_points(polygons, sample_distance)
    
    # Step 2: Find unambiguous contacts
    contact_midpoints = _find_unambiguous_contacts(
        all_boundary_points, buffer_distance, ambiguity_factor
    )
    
    # Step 3: Build and clean polylines for each contact
    result = {}
    
    for contact_name, midpoints in contact_midpoints.items():
        if len(midpoints) < 2:
            continue

        # Convert to simple point list
        points = [(mp.x, mp.y) for mp in midpoints]

        if len(points) < 2:
            continue

        # Remove duplicates
        points = _remove_duplicate_points(points, tolerance=sample_distance / 3)

        # Sort using nearest-neighbor
        points = _sort_points_nearest_neighbor(points, max_gap)

        # Clean the polyline
        if len(points) > 3:
            points = _clean_polyline(points, simplify_tolerance)

        # Terminate at faults (instead of excluding points near faults)
        # This gives clean termination rather than gaps
        if fault_lines and len(points) >= 2:
            points = terminate_at_faults(points, fault_lines, tolerance=fault_exclusion_distance)

        if len(points) >= 2:
            result[contact_name] = points
            logger.debug(f"Contact {contact_name}: {len(points)} cleaned points")
    
    return result


# =============================================================================
# Contact Extractor Class
# =============================================================================

class ContactExtractor:
    """
    Extract contact points between geological units.
    
    This uses an improved point-based approach that:
    - Samples midpoints between adjacent polygon boundaries
    - Only creates contacts where EXACTLY ONE other unit is nearby (no triple junctions)
    - Excludes points near fault lines (faults break contacts)
    - Cleans hooks and backtracking artifacts
    - Returns simplified, ordered points
    """
    
    def __init__(
        self,
        proximity_threshold: float = 10.0,
        sample_distance: float = 2.0,
        min_contact_length: float = 5.0,
        simplify_tolerance: float = 2.0,
        fault_exclusion_distance: float = 15.0,
        ambiguity_factor: float = 1.5,
        use_multi_unit_detection: bool = True
    ):
        """
        Initialize the contact extractor.
        
        Args:
            proximity_threshold: Maximum distance between boundaries to consider adjacent
            sample_distance: Distance between sample points (smaller = more detail)
            min_contact_length: Minimum contact extent (easting span) for validity
            simplify_tolerance: Douglas-Peucker simplification tolerance
            fault_exclusion_distance: Exclude points within this distance of faults
            ambiguity_factor: Multiplier for detecting nearby third units
            use_multi_unit_detection: If True, use unambiguous contact detection
        """
        self.proximity_threshold = proximity_threshold
        self.sample_distance = sample_distance
        self.min_contact_length = min_contact_length
        self.simplify_tolerance = simplify_tolerance
        self.fault_exclusion_distance = fault_exclusion_distance
        self.ambiguity_factor = ambiguity_factor
        self.use_multi_unit_detection = use_multi_unit_detection
        
        self.grouped_contacts: Dict[str, GroupedContact] = {}
    
    def extract_contacts_for_section(
        self,
        units: Dict[str, Dict],
        northing: float,
        section_key: Tuple,
        unit_assignments: Optional[Dict[str, str]] = None,
        inverse_transform=None,
        fault_lines: Optional[List[Dict]] = None,
        model_boundary=None
    ) -> List[ContactPolyline]:
        """
        Extract contact points for a single section.

        Args:
            units: Dictionary of unit data {unit_name: {"vertices": [...], ...}}
            northing: Section northing value
            section_key: (pdf_path, page_num) tuple
            unit_assignments: Optional mapping of unit names to formation names
            inverse_transform: Optional function (easting, rl) -> (pdf_x, pdf_y)
            fault_lines: Optional list of fault data dicts with "vertices" key
            model_boundary: Optional ModelBoundary object for clipping contacts

        Returns:
            List of ContactPolyline objects (each containing points for one contact pair)
        """
        if unit_assignments is None:
            unit_assignments = {}
        
        # Build Shapely polygons
        polygons = {}
        for unit_name, unit_data in units.items():
            vertices = unit_data.get("vertices", [])
            if len(vertices) < 6:
                continue
            
            coords = vertices_to_coords(vertices)
            if len(coords) < 3:
                continue
            
            # Close polygon if needed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    polygons[unit_name] = poly
            except Exception as e:
                logger.warning(f"Could not create polygon for {unit_name}: {e}")
        
        if len(polygons) < 2:
            return []
        
        # Build fault LineStrings
        fault_geometries = []
        if fault_lines:
            for fault in fault_lines:
                fv = fault.get("vertices", [])
                if len(fv) >= 4:
                    fc = vertices_to_coords(fv)
                    if len(fc) >= 2:
                        try:
                            fault_geometries.append(LineString(fc))
                        except:
                            pass
        
        logger.debug(f"Section N={northing}: {len(polygons)} polygons, {len(fault_geometries)} faults")
        
        contacts = []
        
        if self.use_multi_unit_detection and len(polygons) > 2:
            # Use multi-unit aware extraction
            contact_points = extract_contacts_multi_unit(
                polygons=polygons,
                fault_lines=fault_geometries if fault_geometries else None,
                buffer_distance=self.proximity_threshold,
                sample_distance=self.sample_distance,
                simplify_tolerance=self.simplify_tolerance,
                max_gap=50.0,
                fault_exclusion_distance=self.fault_exclusion_distance,
                ambiguity_factor=self.ambiguity_factor
            )
            
            for contact_name, points in contact_points.items():
                if len(points) < 2:
                    continue

                # Clip to model boundary if available
                if model_boundary and len(points) >= 2:
                    # Clip to topography/boundary polygon
                    if hasattr(model_boundary, 'boundary_polygon') and model_boundary.boundary_polygon:
                        points = clip_contact_to_boundary(points, model_boundary.boundary_polygon)
                    # Also terminate at topography surface
                    if hasattr(model_boundary, 'topography') and model_boundary.topography:
                        points = terminate_at_topography(points, model_boundary.topography)

                if len(points) < 2:
                    continue

                # Check minimum length
                total_span = max(p[0] for p in points) - min(p[0] for p in points)
                if total_span < self.min_contact_length:
                    continue

                # Parse unit names from contact name
                unit_names = contact_name.split("-")
                if len(unit_names) != 2:
                    continue

                name1, name2 = unit_names

                # Get formation names
                form1 = self._get_formation_name(name1, units.get(name1, {}), unit_assignments)
                form2 = self._get_formation_name(name2, units.get(name2, {}), unit_assignments)

                # Convert points to flat vertex list
                vertices = coords_to_vertices(points)
                
                # Compute PDF coordinates if inverse transform available
                pdf_vertices = None
                if inverse_transform is not None:
                    try:
                        pdf_vertices = []
                        for x, y in points:
                            pdf_x, pdf_y = inverse_transform(x, y)
                            pdf_vertices.extend([pdf_x, pdf_y])
                    except Exception as e:
                        logger.warning(f"Failed to inverse transform contact: {e}")
                        pdf_vertices = None
                
                contact = ContactPolyline(
                    vertices=vertices,
                    northing=northing,
                    section_key=section_key,
                    source_unit1=name1,
                    source_unit2=name2,
                    pdf_vertices=pdf_vertices
                )
                contacts.append(contact)
                
                # Add to grouped contacts
                self._add_to_group(contact, form1, form2)
                
                logger.debug(f"Contact {name1}-{name2}: {len(points)} points")
        else:
            # Fall back to pairwise extraction
            contacts = self._extract_pairwise(
                polygons, units, northing, section_key,
                unit_assignments, inverse_transform, fault_geometries,
                model_boundary
            )

        logger.info(f"Extracted {len(contacts)} contacts for section N={northing}")
        return contacts

    def _extract_pairwise(
        self,
        polygons: Dict[str, Polygon],
        units: Dict[str, Dict],
        northing: float,
        section_key: Tuple,
        unit_assignments: Dict[str, str],
        inverse_transform,
        fault_geometries: List[LineString],
        model_boundary=None
    ) -> List[ContactPolyline]:
        """Extract contacts using pairwise boundary comparison (legacy method)."""
        contacts = []
        processed_pairs = set()
        
        unit_names = list(polygons.keys())
        for i, name1 in enumerate(unit_names):
            poly1 = polygons[name1]
            
            for name2 in unit_names[i + 1:]:
                pair_key = tuple(sorted([name1, name2]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                poly2 = polygons[name2]
                
                # Quick distance check
                if poly1.distance(poly2) > self.proximity_threshold:
                    continue
                
                # Get exterior rings (handles both Polygon and MultiPolygon)
                exteriors1 = _get_exterior_rings(poly1)
                exteriors2 = _get_exterior_rings(poly2)

                if not exteriors1 or not exteriors2:
                    continue

                # Extract contact points from all combinations of exterior rings
                all_points = []
                for ext1 in exteriors1:
                    for ext2 in exteriors2:
                        pts = extract_contact_points(
                            boundary1=ext1,
                            boundary2=ext2,
                            fault_lines=fault_geometries if fault_geometries else None,
                            proximity_threshold=self.proximity_threshold,
                            sample_distance=self.sample_distance,
                            fault_exclusion_distance=self.fault_exclusion_distance
                        )
                        all_points.extend(pts)

                # Remove duplicates and clean
                if all_points:
                    all_points = _remove_duplicate_points(all_points, tolerance=self.sample_distance / 3)
                    if len(all_points) > 2:
                        all_points = _sort_points_nearest_neighbor(all_points, max_gap=50.0)
                        all_points = _clean_polyline(all_points, simplify_tolerance=self.simplify_tolerance)

                points = all_points

                if len(points) < 2:
                    continue

                # Clip to model boundary if available
                if model_boundary and len(points) >= 2:
                    # Clip to topography/boundary polygon
                    if hasattr(model_boundary, 'boundary_polygon') and model_boundary.boundary_polygon:
                        points = clip_contact_to_boundary(points, model_boundary.boundary_polygon)
                    # Also terminate at topography surface
                    if hasattr(model_boundary, 'topography') and model_boundary.topography:
                        points = terminate_at_topography(points, model_boundary.topography)

                if len(points) < 2:
                    continue

                # Check minimum length
                total_span = max(p[0] for p in points) - min(p[0] for p in points)
                if total_span < self.min_contact_length:
                    continue

                # Get formation names
                form1 = self._get_formation_name(name1, units.get(name1, {}), unit_assignments)
                form2 = self._get_formation_name(name2, units.get(name2, {}), unit_assignments)

                # Convert points to flat vertex list
                vertices = coords_to_vertices(points)
                
                # Compute PDF coordinates if inverse transform available
                pdf_vertices = None
                if inverse_transform is not None:
                    try:
                        pdf_vertices = []
                        for x, y in points:
                            pdf_x, pdf_y = inverse_transform(x, y)
                            pdf_vertices.extend([pdf_x, pdf_y])
                    except Exception as e:
                        logger.warning(f"Failed to inverse transform contact: {e}")
                        pdf_vertices = None
                
                contact = ContactPolyline(
                    vertices=vertices,
                    northing=northing,
                    section_key=section_key,
                    source_unit1=name1,
                    source_unit2=name2,
                    pdf_vertices=pdf_vertices
                )
                contacts.append(contact)
                
                # Add to grouped contacts
                self._add_to_group(contact, form1, form2)
                
                logger.debug(f"Contact {name1}-{name2}: {len(points)} points")
        
        return contacts
    
    def _get_formation_name(
        self,
        unit_name: str,
        unit_data: Dict,
        assignments: Dict[str, str]
    ) -> str:
        """Get the formation name for a unit, preferring assigned names."""
        # Check explicit assignments
        if unit_name in assignments:
            return assignments[unit_name]
        
        # Check unit_assignment field
        if "unit_assignment" in unit_data and unit_data["unit_assignment"]:
            return unit_data["unit_assignment"]
        
        # Check formation field
        if "formation" in unit_data and unit_data["formation"]:
            formation = unit_data["formation"]
            if formation not in ("UNIT", "UNKNOWN", ""):
                return formation
        
        # Strip numbers from unit name as fallback
        base_name = re.sub(r'\d+$', '', unit_name)
        return base_name if base_name else unit_name
    
    def _add_to_group(self, contact: ContactPolyline, formation1: str, formation2: str):
        """Add a contact to the appropriate grouped contact."""
        names = sorted([formation1, formation2])
        group_key = f"{names[0]}-{names[1]}"
        
        if group_key not in self.grouped_contacts:
            self.grouped_contacts[group_key] = GroupedContact(
                formation1=names[0],
                formation2=names[1]
            )
        
        self.grouped_contacts[group_key].polylines.append(contact)
    
    def extract_all_sections(
        self,
        all_sections_data: Dict[Tuple, Dict],
        unit_assignments: Optional[Dict[str, str]] = None,
        model_boundaries: Optional[Dict] = None
    ) -> Dict[str, GroupedContact]:
        """
        Extract contacts from all sections and group by formation pair.

        Args:
            all_sections_data: Dictionary {(pdf_path, page_num): section_data}
            unit_assignments: Optional global unit assignments
            model_boundaries: Optional dict {northing: ModelBoundary} for clipping

        Returns:
            Dictionary of grouped contacts {group_name: GroupedContact}
        """
        self.grouped_contacts = {}

        if unit_assignments is None:
            unit_assignments = {}
        if model_boundaries is None:
            model_boundaries = {}

        total_contacts = 0

        for section_key, section_data in all_sections_data.items():
            units = section_data.get("units", {})
            northing = section_data.get("northing", 0)

            if len(units) < 2:
                continue

            # Build section-specific assignments
            section_assignments = dict(unit_assignments)
            for unit_name, unit_data in units.items():
                if "unit_assignment" in unit_data and unit_data["unit_assignment"]:
                    section_assignments[unit_name] = unit_data["unit_assignment"]

            inverse_transform = section_data.get("inverse_transform")

            # Get faults from polylines
            fault_lines = []
            polylines = section_data.get("polylines", {})
            for pl_name, pl_data in polylines.items():
                if pl_data.get("is_fault") or "fault" in pl_name.lower():
                    fault_lines.append(pl_data)

            # Also check direct faults key
            if section_data.get("faults"):
                fault_lines.extend(section_data["faults"])

            # Get model boundary for this section if available
            boundary = model_boundaries.get(northing)

            contacts = self.extract_contacts_for_section(
                units=units,
                northing=northing,
                section_key=section_key,
                unit_assignments=section_assignments,
                inverse_transform=inverse_transform,
                fault_lines=fault_lines if fault_lines else None,
                model_boundary=boundary
            )

            total_contacts += len(contacts)

        logger.info(f"Extracted {total_contacts} total contacts in {len(self.grouped_contacts)} groups")
        return self.grouped_contacts
    
    def get_contact_groups(self) -> Dict[str, GroupedContact]:
        """Return the grouped contacts."""
        return self.grouped_contacts
    
    def clear(self):
        """Clear all extracted contacts."""
        self.grouped_contacts = {}


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_contacts_grouped(
    all_sections_data: Dict[Tuple, Dict],
    unit_assignments: Optional[Dict[str, str]] = None,
    sample_distance: float = 2.0,
    proximity_threshold: float = 10.0,
    min_contact_length: float = 5.0,
    simplify_tolerance: float = 2.0,
    buffer_distance: float = None,
    fault_exclusion_distance: float = 15.0,
    model_boundaries: Optional[Dict] = None
) -> Dict[str, GroupedContact]:
    """
    Convenience function to extract grouped contacts from section data.

    Args:
        all_sections_data: Dictionary {(pdf_path, page_num): section_data}
        unit_assignments: Optional mapping of unit names to formation names
        sample_distance: Distance between sample points
        proximity_threshold: Maximum distance between boundaries to consider adjacent
        min_contact_length: Minimum contact extent for validity
        simplify_tolerance: Douglas-Peucker simplification tolerance
        buffer_distance: Alias for proximity_threshold
        fault_exclusion_distance: Exclude points within this distance of faults
        model_boundaries: Optional dict {northing: ModelBoundary} for clipping contacts

    Returns:
        Dictionary of grouped contacts {group_name: GroupedContact}
    """
    if buffer_distance is not None:
        proximity_threshold = buffer_distance

    extractor = ContactExtractor(
        proximity_threshold=proximity_threshold,
        sample_distance=sample_distance,
        min_contact_length=min_contact_length,
        simplify_tolerance=simplify_tolerance,
        fault_exclusion_distance=fault_exclusion_distance
    )

    return extractor.extract_all_sections(all_sections_data, unit_assignments, model_boundaries)


def extract_single_contact(
    poly1,
    poly2,
    name1: str,
    name2: str,
    form1: str,
    form2: str,
    northing: float,
    buffer_distance: float = 10.0,
    simplify_tolerance: float = 2.0,
    fault_lines: Optional[List[LineString]] = None
) -> Optional[Dict]:
    """
    Extract a single contact between two polygons.

    Convenience function for section_viewer and other modules.
    Handles both Polygon and MultiPolygon geometries.

    Returns:
        Contact dictionary or None if no contact found
    """
    try:
        # Get exterior rings (handles both Polygon and MultiPolygon)
        exteriors1 = _get_exterior_rings(poly1)
        exteriors2 = _get_exterior_rings(poly2)

        if not exteriors1 or not exteriors2:
            return None

        # Extract contact points from all combinations of exterior rings
        all_points = []
        for ext1 in exteriors1:
            for ext2 in exteriors2:
                pts = extract_contact_points(
                    boundary1=ext1,
                    boundary2=ext2,
                    fault_lines=fault_lines,
                    proximity_threshold=buffer_distance,
                    sample_distance=2.0,
                    fault_exclusion_distance=15.0
                )
                all_points.extend(pts)

        # Remove duplicates and clean
        if all_points:
            all_points = _remove_duplicate_points(all_points, tolerance=2.0 / 3)
            if len(all_points) > 2:
                all_points = _sort_points_nearest_neighbor(all_points, max_gap=50.0)
                all_points = _clean_polyline(all_points, simplify_tolerance=simplify_tolerance)

        points = all_points

        if len(points) < 2:
            return None

        vertices = coords_to_vertices(points)

        return {
            'name': f"{name1}-{name2}",
            'unit1': name1,
            'unit2': name2,
            'formation1': form1,
            'formation2': form2,
            'vertices': vertices,
            'northing': northing,
            'type': 'Contact'
        }

    except Exception as e:
        logger.warning(f"Error extracting contact between {name1} and {name2}: {e}")
        return None


def clip_contact_to_boundary(
    contact_coords: List[Tuple[float, float]],
    boundary_polygon: Polygon
) -> List[Tuple[float, float]]:
    """
    Clip a contact line to a model boundary polygon.

    This cleanly terminates the contact at the boundary edges,
    eliminating hooks and artifacts at section edges.

    Args:
        contact_coords: List of (easting, rl) tuples
        boundary_polygon: Shapely Polygon representing model boundary

    Returns:
        Clipped contact coordinates
    """
    if not boundary_polygon or len(contact_coords) < 2:
        return contact_coords

    try:
        contact_line = LineString(contact_coords)
        clipped = contact_line.intersection(boundary_polygon)

        if clipped.is_empty:
            return []

        if clipped.geom_type == 'LineString':
            return list(clipped.coords)
        elif clipped.geom_type == 'MultiLineString':
            # Return the longest continuous segment
            longest = max(clipped.geoms, key=lambda g: g.length)
            return list(longest.coords)
        else:
            return contact_coords

    except Exception as e:
        logger.warning(f"Error clipping contact to boundary: {e}")
        return contact_coords


def clip_contacts_to_boundary(
    contacts: List[Dict],
    boundary_polygon: Polygon
) -> List[Dict]:
    """
    Clip multiple contacts to a model boundary.

    Args:
        contacts: List of contact dictionaries with 'vertices' key
        boundary_polygon: Shapely Polygon for clipping

    Returns:
        List of contacts with clipped vertices
    """
    clipped_contacts = []

    for contact in contacts:
        vertices = contact.get('vertices', [])
        coords = vertices_to_coords(vertices)

        if len(coords) < 2:
            continue

        clipped_coords = clip_contact_to_boundary(coords, boundary_polygon)

        if len(clipped_coords) >= 2:
            new_contact = dict(contact)
            new_contact['vertices'] = coords_to_vertices(clipped_coords)
            clipped_contacts.append(new_contact)

    return clipped_contacts


def terminate_at_topography(
    contact_coords: List[Tuple[float, float]],
    topography_line: LineString,
    tolerance: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Terminate a contact line where it meets the topography surface.

    The contact should not extend above the topography.

    Args:
        contact_coords: List of (easting, rl) tuples
        topography_line: LineString representing the surface
        tolerance: Distance tolerance for termination

    Returns:
        Trimmed contact coordinates
    """
    if not topography_line or len(contact_coords) < 2:
        return contact_coords

    result = []

    for i, (e, rl) in enumerate(contact_coords):
        point = Point(e, rl)

        # Check if this point is above topography
        # Find nearest point on topography at same easting
        try:
            # Get topography RL at this easting by interpolation
            topo_coords = list(topography_line.coords)
            topo_rl = None

            for j in range(len(topo_coords) - 1):
                x1, y1 = topo_coords[j]
                x2, y2 = topo_coords[j + 1]

                # Check if easting is within this segment
                if min(x1, x2) <= e <= max(x1, x2):
                    if abs(x2 - x1) > 0.001:
                        t = (e - x1) / (x2 - x1)
                        topo_rl = y1 + t * (y2 - y1)
                        break

            if topo_rl is not None:
                # If contact point is above topography, skip it
                if rl > topo_rl + tolerance:
                    continue

            result.append((e, rl))

        except Exception:
            result.append((e, rl))

    return result


def terminate_at_faults(
    contact_coords: List[Tuple[float, float]],
    fault_lines: List[LineString],
    tolerance: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Terminate a contact line where it meets fault lines.

    Instead of simply excluding points near faults (which creates gaps),
    this function clips the contact polyline at fault intersections,
    giving clean termination points.

    Args:
        contact_coords: List of (easting, rl) tuples
        fault_lines: List of Shapely LineString objects representing faults
        tolerance: Buffer distance for detecting intersection

    Returns:
        List of clipped contact coordinates (may return multiple segments)
    """
    if not fault_lines or len(contact_coords) < 2:
        return contact_coords

    try:
        contact_line = LineString(contact_coords)

        # Create a union of all fault buffers
        fault_buffers = []
        for fault in fault_lines:
            if fault and not fault.is_empty:
                fault_buffers.append(fault.buffer(tolerance))

        if not fault_buffers:
            return contact_coords

        fault_union = unary_union(fault_buffers)

        # Get the difference - parts of contact NOT intersecting faults
        result = contact_line.difference(fault_union)

        if result.is_empty:
            return []

        if result.geom_type == 'LineString':
            return list(result.coords)
        elif result.geom_type == 'MultiLineString':
            # Return the longest continuous segment
            longest = max(result.geoms, key=lambda g: g.length)
            return list(longest.coords)
        else:
            return contact_coords

    except Exception as e:
        logger.warning(f"Error terminating contact at faults: {e}")
        return contact_coords


def clip_contact_at_faults(
    contact_coords: List[Tuple[float, float]],
    fault_lines: List[LineString],
    snap_distance: float = 10.0
) -> List[List[Tuple[float, float]]]:
    """
    Clip a contact line at fault intersections, returning multiple segments.

    This is useful when a contact crosses multiple faults and should be
    split into separate segments that terminate cleanly at each fault.

    Args:
        contact_coords: List of (easting, rl) tuples
        fault_lines: List of Shapely LineString objects representing faults
        snap_distance: Distance to snap endpoints to fault lines

    Returns:
        List of contact segments, each a list of (easting, rl) tuples
    """
    if not fault_lines or len(contact_coords) < 2:
        return [contact_coords]

    try:
        contact_line = LineString(contact_coords)
        segments = []

        # Find all intersection points with faults
        cut_points = []
        for fault in fault_lines:
            if fault and not fault.is_empty:
                intersection = contact_line.intersection(fault)
                if intersection.is_empty:
                    continue

                if intersection.geom_type == 'Point':
                    cut_points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    cut_points.extend(intersection.geoms)

        if not cut_points:
            return [contact_coords]

        # Sort cut points along the contact line
        cut_distances = []
        for pt in cut_points:
            d = contact_line.project(pt)
            cut_distances.append((d, pt))
        cut_distances.sort(key=lambda x: x[0])

        # Split the line at cut points
        current_start = 0.0
        for d, pt in cut_distances:
            if d > current_start + 1.0:  # Minimum segment length
                # Extract segment from current_start to d
                segment_coords = []
                for i, (e, rl) in enumerate(contact_coords):
                    pt_on_line = contact_line.project(Point(e, rl))
                    if current_start <= pt_on_line <= d:
                        segment_coords.append((e, rl))

                if len(segment_coords) >= 2:
                    # Snap endpoint to fault intersection
                    segment_coords[-1] = (pt.x, pt.y)
                    segments.append(segment_coords)

            current_start = d

        # Add final segment
        if current_start < contact_line.length - 1.0:
            segment_coords = []
            for e, rl in contact_coords:
                pt_on_line = contact_line.project(Point(e, rl))
                if pt_on_line >= current_start:
                    segment_coords.append((e, rl))

            if len(segment_coords) >= 2:
                segments.append(segment_coords)

        return segments if segments else [contact_coords]

    except Exception as e:
        logger.warning(f"Error clipping contact at faults: {e}")
        return [contact_coords]