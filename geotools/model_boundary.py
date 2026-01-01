# geotools/model_boundary.py
"""
Model boundary extraction from PDF page paths.

Extracts:
1. Topography line - the surface/upper limit for geological features
2. Bounding box - the section's extent limits
3. Model boundary - closed polygon for clipping contacts and units

These boundaries can be used to:
- Clip contacts cleanly at edges (eliminates hooks)
- Terminate polygon boundaries against topography
- Export model extents for 3D modeling software
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class ModelBoundary:
    """
    Represents the model boundary for a cross-section.

    Contains:
    - topography: The surface line (top of geology)
    - bounding_box: The rectangular extent of the section
    - model_polygon: Closed polygon for clipping (box with topo as upper edge)
    """
    # PDF coordinates (before georeferencing)
    pdf_topography: List[Tuple[float, float]] = field(default_factory=list)
    pdf_bounding_box: Tuple[float, float, float, float] = None  # (x_min, y_min, x_max, y_max)

    # Real-world coordinates (after georeferencing)
    topography: List[Tuple[float, float]] = field(default_factory=list)  # [(easting, rl), ...]
    bounding_box: Tuple[float, float, float, float] = None  # (e_min, rl_min, e_max, rl_max)

    # Derived geometries
    topography_line: LineString = None
    boundary_polygon: Polygon = None

    # Section info
    northing: float = 0.0
    section_key: Tuple = None

    def build_geometries(self):
        """Build Shapely geometries from coordinates."""
        if len(self.topography) >= 2:
            self.topography_line = LineString(self.topography)

        if self.bounding_box and len(self.topography) >= 2:
            # Create model polygon: rectangle with topography as upper boundary
            e_min, rl_min, e_max, rl_max = self.bounding_box

            # Build polygon clockwise from bottom-left
            coords = []

            # Bottom edge (left to right)
            coords.append((e_min, rl_min))
            coords.append((e_max, rl_min))

            # Right edge (bottom to top)
            coords.append((e_max, rl_max))

            # Top edge: use topography if available, else straight line
            if self.topography_line:
                # Get topography points sorted by easting (right to left for clockwise)
                topo_points = list(self.topography_line.coords)
                topo_points.sort(key=lambda p: -p[0])  # Descending easting
                coords.extend(topo_points)
            else:
                coords.append((e_min, rl_max))

            # Left edge (top to bottom) - implicit close
            coords.append((e_min, rl_min))

            try:
                self.boundary_polygon = Polygon(coords)
                if not self.boundary_polygon.is_valid:
                    self.boundary_polygon = self.boundary_polygon.buffer(0)
            except Exception as e:
                logger.warning(f"Could not build boundary polygon: {e}")
                # Fall back to simple rectangle
                self.boundary_polygon = box(e_min, rl_min, e_max, rl_max)

    def clip_contact(self, contact_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Clip a contact line to the model boundary.

        This removes any points outside the boundary and ensures
        the contact terminates cleanly at edges.
        """
        if not self.boundary_polygon or len(contact_coords) < 2:
            return contact_coords

        try:
            contact_line = LineString(contact_coords)
            clipped = contact_line.intersection(self.boundary_polygon)

            if clipped.is_empty:
                return []

            if clipped.geom_type == 'LineString':
                return list(clipped.coords)
            elif clipped.geom_type == 'MultiLineString':
                # Return the longest segment
                longest = max(clipped.geoms, key=lambda g: g.length)
                return list(longest.coords)
            else:
                return contact_coords

        except Exception as e:
            logger.warning(f"Error clipping contact: {e}")
            return contact_coords

    def clip_polygon(self, polygon_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Clip a polygon to the model boundary.

        Ensures polygons don't extend past the section limits.
        """
        if not self.boundary_polygon or len(polygon_coords) < 3:
            return polygon_coords

        try:
            poly = Polygon(polygon_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)

            clipped = poly.intersection(self.boundary_polygon)

            if clipped.is_empty:
                return []

            if clipped.geom_type == 'Polygon':
                return list(clipped.exterior.coords)
            elif clipped.geom_type == 'MultiPolygon':
                # Return the largest polygon
                largest = max(clipped.geoms, key=lambda g: g.area)
                return list(largest.exterior.coords)
            else:
                return polygon_coords

        except Exception as e:
            logger.warning(f"Error clipping polygon: {e}")
            return polygon_coords

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'topography': self.topography,
            'bounding_box': self.bounding_box,
            'northing': self.northing,
            'section_key': self.section_key,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelBoundary':
        """Create from dictionary."""
        mb = cls(
            topography=data.get('topography', []),
            bounding_box=data.get('bounding_box'),
            northing=data.get('northing', 0.0),
            section_key=data.get('section_key'),
        )
        mb.build_geometries()
        return mb


def extract_paths_from_page(page) -> List[Dict]:
    """
    Extract all path objects from a PDF page.

    Returns list of path dictionaries with:
    - vertices: List of (x, y) points
    - bbox: Bounding box (x0, y0, x1, y1)
    - is_closed: Whether path is closed
    - color: Stroke color if any
    - fill: Fill color if any
    - items: Raw drawing items
    """
    paths = []

    try:
        drawings = page.get_drawings()

        for idx, drawing in enumerate(drawings):
            items = drawing.get("items", [])
            if not items:
                continue

            # Extract vertices from drawing items
            vertices = []
            for item in items:
                item_type = item[0]

                if item_type == "l":  # Line segment
                    start_pt = item[1]
                    end_pt = item[2]
                    if not vertices:
                        vertices.append((start_pt.x, start_pt.y))
                    vertices.append((end_pt.x, end_pt.y))

                elif item_type == "c":  # Cubic bezier - sample endpoints
                    if len(item) >= 5:
                        p0 = item[1]
                        p3 = item[4]
                        if not vertices:
                            vertices.append((p0.x, p0.y))
                        vertices.append((p3.x, p3.y))

                elif item_type == "re":  # Rectangle
                    rect = item[1]
                    vertices.extend([
                        (rect.x0, rect.y0),
                        (rect.x1, rect.y0),
                        (rect.x1, rect.y1),
                        (rect.x0, rect.y1),
                        (rect.x0, rect.y0),
                    ])

                elif item_type == "qu":  # Quad
                    quad = item[1]
                    vertices.extend([
                        (quad.ul.x, quad.ul.y),
                        (quad.ur.x, quad.ur.y),
                        (quad.lr.x, quad.lr.y),
                        (quad.ll.x, quad.ll.y),
                        (quad.ul.x, quad.ul.y),
                    ])

            if len(vertices) < 2:
                continue

            # Calculate bounding box
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Check if path is closed
            is_closed = drawing.get("closePath", False)
            if vertices[0] == vertices[-1]:
                is_closed = True

            paths.append({
                'index': idx,
                'vertices': vertices,
                'bbox': bbox,
                'width': width,
                'height': height,
                'is_closed': is_closed,
                'color': drawing.get("color"),
                'fill': drawing.get("fill"),
                'stroke_opacity': drawing.get("stroke_opacity", 1.0),
                'fill_opacity': drawing.get("fill_opacity", 1.0),
                'items': items,
                'n_points': len(vertices),
            })

    except Exception as e:
        logger.error(f"Error extracting paths: {e}")

    return paths


def identify_topography(paths: List[Dict], page_width: float, page_height: float) -> Optional[Dict]:
    """
    Identify the topography line from PDF paths.

    Characteristics of topography:
    - Spans most of the page width (>80%)
    - Has many points (irregular surface)
    - Not a closed shape
    - Usually in upper portion of page
    - Often the 2nd path (after any title/border)
    """
    candidates = []

    for path in paths:
        width = path['width']
        n_points = path['n_points']
        bbox = path['bbox']

        # Must span most of page width
        width_ratio = width / page_width if page_width > 0 else 0
        if width_ratio < 0.7:
            continue

        # Should have many points (topography is irregular)
        if n_points < 10:
            continue

        # Should not be a rectangle (4-5 points)
        if n_points <= 5 and path['is_closed']:
            continue

        # Calculate score
        score = width_ratio * 100  # Prefer wider paths
        score += min(n_points, 50)  # Prefer more complex paths

        # Bonus for being in upper portion (PDF y increases downward)
        y_center = (bbox[1] + bbox[3]) / 2
        if y_center < page_height * 0.5:
            score += 20

        candidates.append((score, path))

    if not candidates:
        return None

    # Return highest scoring candidate
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def identify_bounding_box(paths: List[Dict], page_width: float, page_height: float) -> Optional[Tuple[float, float, float, float]]:
    """
    Identify the bounding box (section limits) from PDF paths.

    Characteristics:
    - Could be 4 separate lines forming a rectangle
    - Or a single closed rectangle path
    - Spans most of page width and height
    - Usually black or dark color
    """
    # First, look for a single rectangle that spans most of the page
    for path in paths:
        if not path['is_closed']:
            continue

        width_ratio = path['width'] / page_width if page_width > 0 else 0
        height_ratio = path['height'] / page_height if page_height > 0 else 0

        # Must span most of page
        if width_ratio > 0.8 and height_ratio > 0.7:
            return path['bbox']

    # Look for 4 lines that form a rectangle
    horizontal_lines = []
    vertical_lines = []

    for path in paths:
        if path['n_points'] != 2:
            continue

        v = path['vertices']
        dx = abs(v[1][0] - v[0][0])
        dy = abs(v[1][1] - v[0][1])

        # Horizontal line (wide, not tall)
        if dx > page_width * 0.5 and dy < 5:
            horizontal_lines.append(path)
        # Vertical line (tall, not wide)
        elif dy > page_height * 0.5 and dx < 5:
            vertical_lines.append(path)

    # If we have 2 horizontal and 2 vertical lines, compute bounding box
    if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
        x_coords = []
        y_coords = []

        for path in horizontal_lines + vertical_lines:
            for v in path['vertices']:
                x_coords.append(v[0])
                y_coords.append(v[1])

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    return None


def extract_model_boundary(
    page,
    transform_func=None,
    northing: float = 0.0,
    section_key: Tuple = None
) -> Optional[ModelBoundary]:
    """
    Extract model boundary (topography + bounding box) from a PDF page.

    Args:
        page: PyMuPDF page object
        transform_func: Optional function (pdf_x, pdf_y) -> (easting, rl)
        northing: Section northing value
        section_key: (pdf_path, page_num) tuple

    Returns:
        ModelBoundary object or None if extraction fails
    """
    try:
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Extract all paths
        paths = extract_paths_from_page(page)
        logger.info(f"Found {len(paths)} paths on page")

        if not paths:
            return None

        # Identify topography
        topo_path = identify_topography(paths, page_width, page_height)

        # Identify bounding box
        bbox = identify_bounding_box(paths, page_width, page_height)

        # Create model boundary
        mb = ModelBoundary(
            northing=northing,
            section_key=section_key,
        )

        # Set PDF coordinates
        if topo_path:
            mb.pdf_topography = topo_path['vertices']
            logger.info(f"Found topography line with {len(mb.pdf_topography)} points")

        if bbox:
            mb.pdf_bounding_box = bbox
            logger.info(f"Found bounding box: {bbox}")

        # Transform to real-world coordinates if function provided
        if transform_func:
            if mb.pdf_topography:
                mb.topography = [transform_func(x, y) for x, y in mb.pdf_topography]

            if mb.pdf_bounding_box:
                x0, y0, x1, y1 = mb.pdf_bounding_box
                e0, rl0 = transform_func(x0, y0)
                e1, rl1 = transform_func(x1, y1)
                mb.bounding_box = (
                    min(e0, e1), min(rl0, rl1),
                    max(e0, e1), max(rl0, rl1)
                )
        else:
            # Use PDF coordinates as-is
            mb.topography = mb.pdf_topography
            mb.bounding_box = mb.pdf_bounding_box

        # Build geometries
        mb.build_geometries()

        return mb

    except Exception as e:
        logger.error(f"Error extracting model boundary: {e}")
        return None


def create_model_boundary_from_coord_system(
    coord_system: Dict,
    page=None,
    transform_func=None,
    northing: float = 0.0,
    section_key: Tuple = None
) -> Optional[ModelBoundary]:
    """
    Create a ModelBoundary from coordinate system data.

    This is the preferred method as it uses the precise section boundaries
    identified by the georeferencer from coordinate labels.

    Args:
        coord_system: Dictionary with easting_min/max, rl_min/max, pdf_x_min/max, pdf_y_min/max
        page: Optional PyMuPDF page object for topography extraction
        transform_func: Function (pdf_x, pdf_y) -> (easting, rl)
        northing: Section northing value
        section_key: (pdf_path, page_num) tuple

    Returns:
        ModelBoundary object or None
    """
    if not coord_system:
        return None

    try:
        mb = ModelBoundary(
            northing=northing,
            section_key=section_key,
        )

        # Set real-world bounding box from coord_system
        easting_min = coord_system.get("easting_min")
        easting_max = coord_system.get("easting_max")
        rl_min = coord_system.get("rl_min")
        rl_max = coord_system.get("rl_max")

        if all(v is not None for v in [easting_min, easting_max, rl_min, rl_max]):
            mb.bounding_box = (easting_min, rl_min, easting_max, rl_max)
            logger.debug(f"Boundary box: E=[{easting_min}, {easting_max}], RL=[{rl_min}, {rl_max}]")

        # Set PDF bounding box
        pdf_x_min = coord_system.get("pdf_x_min")
        pdf_x_max = coord_system.get("pdf_x_max")
        pdf_y_min = coord_system.get("pdf_y_min")
        pdf_y_max = coord_system.get("pdf_y_max")

        if all(v is not None for v in [pdf_x_min, pdf_x_max, pdf_y_min, pdf_y_max]):
            mb.pdf_bounding_box = (pdf_x_min, pdf_y_min, pdf_x_max, pdf_y_max)

        # Try to extract topography from page if provided
        if page and transform_func:
            page_rect = page.rect
            paths = extract_paths_from_page(page)

            if paths:
                topo_path = identify_topography(paths, page_rect.width, page_rect.height)

                if topo_path:
                    # Filter topography to only include points within section bounds
                    pdf_verts = topo_path['vertices']

                    # Transform and filter
                    valid_topo = []
                    for px, py in pdf_verts:
                        # Only include points within the PDF section bounds
                        if pdf_x_min and pdf_x_max and pdf_y_min and pdf_y_max:
                            if not (pdf_x_min - 10 <= px <= pdf_x_max + 10):
                                continue

                        try:
                            e, rl = transform_func(px, py)
                            # Only include points within easting bounds
                            if easting_min and easting_max:
                                if easting_min - 50 <= e <= easting_max + 50:
                                    valid_topo.append((e, rl))
                            else:
                                valid_topo.append((e, rl))
                        except:
                            pass

                    if len(valid_topo) >= 2:
                        # Sort by easting
                        valid_topo.sort(key=lambda p: p[0])
                        mb.topography = valid_topo
                        mb.pdf_topography = [(p[0], p[1]) for p in pdf_verts]
                        logger.debug(f"Extracted topography with {len(valid_topo)} points")

        # Build geometries
        mb.build_geometries()

        return mb

    except Exception as e:
        logger.error(f"Error creating model boundary from coord_system: {e}")
        return None


def analyze_page_paths(page) -> Dict:
    """
    Analyze all paths on a page and return a summary.

    Useful for debugging and understanding page structure.
    """
    page_rect = page.rect
    paths = extract_paths_from_page(page)

    summary = {
        'page_size': (page_rect.width, page_rect.height),
        'total_paths': len(paths),
        'paths': [],
    }

    for i, path in enumerate(paths):
        path_info = {
            'index': i,
            'n_points': path['n_points'],
            'bbox': path['bbox'],
            'width': path['width'],
            'height': path['height'],
            'is_closed': path['is_closed'],
            'width_ratio': path['width'] / page_rect.width if page_rect.width > 0 else 0,
            'height_ratio': path['height'] / page_rect.height if page_rect.height > 0 else 0,
            'color': path['color'],
            'fill': path['fill'],
        }
        summary['paths'].append(path_info)

    return summary


def print_path_analysis(page):
    """Print a human-readable analysis of page paths."""
    analysis = analyze_page_paths(page)

    print(f"\n{'='*60}")
    print(f"Page size: {analysis['page_size'][0]:.1f} x {analysis['page_size'][1]:.1f}")
    print(f"Total paths: {analysis['total_paths']}")
    print(f"{'='*60}\n")

    for path in analysis['paths']:
        print(f"Path {path['index']:3d}: {path['n_points']:4d} pts | "
              f"W:{path['width_ratio']*100:5.1f}% H:{path['height_ratio']*100:5.1f}% | "
              f"{'Closed' if path['is_closed'] else 'Open  '} | "
              f"Color:{path['color']} Fill:{path['fill']}")


# =============================================================================
# DXF Export Functions
# =============================================================================

def export_boundary_dxf(
    boundaries: List[ModelBoundary],
    filename: str,
    include_topography: bool = True,
    include_bounding_box: bool = True,
    include_model_polygon: bool = True
) -> int:
    """
    Export model boundaries to DXF format.

    Args:
        boundaries: List of ModelBoundary objects
        filename: Output DXF file path
        include_topography: Export topography lines
        include_bounding_box: Export bounding box rectangles
        include_model_polygon: Export closed model polygons

    Returns:
        Number of features exported
    """
    features_exported = 0

    try:
        with open(filename, 'w') as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            for mb in boundaries:
                northing = mb.northing

                # Export topography as 3D polyline
                if include_topography and mb.topography:
                    f.write("0\nPOLYLINE\n")
                    f.write("8\nTOPOGRAPHY\n")  # Layer name
                    f.write("66\n1\n")  # Vertices follow
                    f.write("70\n8\n")  # 3D polyline flag

                    for e, rl in mb.topography:
                        f.write("0\nVERTEX\n")
                        f.write("8\nTOPOGRAPHY\n")
                        f.write(f"10\n{e:.6f}\n")      # X (Easting)
                        f.write(f"20\n{northing:.6f}\n")  # Y (Northing)
                        f.write(f"30\n{rl:.6f}\n")     # Z (RL)

                    f.write("0\nSEQEND\n")
                    features_exported += 1

                # Export bounding box as 3D polyline
                if include_bounding_box and mb.bounding_box:
                    e_min, rl_min, e_max, rl_max = mb.bounding_box

                    f.write("0\nPOLYLINE\n")
                    f.write("8\nMODEL_EXTENT\n")  # Layer name
                    f.write("66\n1\n")
                    f.write("70\n9\n")  # 3D polyline, closed

                    # Write box corners
                    for e, rl in [(e_min, rl_min), (e_max, rl_min),
                                  (e_max, rl_max), (e_min, rl_max), (e_min, rl_min)]:
                        f.write("0\nVERTEX\n")
                        f.write("8\nMODEL_EXTENT\n")
                        f.write(f"10\n{e:.6f}\n")
                        f.write(f"20\n{northing:.6f}\n")
                        f.write(f"30\n{rl:.6f}\n")

                    f.write("0\nSEQEND\n")
                    features_exported += 1

                # Export model polygon (with topography as top edge)
                if include_model_polygon and mb.boundary_polygon:
                    coords = list(mb.boundary_polygon.exterior.coords)

                    f.write("0\nPOLYLINE\n")
                    f.write("8\nMODEL_BOUNDARY\n")  # Layer name
                    f.write("66\n1\n")
                    f.write("70\n9\n")  # 3D polyline, closed

                    for e, rl in coords:
                        f.write("0\nVERTEX\n")
                        f.write("8\nMODEL_BOUNDARY\n")
                        f.write(f"10\n{e:.6f}\n")
                        f.write(f"20\n{northing:.6f}\n")
                        f.write(f"30\n{rl:.6f}\n")

                    f.write("0\nSEQEND\n")
                    features_exported += 1

            # DXF footer
            f.write("0\nENDSEC\n0\nEOF\n")

        logger.info(f"Exported {features_exported} boundary features to {filename}")

    except Exception as e:
        logger.error(f"Error exporting boundary DXF: {e}")
        raise

    return features_exported


def write_dxf_polyline(
    f,
    coords: List[Tuple[float, float]],
    northing: float,
    layer: str,
    closed: bool = False
):
    """
    Write a single 3D polyline to an open DXF file.

    Args:
        f: Open file handle
        coords: List of (easting, rl) tuples
        northing: Y coordinate (northing)
        layer: DXF layer name
        closed: Whether polyline should be closed
    """
    if len(coords) < 2:
        return

    f.write("0\nPOLYLINE\n")
    f.write(f"8\n{layer}\n")
    f.write("66\n1\n")
    f.write(f"70\n{'9' if closed else '8'}\n")  # 9=closed, 8=open

    for e, rl in coords:
        f.write("0\nVERTEX\n")
        f.write(f"8\n{layer}\n")
        f.write(f"10\n{e:.6f}\n")
        f.write(f"20\n{northing:.6f}\n")
        f.write(f"30\n{rl:.6f}\n")

    f.write("0\nSEQEND\n")
