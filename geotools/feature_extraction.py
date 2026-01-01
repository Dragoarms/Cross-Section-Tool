# geotools\feature_extraction.py
"""
Feature extraction module for geological features.
Handles polygon detection, classification, and contact extraction.

Key filtering rules:
- Only extract LINES with subject='Fault'
- Only extract COLORED polygons (non-grayscale)
- Contacts are simplified polylines between adjacent units
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.strtree import STRtree
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class FeatureExtractor:
    """Extract and classify geological features from PDF annotations."""

    FEATURE_TYPES = {
        "UNIT": "Geological Unit",
        "FAULT": "Fault",
        "CONTACT": "Contact",
        "TOPO": "Topography",
        "UNCONFORMITY": "Unconformity",
        "INTRUSION": "Intrusion",
        "FOLD": "Fold Axis",
    }

    def __init__(self, extraction_filter=None):
        self.features = []
        self.contacts = []
        self.geological_units = {}
        self.faults = []  # Extracted fault lines
        self.feature_colors = {}
        self.extraction_filter = extraction_filter

    def classify_feature_interactive(self, feature: Dict, feature_type: str):
        """Allow user to classify a feature interactively."""
        feature["classification"] = feature_type
        feature["modified"] = True
        logger.info(f"Feature {feature['name']} classified as {feature_type}")

    def set_extraction_filter(self, extraction_filter):
        """Set the extraction filter to use."""
        self.extraction_filter = extraction_filter

    def _is_colored_polygon(self, color) -> Tuple[bool, str]:
        """
        Check if color qualifies as a 'colored' polygon (not grayscale).
        
        Returns:
            (is_colored, reason)
        """
        if not color or len(color) < 3:
            return False, "No color"

        r, g, b = color[:3]

        # Reject black (all values low)
        if r < 0.1 and g < 0.1 and b < 0.1:
            return False, "Black"

        # Reject white (all values high)
        if r > 0.9 and g > 0.9 and b > 0.9:
            return False, "White"

        # Reject grayscale (R â‰ˆ G â‰ˆ B)
        tolerance = 0.15
        if abs(r - g) < tolerance and abs(g - b) < tolerance and abs(r - b) < tolerance:
            return False, f"Grayscale (R={r:.2f}, G={g:.2f}, B={b:.2f})"

        return True, "Colored"

    def _is_fault_line(self, annot) -> Tuple[bool, str]:
        """
        Check if an annotation is a fault line.

        Checks subject, title, author, and content fields for:
        - 'fault' keyword (case insensitive)
        - Shortened versions like 'F1', 'F2', 'F10', etc.

        IMPORTANT: Subject field is prioritized for the fault NAME.
        Other fields are only used to detect if it's a fault, but subject
        is always preferred as the identifier.

        Returns:
            (is_fault, fault_name)
        """
        import re

        if not hasattr(annot, 'info'):
            return False, ""

        info = annot.info

        # Get all field values
        subject = (info.get('subject', '') or '').strip()
        title = (info.get('title', '') or '').strip()
        author = (info.get('author', '') or '').strip()
        content = (info.get('content', '') or '').strip()

        # Pattern for fault: "fault" or "F" followed by number (F1, F2, F10, etc.)
        fault_pattern = re.compile(r'\bfault\b|^f\d+$|^f\s*\d+$|\bf\d+\b', re.IGNORECASE)

        # Check if ANY field indicates this is a fault
        is_fault = False
        for field_value in [subject, title, author, content]:
            if field_value and fault_pattern.search(field_value):
                is_fault = True
                break

        if not is_fault:
            return False, ""

        # PRIORITIZE subject for the fault name
        # Only fall back to other fields if subject is empty or doesn't match pattern
        if subject and fault_pattern.search(subject):
            fault_name = subject
            logger.debug(f"Fault detected, using subject as name: '{fault_name}'")
        elif subject:
            # Subject exists but doesn't match pattern - still use it as name
            fault_name = subject
            logger.debug(f"Fault detected, using subject (non-pattern) as name: '{fault_name}'")
        elif title and fault_pattern.search(title):
            fault_name = title
            logger.debug(f"Fault detected, using title as name: '{fault_name}'")
        elif author and fault_pattern.search(author):
            fault_name = author
            logger.debug(f"Fault detected, using author as name: '{fault_name}'")
        elif content and fault_pattern.search(content):
            fault_name = content
            logger.debug(f"Fault detected, using content as name: '{fault_name}'")
        else:
            # Fallback - use first non-empty field
            fault_name = subject or title or author or content or "Fault"
            logger.debug(f"Fault detected, using fallback name: '{fault_name}'")

        return True, fault_name

    def _get_author_from_annot(self, annot) -> Optional[str]:
        """Get author/title from annotation metadata."""
        if not hasattr(annot, 'info'):
            return None
        info = annot.info
        return info.get('title') or info.get('author')

    def _has_author_tag(self, annot) -> bool:
        """Check if annotation has an author/title tag set."""
        author = self._get_author_from_annot(annot)
        return author is not None and author.strip() != ''

    def extract_annotations(self, page) -> List[Dict]:
        """
        Extract annotations from PDF page.
        
        Filtering rules:
        - ONLY extract items that have an 'author' tag (title/author metadata)
        - Lines/PolyLines: Extract if has author tag, classify as Fault if subject contains 'Fault'
        - Polygons: Extract if has author tag (including grayscale if has author)
        
        Returns:
            List of annotation dictionaries (polygons + faults combined for backward compatibility)
            Faults are also stored separately in self.faults
        """
        annotations = []
        extracted_faults = []

        try:
            annot_count = sum(1 for _ in page.annots())
            logger.info(f"Scanning page: {annot_count} annotations")

            polygons_found = 0
            faults_found = 0
            skipped_no_author = 0
            skipped_too_few_points = 0

            for annot in page.annots():
                annot_type = annot.type[0]
                
                # CRITICAL: Only extract items with author tag
                has_author = self._has_author_tag(annot)
                author = self._get_author_from_annot(annot)
                
                if not has_author:
                    skipped_no_author += 1
                    logger.debug(f"Skipped annotation - no author tag")
                    continue
                
                # Check if this is a line/polyline (types 3, 9, 15)
                # Type 3 = Line, Type 9 = PolyLine, Type 15 = Ink (pencil/freehand)
                is_line_type = annot_type in [3, 9, 15]
                
                # Check if this is a polygon type (types 4, 5, 6, 10)
                is_polygon_type = annot_type in [4, 5, 6, 10]  # Square, Circle, Highlight, Polygon

                # Check if subject indicates this is a fault
                is_fault, fault_name = self._is_fault_line(annot)

                if is_line_type:
                    # Lines with author tag - extract them
                    vertices = self._extract_vertices(annot)
                    if vertices and len(vertices) >= 4:
                        color = self._get_color(annot) or (1.0, 0.0, 0.0)  # Default red for lines
                        
                        if is_fault:
                            # This is a fault line
                            fault_data = {
                                "type": "Fault",
                                "name": fault_name,
                                "vertices": vertices,
                                "color": color,
                                "formation": "FAULT",
                                "classification": "FAULT",
                                "source": "annotation",
                                "author": author,
                            }
                            annotations.append(fault_data)
                            extracted_faults.append(fault_data)
                            faults_found += 1
                            logger.debug(f"Extracted fault: {fault_name} (author: {author})")
                        else:
                            # Regular polyline with author - could be a contact or other feature
                            polyline_data = {
                                "type": "PolyLine",
                                "name": author or "PolyLine",
                                "vertices": vertices,
                                "color": color,
                                "formation": author or "LINE",
                                "classification": "LINE",
                                "source": "annotation",
                                "author": author,
                            }
                            annotations.append(polyline_data)
                            logger.debug(f"Extracted polyline: {author}")
                    else:
                        skipped_too_few_points += 1

                elif is_polygon_type:
                    # Polygons with author tag - extract regardless of color (grayscale OK)
                    vertices = self._extract_vertices(annot)
                    
                    if vertices and len(vertices) >= 6:
                        color = self._get_color(annot) or (0.5, 0.5, 0.5)  # Default gray
                        
                        if is_fault:
                            # Fault polygon
                            fault_data = {
                                "type": "Fault",
                                "name": fault_name,
                                "vertices": vertices,
                                "color": color,
                                "formation": "FAULT",
                                "classification": "FAULT",
                                "source": "annotation",
                                "author": author,
                            }
                            annotations.append(fault_data)
                            extracted_faults.append(fault_data)
                            faults_found += 1
                            logger.debug(f"Extracted fault polygon: {fault_name} (author: {author})")
                        else:
                            # Geological unit polygon - use author as formation name if no subject
                            formation = self._get_formation_from_annot(annot, color)
                            # If formation couldn't be determined from subject, use author
                            if formation == "UNIT" and author:
                                formation = author.strip().upper()
                            
                            # Check for saved assignment from previous session
                            saved_assignment = self._get_saved_assignment(annot)
                            
                            polygon_data = {
                                "type": "Polygon",
                                "vertices": vertices,
                                "color": color,
                                "formation": formation,
                                "unit_number": None,
                                "classification": "UNIT",
                                "source": "annotation",
                                "author": author,
                                "unit_assignment": saved_assignment,  # Pre-populate if saved
                            }
                            annotations.append(polygon_data)
                            polygons_found += 1
                            logger.debug(f"Extracted polygon: {formation} (author: {author}, saved: {saved_assignment})")
                    else:
                        skipped_too_few_points += 1

            # Log annotation summary
            logger.info(f"Annotation extraction summary:")
            logger.info(f"  - Polygons found: {polygons_found}")
            logger.info(f"  - Faults found: {faults_found}")
            logger.info(f"  - Skipped (no author tag): {skipped_no_author}")
            logger.info(f"  - Skipped (too few points): {skipped_too_few_points}")

            # NOTE: We no longer extract from drawings by default since they lack author tags
            # If you need drawing extraction, enable it explicitly with a filter option

            logger.info(f"Total extracted: {len(annotations)} features ({polygons_found} polygons, {faults_found} faults)")

        except Exception as e:
            logger.error(f"Error extracting annotations: {str(e)}", exc_info=True)

        # Store faults separately for easy access
        self.faults = extracted_faults

        return annotations

    def _extract_from_drawings(self, page) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract geological features from PDF drawings (vector graphics).
        
        Only extracts:
        - Colored filled polygons (geological units)
        - Does NOT extract lines from drawings (only annotation-based faults with 'Fault' subject)
        
        Returns:
            (polygons, faults) tuple
        """
        polygons = []
        faults = []  # Currently empty - faults only from annotations with proper subject
        
        try:
            drawings = page.get_drawings()
            logger.debug(f"Found {len(drawings)} drawings on page")
            
            skipped_grayscale = 0
            skipped_unfilled = 0
            skipped_small = 0
            
            for drawing in drawings:
                if "items" not in drawing:
                    continue
                
                # Get fill color - we only want FILLED colored shapes
                fill_color = drawing.get("fill")
                stroke_color = drawing.get("color") or drawing.get("stroke")
                
                # Prefer fill color for polygons
                color = fill_color or stroke_color
                
                if not color:
                    continue
                
                # Convert to tuple if needed
                if hasattr(color, '__iter__') and len(color) >= 3:
                    color = tuple(color[:3])
                else:
                    continue
                
                # Check if colored
                is_colored, reason = self._is_colored_polygon(color)
                if not is_colored:
                    skipped_grayscale += 1
                    continue
                
                # Check if filled (filled shapes are more likely geological units)
                has_fill = fill_color is not None
                is_closed = drawing.get("closePath", False)
                
                # Build vertices from drawing items
                vertices = []
                items = drawing["items"]
                
                for item in items:
                    item_type = item[0]
                    
                    if item_type == "l":  # Line segment
                        start_pt = item[1]
                        end_pt = item[2]
                        
                        if not vertices:
                            vertices.extend([start_pt.x, start_pt.y])
                        vertices.extend([end_pt.x, end_pt.y])
                        
                    elif item_type == "c":  # Cubic bezier
                        if len(item) >= 5:
                            p0 = item[1]
                            p3 = item[4]
                            
                            if not vertices:
                                vertices.extend([p0.x, p0.y])
                            vertices.extend([p3.x, p3.y])
                                
                    elif item_type == "re":  # Rectangle
                        rect = item[1]
                        vertices.extend([rect.x0, rect.y0])
                        vertices.extend([rect.x1, rect.y0])
                        vertices.extend([rect.x1, rect.y1])
                        vertices.extend([rect.x0, rect.y1])
                        vertices.extend([rect.x0, rect.y0])
                
                # Must have at least 3 points for a polygon
                num_points = len(vertices) // 2
                if num_points < 3:
                    skipped_small += 1
                    continue
                
                # Calculate bounding box to filter out small items (legends, drillhole lith boxes)
                xs = [vertices[i] for i in range(0, len(vertices), 2)]
                ys = [vertices[i] for i in range(1, len(vertices), 2)]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                area = width * height
                
                # Filter out small items - minimum 500 sq units (adjustable)
                # This excludes legend boxes and small drillhole lithology rectangles
                MIN_AREA = 500
                MIN_DIMENSION = 10
                if area < MIN_AREA or width < MIN_DIMENSION or height < MIN_DIMENSION:
                    skipped_small += 1
                    logger.debug(f"Skipped small drawing: {width:.1f}x{height:.1f} = {area:.1f} sq units")
                    continue
                
                # Only accept filled or closed shapes with enough points
                if not (has_fill or is_closed):
                    if len(items) < 5:  # Small unfilled shapes likely aren't geological units
                        skipped_unfilled += 1
                        continue
                
                # Create polygon data
                formation = self._identify_formation(color)
                
                polygon_data = {
                    "type": "Polygon",
                    "vertices": vertices,
                    "color": color,
                    "formation": formation,
                    "unit_number": None,
                    "classification": "UNIT",
                    "source": "drawing",
                }
                polygons.append(polygon_data)
            
            if skipped_grayscale or skipped_unfilled or skipped_small:
                logger.debug(f"Drawing extraction skipped: {skipped_grayscale} grayscale, "
                           f"{skipped_unfilled} unfilled, {skipped_small} too small")
                
        except Exception as e:
            logger.warning(f"Error extracting from drawings: {e}")
        
        return polygons, faults

    def _get_formation_from_annot(self, annot, color) -> str:
        """Get formation name from annotation or infer from color."""
        if hasattr(annot, 'info'):
            info = annot.info
            subject = info.get('subject', '') or info.get('content', '')
            if subject and subject.strip():
                # Check for saved assignment (from write_assignments_to_pdf)
                if subject.startswith('Assigned:'):
                    return subject[9:].strip().upper()  # Return saved assignment
                # Don't use 'Fault' as formation name
                if 'fault' not in subject.lower():
                    return subject.strip().upper()

        # Infer from color
        return self._identify_formation(color)

    def _get_saved_assignment(self, annot) -> Optional[str]:
        """
        Check if annotation has a saved assignment from previous session.
        Returns the assignment name if found, None otherwise.
        """
        if not hasattr(annot, 'info'):
            return None
        info = annot.info
        subject = info.get('subject', '') or ''
        if subject.startswith('Assigned:'):
            return subject[9:].strip()
        return None

    def _extract_vertices(self, annot) -> Optional[List[float]]:
        """Extract vertices from PDF annotation."""
        vertices = []
        annot_type = annot.type[0] if hasattr(annot, "type") else -1

        # Try to get vertices attribute first (for PolyLine, Polygon annotations)
        if hasattr(annot, "vertices") and annot.vertices:
            for v in annot.vertices:
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    vertices.extend([float(v[0]), float(v[1])])
                elif hasattr(v, 'x') and hasattr(v, 'y'):
                    # PyMuPDF Point object
                    vertices.extend([float(v.x), float(v.y)])
                else:
                    vertices.append(float(v))
        
        # For Line annotations (type 3), also try the line attribute
        if not vertices and annot_type == 3:
            if hasattr(annot, "line") and annot.line:
                # line attribute returns two Point objects
                line = annot.line
                if len(line) >= 2:
                    vertices = [float(line[0].x), float(line[0].y),
                               float(line[1].x), float(line[1].y)]

        # For Ink annotations (type 15), use ink_list property
        # Ink annotations can have multiple strokes - we concatenate them
        if not vertices and annot_type == 15:
            try:
                # Try to get ink_list (list of point lists for each stroke)
                ink_list = None
                if hasattr(annot, "ink_list"):
                    ink_list = annot.ink_list
                elif hasattr(annot, "get_ink_list"):
                    ink_list = annot.get_ink_list()

                if ink_list:
                    for stroke in ink_list:
                        for point in stroke:
                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                vertices.extend([float(point.x), float(point.y)])
                            elif isinstance(point, (tuple, list)) and len(point) >= 2:
                                vertices.extend([float(point[0]), float(point[1])])
                    logger.debug(f"Extracted {len(vertices)//2} points from Ink annotation")
            except Exception as e:
                logger.warning(f"Error extracting Ink annotation vertices: {e}")

        # Fall back to rect if no vertices found
        if not vertices and hasattr(annot, "rect"):
            rect = annot.rect
            if annot_type == 3:  # Line
                vertices = [rect.x0, rect.y0, rect.x1, rect.y1]
            elif annot_type in [4, 5, 6, 10]:  # Square, Circle, Highlight, Polygon
                vertices = [
                    rect.x0, rect.y0,
                    rect.x1, rect.y0,
                    rect.x1, rect.y1,
                    rect.x0, rect.y1,
                ]

        return vertices if vertices else None

    def _get_color(self, annot) -> Optional[tuple]:
        """Get color from annotation."""
        color = None

        if hasattr(annot, "colors") and annot.colors:
            color = annot.colors.get("stroke") or annot.colors.get("fill")

        if not color and hasattr(annot, "stroke_color"):
            color = annot.stroke_color

        if color and len(color) >= 3:
            return tuple(color[:3])

        return None

    def _identify_formation(self, color) -> str:
        """Identify geological formation from color."""
        if not color or len(color) < 3:
            return "UNIT"

        r, g, b = color[:3]

        # Blue dominant -> BIF
        if b > r and b > g and b > 0.3:
            return "BIF"

        # Green dominant -> SCH or AMP
        if g > r and g > b:
            brightness = (r + g + b) / 3
            if brightness < 0.3:
                return "AMP"
            else:
                return "SCH"

        # Brown/tan (red-green mix, low blue)
        if r > 0.3 and g > 0.2 and b < 0.3 and abs(r - g) < 0.3:
            return "PHY"

        # Red dominant
        if r > g and r > b and r > 0.4:
            return "LAT"  # Laterite or similar

        # Yellow/orange
        if r > 0.6 and g > 0.4 and b < 0.3:
            return "OXI"  # Oxidized

        return "UNIT"

    def number_geological_units(self, annotations: List[Dict]):
        """Assign numbers to geological units of the same type."""
        formation_groups = defaultdict(list)
        for annot in annotations:
            # Only process polygons, not faults
            if annot.get('type') == 'Polygon':
                formation_groups[annot["formation"]].append(annot)

        self.geological_units = {}

        for formation, units in formation_groups.items():
            # Sort by average X coordinate (left to right)
            def get_avg_x(unit):
                xs = [unit["vertices"][i] for i in range(0, len(unit["vertices"]), 2)]
                return sum(xs) / len(xs) if xs else 0

            units.sort(key=get_avg_x)

            for i, unit in enumerate(units, 1):
                unit_name = f"{formation}{i}"
                unit["unit_number"] = unit_name
                unit["name"] = unit_name
                self.geological_units[unit_name] = unit

        logger.info(f"Numbered {len(self.geological_units)} geological units")

    def simplify_line(
        self, coords: List[Tuple[float, float]], tolerance: float = 2.0
    ) -> List[Tuple[float, float]]:
        """Simplify a line using Douglas-Peucker algorithm."""
        try:
            if len(coords) <= 2:
                return coords

            line = LineString(coords)
            simplified = line.simplify(tolerance, preserve_topology=True)

            return list(simplified.coords)

        except Exception as e:
            logger.warning(f"Could not simplify line: {e}")
            return coords

    def get_extraction_summary(self) -> Dict:
        """Get a summary of extracted features."""
        return {
            "geological_units": len(self.geological_units),
            "faults": len(self.faults),
            "contacts": len(self.contacts),
            "formations": list(set(u.get('formation', 'Unknown') for u in self.geological_units.values())),
        }
