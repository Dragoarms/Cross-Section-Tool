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
from typing import Dict, List, Optional, Tuple, Set
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

# Default names that are typically noise/sketches - can be excluded by user
DEFAULT_NOISE_NAMES = {
    'pencil', 'polygon', 'line', 'arrow', 'rectangle', 'square',
    'circle', 'oval', 'ellipse', 'freehand', 'ink', 'shape',
    'highlight', 'underline', 'strikeout', 'squiggly',
    'text', 'note', 'comment', 'stamp', 'caret',
    'drawing', 'sketch', 'markup', 'annotation'
}


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
        self.included_groups: Optional[Set[str]] = None  # If set, only include these groups
        self.excluded_groups: Set[str] = set()  # Groups to exclude

    def set_group_filter(self, included: Optional[Set[str]] = None, excluded: Optional[Set[str]] = None):
        """
        Set which annotation groups to include/exclude.

        Args:
            included: If set, ONLY include these groups (whitelist mode)
            excluded: Groups to exclude (blacklist mode, used if included is None)
        """
        self.included_groups = included
        self.excluded_groups = excluded or set()

    def clear_group_filter(self):
        """Clear all group filters."""
        self.included_groups = None
        self.excluded_groups = set()

    @staticmethod
    def scan_annotation_groups(page) -> Dict[str, Dict]:
        """
        Scan a PDF page and group annotations by their subject/author.

        Returns:
            Dictionary {group_name: {
                'count': int,
                'types': set of annotation type names,
                'is_default_name': bool (True if matches common noise names),
                'has_fault_keyword': bool,
                'sample_colors': list of colors
            }}
        """
        groups = defaultdict(lambda: {
            'count': 0,
            'types': set(),
            'is_default_name': False,
            'has_fault_keyword': False,
            'sample_colors': []
        })

        type_names = {
            0: 'Text', 1: 'Link', 2: 'FreeText', 3: 'Line', 4: 'Square',
            5: 'Circle', 6: 'Polygon', 7: 'PolyLine', 8: 'Highlight',
            9: 'Underline', 10: 'Squiggly', 11: 'StrikeOut', 12: 'Stamp',
            13: 'Caret', 14: 'Ink', 15: 'Popup', 16: 'FileAttachment',
            17: 'Sound', 18: 'Movie', 19: 'Widget', 20: 'Screen',
            21: 'PrinterMark', 22: 'TrapNet', 23: 'Watermark', 24: '3D'
        }

        try:
            for annot in page.annots():
                info = annot.info if hasattr(annot, 'info') else {}

                # Get group name from author/title
                author = info.get('title') or info.get('author') or ''
                if not author or not author.strip():
                    author = '(No Author)'
                else:
                    author = author.strip()

                # Get annotation type
                annot_type = annot.type[0] if hasattr(annot, 'type') else -1
                type_name = type_names.get(annot_type, f'Type{annot_type}')

                # Get subject for fault detection
                subject = info.get('subject', '') or info.get('content', '') or ''

                # Get color
                color = None
                if hasattr(annot, 'colors') and annot.colors:
                    color = annot.colors.get('stroke') or annot.colors.get('fill')

                # Update group info
                groups[author]['count'] += 1
                groups[author]['types'].add(type_name)
                groups[author]['is_default_name'] = author.lower() in DEFAULT_NOISE_NAMES

                if 'fault' in subject.lower() or 'fault' in author.lower():
                    groups[author]['has_fault_keyword'] = True

                if color and len(groups[author]['sample_colors']) < 3:
                    groups[author]['sample_colors'].append(tuple(color[:3]) if len(color) >= 3 else color)

        except Exception as e:
            logger.warning(f"Error scanning annotation groups: {e}")

        return dict(groups)

    @staticmethod
    def scan_all_pages(doc) -> Dict[str, Dict]:
        """
        Scan all pages in a PDF document and aggregate annotation groups.

        Args:
            doc: PyMuPDF document object

        Returns:
            Aggregated groups dictionary
        """
        all_groups = defaultdict(lambda: {
            'count': 0,
            'types': set(),
            'is_default_name': False,
            'has_fault_keyword': False,
            'sample_colors': [],
            'pages': set()
        })

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_groups = FeatureExtractor.scan_annotation_groups(page)

            for group_name, group_info in page_groups.items():
                all_groups[group_name]['count'] += group_info['count']
                all_groups[group_name]['types'].update(group_info['types'])
                all_groups[group_name]['is_default_name'] = group_info['is_default_name']
                all_groups[group_name]['has_fault_keyword'] = group_info['has_fault_keyword']
                all_groups[group_name]['sample_colors'].extend(group_info['sample_colors'])
                all_groups[group_name]['pages'].add(page_num)

        return dict(all_groups)

    def _should_include_annotation(self, author: str) -> bool:
        """Check if an annotation should be included based on group filters."""
        if not author:
            return False

        author_clean = author.strip()

        # Whitelist mode: only include if in included_groups
        if self.included_groups is not None:
            return author_clean in self.included_groups

        # Blacklist mode: exclude if in excluded_groups
        if author_clean in self.excluded_groups:
            return False

        return True

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

        # Reject grayscale (R ≈ G ≈ B)
        tolerance = 0.15
        if abs(r - g) < tolerance and abs(g - b) < tolerance and abs(r - b) < tolerance:
            return False, f"Grayscale (R={r:.2f}, G={g:.2f}, B={b:.2f})"

        return True, "Colored"

    def _is_fault_line(self, annot) -> Tuple[bool, str]:
        """
        Check if an annotation is a fault line (subject contains 'Fault').

        Returns:
            (is_fault, fault_name)
        """
        if not hasattr(annot, 'info'):
            return False, ""

        info = annot.info
        subject = info.get('subject', '') or info.get('content', '') or ''

        # Check if subject contains 'Fault' (case insensitive)
        subject_lower = subject.lower().strip()
        if 'fault' in subject_lower:
            # Extract fault name
            fault_name = subject.strip() if subject.strip() else "Fault"
            return True, fault_name

        return False, ""

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

    def _get_subject_from_annot(self, annot) -> Optional[str]:
        """Get subject line from annotation metadata."""
        if not hasattr(annot, 'info'):
            return None
        info = annot.info
        return info.get('subject', '') or info.get('content', '')

    def _get_visual_properties(self, annot) -> Dict:
        """Extract visual properties from annotation for similarity matching.

        Returns dict with:
        - stroke_color: RGB tuple for stroke/line color
        - fill_color: RGB tuple for fill color
        - line_width: Line/border width
        - dashes: Dash pattern if any
        """
        props = {
            'stroke_color': None,
            'fill_color': None,
            'line_width': None,
            'dashes': None,
        }

        # Get colors
        if hasattr(annot, 'colors') and annot.colors:
            stroke = annot.colors.get('stroke')
            fill = annot.colors.get('fill')
            if stroke and len(stroke) >= 3:
                props['stroke_color'] = tuple(stroke[:3])
            if fill and len(fill) >= 3:
                props['fill_color'] = tuple(fill[:3])

        # Get border/line width
        if hasattr(annot, 'border'):
            border = annot.border
            if border and len(border) >= 1:
                props['line_width'] = border[0] if isinstance(border, (list, tuple)) else border

        # Get dash pattern from border
        if hasattr(annot, 'border') and annot.border:
            border = annot.border
            if isinstance(border, (list, tuple)) and len(border) >= 4:
                # Border format: [width, style, dash_array...]
                props['dashes'] = border[3:] if len(border) > 3 else None

        return props

    def _is_default_subject(self, subject: str) -> bool:
        """Check if subject is a default/generic name that shouldn't be used for grouping."""
        if not subject:
            return True
        subject_lower = subject.lower().strip()
        default_names = {'line', 'polyline', 'polygon', 'circle', 'square', 'rectangle',
                         'shape', 'drawing', 'annotation', 'ink', 'freehand', ''}
        return subject_lower in default_names

    def extract_annotations(self, page) -> List[Dict]:
        """
        Extract annotations from PDF page.

        Now extracts ALL items (not just those with author tags):
        - Items with author tags are marked as 'assigned'
        - Items without author tags are marked as 'unassigned' for manual assignment
        - Uses subject line for naming (not author) to avoid f123456 being detected as faults
        - Stores visual properties for similarity-based selection

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
            lines_found = 0
            unassigned_found = 0
            skipped_filtered = 0
            skipped_too_few_points = 0

            for annot in page.annots():
                annot_type = annot.type[0]

                # Get metadata - author and subject are SEPARATE
                has_author = self._has_author_tag(annot)
                author = self._get_author_from_annot(annot) or ""
                subject = self._get_subject_from_annot(annot) or ""

                # Check group filter (only if author exists)
                if has_author and not self._should_include_annotation(author):
                    skipped_filtered += 1
                    logger.debug(f"Skipped annotation - filtered out group: {author}")
                    continue

                # Get visual properties for similarity matching
                visual_props = self._get_visual_properties(annot)

                # Check if this is a line/polyline/ink (types 3, 7, 14)
                # Type 3: Line, Type 7: PolyLine, Type 14: Ink (freehand drawing)
                is_line_type = annot_type in [3, 7, 14]  # Line, PolyLine, Ink

                # Check if this is a polygon type (types 4, 5, 6, 10)
                is_polygon_type = annot_type in [4, 5, 6, 10]  # Square, Circle, Highlight, Polygon

                # Check if subject indicates this is a fault (use SUBJECT, not author!)
                is_fault, fault_name = self._is_fault_line(annot)

                # Determine if this is an "assigned" item (has meaningful metadata)
                # Items without author OR with default subject are considered unassigned
                is_assigned = has_author and not self._is_default_subject(subject)

                if is_line_type:
                    # Extract line/polyline
                    vertices = self._extract_vertices(annot)
                    if vertices and len(vertices) >= 4:
                        color = self._get_color(annot) or (1.0, 0.0, 0.0)  # Default red for lines

                        if is_fault:
                            # This is a fault line - use SUBJECT for fault_name, not author
                            fault_data = {
                                "type": "Fault",
                                "name": fault_name,  # From subject, not author
                                "vertices": vertices,
                                "color": color,
                                "formation": "FAULT",
                                "classification": "FAULT",
                                "source": "annotation",
                                "author": author,
                                "subject": subject,
                                "is_assigned": True,  # Faults are always assigned
                                "visual_properties": visual_props,
                            }
                            annotations.append(fault_data)
                            extracted_faults.append(fault_data)
                            faults_found += 1
                            logger.debug(f"Extracted fault: {fault_name} (author: {author}, subject: {subject})")
                        else:
                            # Regular polyline - use subject if available, otherwise generic name
                            display_name = subject if subject and not self._is_default_subject(subject) else "PolyLine"
                            polyline_data = {
                                "type": "PolyLine",
                                "name": display_name,
                                "vertices": vertices,
                                "color": color,
                                "formation": display_name if display_name != "PolyLine" else "LINE",
                                "classification": "LINE",
                                "source": "annotation",
                                "author": author,
                                "subject": subject,
                                "is_assigned": is_assigned,
                                "visual_properties": visual_props,
                            }
                            annotations.append(polyline_data)
                            lines_found += 1
                            if not is_assigned:
                                unassigned_found += 1
                            logger.debug(f"Extracted polyline: {display_name} (assigned: {is_assigned})")
                    else:
                        skipped_too_few_points += 1

                elif is_polygon_type:
                    # Extract all polygons - mark as assigned/unassigned based on metadata
                    vertices = self._extract_vertices(annot)

                    if vertices and len(vertices) >= 6:
                        color = self._get_color(annot) or (0.5, 0.5, 0.5)  # Default gray

                        if is_fault:
                            # Fault polygon - use SUBJECT for fault_name
                            fault_data = {
                                "type": "Fault",
                                "name": fault_name,  # From subject
                                "vertices": vertices,
                                "color": color,
                                "formation": "FAULT",
                                "classification": "FAULT",
                                "source": "annotation",
                                "author": author,
                                "subject": subject,
                                "is_assigned": True,  # Faults are always assigned
                                "visual_properties": visual_props,
                            }
                            annotations.append(fault_data)
                            extracted_faults.append(fault_data)
                            faults_found += 1
                            logger.debug(f"Extracted fault polygon: {fault_name} (author: {author}, subject: {subject})")
                        else:
                            # Geological unit polygon - use subject as formation name
                            formation, from_subject = self._get_formation_with_source(annot, color)
                            # If formation couldn't be determined from subject, keep as UNIT
                            # Do NOT fall back to author (which might be f123456)

                            # Check for saved assignment from previous session
                            saved_assignment = self._get_saved_assignment(annot)

                            polygon_data = {
                                "type": "Polygon",
                                "vertices": vertices,
                                "color": color,
                                "formation": formation,
                                "formation_from_subject": from_subject,  # True if explicitly set in PDF
                                "unit_number": None,
                                "classification": "UNIT",
                                "source": "annotation",
                                "author": author,
                                "subject": subject,
                                "unit_assignment": saved_assignment,  # Pre-populate if saved
                                "is_assigned": is_assigned or saved_assignment is not None,
                                "visual_properties": visual_props,
                            }
                            annotations.append(polygon_data)
                            polygons_found += 1
                            if not is_assigned and not saved_assignment:
                                unassigned_found += 1
                            logger.debug(f"Extracted polygon: {formation} (author: {author}, subject: {subject}, assigned: {is_assigned})")
                    else:
                        skipped_too_few_points += 1

            # Log annotation summary
            logger.info(f"Annotation extraction summary:")
            logger.info(f"  - Polygons found: {polygons_found}")
            logger.info(f"  - Faults found: {faults_found}")
            logger.info(f"  - Lines found: {lines_found}")
            logger.info(f"  - Unassigned items: {unassigned_found}")
            logger.info(f"  - Skipped (filtered out): {skipped_filtered}")
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
        formation, _ = self._get_formation_with_source(annot, color)
        return formation

    def _get_formation_with_source(self, annot, color) -> Tuple[str, bool]:
        """
        Get formation name from annotation or infer from color.

        Returns:
            Tuple of (formation_name, is_from_subject)
            is_from_subject is True if the formation came from the PDF subject field
        """
        # Default/generic names that should be ignored and replaced with color-based identification
        DEFAULT_SUBJECT_NAMES = {
            'polygon', 'polyline', 'line', 'shape', 'annotation',
            'square', 'rectangle', 'circle', 'oval', 'ellipse',
            'pencil', 'freehand', 'ink', 'highlight', 'strikeout',
            'underline', 'squiggly', 'stamp', 'caret', 'fileattachment',
            'sound', 'movie', 'widget', 'screen', 'printermark',
            'trapnet', 'watermark', '3d', 'redact', 'projection',
            'unknown', 'none', '', 'default'
        }

        if hasattr(annot, 'info'):
            info = annot.info
            subject = info.get('subject', '') or info.get('content', '')
            if subject and subject.strip():
                subject_lower = subject.strip().lower()

                # Check for saved assignment (from write_assignments_to_pdf)
                if subject.startswith('Assigned:'):
                    return subject[9:].strip().upper(), True  # Return saved assignment

                # Don't use 'Fault' as formation name
                if 'fault' in subject_lower:
                    return self._identify_formation(color), False

                # Skip default/generic annotation type names
                if subject_lower in DEFAULT_SUBJECT_NAMES:
                    # Use color-based identification instead
                    return self._identify_formation(color), False

                # Valid subject - use it
                return subject.strip().upper(), True

        # Infer from color (not from subject)
        return self._identify_formation(color), False

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
        """Get color from annotation, preferring fill color for polygons."""
        color = None

        if hasattr(annot, "colors") and annot.colors:
            # Prefer fill color over stroke - fill distinguishes geological units
            color = annot.colors.get("fill") or annot.colors.get("stroke")

        # Also check for fill_color attribute
        if not color and hasattr(annot, "fill_color") and annot.fill_color:
            color = annot.fill_color

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
