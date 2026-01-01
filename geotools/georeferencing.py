# geotools\georeferencing.py
"""
Coordinate detection and georeferencing module for geological cross-sections.
Handles coordinate extraction, transformation, and GeoTIFF export.
"""

import re
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import fitz
import rasterio
from rasterio.transform import from_origin
from rasterio.control import GroundControlPoint

from .utils.debug_utils import DebugLogger, debug_trace, debug_value, log_state

# Set up module logger
debug_logger = DebugLogger()
logger = debug_logger.setup_module_logger("georeferencing", level="INFO")


class GeoReferencer:
    """Handles coordinate detection and transformation for PDF cross-sections."""

    def __init__(self):
        self.patterns = {
            "easting": re.compile(r"^(29\d{4}|30\d{4})$"),
            "rl": re.compile(r"^(\d{3,4})$"),
            "northing": re.compile(r"(\d{3}[,]?\d{3})\s*(West|East|North|South|N|S|E|W)", re.I),
        }
        self.coord_system = None
        self.potential_northings = []

    @debug_trace(logger, log_args=True, log_result=False)
    def detect_coordinates(
        self, page: fitz.Page, filename: Optional[Path] = None
    ) -> Optional[Dict]:
        """
        Detect coordinate system from PDF page.
        Now includes scale detection for full-page georeferencing.
        """
        debug_value(logger, "page.rect", page.rect if page else None)
        debug_value(logger, "filename", str(filename) if filename else None)

        coord_system = {
            "northing": None,
            "northing_text": None,
            "easting_labels": [],
            "rl_labels": [],
            "page_rect": page.rect,
            "scale_x": None,  # pixels per meter
            "scale_y": None,  # pixels per meter
        }

        # Patterns for finding northing from Location box or section title
        northing_patterns = [
            # Location box patterns - W: or E: followed by easting, then northing
            re.compile(r"[WE]:\s*\d+\s*,\s*(\d{6})", re.I),  # Matches "W: 294952, 114500"
            re.compile(r"[WE]:\s*\d+\s*,\s*(\d{3},\d{3})", re.I),  # Matches with comma separator
            # Title patterns - ANY two uppercase letters followed by patterns
            re.compile(r"[A-Z]{2}\s+(\d{3},\d{3})\s+Line\s+Cross\s+Section", re.I),
            re.compile(r"[A-Z]{2}\s+Line\s+(\d{3},\d{3})\s+Cross\s+Section", re.I),
            re.compile(r"[A-Z]{2}\s+(\d{3},\d{3})\s+Cross\s+Section", re.I),
            re.compile(r"^\s*(\d{3},\d{3})\s+Line\s+Cross\s+Section", re.I),
            # Alternative patterns
            re.compile(r"Line\s+(\d{3},\d{3})(?:\s|$)", re.I),
            re.compile(r"Section\s+(\d{3},\d{3})(?:\s|$)", re.I),
        ]

        # Also look for scale information
        scale_pattern = re.compile(r"Scale[:\s]*1[:\s]*(\d+[,]?\d*)", re.I)
        scale_value = None

        self.potential_northings = []

        # Extract all text blocks
        text_blocks = page.get_text("dict")

        # First pass: look for Location box - collect all spans that might be part of it
        location_texts = []
        all_text_spans = []
        location_bbox = None

        # First, collect ALL text spans with their positions
        for block in text_blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        bbox = span.get("bbox", [])
                        if text and bbox:
                            all_text_spans.append(
                                {
                                    "text": text,
                                    "bbox": bbox,
                                    "y": ((bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0),
                                }
                            )
                            # Track where "Location" label is
                            if "Location" in text:
                                location_bbox = bbox

        # Now find W: and E: lines near the Location label (or standalone)
        if location_bbox:
            location_y = (location_bbox[1] + location_bbox[3]) / 2
            # Collect W: and E: lines within ~100 pixels of Location
            for span in all_text_spans:
                if abs(span["y"] - location_y) < 100:
                    if re.match(r"[WE]:\s*\d+", span["text"], re.I):
                        location_texts.append(span["text"])
        else:
            # No "Location" label found, look for standalone W: and E: patterns
            for span in all_text_spans:
                if re.match(r"[WE]:\s*\d+\s*,\s*\d+", span["text"], re.I):
                    location_texts.append(span["text"])

        # Process each text block (continue with existing logic)
        for block in text_blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                # Build block text for additional processing
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_text += text + " "

                # Also add full block text if it contains Location info (for backward compatibility)
                if "Location" in block_text and block_text not in location_texts:
                    location_texts.append(block_text)
                    # Try to extract northing from Location block
                    for pattern in northing_patterns[:2]:  # Use first two patterns for Location box
                        match = pattern.search(block_text)
                        if match:
                            try:
                                num_str = match.group(1).replace(",", "")
                                value = float(num_str)
                                if 100000 <= value <= 999999:
                                    coord_system["northing"] = value
                                    coord_system["northing_text"] = (
                                        f"From Location: {block_text[:50]}"
                                    )
                                    logger.info(f"Found northing from Location box: {value}")
                                    break
                            except:
                                pass

                # Also process individual spans for other coordinates
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        bbox = span.get("bbox", [])
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        is_bold = bool(font_flags & 2**4)

                        if not text or len(bbox) < 4:
                            continue

                        # Check for coordinate patterns - need to distinguish by position
                        # Numbers at bottom of page (high y value) are likely eastings
                        # Numbers on left side (low x value) are likely RLs

                        # Skip if text contains 'E' suffix (it's an easting)
                        if "E" in text.upper():
                            # It's an easting label, process as easting only
                            clean_text = re.sub(r"[EeNnSsWw]", "", text).strip()
                            if self.patterns["easting"].match(clean_text):
                                val = float(clean_text)
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2
                                coord_system["easting_labels"].append(
                                    {
                                        "value": val,
                                        "x": center_x,
                                        "y": center_y,
                                        "bbox": bbox,
                                    }
                                )
                        elif self.patterns["easting"].match(text):
                            val = float(text)
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2

                            # If near bottom of page (y > 70% of page height), it's an easting
                            if center_y > page.rect.height * 0.7:
                                coord_system["easting_labels"].append(
                                    {
                                        "value": val,
                                        "x": center_x,
                                        "y": center_y,
                                        "bbox": bbox,
                                    }
                                )
                            # Check if it could be a northing (6 digits, with decimal)
                            elif 100000 <= val <= 999999 and "." not in text:
                                # Integer 6-digit numbers are less likely to be northings
                                # Real northings usually have decimals
                                pass

                        elif self.patterns["rl"].match(text):
                            rl_val = float(text)
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2

                            # Check if it's on left side (x < 20% of page width)
                            if center_x < page.rect.width * 0.2 and 0 <= rl_val <= 9999:
                                coord_system["rl_labels"].append(
                                    {
                                        "value": rl_val,
                                        "x": center_x,
                                        "y": center_y,
                                        "bbox": bbox,
                                    }
                                )

                        # Look for northing in section titles or explicit labels
                        # Skip if it's just a standalone number with N (like "112000N" on a map)
                        if not coord_system["northing"]:
                            # Skip simple "numberN" patterns that are likely map labels
                            if re.match(r"^\d+N$", text):
                                continue

                            for pattern in northing_patterns[
                                2:
                            ]:  # Skip Location patterns (already processed)
                                match = pattern.search(text)
                                if match:
                                    try:
                                        num_str = match.group(1).replace(",", "")
                                        value = float(num_str)
                                        if 100000 <= value <= 999999:
                                            self.potential_northings.append((value, text))
                                            if not coord_system["northing"]:
                                                coord_system["northing"] = value
                                                coord_system["northing_text"] = text
                                                logger.info(
                                                    f"Found northing: {value} from text: '{text}'"
                                                )
                                                break
                                    except:
                                        pass

        # Try filename if no northing found
        if not coord_system["northing"] and filename:
            filename_str = filename.stem if isinstance(filename, Path) else str(filename)
            for pattern in northing_patterns:
                match = pattern.search(filename_str)
                if match:
                    try:
                        num_str = match.group(1).replace(",", "")
                        value = float(num_str)
                        if 100000 <= value <= 999999:
                            coord_system["northing"] = value
                            coord_system["northing_text"] = f"From filename: {filename_str}"
                            logger.info(f"Found northing from filename: {value}")
                            break
                    except:
                        pass

        # No default northing - must be found or provided
        if not coord_system["northing"]:
            logger.warning(
                f"No northing found for page {page.number if hasattr(page, 'number') else 'unknown'}"
            )
            return None  # Return None if no northing found

        # Validate and calculate ranges
        debug_value(logger, "easting_labels_count", len(coord_system["easting_labels"]))
        debug_value(logger, "rl_labels_count", len(coord_system["rl_labels"]))

        if len(coord_system["easting_labels"]) >= 2 and len(coord_system["rl_labels"]) >= 2:
            # Sort labels
            coord_system["easting_labels"].sort(key=lambda x: x["x"])
            coord_system["rl_labels"].sort(key=lambda x: x["y"], reverse=True)

            # Calculate ranges
            coord_system["easting_min"] = min(e["value"] for e in coord_system["easting_labels"])
            coord_system["easting_max"] = max(e["value"] for e in coord_system["easting_labels"])
            coord_system["rl_min"] = min(r["value"] for r in coord_system["rl_labels"])
            coord_system["rl_max"] = max(r["value"] for r in coord_system["rl_labels"])

            # Store PDF coordinate ranges
            coord_system["pdf_x_min"] = min(e["x"] for e in coord_system["easting_labels"])
            coord_system["pdf_x_max"] = max(e["x"] for e in coord_system["easting_labels"])
            coord_system["pdf_y_min"] = min(r["y"] for r in coord_system["rl_labels"])
            coord_system["pdf_y_max"] = max(r["y"] for r in coord_system["rl_labels"])

            # Fit linear mappings using all labels (PEP-8)
            # Easting: value = ax * page_x + bx
            xs = np.array([e["x"] for e in coord_system["easting_labels"]], dtype=float)
            es = np.array([e["value"] for e in coord_system["easting_labels"]], dtype=float)
            ax, bx = np.polyfit(xs, es, 1)

            # RL: value = cy * page_y + dy  (page_y increases downward)
            ys = np.array([r["y"] for r in coord_system["rl_labels"]], dtype=float)
            rs = np.array([r["value"] for r in coord_system["rl_labels"]], dtype=float)
            cy, dy = np.polyfit(ys, rs, 1)

            coord_system["ax"] = float(ax)
            coord_system["bx"] = float(bx)
            coord_system["cy"] = float(cy)
            coord_system["dy"] = float(dy)

            # Optional: derived "scale" near label range for info only
            coord_system["scale_x"] = 1.0 / abs(ax) if ax != 0 else None
            coord_system["scale_y"] = 1.0 / abs(cy) if cy != 0 else None

            logger.info(
                f"Coordinate ranges: E[{coord_system['easting_min']}-{coord_system['easting_max']}], "
                f"RL[{coord_system['rl_min']}-{coord_system['rl_max']}]"
            )

            return coord_system
        else:
            return None

    def build_transformation(self, extrapolate=True):
        """
        Build transformation function using linear fits over all labels.
        For vertical cross-sections:
        - Page X axis -> Easting (real-world X)
        - Page Y axis -> RL/Elevation (real-world Z)
        - Northing is constant (real-world Y)
        """
        if not self.coord_system:
            return None

        ax = self.coord_system.get("ax")
        bx = self.coord_system.get("bx")
        cy = self.coord_system.get("cy")
        dy = self.coord_system.get("dy")

        if ax is not None and bx is not None and cy is not None and dy is not None:
            # Validate the transformation parameters
            # ax should be positive (easting increases left to right)
            # cy should be negative (RL decreases top to bottom in page coords)

            if abs(ax) < 0.01:  # Scale is way off
                logger.warning(f"Easting scale seems incorrect: ax={ax}")
            if cy > 0:  # Y axis is inverted in PDF
                logger.warning(f"RL scale may be inverted: cy={cy}")

            def transform(x, y):
                # x,y are page coordinates
                easting = ax * float(x) + bx
                rl = cy * float(y) + dy
                northing = self.coord_system["northing"]
                return easting, northing, rl  # X, Y, Z

            return transform

        # Fallback to bounds-based interpolation if linear fits are missing
        def transform(x, y):
            easting = np.interp(
                x,
                [self.coord_system["pdf_x_min"], self.coord_system["pdf_x_max"]],
                [self.coord_system["easting_min"], self.coord_system["easting_max"]],
            )
            rl = np.interp(
                y,
                [self.coord_system["pdf_y_min"], self.coord_system["pdf_y_max"]],
                [self.coord_system["rl_max"], self.coord_system["rl_min"]],
            )
            northing = self.coord_system["northing"]
            return easting, northing, rl  # X, Y, Z

        return transform

    def build_inverse_transformation(self):
        """
        Build inverse transformation function (real-world coords -> PDF coords).
        
        For vertical cross-sections:
        - Easting (real-world X) -> Page X
        - RL/Elevation (real-world Z) -> Page Y
        
        Returns:
            Function that takes (easting, rl) and returns (pdf_x, pdf_y)
        """
        if not self.coord_system:
            return None

        ax = self.coord_system.get("ax")
        bx = self.coord_system.get("bx")
        cy = self.coord_system.get("cy")
        dy = self.coord_system.get("dy")

        if ax is not None and bx is not None and cy is not None and dy is not None:
            # Forward transform: easting = ax * x + bx  =>  x = (easting - bx) / ax
            # Forward transform: rl = cy * y + dy       =>  y = (rl - dy) / cy
            
            if abs(ax) < 1e-10 or abs(cy) < 1e-10:
                logger.warning("Cannot build inverse transform: degenerate scale")
                return None

            def inverse_transform(easting: float, rl: float) -> Tuple[float, float]:
                """Convert real-world (easting, rl) to PDF page coordinates (x, y)."""
                pdf_x = (float(easting) - bx) / ax
                pdf_y = (float(rl) - dy) / cy
                return pdf_x, pdf_y

            return inverse_transform

        # Fallback to bounds-based interpolation
        pdf_x_min = self.coord_system.get("pdf_x_min")
        pdf_x_max = self.coord_system.get("pdf_x_max")
        pdf_y_min = self.coord_system.get("pdf_y_min")
        pdf_y_max = self.coord_system.get("pdf_y_max")
        e_min = self.coord_system.get("easting_min")
        e_max = self.coord_system.get("easting_max")
        rl_min = self.coord_system.get("rl_min")
        rl_max = self.coord_system.get("rl_max")

        if all(v is not None for v in [pdf_x_min, pdf_x_max, pdf_y_min, pdf_y_max, 
                                        e_min, e_max, rl_min, rl_max]):
            def inverse_transform(easting: float, rl: float) -> Tuple[float, float]:
                """Convert real-world (easting, rl) to PDF page coordinates (x, y)."""
                pdf_x = np.interp(easting, [e_min, e_max], [pdf_x_min, pdf_x_max])
                # Note: RL increases upward but PDF Y increases downward
                pdf_y = np.interp(rl, [rl_min, rl_max], [pdf_y_max, pdf_y_min])
                return float(pdf_x), float(pdf_y)

            return inverse_transform

        return None

    def export_geotiff(
        self,
        pdf_path: Path,
        output_path: Path,
        render_scale: float = 2.0,
        page_num: int = None,
    ) -> bool:
        """Export PDF page(s) as georeferenced GeoTIFF(s).
        If page_num is None, exports all pages as separate files."""
        try:
            doc = fitz.open(str(pdf_path))

            # Determine which pages to export
            if page_num is not None:
                # Single page export
                pages_to_export = [page_num]
                output_files = [output_path]
            else:
                # Multi-page export - create one file per page
                pages_to_export = range(len(doc))
                output_files = []

                # Pre-detect coordinates for all pages to get northings
                page_northings = {}
                for p in pages_to_export:
                    if p < len(doc):
                        page = doc[p]
                        coord_sys = self.detect_coordinates(page, pdf_path)
                        if coord_sys and coord_sys.get("northing"):
                            page_northings[p] = coord_sys["northing"]

                for p in pages_to_export:
                    if len(doc) > 1:
                        # Use northing for filename if available
                        if p in page_northings:
                            northing_val = page_northings[p]
                            # Try to extract section prefix from detected text
                            coord_sys = self.detect_coordinates(doc[p], pdf_path)
                            section_prefix = ""
                            if coord_sys and coord_sys.get("northing_text"):
                                # Extract prefix like "KM" from text like "KM 122,800 Cross Section"
                                import re

                                match = re.match(r"^([A-Z]+)\s+", coord_sys["northing_text"])
                                if match:
                                    section_prefix = f"{match.group(1)}_"

                            # Format northing as integer (no decimals)
                            output_file = (
                                output_path.parent
                                / f"{section_prefix}{int(northing_val)}{output_path.suffix}"
                            )
                        else:
                            # Fallback to page number if no northing found
                            output_file = (
                                output_path.parent
                                / f"{output_path.stem}_page{p+1}{output_path.suffix}"
                            )
                    else:
                        # Single page - use northing if available
                        if p in page_northings:
                            northing_val = page_northings[p]
                            output_file = (
                                output_path.parent / f"{int(northing_val)}{output_path.suffix}"
                            )
                        else:
                            output_file = output_path
                    output_files.append(output_file)

            success_count = 0
            for page_idx, out_path in zip(pages_to_export, output_files):
                if page_idx >= len(doc):
                    logger.error(f"Page {page_idx} does not exist in PDF")
                    continue

                page = doc[page_idx]

                # Detect coordinates FOR THIS PAGE
                self.coord_system = self.detect_coordinates(page, pdf_path)
                if not self.coord_system:
                    logger.error(
                        f"Could not detect coordinates for {pdf_path} page {page_idx} - skipping"
                    )
                    continue

                # Render page
                mat = fitz.Matrix(render_scale, render_scale)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Calculate geotransform for vertical cross-section (not passed when using GCPs)
                # Ensure these are defined for metadata regardless of branch
                e_min, e_max = (
                    self.coord_system["easting_min"],
                    self.coord_system["easting_max"],
                )
                rl_min, rl_max = (
                    self.coord_system["rl_min"],
                    self.coord_system["rl_max"],
                )

                if self.coord_system.get("scale_x") and self.coord_system.get("scale_y"):
                    # Optional: compute transform info (informational only when GCPs are used)
                    pixel_width = 1.0 / (self.coord_system["scale_x"] * render_scale)
                    pixel_height = 1.0 / (self.coord_system["scale_y"] * render_scale)
                    transform_func = self.build_transformation(extrapolate=True)
                    origin_e, _, origin_rl = transform_func(0, 0)
                    _transform = from_origin(
                        origin_e, origin_rl, pixel_width, pixel_height
                    )  # not applied
                else:
                    pixel_width = (e_max - e_min) / pix.width
                    pixel_height = (rl_max - rl_min) / pix.height
                    _transform = from_origin(
                        e_min, rl_max, pixel_width, pixel_height
                    )  # not applied

                # For vertical sections, define GCPs using the fitted mappings (PEP-8).
                # Convert image pixels (col,row) to page coords by dividing by render_scale.
                transform_func = self.build_transformation(extrapolate=True)
                if not transform_func:
                    logger.error("No transformation function available for GCPs")
                    doc.close()
                    return False

                northing = self.coord_system["northing"]

                # Define corners more carefully
                # Top-left in image = min easting, max RL
                # Top-right in image = max easting, max RL
                # Bottom-left in image = min easting, min RL
                # Bottom-right in image = max easting, min RL

                # Use the actual page rect for more accurate mapping
                page_width = page.rect.width
                page_height = page.rect.height

                # Use 3 corner points for Leapfrog (forms a triangle)
                corners_page = [
                    (0, 0),  # top-left
                    (page_width, 0),  # top-right
                    (0, page_height),  # bottom-left
                    # Skip bottom-right to have exactly 3 points
                ]

                # Convert to image pixels and then to real-world
                gcps = []
                for i, (px, py) in enumerate(corners_page):
                    # Page coord to image pixel
                    img_col = px * render_scale
                    img_row = py * render_scale

                    # Ensure within bounds
                    img_col = min(max(0, img_col), pix.width - 1)
                    img_row = min(max(0, img_row), pix.height - 1)

                    # Transform to real-world
                    e, n, rl = transform_func(px, py)

                    gcps.append(
                        GroundControlPoint(
                            row=int(img_row),
                            col=int(img_col),
                            x=float(e),
                            y=float(n),
                            z=float(rl),
                        )
                    )

                    logger.debug(
                        f"GCP {i}: img({img_col:.0f},{img_row:.0f}) -> world({e:.1f},{n:.1f},{rl:.1f})"
                    )

                # Remove the duplicate/broken GCP definition
                # The correct GCPs are already defined above

                # Write GeoTIFF with GCPs for proper 3D georeferencing
                # Using UTM Zone 33N WGS84 (EPSG:32633) for your project
                with rasterio.open(
                    str(out_path),  # CRITICAL: Use out_path from loop, not output_path!
                    "w",
                    driver="GTiff",
                    height=pix.height,
                    width=pix.width,
                    count=3 if img.shape[2] >= 3 else 1,
                    dtype=img.dtype,
                    crs="EPSG:32633",  # UTM Zone 33N WGS84
                    gcps=gcps,
                ) as dst:
                    if img.shape[2] >= 3:
                        dst.write(img[:, :, 0], 1)
                        dst.write(img[:, :, 1], 2)
                        dst.write(img[:, :, 2], 3)
                    else:
                        dst.write(img[:, :, 0], 1)

                    # Add metadata for vertical cross-section
                    dst.update_tags(
                        northing=str(northing),
                        source_pdf=str(pdf_path.name),
                        rl_min=str(rl_min),
                        rl_max=str(rl_max),
                        easting_min=str(e_min),
                        easting_max=str(e_max),
                        section_type="vertical_cross_section",
                        section_orientation="east_west",
                        coordinate_system="X=Easting, Y=Northing(constant), Z=RL/Elevation",
                        vertical_section_line=f"Northing={northing}",
                    )

                    # Add page number to metadata if multi-page
                    if page_num is None and len(doc) > 1:
                        dst.update_tags(page_number=str(page_idx + 1), total_pages=str(len(doc)))

                    logger.info(f"Successfully exported GeoTIFF: {out_path}")
                    success_count += 1

            doc.close()
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to export GeoTIFF: {e}")
            return False
