# geotools\batch_processor.py
"""
Batch processor for converting multiple PDFs to GeoTIFFs and DXFs.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .georeferencing import GeoReferencer
from .feature_extraction import FeatureExtractor
from .section_correlation import SectionCorrelator
import fitz

from .utils.debug_utils import DebugLogger, debug_trace, debug_value, log_state

# Set up module logger
debug_logger = DebugLogger()
logger = debug_logger.setup_module_logger("batch_processor", level="INFO")


class BatchProcessor:
    """Handle batch processing of PDF to GeoTIFF and DXF conversion."""

    def __init__(self, render_scale: float = 2.0, max_workers: int = 4):
        self.render_scale = render_scale
        self.max_workers = max_workers
        self.georeferencer = GeoReferencer()
        self.feature_extractor = FeatureExtractor()
        self.northing_overrides = {}  # Store user-provided northings
        self.correlator = SectionCorrelator()  # For cross-section correlation
        self.section_data = {}  # Store section data for correlation

    def scan_for_missing_northings(
        self, pdf_files: List[Path]
    ) -> Dict[Path, Dict[int, Optional[float]]]:
        """
        Scan PDFs to detect which pages are missing northing values.
        Returns dict of {pdf_path: {page_num: detected_northing or None}}
        """
        northings = {}

        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                pdf_northings = {}

                # Check all pages for northing
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    coord_system = self.georeferencer.detect_coordinates(page, pdf_path)

                    if coord_system and coord_system.get("northing"):
                        pdf_northings[page_num] = coord_system["northing"]
                        logger.info(
                            f"Found northing {coord_system['northing']} for {pdf_path.name} page {page_num + 1}"
                        )
                    else:
                        pdf_northings[page_num] = None
                        logger.warning(f"No northing found for {pdf_path.name} page {page_num + 1}")

                northings[pdf_path] = pdf_northings
                doc.close()

            except Exception as e:
                logger.error(f"Error scanning {pdf_path}: {e}")
                northings[pdf_path] = {}

        return northings

    def process_batch(
        self, pdf_files: List[Path], output_dir: Path, progress_callback=None
    ) -> Dict[str, bool]:
        """
        Process multiple PDFs in parallel.
        Returns dict of {filename: success_bool}
        """
        results = {}
        total = len(pdf_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {}
            for pdf_path in pdf_files:
                # Use a placeholder name - the export_geotiff method will handle naming
                output_path = output_dir / f"section.tif"
                future = executor.submit(
                    self.georeferencer.export_geotiff,
                    pdf_path,
                    output_path,
                    self.render_scale,
                    None,  # Export all pages
                )
                futures[future] = pdf_path

            # Process completed tasks
            for i, future in enumerate(as_completed(futures)):
                pdf_path = futures[future]
                try:
                    success = future.result()
                    results[pdf_path.name] = success

                    if progress_callback:
                        progress_callback(i + 1, total, pdf_path.name)

                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    results[pdf_path.name] = False

        return results

    def process_batch_to_dxf(
        self,
        pdf_files: List[Path],
        output_dir: Path,
        progress_callback=None,
        export_units: bool = True,
        export_contacts: bool = True,
    ) -> Dict[str, Dict]:
        """
        Process multiple PDFs and export combined features to DXF.

        Args:
            pdf_files: List of PDF paths to process
            output_dir: Output directory for DXF files
            progress_callback: Optional progress callback
            export_units: Whether to export geological units
            export_contacts: Whether to export contacts

        Returns dict of {pdf_filename: {feature_name: success_bool}}
        """
        results = {}
        total = len(pdf_files)

        # Collect all features across all PDFs and pages
        all_features_by_type = {}  # {feature_type: [(feature_data, pdf_name, page_num, northing)]}
        all_contacts_by_type = {}  # {contact_type: [(contact_data, pdf_name, page_num, northing)]}

        for i, pdf_path in enumerate(pdf_files):
            try:
                results[pdf_path.name] = {}

                # Open PDF
                doc = fitz.open(str(pdf_path))
                if len(doc) == 0:
                    logger.warning(f"PDF has no pages: {pdf_path}")
                    continue

                # Process each page
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # Detect coordinates for this page
                    coord_system = self.georeferencer.detect_coordinates(page, pdf_path)

                    # Store page-specific northing
                    page_northing = None

                    if not coord_system:
                        logger.warning(
                            f"Could not detect coordinates for {pdf_path} page {page_num}"
                        )
                        # Check for user-provided northing - but this should be per-page for multi-page PDFs
                        user_northing = None
                        if pdf_path in self.northing_overrides:
                            # Support both single northing and per-page northings
                            override = self.northing_overrides[pdf_path]
                            if isinstance(override, dict):
                                # Per-page overrides
                                user_northing = override.get(page_num, None)
                            else:
                                # Single override for all pages (legacy support)
                                user_northing = override

                        # For multi-page PDFs, might need different northing per page
                        if user_northing:
                            # If user provided a single northing, use it for this page
                            # In future, could support per-page northing overrides
                            page_northing = user_northing
                        if user_northing:
                            # Create minimal coord_system with user northing
                            coord_system = {
                                "northing": user_northing,
                                "northing_text": f"User provided: {user_northing}",
                                "easting_labels": [],
                                "rl_labels": [],
                                "page_rect": page.rect,
                            }
                            self.georeferencer.coord_system = coord_system

                            # Create basic transform with user northing
                            def transform(x, y):
                                return x, user_northing, y

                        else:
                            logger.error(
                                f"No coordinate system or northing for {pdf_path} page {page_num}"
                            )
                            results[pdf_path.name][f"page_{page_num}"] = {"error": "No coordinates"}
                            continue  # Skip this page, not the whole PDF
                    else:
                        self.georeferencer.coord_system = coord_system
                        transform = self.georeferencer.build_transformation()
                        if not transform:
                            logger.error(f"Cannot create transform for {pdf_path} page {page_num}")
                            results[pdf_path.name][f"page_{page_num}"] = {"error": "No transform"}
                            continue  # Skip this page

                    # Extract features for this page
                    annotations = self.feature_extractor.extract_annotations(page)
                    self.feature_extractor.number_geological_units(annotations)

                    # Extract section prefix from northing text if available
                    section_prefix = ""
                    if coord_system and coord_system.get("northing_text"):
                        import re

                        # Extract prefix like "KM" from text like "KM 122,800 Cross Section"
                        match = re.match(r"^([A-Z]+)\s+", coord_system["northing_text"])
                        if match:
                            section_prefix = match.group(1)

                    # Collect geological units by formation type
                    for (
                        unit_name,
                        unit,
                    ) in self.feature_extractor.geological_units.items():
                        formation = unit.get("formation", "UNKNOWN")
                        if formation not in all_features_by_type:
                            all_features_by_type[formation] = []

                        # Get the actual northing for this specific page
                        page_specific_northing = (
                            coord_system.get("northing", 0) if coord_system else page_northing
                        )

                        all_features_by_type[formation].append(
                            {
                                "unit": unit,
                                "unit_name": unit_name,
                                "pdf_name": pdf_path.stem,
                                "page_num": page_num,
                                "northing": page_specific_northing,
                                "section_prefix": section_prefix,
                                "transform": transform,
                                "author": unit.get("author", "UNKNOWN"),
                            }
                        )
                        results[pdf_path.name][f"{unit_name}_p{page_num}"] = True

                    # Extract and collect contacts using new centerline method
                    from .contact_extraction import extract_contacts_grouped
                    
                    page_northing_val = coord_system.get("northing", 0) if coord_system else page_northing
                    temp_section_data = {
                        (pdf_path, page_num): {
                            "units": dict(self.feature_extractor.geological_units),
                            "northing": page_northing_val,
                        }
                    }
                    grouped = extract_contacts_grouped(temp_section_data)
                    
                    # Flatten grouped contacts to list format
                    contacts = []
                    for group_name, group in grouped.items():
                        for polyline in group.polylines:
                            contacts.append({
                                "name": group_name,
                                "vertices": polyline.vertices,
                                "formation1": group.formation1,
                                "formation2": group.formation2,
                            })

                    for contact in contacts:
                        try:
                            # Create contact type key based on formations
                            # Normalize contact type name (sort formations alphabetically)
                            f1 = contact.get('formation1', 'UNK')
                            f2 = contact.get('formation2', 'UNK')
                            contact_type = "-".join(sorted([f1, f2]))
                            if contact_type not in all_contacts_by_type:
                                all_contacts_by_type[contact_type] = []

                            all_contacts_by_type[contact_type].append(
                                {
                                    "contact": contact,
                                    "pdf_name": pdf_path.stem,
                                    "page_num": page_num,
                                    "northing": page_northing_val,
                                    "section_prefix": section_prefix,
                                    "author": contact.get("author", "UNKNOWN"),
                                }
                            )
                            results[pdf_path.name][f"contact_{contact['name']}_p{page_num}"] = True
                        except Exception as e:
                            logger.error(
                                f"Failed to process contact {contact['name']} from {pdf_path} page {page_num}: {e}"
                            )
                            results[pdf_path.name][f"contact_{contact['name']}_p{page_num}"] = False

                    # Clear feature extractor for next page
                    self.feature_extractor.geological_units = {}
                    self.feature_extractor.contacts = []

                # End of page loop
                doc.close()

                if progress_callback:
                    progress_callback(i + 1, total, pdf_path.name)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results[pdf_path.name] = {"error": str(e)}

        # Export combined DXF files by feature type
        if export_units:
            self._export_combined_features(all_features_by_type, output_dir)
        if export_contacts:
            self._export_combined_contacts(all_contacts_by_type, output_dir)

        return results

    def _check_and_fix_overlaps(self, features_by_type: Dict) -> Dict:
        """Check for overlapping units and trim if necessary.

        This function operates in real-world coordinates and:
        - groups units by northing (section)
        - finds polygon-polygon overlaps
        - trims the lower-priority polygon (by area) where they intersect
        It is robust to Polygon and MultiPolygon outputs from Shapely.
        """
        from shapely.geometry import Polygon, MultiPolygon

        # Group by northing (section)
        sections = {}
        for formation, feature_list in features_by_type.items():
            for feature_data in feature_list:
                northing = feature_data["northing"]
                if northing not in sections:
                    sections[northing] = []
                sections[northing].append(feature_data)

        # Check for overlaps within each section
        for northing, features in sections.items():
            polygons = []
            for feature_data in features:
                unit = feature_data["unit"]
                coords = []
                for i in range(0, len(unit["vertices"]), 2):
                    if i + 1 < len(unit["vertices"]):
                        coords.append((unit["vertices"][i], unit["vertices"][i + 1]))

                if len(coords) >= 3:
                    # Close polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])

                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and not poly.is_empty:
                            polygons.append((poly, feature_data))
                    except Exception:
                        # Skip invalid geometries
                        continue

            # Check for overlaps
            for i in range(len(polygons)):
                for j in range(i + 1, len(polygons)):
                    poly1, data1 = polygons[i]
                    poly2, data2 = polygons[j]

                    if poly1.intersects(poly2) and not poly1.touches(poly2):
                        overlap = poly1.intersection(poly2)
                        if overlap.is_empty:
                            continue

                        overlap_area = overlap.area
                        if overlap_area <= 0:
                            continue

                        # Log warning about overlap
                        logger.warning(
                            f"Overlap detected between {data1['unit_name']} and {data2['unit_name']} at northing {northing}"
                        )

                        # Trim the overlapping area from both polygons
                        # Keep the unit with higher priority (based on area here)
                        if poly1.area > poly2.area:
                            # Trim from poly2
                            poly2_new = poly2.difference(overlap)
                            if not poly2_new.is_valid or poly2_new.is_empty:
                                continue

                            # If we get a MultiPolygon, keep the largest component
                            if isinstance(poly2_new, MultiPolygon):
                                largest = max(poly2_new.geoms, key=lambda g: g.area)
                                poly2_new = largest

                            if not hasattr(poly2_new, "exterior"):
                                continue

                            new_coords = list(poly2_new.exterior.coords)
                            new_vertices = []
                            for x, y in new_coords:
                                new_vertices.extend([x, y])
                            data2["unit"]["vertices"] = new_vertices
                        else:
                            # Trim from poly1
                            poly1_new = poly1.difference(overlap)
                            if not poly1_new.is_valid or poly1_new.is_empty:
                                continue

                            if isinstance(poly1_new, MultiPolygon):
                                largest = max(poly1_new.geoms, key=lambda g: g.area)
                                poly1_new = largest

                            if not hasattr(poly1_new, "exterior"):
                                continue

                            new_coords = list(poly1_new.exterior.coords)
                            new_vertices = []
                            for x, y in new_coords:
                                new_vertices.extend([x, y])
                            data1["unit"]["vertices"] = new_vertices

        return features_by_type

    def _export_combined_features(self, features_by_type: Dict, output_dir: Path):
        """Export all features of same type to single DXF files."""
        # Check and fix overlaps first
        features_by_type = self._check_and_fix_overlaps(features_by_type)

        for formation, feature_list in features_by_type.items():
            if not feature_list:
                continue

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in formation)

            # Get unique authors from this formation's features
            authors = set()
            for feature in feature_list:
                author = feature.get("author")
                if author and author != "UNKNOWN":
                    authors.add(author)

            # Include authors in filename if found
            if authors:
                author_suffix = "_" + "_".join(sorted(authors)[:3])  # Limit to 3 authors
                author_suffix = "".join(
                    c if c.isalnum() or c in "-_" else "" for c in author_suffix
                )
            else:
                author_suffix = ""

            dxf_path = output_dir / f"all_{safe_name}_units{author_suffix}.dxf"

            try:
                with open(dxf_path, "w") as f:
                    # DXF header
                    f.write("0\nSECTION\n2\nENTITIES\n")

                    for feature_data in feature_list:
                        unit = feature_data["unit"]
                        northing = feature_data["northing"]
                        transform = feature_data["transform"]

                        # Create 3D polyline from unit vertices
                        coords = []
                        for i in range(0, len(unit["vertices"]), 2):
                            if i + 1 < len(unit["vertices"]):
                                x, y = unit["vertices"][i], unit["vertices"][i + 1]
                                if transform:
                                    easting, _, rl = transform(x, y)
                                else:
                                    easting, rl = x, y
                                coords.append((easting, northing, rl))

                        if len(coords) >= 2:
                            # Close polygon if needed
                            if coords[0] != coords[-1] and unit["type"] in [
                                "Polygon",
                                "PolyLine",
                            ]:
                                coords.append(coords[0])

                            # Write polyline with metadata in layer name
                            northing_str = (
                                str(int(feature_data["northing"]))
                                if feature_data["northing"]
                                else "UNK"
                            )
                            section_prefix = feature_data.get("section_prefix", "")
                            author_str = feature_data.get("author", "UNK")
                            if author_str and author_str != "UNKNOWN":
                                # Clean author string for use in layer name
                                author_str = "".join(
                                    c if c.isalnum() or c in "-_" else "" for c in author_str
                                )
                            else:
                                author_str = "UNK"

                            # Format: FORMATION_PREFIX_NORTHING_AUTHOR (e.g., AMP_KM_123400_CST)
                            if section_prefix:
                                layer_name = (
                                    f"{formation}_{section_prefix}_{northing_str}_{author_str}"
                                )
                            else:
                                layer_name = f"{formation}_{northing_str}_{author_str}"
                            f.write("0\nPOLYLINE\n")
                            f.write(
                                f"8\n{layer_name}\n"
                            )  # Layer name includes formation, northing, and author
                            f.write("66\n1\n")  # Vertices follow
                            f.write("70\n8\n")  # 3D polyline

                            for x, y, z in coords:
                                f.write("0\nVERTEX\n")
                                f.write(f"8\n{layer_name}\n")
                                f.write(f"10\n{x:.2f}\n")  # X = Easting
                                f.write(f"20\n{y:.2f}\n")  # Y = Northing
                                f.write(f"30\n{z:.2f}\n")  # Z = RL

                            f.write("0\nSEQEND\n")

                    f.write("0\nENDSEC\n0\nEOF\n")

                logger.info(f"Exported {len(feature_list)} {formation} units to {dxf_path}")

            except Exception as e:
                logger.error(f"Failed to export combined {formation} features: {e}")

    def _export_combined_contacts(self, contacts_by_type: Dict, output_dir: Path):
        """Export all contacts of same type to single DXF files."""
        for contact_type, contact_list in contacts_by_type.items():
            if not contact_list:
                continue

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in contact_type)
            dxf_path = output_dir / f"all_{safe_name}_contacts.dxf"

            try:
                with open(dxf_path, "w") as f:
                    # DXF header
                    f.write("0\nSECTION\n2\nENTITIES\n")

                    for contact_data in contact_list:
                        contact = contact_data["contact"]
                        northing = contact_data["northing"]

                        # Contact vertices are already in real-world coords
                        coords = []
                        for i in range(0, len(contact["vertices"]), 2):
                            if i + 1 < len(contact["vertices"]):
                                easting = contact["vertices"][i]
                                rl = contact["vertices"][i + 1]
                                coords.append((easting, northing, rl))

                        if len(coords) >= 2:
                            northing_str = (
                                str(int(contact_data["northing"]))
                                if contact_data["northing"]
                                else "UNK"
                            )
                            section_prefix = contact_data.get("section_prefix", "")
                            author_str = contact_data.get("author", "UNK")
                            if author_str and author_str != "UNKNOWN":
                                # Clean author string for use in layer name
                                author_str = "".join(
                                    c if c.isalnum() or c in "-_" else "" for c in author_str
                                )
                            else:
                                author_str = "UNK"

                            # Format: CONTACTTYPE_PREFIX_NORTHING_AUTHOR
                            if section_prefix:
                                layer_name = (
                                    f"{contact_type}_{section_prefix}_{northing_str}_{author_str}"
                                )
                            else:
                                layer_name = f"{contact_type}_{northing_str}_{author_str}"
                            f.write("0\nPOLYLINE\n")
                            f.write(f"8\n{layer_name}\n")  # Layer name
                            f.write("66\n1\n")  # Vertices follow
                            f.write("70\n8\n")  # 3D polyline

                            for x, y, z in coords:
                                f.write("0\nVERTEX\n")
                                f.write(f"8\n{layer_name}\n")
                                f.write(f"10\n{x:.2f}\n")  # X = Easting
                                f.write(f"20\n{y:.2f}\n")  # Y = Northing
                                f.write(f"30\n{z:.2f}\n")  # Z = RL

                            f.write("0\nSEQEND\n")

                    f.write("0\nENDSEC\n0\nEOF\n")

                logger.info(f"Exported {len(contact_list)} {contact_type} contacts to {dxf_path}")

            except Exception as e:
                logger.error(f"Failed to export combined {contact_type} contacts: {e}")

    def _export_single_feature_dxf(self, dxf_path: Path, unit: Dict, unit_name: str, transform):
        """Export a single geological unit to DXF."""
        with open(dxf_path, "w") as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            # Create 3D polyline from unit vertices
            coords = []
            for i in range(0, len(unit["vertices"]), 2):
                if i + 1 < len(unit["vertices"]):
                    x, y = unit["vertices"][i], unit["vertices"][i + 1]
                    easting, northing, rl = transform(x, y)
                    coords.append((easting, northing, rl))

            if len(coords) >= 2:
                # Close polygon if needed
                if coords[0] != coords[-1] and unit["type"] in ["Polygon", "PolyLine"]:
                    coords.append(coords[0])

                f.write("0\nPOLYLINE\n")
                f.write(f"8\n{unit_name}\n")  # Layer name
                f.write("66\n1\n")  # Vertices follow
                f.write("70\n8\n")  # 3D polyline

                for x, y, z in coords:
                    f.write("0\nVERTEX\n")
                    f.write(f"8\n{unit_name}\n")
                    f.write(f"10\n{x:.2f}\n")  # X = Easting
                    f.write(f"20\n{y:.2f}\n")  # Y = Northing
                    f.write(f"30\n{z:.2f}\n")  # Z = RL

                f.write("0\nSEQEND\n")

            f.write("0\nENDSEC\n0\nEOF\n")

    def _export_single_contact_dxf(self, dxf_path: Path, contact: Dict, transform):
        """Export a single contact to DXF."""
        with open(dxf_path, "w") as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            # Get the constant northing value
            northing = (
                self.georeferencer.coord_system.get("northing", 0)
                if self.georeferencer.coord_system
                else 0
            )

            # Create 3D polyline from contact vertices
            # Contact vertices are stored as (easting, rl) pairs from extraction
            coords = []
            for i in range(0, len(contact["vertices"]), 2):
                if i + 1 < len(contact["vertices"]):
                    easting = contact["vertices"][i]
                    rl = contact["vertices"][i + 1]
                    # Proper 3D coordinates: X=Easting, Y=Northing(constant), Z=RL
                    coords.append((easting, northing, rl))

            if len(coords) >= 2:
                f.write("0\nPOLYLINE\n")
                f.write(f"8\n{contact['name']}\n")  # Layer name
                f.write("66\n1\n")  # Vertices follow
                f.write("70\n8\n")  # 3D polyline

                for x, y, z in coords:
                    f.write("0\nVERTEX\n")
                    f.write(f"8\n{contact['name']}\n")
                    f.write(f"10\n{x:.2f}\n")  # X = Easting
                    f.write(f"20\n{y:.2f}\n")  # Y = Northing
                    f.write(f"30\n{z:.2f}\n")  # Z = RL

                f.write("0\nSEQEND\n")

            f.write("0\nENDSEC\n0\nEOF\n")

    def process_batch_with_correlation(
        self,
        pdf_files: List[Path],
        output_dir: Path,
        correlate: bool = True,
        export_ties: bool = True,
        progress_callback=None,
    ) -> Dict:
        """
        Process multiple PDFs and correlate stratigraphy between sections.

        Args:
            pdf_files: List of PDF paths to process
            output_dir: Output directory for exports
            correlate: Whether to perform correlation
            export_ties: Whether to export tie lines
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with processing results and correlations
        """
        results = {
            "sections": {},
            "correlations": [],
            "tie_lines": [],
            "unified_export": None,
        }

        total = len(pdf_files)

        # First scan for missing northings
        self.status_var.set("Scanning PDFs for coordinate information...")
        northings = self.scan_for_missing_northings(pdf_files)

        # Process each PDF and collect data
        for i, pdf_path in enumerate(pdf_files):
            try:
                # Open PDF
                doc = fitz.open(str(pdf_path))
                if len(doc) == 0:
                    logger.warning(f"PDF has no pages: {pdf_path}")
                    continue

                page = doc[0]

                # Detect coordinates
                coord_system = self.georeferencer.detect_coordinates(page, pdf_path)

                if not coord_system:
                    # Check for user-provided northing
                    user_northing = self.northing_overrides.get(pdf_path, None)
                    if user_northing:
                        coord_system = {
                            "northing": user_northing,
                            "northing_text": f"User provided: {user_northing}",
                            "easting_labels": [],
                            "rl_labels": [],
                            "page_rect": page.rect,
                        }

                if not coord_system:
                    logger.error(f"No coordinate system for {pdf_path}")
                    continue

                self.georeferencer.coord_system = coord_system
                northing = coord_system["northing"]

                # Extract features
                annotations = self.feature_extractor.extract_annotations(page)
                self.feature_extractor.number_geological_units(annotations)

                # Build transformation
                transform = self.georeferencer.build_transformation()

                # Extract contacts using new centerline method
                from .contact_extraction import extract_contacts_grouped
                
                # Build temporary section data for contact extraction
                temp_section_data = {
                    (pdf_path, 0): {
                        "units": dict(self.feature_extractor.geological_units),
                        "northing": northing,
                    }
                }
                grouped = extract_contacts_grouped(temp_section_data)
                
                # Flatten grouped contacts to list
                contacts = []
                for group_name, group in grouped.items():
                    for polyline in group.polylines:
                        contacts.append({
                            "name": group_name,
                            "vertices": polyline.vertices,
                            "formation1": group.formation1,
                            "formation2": group.formation2,
                        })

                # Store section data
                self.section_data[northing] = {
                    "pdf_path": pdf_path,
                    "units": dict(self.feature_extractor.geological_units),
                    "contacts": contacts,
                    "coord_system": coord_system,
                }

                # Add to correlator if enabled
                if correlate:
                    self.correlator.add_section(
                        northing,
                        self.feature_extractor.geological_units,
                        contacts,
                        pdf_path.name,
                    )

                results["sections"][pdf_path.name] = {
                    "northing": northing,
                    "num_units": len(self.feature_extractor.geological_units),
                    "num_contacts": len(contacts),
                }

                doc.close()

                if progress_callback:
                    progress_callback(i + 1, total, pdf_path.name)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results["sections"][pdf_path.name] = {"error": str(e)}

        # Perform correlation if requested
        if correlate and len(self.correlator.sections) >= 2:
            logger.info("Finding correlations between sections...")
            correlations = self.correlator.find_correlations()
            results["correlations"] = correlations

            if export_ties:
                tie_lines = self.correlator.generate_tie_lines()
                results["tie_lines"] = tie_lines

                # Export tie lines to DXF
                tie_file = output_dir / "tie_lines.dxf"
                self.correlator.export_tie_lines_dxf(tie_file)

                # Export correlation data
                corr_file = output_dir / "correlations.json"
                self.correlator.export_correlations(corr_file)

        # Export unified dataset
        unified_file = output_dir / "unified_sections.csv"
        self._export_unified_csv(unified_file, self.section_data)
        results["unified_export"] = str(unified_file)

        # Export unified DXF with all sections
        unified_dxf = output_dir / "all_sections.dxf"
        self._export_unified_dxf(unified_dxf, self.section_data)

        return results

    def _export_unified_csv(self, output_path: Path, section_data: Dict):
        """Export all sections to a single CSV file."""
        with open(output_path, "w") as f:
            f.write("Section,Northing,Feature,Type,Formation,Easting,RL,VertexNum\n")

            for northing, data in sorted(section_data.items()):
                pdf_name = data["pdf_path"].stem

                # Export units
                for unit_name, unit in data["units"].items():
                    formation = unit.get("formation", "UNKNOWN")

                    for vertex_num, i in enumerate(range(0, len(unit["vertices"]), 2)):
                        if i + 1 < len(unit["vertices"]):
                            easting = unit["vertices"][i]
                            rl = unit["vertices"][i + 1]

                            f.write(
                                f"{pdf_name},{northing},{unit_name},Unit,"
                                f"{formation},{easting:.2f},{rl:.2f},{vertex_num+1}\n"
                            )

                # Export contacts
                for contact in data["contacts"]:
                    for vertex_num, i in enumerate(range(0, len(contact["vertices"]), 2)):
                        if i + 1 < len(contact["vertices"]):
                            easting = contact["vertices"][i]
                            rl = contact["vertices"][i + 1]

                            f.write(
                                f"{pdf_name},{northing},{contact['name']},Contact,"
                                f"Contact,{easting:.2f},{rl:.2f},{vertex_num+1}\n"
                            )

        logger.info(f"Exported unified CSV to {output_path}")

    def _export_unified_dxf(self, output_path: Path, section_data: Dict):
        """Export all sections to a single DXF file."""
        with open(output_path, "w") as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            for northing, data in sorted(section_data.items()):
                # Export units as 3D polylines
                for unit_name, unit in data["units"].items():
                    coords = []
                    for i in range(0, len(unit["vertices"]), 2):
                        if i + 1 < len(unit["vertices"]):
                            coords.append((unit["vertices"][i], northing, unit["vertices"][i + 1]))

                    if len(coords) >= 2:
                        # Close polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])

                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\n{unit.get('formation', 'UNKNOWN')}\n")  # Layer
                        f.write("66\n1\n")  # Vertices follow
                        f.write("70\n8\n")  # 3D polyline

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\n{unit.get('formation', 'UNKNOWN')}\n")
                            f.write(f"10\n{x:.2f}\n")  # X = Easting
                            f.write(f"20\n{y:.2f}\n")  # Y = Northing
                            f.write(f"30\n{z:.2f}\n")  # Z = RL

                        f.write("0\nSEQEND\n")

                # Export contacts
                for contact in data["contacts"]:
                    coords = []
                    for i in range(0, len(contact["vertices"]), 2):
                        if i + 1 < len(contact["vertices"]):
                            coords.append(
                                (
                                    contact["vertices"][i],
                                    northing,
                                    contact["vertices"][i + 1],
                                )
                            )

                    if len(coords) >= 2:
                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\nCONTACTS\n")  # Layer
                        f.write("66\n1\n")
                        f.write("70\n8\n")

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\nCONTACTS\n")
                            f.write(f"10\n{x:.2f}\n")
                            f.write(f"20\n{y:.2f}\n")
                            f.write(f"30\n{z:.2f}\n")

                        f.write("0\nSEQEND\n")

            f.write("0\nENDSEC\n0\nEOF\n")

        logger.info(f"Exported unified DXF to {output_path}")

    def validate_pdfs(self, pdf_files: List[Path]) -> List[Path]:
        """Validate PDFs before processing."""
        valid = []
        for pdf_path in pdf_files:
            if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
                # Quick check if coordinates can be detected
                try:
                    import fitz

                    doc = fitz.open(str(pdf_path))
                    if len(doc) > 0:
                        valid.append(pdf_path)
                    doc.close()
                except:
                    logger.warning(f"Could not validate {pdf_path}")

        return valid

    def process_batch_with_correlation(
        self,
        pdf_files: List[Path],
        output_dir: Path,
        correlate: bool = True,
        export_ties: bool = True,
        progress_callback=None,
    ) -> Dict:
        """
        Process multiple PDFs and correlate stratigraphy between sections.

        Args:
            pdf_files: List of PDF paths to process
            output_dir: Output directory for exports
            correlate: Whether to perform correlation
            export_ties: Whether to export tie lines
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with processing results and correlations
        """
        results = {
            "sections": {},
            "correlations": [],
            "tie_lines": [],
            "unified_export": None,
        }

        total = len(pdf_files)
        correlator = SectionCorrelator()
        section_data = {}

        # Process each PDF and collect data
        for i, pdf_path in enumerate(pdf_files):
            try:
                # Open PDF
                doc = fitz.open(str(pdf_path))
                if len(doc) == 0:
                    logger.warning(f"PDF has no pages: {pdf_path}")
                    continue

                page = doc[0]

                # Detect coordinates
                coord_system = self.georeferencer.detect_coordinates(page, pdf_path)

                if not coord_system:
                    # Check for user-provided northing
                    user_northing = self.northing_overrides.get(pdf_path, None)
                    if user_northing:
                        coord_system = {
                            "northing": user_northing,
                            "northing_text": f"User provided: {user_northing}",
                            "easting_labels": [],
                            "rl_labels": [],
                            "page_rect": page.rect,
                        }

                if not coord_system:
                    logger.error(f"No coordinate system for {pdf_path}")
                    continue

                self.georeferencer.coord_system = coord_system
                northing = coord_system["northing"]

                # Extract features
                annotations = self.feature_extractor.extract_annotations(page)
                self.feature_extractor.number_geological_units(annotations)

                # Build transformation
                transform = self.georeferencer.build_transformation()

                # Convert units to real-world coordinates
                units_rw = {}
                for unit_name, unit in self.feature_extractor.geological_units.items():
                    unit_rw = unit.copy()
                    # Transform vertices to real-world coords
                    if transform:
                        rw_vertices = []
                        for j in range(0, len(unit["vertices"]), 2):
                            if j + 1 < len(unit["vertices"]):
                                x, y = unit["vertices"][j], unit["vertices"][j + 1]
                                e, n, r = transform(x, y)
                                rw_vertices.extend([e, r])  # Store as (easting, rl) pairs
                    units_rw[unit_name] = unit_rw

                # Extract contacts using new centerline method
                from .contact_extraction import extract_contacts_grouped
                
                # Build temporary section data for contact extraction
                temp_section_data = {
                    (pdf_path, 0): {
                        "units": dict(self.feature_extractor.geological_units),
                        "northing": northing,
                    }
                }
                grouped = extract_contacts_grouped(temp_section_data)
                
                # Flatten grouped contacts to list
                contacts = []
                for group_name, group in grouped.items():
                    for polyline in group.polylines:
                        contacts.append({
                            "name": group_name,
                            "vertices": polyline.vertices,
                            "formation1": group.formation1,
                            "formation2": group.formation2,
                        })

                # Store section data
                section_data[northing] = {
                    "pdf_path": pdf_path,
                    "units": units_rw,
                    "contacts": contacts,
                    "coord_system": coord_system,
                }

                # Add to correlator if enabled
                if correlate:
                    correlator.add_section(northing, units_rw, contacts, pdf_path.name)

                results["sections"][pdf_path.name] = {
                    "northing": northing,
                    "num_units": len(units_rw),
                    "num_contacts": len(contacts),
                }

                doc.close()

                if progress_callback:
                    progress_callback(i + 1, total, pdf_path.name)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results["sections"][pdf_path.name] = {"error": str(e)}

        # Perform correlation if requested
        if correlate and len(correlator.sections) >= 2:
            logger.info("Finding correlations between sections...")
            correlations = correlator.find_correlations()
            results["correlations"] = correlations

            if export_ties:
                tie_lines = correlator.generate_tie_lines()
                results["tie_lines"] = tie_lines

                # Export tie lines to DXF
                tie_file = output_dir / "tie_lines.dxf"
                correlator.export_tie_lines_dxf(tie_file)

                # Export correlation data
                corr_file = output_dir / "correlations.json"
                correlator.export_correlations(corr_file)

        # Export unified dataset
        unified_csv = output_dir / "unified_sections.csv"
        self._export_unified_csv(unified_csv, section_data)
        results["unified_export"] = str(unified_csv)

        # Export unified DXF with all sections
        unified_dxf = output_dir / "all_sections.dxf"
        self._export_unified_dxf(unified_dxf, section_data)

        return results

    def _export_unified_csv(self, output_path: Path, section_data: Dict):
        """Export all sections to a single CSV file."""
        with open(output_path, "w") as f:
            f.write("Section,Northing,Feature,Type,Formation,Easting,RL,VertexNum\n")

            for northing, data in sorted(section_data.items()):
                pdf_name = data["pdf_path"].stem

                # Export units
                for unit_name, unit in data["units"].items():
                    formation = unit.get("formation", "UNKNOWN")

                    for vertex_num, i in enumerate(range(0, len(unit["vertices"]), 2)):
                        if i + 1 < len(unit["vertices"]):
                            easting = unit["vertices"][i]
                            rl = unit["vertices"][i + 1]

                            f.write(
                                f"{pdf_name},{northing},{unit_name},Unit,"
                                f"{formation},{easting:.2f},{rl:.2f},{vertex_num+1}\n"
                            )

                # Export contacts
                for contact in data["contacts"]:
                    for vertex_num, i in enumerate(range(0, len(contact["vertices"]), 2)):
                        if i + 1 < len(contact["vertices"]):
                            easting = contact["vertices"][i]
                            rl = contact["vertices"][i + 1]

                            f.write(
                                f"{pdf_name},{northing},{contact['name']},Contact,"
                                f"Contact,{easting:.2f},{rl:.2f},{vertex_num+1}\n"
                            )

        logger.info(f"Exported unified CSV to {output_path}")

    def _export_unified_dxf(self, output_path: Path, section_data: Dict):
        """Export all sections to a single DXF file."""
        with open(output_path, "w") as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            for northing, data in sorted(section_data.items()):
                # Export units as 3D polylines
                for unit_name, unit in data["units"].items():
                    coords = []
                    for i in range(0, len(unit["vertices"]), 2):
                        if i + 1 < len(unit["vertices"]):
                            coords.append((unit["vertices"][i], northing, unit["vertices"][i + 1]))

                    if len(coords) >= 2:
                        # Close polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])

                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\n{unit.get('formation', 'UNKNOWN')}\n")  # Layer
                        f.write("66\n1\n")  # Vertices follow
                        f.write("70\n8\n")  # 3D polyline

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\n{unit.get('formation', 'UNKNOWN')}\n")
                            f.write(f"10\n{x:.2f}\n")  # X = Easting
                            f.write(f"20\n{y:.2f}\n")  # Y = Northing
                            f.write(f"30\n{z:.2f}\n")  # Z = RL

                        f.write("0\nSEQEND\n")

                # Export contacts
                for contact in data["contacts"]:
                    coords = []
                    for i in range(0, len(contact["vertices"]), 2):
                        if i + 1 < len(contact["vertices"]):
                            coords.append(
                                (
                                    contact["vertices"][i],
                                    northing,
                                    contact["vertices"][i + 1],
                                )
                            )

                    if len(coords) >= 2:
                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\nCONTACTS\n")  # Layer
                        f.write("66\n1\n")
                        f.write("70\n8\n")

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\nCONTACTS\n")
                            f.write(f"10\n{x:.2f}\n")
                            f.write(f"20\n{y:.2f}\n")
                            f.write(f"30\n{z:.2f}\n")

                        f.write("0\nSEQEND\n")

            f.write("0\nENDSEC\n0\nEOF\n")

        logger.info(f"Exported unified DXF to {output_path}")
