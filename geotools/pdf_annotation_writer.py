# geotools\pdf_annotation_writer.py
"""
PDF Annotation Writer for Geological Cross-Section Analysis Tool

This module provides functionality to:
1. Duplicate existing PDF annotations with user-assigned names in the Subject field
2. Add extracted contacts as new PolyLine annotations
3. Preserve original annotations untouched

The approach creates new annotations alongside originals, allowing PDF-XChange Editor
to display and manage the assigned names while keeping the source annotations intact.

Usage:
    writer = PDFAnnotationWriter()
    writer.write_assignments_to_pdf(pdf_path, page_num, units, polylines, contacts)
"""

import fitz
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFAnnotationWriter:
    """Write geological annotations back to PDF files."""
    
    # Offset to apply to duplicated annotations to avoid exact overlap
    DUPLICATE_OFFSET = 0.5  # PDF points
    
    # Default colors for new annotations
    DEFAULT_CONTACT_COLOR = (0.0, 0.0, 0.0)  # Black
    DEFAULT_FAULT_COLOR = (1.0, 0.0, 0.0)    # Red
    
    # Annotation type mappings
    ANNOT_TYPE_POLYGON = 5    # Circle (also used for polygons in some cases)
    ANNOT_TYPE_POLYLINE = 9   # PolyLine
    ANNOT_TYPE_LINE = 3       # Line
    
    def __init__(self, create_backups: bool = True, duplicate_mode: bool = True):
        """
        Initialize the annotation writer.
        
        Args:
            create_backups: If True, create .backup.pdf files before modifying
            duplicate_mode: If True, create new annotations with assignments.
                           If False, modify original annotations in place.
        """
        self.create_backups = create_backups
        self.duplicate_mode = duplicate_mode
    
    def write_all_assignments(
        self,
        pdf_path: Path,
        sections_data: Dict[int, Dict],
        contacts_by_page: Optional[Dict[int, List[Dict]]] = None,
        author_prefix: str = "Assigned"
    ) -> Tuple[bool, str]:
        """
        Write all assignments for a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            sections_data: Dict mapping page_num -> section data containing:
                          - 'units': Dict of unit_name -> unit data with 'unit_assignment'
                          - 'polylines': Dict of line_name -> polyline data with 'fault_assignment'
            contacts_by_page: Optional dict mapping page_num -> list of contact dicts
            author_prefix: Prefix for the author field of new annotations
            
        Returns:
            (success, message) tuple
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return False, f"File not found: {pdf_path}"
        
        try:
            # Create backup if requested
            if self.create_backups:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = pdf_path.with_suffix(f".backup_{timestamp}.pdf")
                shutil.copy2(pdf_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Open PDF
            doc = fitz.open(str(pdf_path))
            
            modified_pages = 0
            total_annotations_added = 0
            total_annotations_modified = 0
            
            for page_num, section_data in sections_data.items():
                if page_num >= len(doc):
                    logger.warning(f"Page {page_num} not found in {pdf_path}")
                    continue
                
                page = doc[page_num]
                
                # Process units (polygons)
                units = section_data.get('units', {})
                for unit_name, unit_data in units.items():
                    assignment = unit_data.get('unit_assignment')
                    if not assignment:
                        continue
                    
                    if self.duplicate_mode:
                        # Create duplicate annotation with assignment
                        added = self._add_polygon_annotation(
                            page, unit_data, assignment, author_prefix
                        )
                        if added:
                            total_annotations_added += 1
                    else:
                        # Modify original annotation
                        modified = self._modify_annotation_by_author(
                            page, unit_data.get('author'), assignment
                        )
                        if modified:
                            total_annotations_modified += 1
                
                # Process polylines (faults)
                polylines = section_data.get('polylines', {})
                for line_name, line_data in polylines.items():
                    assignment = line_data.get('fault_assignment')
                    if not assignment:
                        continue
                    
                    if self.duplicate_mode:
                        # Create duplicate annotation with assignment
                        added = self._add_polyline_annotation(
                            page, line_data, assignment, author_prefix
                        )
                        if added:
                            total_annotations_added += 1
                    else:
                        # Modify original annotation
                        modified = self._modify_annotation_by_author(
                            page, line_data.get('author'), assignment
                        )
                        if modified:
                            total_annotations_modified += 1
                
                # Add contacts if provided
                if contacts_by_page and page_num in contacts_by_page:
                    contacts = contacts_by_page[page_num]
                    for contact in contacts:
                        added = self._add_contact_annotation(page, contact, author_prefix)
                        if added:
                            total_annotations_added += 1
                
                if units or polylines or (contacts_by_page and page_num in contacts_by_page):
                    modified_pages += 1
            
            # Save to new file with _annotated suffix to avoid encryption issues
            output_path = pdf_path.with_stem(f"{pdf_path.stem}_annotated")
            
            # Save the document to the new file
            doc.save(str(output_path), garbage=3, deflate=True)
            doc.close()
            
            msg = f"Modified {modified_pages} page(s). "
            if self.duplicate_mode:
                msg += f"Added {total_annotations_added} new annotation(s)."
            else:
                msg += f"Updated {total_annotations_modified} annotation(s)."
            msg += f"\n\nSaved to: {output_path.name}"
            
            logger.info(f"Saved annotated PDF: {output_path}")
            
            return True, msg
            
        except Exception as e:
            logger.error(f"Error writing to {pdf_path}: {e}", exc_info=True)
            return False, str(e)
    
    def _add_polygon_annotation(
        self,
        page,
        unit_data: Dict,
        assignment: str,
        author_prefix: str
    ) -> bool:
        """
        Add a new polygon annotation with the assignment in the Subject field.
        
        Args:
            page: PyMuPDF page object
            unit_data: Dict containing 'vertices', 'color', 'author'
            assignment: The assigned unit name
            author_prefix: Prefix for author field
            
        Returns:
            True if annotation was added successfully
        """
        vertices = unit_data.get('vertices', [])
        if not vertices or len(vertices) < 6:
            return False
        
        # Convert flat vertex list to list of points
        points = self._vertices_to_points(vertices)
        if len(points) < 3:
            return False
        
        # Apply small offset to distinguish from original
        points = [(p[0] + self.DUPLICATE_OFFSET, p[1] + self.DUPLICATE_OFFSET) 
                  for p in points]
        
        try:
            # Add polygon annotation
            annot = page.add_polygon_annot(points)
            
            # Set colors
            color = unit_data.get('color')
            if color:
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    stroke_color = tuple(color[:3])
                    fill_color = tuple(c * 0.3 + 0.7 for c in color[:3])  # Lighter fill
                    annot.set_colors(stroke=stroke_color, fill=fill_color)
            
            # Set border
            annot.set_border(width=1)
            
            # Set opacity
            annot.set_opacity(0.5)
            
            # Set metadata
            original_author = unit_data.get('author', '')
            info = {
                'title': f"{author_prefix}_{original_author}" if original_author else author_prefix,
                'subject': assignment,
                'content': f"Unit: {assignment}\nOriginal: {original_author}",
            }
            annot.set_info(info)
            
            # Update appearance
            annot.update()
            
            logger.debug(f"Added polygon annotation: {assignment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add polygon annotation: {e}")
            return False
    
    def _add_polyline_annotation(
        self,
        page,
        line_data: Dict,
        assignment: str,
        author_prefix: str
    ) -> bool:
        """
        Add a new polyline annotation with the assignment in the Subject field.
        
        Args:
            page: PyMuPDF page object
            line_data: Dict containing 'vertices', 'color', 'author'
            assignment: The assigned fault name
            author_prefix: Prefix for author field
            
        Returns:
            True if annotation was added successfully
        """
        vertices = line_data.get('vertices', [])
        if not vertices or len(vertices) < 4:
            return False
        
        # Convert flat vertex list to list of points
        points = self._vertices_to_points(vertices)
        if len(points) < 2:
            return False
        
        # Apply small offset
        points = [(p[0] + self.DUPLICATE_OFFSET, p[1] + self.DUPLICATE_OFFSET) 
                  for p in points]
        
        try:
            # Add polyline annotation
            annot = page.add_polyline_annot(points)
            
            # Set colors
            color = line_data.get('color', self.DEFAULT_FAULT_COLOR)
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                stroke_color = tuple(color[:3])
                annot.set_colors(stroke=stroke_color)
            else:
                annot.set_colors(stroke=self.DEFAULT_FAULT_COLOR)
            
            # Set border
            annot.set_border(width=2)
            
            # Set metadata
            original_author = line_data.get('author', '')
            info = {
                'title': f"{author_prefix}_{original_author}" if original_author else author_prefix,
                'subject': f"Fault:{assignment}",
                'content': f"Fault: {assignment}\nOriginal: {original_author}",
            }
            annot.set_info(info)
            
            # Update appearance
            annot.update()
            
            logger.debug(f"Added polyline annotation: {assignment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add polyline annotation: {e}")
            return False
    
    def _add_contact_annotation(
        self,
        page,
        contact: Dict,
        author_prefix: str
    ) -> bool:
        """
        Add a contact as a new polyline annotation.
        
        Args:
            page: PyMuPDF page object
            contact: Dict containing 'vertices', 'unit1', 'unit2', optional 'color'
            author_prefix: Prefix for author field
            
        Returns:
            True if annotation was added successfully
        """
        # Contacts may have vertices in real-world coordinates
        # Check if we have PDF coordinates or need transformation
        vertices = contact.get('vertices', [])
        pdf_vertices = contact.get('pdf_vertices', vertices)  # Use pdf_vertices if available
        
        if not pdf_vertices or len(pdf_vertices) < 4:
            return False
        
        points = self._vertices_to_points(pdf_vertices)
        if len(points) < 2:
            return False
        
        try:
            # Add polyline annotation for contact
            annot = page.add_polyline_annot(points)
            
            # Set colors - contacts typically black or dark gray
            color = contact.get('color', self.DEFAULT_CONTACT_COLOR)
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                stroke_color = tuple(color[:3])
            else:
                stroke_color = self.DEFAULT_CONTACT_COLOR
            annot.set_colors(stroke=stroke_color)
            
            # Set border - dashed for contacts
            annot.set_border(width=1, dashes=[3, 2])
            
            # Set metadata
            unit1 = contact.get('unit1', 'Unknown')
            unit2 = contact.get('unit2', 'Unknown')
            contact_name = f"{unit1}_{unit2}"
            
            info = {
                'title': f"{author_prefix}_Contact",
                'subject': f"Contact:{contact_name}",
                'content': f"Contact between {unit1} and {unit2}",
            }
            annot.set_info(info)
            
            # Update appearance
            annot.update()
            
            logger.debug(f"Added contact annotation: {contact_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add contact annotation: {e}")
            return False
    
    def _modify_annotation_by_author(
        self,
        page,
        author: str,
        assignment: str
    ) -> bool:
        """
        Modify an existing annotation's Subject field by matching author tag.
        
        Args:
            page: PyMuPDF page object
            author: The author tag to match
            assignment: The assignment to write to Subject
            
        Returns:
            True if annotation was found and modified
        """
        if not author:
            return False
        
        try:
            for annot in page.annots():
                info = annot.info if hasattr(annot, 'info') else {}
                existing_author = info.get('title') or info.get('author', '')
                
                if existing_author == author:
                    # Found matching annotation
                    info['subject'] = f"Assigned:{assignment}"
                    annot.set_info(info)
                    annot.update()
                    logger.debug(f"Modified annotation {author} -> {assignment}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error modifying annotation: {e}")
        
        return False
    
    def _vertices_to_points(self, vertices: List[float]) -> List[Tuple[float, float]]:
        """
        Convert flat vertex list [x1, y1, x2, y2, ...] to list of point tuples.
        
        Args:
            vertices: Flat list of coordinates
            
        Returns:
            List of (x, y) tuples
        """
        points = []
        for i in range(0, len(vertices) - 1, 2):
            points.append((vertices[i], vertices[i + 1]))
        return points
    
    def add_contacts_to_pdf(
        self,
        pdf_path: Path,
        page_num: int,
        contacts: List[Dict],
        coord_transform_func=None
    ) -> Tuple[bool, str]:
        """
        Add extracted contacts to a PDF as polyline annotations.
        
        This is a convenience method for adding contacts without affecting
        other annotations.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            contacts: List of contact dicts with 'vertices', 'unit1', 'unit2'
            coord_transform_func: Optional function to transform real-world coords to PDF coords
                                 Function signature: (easting, rl) -> (pdf_x, pdf_y)
            
        Returns:
            (success, message) tuple
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return False, f"File not found: {pdf_path}"
        
        if not contacts:
            return True, "No contacts to add"
        
        try:
            # Create backup
            if self.create_backups:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = pdf_path.with_suffix(f".contacts_{timestamp}.pdf")
                shutil.copy2(pdf_path, backup_path)
            
            doc = fitz.open(str(pdf_path))
            
            if page_num >= len(doc):
                doc.close()
                return False, f"Page {page_num} not found"
            
            page = doc[page_num]
            added_count = 0
            
            for contact in contacts:
                # Transform coordinates if function provided
                if coord_transform_func and 'vertices' in contact:
                    vertices = contact['vertices']
                    pdf_vertices = []
                    for i in range(0, len(vertices) - 1, 2):
                        easting, rl = vertices[i], vertices[i + 1]
                        pdf_x, pdf_y = coord_transform_func(easting, rl)
                        pdf_vertices.extend([pdf_x, pdf_y])
                    contact['pdf_vertices'] = pdf_vertices
                
                if self._add_contact_annotation(page, contact, "ExtractedContact"):
                    added_count += 1
            
            doc.save(str(pdf_path), incremental=True, deflate=True)
            doc.close()
            
            return True, f"Added {added_count} contact annotation(s)"
            
        except Exception as e:
            logger.error(f"Error adding contacts to {pdf_path}: {e}", exc_info=True)
            return False, str(e)


def write_assignments_batch(
    pdf_files: List[Path],
    all_sections_data: Dict[Tuple[Any, int], Dict],
    contacts_data: Optional[Dict[Tuple[Any, int], List[Dict]]] = None,
    create_backups: bool = True,
    duplicate_mode: bool = True
) -> Dict[str, Any]:
    """
    Batch write assignments to multiple PDF files.
    
    Args:
        pdf_files: List of PDF file paths
        all_sections_data: Dict mapping (pdf_path, page_num) -> section data
        contacts_data: Optional dict mapping (pdf_path, page_num) -> contacts list
        create_backups: Whether to create backup files
        duplicate_mode: Whether to create new annotations (True) or modify existing (False)
        
    Returns:
        Dict with 'success', 'modified_files', 'errors' keys
    """
    writer = PDFAnnotationWriter(create_backups=create_backups, duplicate_mode=duplicate_mode)
    
    # Group data by PDF file
    pdf_data = {}
    for (pdf_path, page_num), section_data in all_sections_data.items():
        pdf_str = str(pdf_path)
        if pdf_str not in pdf_data:
            pdf_data[pdf_str] = {}
        pdf_data[pdf_str][page_num] = section_data
    
    # Group contacts by PDF file
    pdf_contacts = {}
    if contacts_data:
        for (pdf_path, page_num), contacts in contacts_data.items():
            pdf_str = str(pdf_path)
            if pdf_str not in pdf_contacts:
                pdf_contacts[pdf_str] = {}
            pdf_contacts[pdf_str][page_num] = contacts
    
    results = {
        'success': True,
        'modified_files': 0,
        'errors': []
    }
    
    for pdf_str, sections in pdf_data.items():
        contacts = pdf_contacts.get(pdf_str)
        success, msg = writer.write_all_assignments(
            Path(pdf_str), sections, contacts
        )
        
        if success:
            results['modified_files'] += 1
            logger.info(f"{pdf_str}: {msg}")
        else:
            results['errors'].append(f"{pdf_str}: {msg}")
            results['success'] = False
    
    return results


def heal_contact_gaps(
    contacts: List[Dict],
    max_gap_distance: float = 10.0,
    snap_to_endpoints: bool = True
) -> List[Dict]:
    """
    Heal gaps between adjacent contacts by extending/connecting endpoints.

    This is useful when contacts don't quite meet due to extraction artifacts.

    Args:
        contacts: List of contact dicts with 'vertices' and 'name'
        max_gap_distance: Maximum distance to heal (in coordinate units)
        snap_to_endpoints: If True, snap nearby endpoints together

    Returns:
        List of contacts with healed gaps
    """
    if len(contacts) < 2:
        return contacts

    healed_contacts = [dict(c) for c in contacts]  # Deep copy
    healed_count = 0

    # Get endpoints for all contacts
    endpoints = []
    for i, contact in enumerate(healed_contacts):
        vertices = contact.get('vertices', [])
        if len(vertices) < 4:
            continue

        # Start point
        endpoints.append({
            'contact_idx': i,
            'is_start': True,
            'x': vertices[0],
            'y': vertices[1],
        })
        # End point
        endpoints.append({
            'contact_idx': i,
            'is_start': False,
            'x': vertices[-2],
            'y': vertices[-1],
        })

    # Find and heal gaps
    import numpy as np

    for i, ep1 in enumerate(endpoints):
        for j, ep2 in enumerate(endpoints):
            if i >= j:  # Skip self and already checked
                continue
            if ep1['contact_idx'] == ep2['contact_idx']:  # Same contact
                continue

            # Calculate distance
            dist = np.sqrt((ep1['x'] - ep2['x'])**2 + (ep1['y'] - ep2['y'])**2)

            if dist <= max_gap_distance and dist > 0.1:
                # Heal the gap by averaging the positions
                mid_x = (ep1['x'] + ep2['x']) / 2
                mid_y = (ep1['y'] + ep2['y']) / 2

                # Update the contact vertices
                contact1 = healed_contacts[ep1['contact_idx']]
                contact2 = healed_contacts[ep2['contact_idx']]
                vertices1 = contact1['vertices']
                vertices2 = contact2['vertices']

                if snap_to_endpoints:
                    # Move both endpoints to the midpoint
                    if ep1['is_start']:
                        vertices1[0] = mid_x
                        vertices1[1] = mid_y
                    else:
                        vertices1[-2] = mid_x
                        vertices1[-1] = mid_y

                    if ep2['is_start']:
                        vertices2[0] = mid_x
                        vertices2[1] = mid_y
                    else:
                        vertices2[-2] = mid_x
                        vertices2[-1] = mid_y

                    healed_count += 1
                    logger.debug(
                        f"Healed gap ({dist:.1f} units) between "
                        f"{contact1.get('name', 'unknown')} and {contact2.get('name', 'unknown')}"
                    )

    logger.info(f"Healed {healed_count} contact gaps")
    return healed_contacts


def extend_contacts_to_boundaries(
    contacts: List[Dict],
    left_boundary: float,
    right_boundary: float,
    extension_distance: float = 20.0
) -> List[Dict]:
    """
    Extend contacts that nearly reach the section boundaries.

    Args:
        contacts: List of contact dicts with 'vertices'
        left_boundary: Left edge X coordinate
        right_boundary: Right edge X coordinate
        extension_distance: Max distance from boundary to extend

    Returns:
        List of contacts with extended endpoints
    """
    import numpy as np

    extended_contacts = [dict(c) for c in contacts]
    extended_count = 0

    for contact in extended_contacts:
        vertices = contact.get('vertices', [])
        if len(vertices) < 4:
            continue

        # Check start point (left end)
        start_x, start_y = vertices[0], vertices[1]
        if abs(start_x - left_boundary) <= extension_distance:
            # Extend to left boundary
            if len(vertices) >= 4:
                # Use direction from first segment
                dx = vertices[0] - vertices[2]
                dy = vertices[1] - vertices[3]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Extend to boundary
                    extend_dist = start_x - left_boundary
                    new_x = left_boundary
                    new_y = start_y + (dy / dx) * extend_dist if abs(dx) > 0.001 else start_y
                    vertices[0] = new_x
                    vertices[1] = new_y
                    extended_count += 1

        # Check end point (right end)
        end_x, end_y = vertices[-2], vertices[-1]
        if abs(end_x - right_boundary) <= extension_distance:
            # Extend to right boundary
            if len(vertices) >= 4:
                dx = vertices[-2] - vertices[-4]
                dy = vertices[-1] - vertices[-3]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    extend_dist = right_boundary - end_x
                    new_x = right_boundary
                    new_y = end_y + (dy / dx) * extend_dist if abs(dx) > 0.001 else end_y
                    vertices[-2] = new_x
                    vertices[-1] = new_y
                    extended_count += 1

    logger.info(f"Extended {extended_count} contact endpoints to boundaries")
    return extended_contacts


if __name__ == "__main__":
    # Test/demo code
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python pdf_annotation_writer.py <pdf_file>")
        print("\nThis module provides PDF annotation writing capabilities.")
        print("Import it in your main application to use its functions.")
        sys.exit(1)

    # Simple test - list existing annotations
    pdf_path = sys.argv[1]
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        print(f"\nPage {page_num}:")
        for annot in page.annots():
            info = annot.info
            print(f"  Type: {annot.type[1]}")
            print(f"    Author: {info.get('title', 'N/A')}")
            print(f"    Subject: {info.get('subject', 'N/A')}")
            print(f"    Vertices: {len(annot.vertices) if annot.vertices else 0} points")

    doc.close()
