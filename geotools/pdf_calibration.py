# geotools\pdf_calibration.py
"""
PDF Calibration Tool for inspecting and filtering PDF contents.
Helps identify geological features vs background elements (grids, borders, etc.)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import fitz
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExtractionFilter:
    """Configurable filter for PDF extraction."""
    
    def __init__(self):
        # Color filters
        self.excluded_colors = set()  # Colors to exclude (as hex strings)
        self.included_colors = set()  # If set, ONLY these colors are included
        
        # Grayscale filter
        self.exclude_grayscale = True
        self.grayscale_tolerance = 0.15  # How close R,G,B must be to be "grayscale"
        
        # Point count filters
        self.min_points = 3  # Minimum vertices for a valid feature
        self.max_points = 10000  # Maximum (to filter out very complex items)
        
        # Size filters (in PDF coordinates)
        self.min_width = 10.0
        self.min_height = 10.0
        
        # Type filters
        self.excluded_types = set()  # e.g., {"l"} to exclude line segments
        
        # Metadata filters
        self.require_annotation = False  # Only extract PDF annotations, not drawings
        
        # Author/subject filters
        self.included_authors = set()
        self.excluded_authors = set()
        self.included_subjects = set()
        self.excluded_subjects = set()
        
    def color_to_hex(self, color: Tuple) -> str:
        """Convert RGB tuple to hex string."""
        if not color or len(color) < 3:
            return "#000000"
        r, g, b = color[:3]
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex string to RGB tuple (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    
    def is_grayscale(self, color: Tuple) -> bool:
        """Check if color is grayscale (R≈G≈B)."""
        if not color or len(color) < 3:
            return True
        r, g, b = color[:3]
        return (abs(r - g) < self.grayscale_tolerance and 
                abs(g - b) < self.grayscale_tolerance and 
                abs(r - b) < self.grayscale_tolerance)
    
    def should_include(self, item: Dict) -> Tuple[bool, str]:
        """
        Check if an item should be included based on filter rules.
        Returns (include: bool, reason: str)
        """
        color = item.get("color")
        vertices = item.get("vertices", [])
        item_type = item.get("type", "")
        author = item.get("author", "")
        subject = item.get("subject", "")
        
        # Check color exclusions
        if color:
            hex_color = self.color_to_hex(color)
            
            if hex_color in self.excluded_colors:
                return False, f"Color {hex_color} is excluded"
            
            if self.included_colors and hex_color not in self.included_colors:
                return False, f"Color {hex_color} not in allowed list"
            
            if self.exclude_grayscale and self.is_grayscale(color):
                # Allow pure black and white through (they're usually intentional)
                r, g, b = color[:3]
                if not (r < 0.1 and g < 0.1 and b < 0.1):  # Not black
                    if not (r > 0.9 and g > 0.9 and b > 0.9):  # Not white
                        return False, "Grayscale color excluded"
        
        # Check point count
        num_points = len(vertices) // 2
        if num_points < self.min_points:
            return False, f"Too few points ({num_points} < {self.min_points})"
        if num_points > self.max_points:
            return False, f"Too many points ({num_points} > {self.max_points})"
        
        # Check bounding box size
        if vertices and len(vertices) >= 4:
            xs = [vertices[i] for i in range(0, len(vertices), 2)]
            ys = [vertices[i] for i in range(1, len(vertices), 2)]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            
            if width < self.min_width and height < self.min_height:
                return False, f"Too small ({width:.1f}x{height:.1f})"
        
        # Check type exclusions
        if item_type in self.excluded_types:
            return False, f"Type {item_type} is excluded"
        
        # Check author/subject filters
        if self.included_authors and author not in self.included_authors:
            return False, f"Author '{author}' not in allowed list"
        if author in self.excluded_authors:
            return False, f"Author '{author}' is excluded"
        
        if self.included_subjects and subject not in self.included_subjects:
            return False, f"Subject '{subject}' not in allowed list"
        if subject in self.excluded_subjects:
            return False, f"Subject '{subject}' is excluded"
        
        return True, "Passed all filters"
    
    def save(self, filepath: Path):
        """Save filter configuration to JSON."""
        config = {
            "excluded_colors": list(self.excluded_colors),
            "included_colors": list(self.included_colors),
            "exclude_grayscale": self.exclude_grayscale,
            "grayscale_tolerance": self.grayscale_tolerance,
            "min_points": self.min_points,
            "max_points": self.max_points,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "excluded_types": list(self.excluded_types),
            "require_annotation": self.require_annotation,
            "included_authors": list(self.included_authors),
            "excluded_authors": list(self.excluded_authors),
            "included_subjects": list(self.included_subjects),
            "excluded_subjects": list(self.excluded_subjects),
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved filter config to {filepath}")
    
    def load(self, filepath: Path):
        """Load filter configuration from JSON."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.excluded_colors = set(config.get("excluded_colors", []))
        self.included_colors = set(config.get("included_colors", []))
        self.exclude_grayscale = config.get("exclude_grayscale", True)
        self.grayscale_tolerance = config.get("grayscale_tolerance", 0.15)
        self.min_points = config.get("min_points", 3)
        self.max_points = config.get("max_points", 10000)
        self.min_width = config.get("min_width", 10.0)
        self.min_height = config.get("min_height", 10.0)
        self.excluded_types = set(config.get("excluded_types", []))
        self.require_annotation = config.get("require_annotation", False)
        self.included_authors = set(config.get("included_authors", []))
        self.excluded_authors = set(config.get("excluded_authors", []))
        self.included_subjects = set(config.get("included_subjects", []))
        self.excluded_subjects = set(config.get("excluded_subjects", []))
        
        logger.info(f"Loaded filter config from {filepath}")


class PDFCalibrationDialog:
    """Dialog for inspecting PDF contents and configuring extraction filters."""
    
    def __init__(self, parent, pdf_path: Path = None, extraction_filter: ExtractionFilter = None):
        self.parent = parent
        self.pdf_path = pdf_path
        self.filter = extraction_filter or ExtractionFilter()
        
        # Data collected from PDF
        self.annotations = []
        self.drawings = []
        self.color_stats = defaultdict(lambda: {"count": 0, "items": [], "point_counts": []})
        self.author_stats = defaultdict(int)
        self.subject_stats = defaultdict(int)
        self.type_stats = defaultdict(int)
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("PDF Calibration Tool")
        self.dialog.geometry("1200x800")
        self.dialog.transient(parent)
        
        self.setup_ui()
        
        if pdf_path:
            self.load_pdf(pdf_path)
    
    def setup_ui(self):
        """Create the dialog UI."""
        # Main paned window
        main_pane = ttk.PanedWindow(self.dialog, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Statistics and categories
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        
        # PDF selection
        pdf_frame = ttk.LabelFrame(left_frame, text="PDF File")
        pdf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pdf_label = ttk.Label(pdf_frame, text="No PDF loaded")
        self.pdf_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(pdf_frame, text="Browse...", command=self.browse_pdf).pack(side=tk.RIGHT, padx=5, pady=5)
        ttk.Button(pdf_frame, text="Scan PDF", command=self.scan_pdf).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Page selection
        page_frame = ttk.Frame(left_frame)
        page_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(page_frame, text="Pages to scan:").pack(side=tk.LEFT)
        self.scan_mode_var = tk.StringVar(value="all")
        ttk.Radiobutton(page_frame, text="All pages", variable=self.scan_mode_var, value="all").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(page_frame, text="First page only", variable=self.scan_mode_var, value="first").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(page_frame, text="Specific page:", variable=self.scan_mode_var, value="specific").pack(side=tk.LEFT, padx=5)
        self.specific_page_var = tk.IntVar(value=1)
        self.page_spinbox = ttk.Spinbox(page_frame, from_=1, to=100, width=5, textvariable=self.specific_page_var)
        self.page_spinbox.pack(side=tk.LEFT)
        
        # Statistics notebook
        stats_notebook = ttk.Notebook(left_frame)
        stats_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Colors tab
        colors_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(colors_frame, text="Colors")
        
        # Color tree with checkboxes for exclude/include
        color_tree_frame = ttk.Frame(colors_frame)
        color_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.color_tree = ttk.Treeview(color_tree_frame, 
                                        columns=("hex", "count", "avg_points", "action"),
                                        show="headings", height=15)
        self.color_tree.heading("hex", text="Color (Hex)")
        self.color_tree.heading("count", text="Count")
        self.color_tree.heading("avg_points", text="Avg Points")
        self.color_tree.heading("action", text="Action")
        self.color_tree.column("hex", width=100)
        self.color_tree.column("count", width=80)
        self.color_tree.column("avg_points", width=80)
        self.color_tree.column("action", width=80)
        
        color_scroll = ttk.Scrollbar(color_tree_frame, orient=tk.VERTICAL, command=self.color_tree.yview)
        self.color_tree.configure(yscrollcommand=color_scroll.set)
        self.color_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        color_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Color action buttons
        color_btn_frame = ttk.Frame(colors_frame)
        color_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(color_btn_frame, text="Exclude Selected", command=self.exclude_selected_color).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_btn_frame, text="Include Only Selected", command=self.include_only_selected_color).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_btn_frame, text="Clear Color Filters", command=self.clear_color_filters).pack(side=tk.LEFT, padx=2)
        
        # Grayscale filter
        self.exclude_gray_var = tk.BooleanVar(value=self.filter.exclude_grayscale)
        ttk.Checkbutton(colors_frame, text="Exclude grayscale colors (grids, borders)", 
                       variable=self.exclude_gray_var, 
                       command=self.update_grayscale_filter).pack(anchor=tk.W, padx=5)
        
        # Authors tab
        authors_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(authors_frame, text="Authors")
        
        self.author_tree = ttk.Treeview(authors_frame, columns=("author", "count"), show="headings", height=10)
        self.author_tree.heading("author", text="Author")
        self.author_tree.heading("count", text="Count")
        self.author_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Subjects tab
        subjects_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(subjects_frame, text="Subjects")
        
        self.subject_tree = ttk.Treeview(subjects_frame, columns=("subject", "count"), show="headings", height=10)
        self.subject_tree.heading("subject", text="Subject")
        self.subject_tree.heading("count", text="Count")
        self.subject_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Types tab
        types_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(types_frame, text="Types")
        
        self.type_tree = ttk.Treeview(types_frame, columns=("type", "count"), show="headings", height=10)
        self.type_tree.heading("type", text="Type")
        self.type_tree.heading("count", text="Count")
        self.type_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Filter settings and preview
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        
        # Filter settings
        filter_frame = ttk.LabelFrame(right_frame, text="Filter Settings")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Point count filters
        points_frame = ttk.Frame(filter_frame)
        points_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(points_frame, text="Min Points:").pack(side=tk.LEFT)
        self.min_points_var = tk.IntVar(value=self.filter.min_points)
        ttk.Spinbox(points_frame, from_=1, to=100, width=8, textvariable=self.min_points_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(points_frame, text="Max Points:").pack(side=tk.LEFT, padx=(20, 0))
        self.max_points_var = tk.IntVar(value=self.filter.max_points)
        ttk.Spinbox(points_frame, from_=10, to=100000, width=8, textvariable=self.max_points_var).pack(side=tk.LEFT, padx=5)
        
        # Size filters
        size_frame = ttk.Frame(filter_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(size_frame, text="Min Width:").pack(side=tk.LEFT)
        self.min_width_var = tk.DoubleVar(value=self.filter.min_width)
        ttk.Spinbox(size_frame, from_=0, to=1000, width=8, textvariable=self.min_width_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="Min Height:").pack(side=tk.LEFT, padx=(20, 0))
        self.min_height_var = tk.DoubleVar(value=self.filter.min_height)
        ttk.Spinbox(size_frame, from_=0, to=1000, width=8, textvariable=self.min_height_var).pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(filter_frame, text="Apply Filters", command=self.apply_filters).pack(pady=5)
        
        # Preview/Results
        results_frame = ttk.LabelFrame(right_frame, text="Filter Results Preview")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, width=50)
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Save Filter Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Filter Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Apply & Close", command=self.apply_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_pdf(self):
        """Browse for a PDF file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filepath:
            self.pdf_path = Path(filepath)
            self.pdf_label.config(text=self.pdf_path.name)
    
    def load_pdf(self, pdf_path: Path):
        """Load a PDF file."""
        self.pdf_path = pdf_path
        self.pdf_label.config(text=pdf_path.name)
    
    def scan_pdf(self):
        """Scan the PDF and collect statistics."""
        if not self.pdf_path:
            messagebox.showwarning("Warning", "Please select a PDF file first")
            return
        
        try:
            doc = fitz.open(str(self.pdf_path))
            total_pages = len(doc)
            
            # Update page spinbox max
            self.page_spinbox.config(to=total_pages)
            
            # Determine which pages to scan
            scan_mode = self.scan_mode_var.get()
            if scan_mode == "all":
                pages_to_scan = list(range(total_pages))
            elif scan_mode == "first":
                pages_to_scan = [0]
            else:  # specific
                page_num = self.specific_page_var.get() - 1  # Convert to 0-based
                if 0 <= page_num < total_pages:
                    pages_to_scan = [page_num]
                else:
                    messagebox.showwarning("Warning", f"Page {page_num + 1} does not exist (PDF has {total_pages} pages)")
                    doc.close()
                    return
            
            # Clear previous data
            self.annotations = []
            self.drawings = []
            self.color_stats = defaultdict(lambda: {"count": 0, "items": [], "point_counts": []})
            self.author_stats = defaultdict(int)
            self.subject_stats = defaultdict(int)
            self.type_stats = defaultdict(int)
            
            # Scan selected pages
            for page_idx in pages_to_scan:
                page = doc[page_idx]
                
                # Collect annotations
                for annot in page.annots():
                    if annot.type[0] in [3, 4, 6, 9, 10]:  # Drawing types
                        annot_data = self._extract_annot_data(annot)
                        if annot_data:
                            annot_data["page"] = page_idx + 1
                            self.annotations.append(annot_data)
                            self._update_stats(annot_data, source="annotation")
                
                # Collect drawings
                drawings = page.get_drawings()
                for draw_idx, drawing in enumerate(drawings):
                    draw_data = self._extract_drawing_data(drawing, draw_idx)
                    if draw_data:
                        draw_data["page"] = page_idx + 1
                        self.drawings.append(draw_data)
                        self._update_stats(draw_data, source="drawing")
            
            doc.close()
            
            # Update UI
            self._update_color_tree()
            self._update_author_tree()
            self._update_subject_tree()
            self._update_type_tree()
            self._update_results_preview()
            
            pages_scanned = len(pages_to_scan)
            messagebox.showinfo("Scan Complete", 
                              f"Scanned {pages_scanned} page(s) of {total_pages} total\n\n"
                              f"Found:\n"
                              f"  {len(self.annotations)} annotations\n"
                              f"  {len(self.drawings)} drawings\n"
                              f"  {len(self.color_stats)} unique colors")
            
        except Exception as e:
            logger.error(f"Error scanning PDF: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to scan PDF: {str(e)}")
    
    def _extract_annot_data(self, annot) -> Optional[Dict]:
        """Extract data from a PDF annotation."""
        try:
            # Get color
            color = None
            if hasattr(annot, "colors") and annot.colors:
                color = annot.colors.get("stroke") or annot.colors.get("fill")
            if not color and hasattr(annot, "stroke_color"):
                color = annot.stroke_color
            
            if color:
                color = tuple(color[:3])
            
            # Get vertices
            vertices = []
            if hasattr(annot, "vertices") and annot.vertices:
                for v in annot.vertices:
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        vertices.extend([float(v[0]), float(v[1])])
                    else:
                        vertices.append(float(v))
            elif hasattr(annot, "rect"):
                rect = annot.rect
                vertices = [rect.x0, rect.y0, rect.x1, rect.y0, rect.x1, rect.y1, rect.x0, rect.y1]
            
            # Get metadata
            author = None
            subject = None
            if hasattr(annot, "info"):
                info = annot.info
                author = info.get("title") or info.get("author")
                subject = info.get("subject") or info.get("content")
            
            return {
                "type": annot.type[1],
                "color": color,
                "vertices": vertices,
                "author": author,
                "subject": subject,
                "source": "annotation"
            }
        except Exception as e:
            logger.warning(f"Error extracting annotation: {e}")
            return None
    
    def _extract_drawing_data(self, drawing: Dict, index: int) -> Optional[Dict]:
        """Extract data from a PDF drawing."""
        try:
            # Get color
            color = drawing.get("color") or drawing.get("stroke")
            if color:
                color = tuple(color[:3]) if len(color) >= 3 else None
            
            # Count items and points
            items = drawing.get("items", [])
            item_types = [item[0] for item in items]
            
            # Estimate vertex count
            vertex_count = 0
            for item in items:
                if item[0] == "l":  # Line
                    vertex_count += 2
                elif item[0] == "c":  # Cubic bezier
                    vertex_count += 4
                elif item[0] == "q":  # Quadratic bezier
                    vertex_count += 3
                elif item[0] == "re":  # Rectangle
                    vertex_count += 4
            
            # Get bounding rect
            rect = drawing.get("rect")
            
            return {
                "type": f"drawing_{','.join(set(item_types))}",
                "color": color,
                "vertices": [0] * (vertex_count * 2),  # Placeholder
                "item_types": item_types,
                "item_count": len(items),
                "rect": rect,
                "source": "drawing",
                "index": index
            }
        except Exception as e:
            logger.warning(f"Error extracting drawing: {e}")
            return None
    
    def _update_stats(self, item: Dict, source: str):
        """Update statistics with an item."""
        color = item.get("color")
        if color:
            hex_color = self.filter.color_to_hex(color)
            self.color_stats[hex_color]["count"] += 1
            self.color_stats[hex_color]["items"].append(item)
            num_points = len(item.get("vertices", [])) // 2
            self.color_stats[hex_color]["point_counts"].append(num_points)
        
        author = item.get("author") or "(none)"
        self.author_stats[author] += 1
        
        subject = item.get("subject") or "(none)"
        self.subject_stats[subject] += 1
        
        item_type = item.get("type", "unknown")
        self.type_stats[item_type] += 1
    
    def _update_color_tree(self):
        """Update the color tree view."""
        for item in self.color_tree.get_children():
            self.color_tree.delete(item)
        
        # Sort by count descending
        sorted_colors = sorted(self.color_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        
        for hex_color, stats in sorted_colors:
            count = stats["count"]
            point_counts = stats["point_counts"]
            avg_points = sum(point_counts) / len(point_counts) if point_counts else 0
            
            # Determine action status
            action = "Include"
            if hex_color in self.filter.excluded_colors:
                action = "EXCLUDED"
            elif self.filter.included_colors and hex_color not in self.filter.included_colors:
                action = "Not included"
            elif self.filter.exclude_grayscale:
                rgb = self.filter.hex_to_rgb(hex_color)
                if self.filter.is_grayscale(rgb):
                    action = "Grayscale"
            
            self.color_tree.insert("", "end", values=(hex_color, count, f"{avg_points:.1f}", action))
    
    def _update_author_tree(self):
        """Update the author tree view."""
        for item in self.author_tree.get_children():
            self.author_tree.delete(item)
        
        for author, count in sorted(self.author_stats.items(), key=lambda x: x[1], reverse=True):
            self.author_tree.insert("", "end", values=(author, count))
    
    def _update_subject_tree(self):
        """Update the subject tree view."""
        for item in self.subject_tree.get_children():
            self.subject_tree.delete(item)
        
        for subject, count in sorted(self.subject_stats.items(), key=lambda x: x[1], reverse=True):
            self.subject_tree.insert("", "end", values=(subject, count))
    
    def _update_type_tree(self):
        """Update the type tree view."""
        for item in self.type_tree.get_children():
            self.type_tree.delete(item)
        
        for item_type, count in sorted(self.type_stats.items(), key=lambda x: x[1], reverse=True):
            self.type_tree.insert("", "end", values=(item_type, count))
    
    def _update_results_preview(self):
        """Update the results preview with filter statistics."""
        self.results_text.delete(1.0, tk.END)
        
        total_items = len(self.annotations) + len(self.drawings)
        included = 0
        excluded_reasons = defaultdict(int)
        
        all_items = self.annotations + self.drawings
        
        for item in all_items:
            include, reason = self.filter.should_include(item)
            if include:
                included += 1
            else:
                excluded_reasons[reason] += 1
        
        self.results_text.insert(tk.END, f"=== Filter Preview ===\n\n")
        self.results_text.insert(tk.END, f"Total items scanned: {total_items}\n")
        self.results_text.insert(tk.END, f"Would be INCLUDED: {included}\n")
        self.results_text.insert(tk.END, f"Would be EXCLUDED: {total_items - included}\n\n")
        
        if excluded_reasons:
            self.results_text.insert(tk.END, f"Exclusion reasons:\n")
            for reason, count in sorted(excluded_reasons.items(), key=lambda x: x[1], reverse=True):
                self.results_text.insert(tk.END, f"  {count:5d} - {reason}\n")
        
        # Show color breakdown
        self.results_text.insert(tk.END, f"\n=== Color Summary ===\n")
        for hex_color, stats in sorted(self.color_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:15]:
            rgb = self.filter.hex_to_rgb(hex_color)
            is_gray = self.filter.is_grayscale(rgb)
            gray_marker = " [GRAY]" if is_gray else ""
            excluded_marker = " [EXCLUDED]" if hex_color in self.filter.excluded_colors else ""
            self.results_text.insert(tk.END, f"  {hex_color}: {stats['count']} items{gray_marker}{excluded_marker}\n")
    
    def exclude_selected_color(self):
        """Add selected color(s) to exclusion list."""
        selection = self.color_tree.selection()
        for item in selection:
            values = self.color_tree.item(item, "values")
            hex_color = values[0]
            self.filter.excluded_colors.add(hex_color)
            self.filter.included_colors.discard(hex_color)
        
        self._update_color_tree()
        self._update_results_preview()
    
    def include_only_selected_color(self):
        """Set included colors to only the selected ones."""
        selection = self.color_tree.selection()
        if selection:
            self.filter.included_colors = set()
            for item in selection:
                values = self.color_tree.item(item, "values")
                hex_color = values[0]
                self.filter.included_colors.add(hex_color)
                self.filter.excluded_colors.discard(hex_color)
        
        self._update_color_tree()
        self._update_results_preview()
    
    def clear_color_filters(self):
        """Clear all color filters."""
        self.filter.excluded_colors.clear()
        self.filter.included_colors.clear()
        self._update_color_tree()
        self._update_results_preview()
    
    def update_grayscale_filter(self):
        """Update grayscale filter setting."""
        self.filter.exclude_grayscale = self.exclude_gray_var.get()
        self._update_color_tree()
        self._update_results_preview()
    
    def apply_filters(self):
        """Apply current filter settings from UI."""
        self.filter.min_points = self.min_points_var.get()
        self.filter.max_points = self.max_points_var.get()
        self.filter.min_width = self.min_width_var.get()
        self.filter.min_height = self.min_height_var.get()
        self.filter.exclude_grayscale = self.exclude_gray_var.get()
        
        self._update_results_preview()
    
    def save_config(self):
        """Save filter configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.apply_filters()
            self.filter.save(Path(filepath))
            messagebox.showinfo("Saved", f"Filter configuration saved to {filepath}")
    
    def load_config(self):
        """Load filter configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.filter.load(Path(filepath))
            # Update UI with loaded values
            self.min_points_var.set(self.filter.min_points)
            self.max_points_var.set(self.filter.max_points)
            self.min_width_var.set(self.filter.min_width)
            self.min_height_var.set(self.filter.min_height)
            self.exclude_gray_var.set(self.filter.exclude_grayscale)
            
            self._update_color_tree()
            self._update_results_preview()
            messagebox.showinfo("Loaded", f"Filter configuration loaded from {filepath}")
    
    def apply_and_close(self):
        """Apply filters and close dialog."""
        self.apply_filters()
        self.dialog.destroy()
    
    def get_filter(self) -> ExtractionFilter:
        """Get the configured filter."""
        return self.filter


def open_calibration_dialog(parent, pdf_path: Path = None, existing_filter: ExtractionFilter = None) -> Optional[ExtractionFilter]:
    """
    Open the calibration dialog and return the configured filter.

    Args:
        parent: Parent window
        pdf_path: Optional PDF path to pre-load
        existing_filter: Optional existing filter to edit

    Returns:
        ExtractionFilter if user clicked Apply, None if cancelled
    """
    dialog = PDFCalibrationDialog(parent, pdf_path, existing_filter)
    parent.wait_window(dialog.dialog)
    return dialog.get_filter()


class AnnotationGroupFilterDialog:
    """
    Dialog for selecting which annotation groups to include/exclude during extraction.
    Shows groups by author/title with checkboxes.
    """

    def __init__(self, parent, groups: Dict[str, Dict],
                 included: Optional[set] = None, excluded: Optional[set] = None):
        """
        Args:
            parent: Parent window
            groups: Dictionary from FeatureExtractor.scan_annotation_groups() or scan_all_pages()
            included: Pre-selected included groups (whitelist mode)
            excluded: Pre-selected excluded groups (blacklist mode)
        """
        self.parent = parent
        self.groups = groups
        self.result_included = None  # Will be set if user clicks Apply
        self.result_excluded = None
        self.cancelled = True

        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Filter Annotation Groups")
        self.dialog.geometry("500x450")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Variables for checkboxes
        self.check_vars = {}

        self._create_widgets(included, excluded)

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

    def _create_widgets(self, included, excluded):
        """Create dialog widgets."""
        # Instructions
        ttk.Label(
            self.dialog,
            text="Select which annotation groups to include in extraction.\n"
                 "Groups with default names (Pencil, Polygon, etc.) are typically noise.",
            wraplength=480
        ).pack(pady=10, padx=10)

        # Quick selection buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Select None", command=self._select_none).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Select Non-Default", command=self._select_non_default).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Invert", command=self._invert_selection).pack(side=tk.LEFT, padx=2)

        # Scrollable frame for checkboxes
        container = ttk.Frame(self.dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Add checkboxes for each group
        # Sort by count (descending), then by name
        sorted_groups = sorted(
            self.groups.items(),
            key=lambda x: (-x[1]['count'], x[0].lower())
        )

        for group_name, info in sorted_groups:
            var = tk.BooleanVar()

            # Determine default checked state
            if included is not None:
                var.set(group_name in included)
            elif excluded is not None:
                var.set(group_name not in excluded)
            else:
                # Default: exclude noise names
                var.set(not info.get('is_default_name', False))

            self.check_vars[group_name] = var

            # Create checkbox with info
            row_frame = ttk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X, pady=1)

            cb = ttk.Checkbutton(row_frame, variable=var)
            cb.pack(side=tk.LEFT)

            # Color indicator if available
            colors = info.get('sample_colors', [])
            if colors:
                color = colors[0]
                if isinstance(color, (tuple, list)) and len(color) >= 3:
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    )
                    color_label = tk.Label(row_frame, bg=hex_color, width=2, height=1)
                    color_label.pack(side=tk.LEFT, padx=2)

            # Group name and count
            types_str = ', '.join(sorted(info.get('types', set())))[:30]
            label_text = f"{group_name} ({info['count']})"
            if info.get('is_default_name'):
                label_text += " [DEFAULT]"
            if info.get('has_fault_keyword'):
                label_text += " [FAULT]"

            ttk.Label(row_frame, text=label_text).pack(side=tk.LEFT, padx=5)

            if types_str:
                ttk.Label(row_frame, text=f"({types_str})", foreground='gray').pack(side=tk.LEFT)

        # Buttons
        btn_frame2 = ttk.Frame(self.dialog)
        btn_frame2.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame2, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame2, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=5)

    def _select_all(self):
        for var in self.check_vars.values():
            var.set(True)

    def _select_none(self):
        for var in self.check_vars.values():
            var.set(False)

    def _select_non_default(self):
        for group_name, var in self.check_vars.items():
            info = self.groups.get(group_name, {})
            var.set(not info.get('is_default_name', False))

    def _invert_selection(self):
        for var in self.check_vars.values():
            var.set(not var.get())

    def _apply(self):
        # Build included set (whitelist of checked items)
        self.result_included = {
            name for name, var in self.check_vars.items() if var.get()
        }
        self.result_excluded = {
            name for name, var in self.check_vars.items() if not var.get()
        }
        self.cancelled = False
        self.dialog.destroy()

    def _cancel(self):
        self.cancelled = True
        self.dialog.destroy()


def open_annotation_filter_dialog(
    parent,
    groups: Dict[str, Dict],
    included: Optional[set] = None,
    excluded: Optional[set] = None
) -> Tuple[Optional[set], Optional[set], bool]:
    """
    Open the annotation group filter dialog.

    Args:
        parent: Parent window
        groups: Dictionary from FeatureExtractor.scan_annotation_groups()
        included: Pre-selected included groups
        excluded: Pre-selected excluded groups

    Returns:
        (included_set, excluded_set, cancelled)
    """
    dialog = AnnotationGroupFilterDialog(parent, groups, included, excluded)
    parent.wait_window(dialog.dialog)
    return dialog.result_included, dialog.result_excluded, dialog.cancelled