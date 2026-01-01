# geotools\main_gui.py
"""
Main GUI for geological cross-section tool suite.
Enhanced with improved section viewer, 3D visualization, and unit assignment capabilities.
"""

import tkinter as tk
from tkinter import (
    filedialog,
    messagebox,
    ttk,
    scrolledtext,
    simpledialog,
    colorchooser,
)
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
import logging
import sys
import fitz
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


# Import our modules
from .georeferencing import GeoReferencer
from .feature_extraction import FeatureExtractor
from .strat_column_v2 import StratColumnV2, StratUnit, Prospect, Fault, FaultType
from .auto_labeler import AutoLabeler
from .batch_processor import BatchProcessor
from .section_correlation import SectionCorrelator
from .pdf_calibration import PDFCalibrationDialog, ExtractionFilter, open_calibration_dialog
from .contact_extraction import ContactExtractor, GroupedContact, extract_contacts_grouped
from .tie_line_editor import TieLineEditor, open_tie_line_editor
from .pdf_annotation_writer import PDFAnnotationWriter, write_assignments_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set to DEBUG for development
if __name__ == "__main__" or "debug" in sys.argv:
    logger.setLevel(logging.DEBUG)


class GeologicalCrossSectionGUI:
    """Main GUI application for geological cross-section analysis with integrated section viewer."""

    # Default config file paths
    CONFIG_DIR = Path.home() / ".geo_cross_section"
    DEFAULT_STRAT_CONFIG = CONFIG_DIR / "strat_column.json"
    DEFAULT_FILTER_CONFIG = CONFIG_DIR / "extraction_filter.json"

    def __init__(self, root):
        self.root = root
        self.root.title("Geological Cross-Section Tool Suite - Enhanced Edition")
        self.root.geometry("1800x1000")

        # Ensure config directory exists
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.georeferencer = GeoReferencer()
        self.feature_extractor = FeatureExtractor()
        self.strat_column = StratColumnV2()
        self.auto_labeler = AutoLabeler(self.strat_column)
        self.batch_processor = BatchProcessor()
        self.extraction_filter = ExtractionFilter()  # Configurable filter for PDF extraction
        
        # Track user-assigned polygons (for auto-labeling to respect)
        self.user_assigned_polygons = set()  # Polygon names explicitly assigned by user

        # Auto-load saved config if exists
        self._load_saved_config()

        # State variables
        self.pdf_list = []  # List of PDF paths
        self.current_pdf_index = -1
        self.current_pdf = None
        self.current_page = None
        self.current_page_num = 0
        self.pdf_path = None
        self.selected_feature = None
        self.selected_feature_name = None
        self.annotations = []
        self.selected_items = {}
        self.show_grid = False
        self.show_coordinates_var = tk.BooleanVar(value=True)
        self.current_strat_unit = None

        # Section viewer state
        self.all_sections_data = {}  # {(pdf_path, page_num): section_data}
        self.all_geological_units = {}  # All units across sections
        self.all_contacts = []  # All contacts (legacy format)
        self.grouped_contacts = {}  # New grouped contacts: {name: GroupedContact}
        self.selected_units = set()  # Selected units for assignment
        self.unit_patches = {}  # Maps matplotlib patches to unit data
        self.polyline_patches = {}  # Maps matplotlib lines to polyline data
        self.selected_polylines = set()  # Selected polylines for fault assignment
        
        # Classification mode state
        self.classification_mode = "none"  # "none", "polygon", "fault"
        self.current_unit_assignment = None  # Unit name to assign when clicking
        self.current_fault_assignment = None  # Fault name to assign when clicking
        
        # Defined units and faults (user-created)
        self.defined_units = {}  # {name: {'color': (r,g,b), 'name': name}}
        self.defined_faults = {}  # {name: {'color': hex_color, 'name': name, 'type': str}}
        self.fault_colors = {"F1": "#FF0000", "F2": "#0000FF", "F3": "#00AA00", "F4": "#FF8800"}
        
        # UI button references
        self.unit_buttons = {}
        self.fault_buttons = {}

        # View parameters
        self.current_center_section = 0
        self.sections_to_display = 1  # Number of sections to show
        self.vertical_spacing = 50  # Spacing between sections
        self.northings = []  # Sorted list of northings

        # Display options
        self.pdf_alpha_var = tk.DoubleVar(value=0.3)
        self.show_pdf_var = tk.BooleanVar(value=False)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.hide_empty_var = tk.BooleanVar(value=True)
        self.section_count_var = tk.IntVar(value=1)
        self.view_mode = tk.StringVar(value="section")  # "section" or "3d"

        # Pan/drag state
        self.pan_active = False
        self.pan_start = None

        # 3D display options
        self.display_mode_3d = tk.StringVar(value="wireframe")

        # Buffers for solid meshes built between sections
        # Each entry in mesh_vertices is [x, y, z]; mesh_faces are [i0, i1, i2] indices
        self.mesh_vertices = []
        self.mesh_faces = []

        self.setup_ui()

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _load_saved_config(self):
        """Load saved strat column and filter config if they exist."""
        try:
            if self.DEFAULT_STRAT_CONFIG.exists():
                self.strat_column.load(self.DEFAULT_STRAT_CONFIG)
                logger.info(f"Loaded strat column from {self.DEFAULT_STRAT_CONFIG}")
        except Exception as e:
            logger.warning(f"Could not load strat column config: {e}")
        
        try:
            if self.DEFAULT_FILTER_CONFIG.exists():
                self.extraction_filter.load(self.DEFAULT_FILTER_CONFIG)
                logger.info(f"Loaded extraction filter from {self.DEFAULT_FILTER_CONFIG}")
        except Exception as e:
            logger.warning(f"Could not load filter config: {e}")

    def _save_config_on_exit(self):
        """Save current config before exit."""
        try:
            self.strat_column.save(self.DEFAULT_STRAT_CONFIG)
            logger.info(f"Saved strat column to {self.DEFAULT_STRAT_CONFIG}")
        except Exception as e:
            logger.warning(f"Could not save strat column config: {e}")
        
        try:
            self.extraction_filter.save(self.DEFAULT_FILTER_CONFIG)
            logger.info(f"Saved extraction filter to {self.DEFAULT_FILTER_CONFIG}")
        except Exception as e:
            logger.warning(f"Could not save filter config: {e}")

    def setup_ui(self):
        """Create the main UI layout - streamlined without menu bar."""
        # NO menu bar - all actions via toolbar buttons
        
        # Create main notebook for tabbed interface
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Section Viewer (main workflow)
        self.create_section_viewer_tab()

        # Tab 2: Tie Lines (embedded editor)
        self.create_tie_lines_tab()

        # Tab 3: 3D visualization
        self.create_3d_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Import PDFs to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open PDF(s)", command=self.open_pdf, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Folder", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export GeoTIFF", command=self.export_geotiff)
        file_menu.add_command(label="Export Features (CSV)", command=self.export_csv)
        file_menu.add_command(label="Export Features (DXF)", command=self.export_dxf)
        file_menu.add_separator()
        file_menu.add_command(label="Batch Export GeoTIFFs", command=self.batch_export)
        file_menu.add_command(label="Batch Export DXFs", command=self.batch_export_dxf)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Write Assignments to PDFs", command=self.write_assignments_to_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Process menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Process All Pages", command=self.process_all)
        process_menu.add_command(label="Detect Coordinates", command=self.detect_coordinates)
        process_menu.add_command(label="Set Northing Manually", command=self.set_northing)
        process_menu.add_separator()
        process_menu.add_command(label="Extract Features", command=self.extract_features)
        process_menu.add_command(
            label="Find Contacts (Buffered)", command=lambda: self.find_contacts("buffered")
        )
        process_menu.add_command(
            label="Find Contacts (Dense)", command=lambda: self.find_contacts("dense")
        )
        process_menu.add_command(
            label="Find Contacts (Simplified for Leapfrog)", command=lambda: self.find_contacts("simplified")
        )
        process_menu.add_separator()
        process_menu.add_command(label="Correlate Sections", command=self.correlate_sections)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(
            label="Show Grid", variable=self.show_grid_var, command=self.update_display
        )
        view_menu.add_checkbutton(
            label="Show Coordinates",
            variable=self.show_coordinates_var,
            command=self.update_display,
        )
        view_menu.add_checkbutton(
            label="Show PDF Background", variable=self.show_pdf_var, command=self.update_display
        )
        view_menu.add_separator()
        view_menu.add_command(label="Debug Info", command=self.show_debug_info)

        # Strat menu
        strat_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Stratigraphy", menu=strat_menu)
        strat_menu.add_command(label="Load Strat Column", command=self.load_strat_column)
        strat_menu.add_command(label="Save Strat Column", command=self.save_strat_column)
        strat_menu.add_separator()
        strat_menu.add_command(label="Add Unit", command=self.add_strat_unit)
        strat_menu.add_command(label="Remove Unit", command=self.remove_strat_unit)
        strat_menu.add_separator()
        strat_menu.add_command(label="Auto-detect Faults", command=self.detect_faults_from_contacts)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="PDF Calibration...", command=self.open_calibration_tool)
        tools_menu.add_command(label="Load Filter Config...", command=self.load_filter_config)
        tools_menu.add_command(label="Save Filter Config...", command=self.save_filter_config)
        tools_menu.add_separator()
        tools_menu.add_command(label="Extract Contacts (Grouped)", command=self.extract_contacts_grouped)
        tools_menu.add_command(label="Open Tie Line Editor...", command=self.open_tie_line_editor)

        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.open_pdf())

    # PART 2: PDF Viewer Tab
    def create_pdf_tab(self):
        """Create the traditional PDF viewer tab."""
        pdf_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(pdf_frame, text="PDF Viewer")

        # Top toolbar
        toolbar = ttk.Frame(pdf_frame)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # File operations
        ttk.Button(toolbar, text="Open PDF", command=self.open_pdf, width=12).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Open Folder", command=self.open_folder, width=12).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Page navigation
        ttk.Button(toolbar, text="<", command=self.prev_page, width=3).pack(side=tk.LEFT, padx=2)
        self.page_label = ttk.Label(toolbar, text="Page 1/1")
        self.page_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text=">", command=self.next_page, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Processing
        ttk.Button(toolbar, text="Process All", command=self.process_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Detect Coords", command=self.detect_coordinates).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Extract Features", command=self.extract_features).pack(
            side=tk.LEFT, padx=2
        )

        # Main content with paned windows
        main_paned = ttk.PanedWindow(pdf_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - PDF viewer
        viewer_frame = ttk.Frame(main_paned)
        main_paned.add(viewer_frame, weight=3)

        # Create matplotlib figure for PDF display
        self.fig_pdf = plt.figure(figsize=(10, 12))
        self.ax_pdf = self.fig_pdf.add_subplot(111)

        # Canvas for PDF
        canvas_frame = ttk.Frame(viewer_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_pdf = FigureCanvasTkAgg(self.fig_pdf, master=canvas_frame)
        self.canvas_pdf.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.nav_toolbar_pdf = NavigationToolbar2Tk(self.canvas_pdf, toolbar_frame)
        self.nav_toolbar_pdf.update()

        # Bind mouse events
        self.canvas_pdf.mpl_connect("button_press_event", self.on_pdf_click)

        # Right panel - Feature list and properties
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Feature tree
        self.create_feature_panel(right_frame)

    def create_feature_panel(self, parent):
        """Create feature list panel."""
        # Feature tree
        tree_frame = ttk.LabelFrame(parent, text="Features", padding=5)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Selection buttons
        button_frame = ttk.Frame(tree_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Select None", command=self.select_none).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Invert", command=self.invert_selection).pack(
            side=tk.LEFT, padx=2
        )

        # Feature tree
        self.feature_tree = ttk.Treeview(
            tree_frame, columns=("type", "formation", "points"), height=15
        )
        self.feature_tree.heading("#0", text="Selected")
        self.feature_tree.heading("type", text="Type")
        self.feature_tree.heading("formation", text="Formation")
        self.feature_tree.heading("points", text="Points")
        self.feature_tree.column("#0", width=60)
        self.feature_tree.column("type", width=80)
        self.feature_tree.column("formation", width=100)
        self.feature_tree.column("points", width=60)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=scrollbar.set)

        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.feature_tree.bind("<Button-1>", self.on_tree_click)

        # Coordinate info
        coord_frame = ttk.LabelFrame(parent, text="Coordinates", padding=5)
        coord_frame.pack(fill=tk.X, padx=5, pady=5)

        self.coord_text = tk.Text(coord_frame, height=8, width=40)
        self.coord_text.pack(fill=tk.X)

    # Part 3: Section Viewer Tab

    def create_section_viewer_tab(self):
        """Create the enhanced section viewer tab with workflow buttons."""
        section_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(section_frame, text="Section Viewer")

        # Top control panel
        control_panel = ttk.Frame(section_frame)
        control_panel.pack(fill=tk.X, padx=5, pady=5)

        # WORKFLOW BUTTONS - Step 1: Import
        import_frame = ttk.LabelFrame(control_panel, text="1. Import", padding=3)
        import_frame.pack(side=tk.LEFT, padx=3)
        ttk.Button(import_frame, text="Import PDF(s)", command=self.open_pdf, width=13).pack(side=tk.LEFT, padx=2)
        ttk.Button(import_frame, text="Folder", command=self.open_folder, width=8).pack(side=tk.LEFT, padx=2)

        # WORKFLOW BUTTONS - Step 2: Process
        process_frame = ttk.LabelFrame(control_panel, text="2. Process", padding=3)
        process_frame.pack(side=tk.LEFT, padx=3)
        ttk.Button(process_frame, text=" Process All", command=self.process_all, width=11).pack(side=tk.LEFT, padx=2)

        # WORKFLOW BUTTONS - Step 3: Contacts
        contacts_frame = ttk.LabelFrame(control_panel, text="3. Contacts", padding=3)
        contacts_frame.pack(side=tk.LEFT, padx=3)
        ttk.Button(contacts_frame, text="Extract", command=self.extract_contacts_grouped, width=9).pack(side=tk.LEFT, padx=2)
        ttk.Button(contacts_frame, text="Tie Lines", command=self.switch_to_tie_lines_tab, width=10).pack(side=tk.LEFT, padx=2)

        # Navigation frame
        nav_frame = ttk.LabelFrame(control_panel, text="Navigation", padding=3)
        nav_frame.pack(side=tk.LEFT, fill=tk.X, padx=3)

        ttk.Label(nav_frame, text="Current Section:").pack(side=tk.LEFT, padx=5)
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(
            nav_frame, textvariable=self.section_var, state="readonly", width=20
        )
        self.section_combo.pack(side=tk.LEFT, padx=5)
        self.section_combo.bind("<<ComboboxSelected>>", self.on_section_changed)

        ttk.Button(nav_frame, text="< Previous", command=self.prev_section).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_frame, text="Next >", command=self.next_section).pack(side=tk.LEFT, padx=2)
        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(nav_frame, text="First", command=lambda: self.jump_to_section(0)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            nav_frame,
            text="Last",
            command=lambda: (
                self.jump_to_section(len(self.northings) - 1) if self.northings else None
            ),
        ).pack(side=tk.LEFT, padx=2)

        # Display options frame
        display_frame = ttk.LabelFrame(control_panel, text="Display Options", padding=5)
        display_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)

        ttk.Label(display_frame, text="Transparency:").pack(side=tk.LEFT, padx=5)
        alpha_scale = ttk.Scale(
            display_frame,
            from_=0.0,
            to=1.0,
            variable=self.pdf_alpha_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=lambda x: self.update_section_display(),
        )
        alpha_scale.pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(
            display_frame,
            text="Show PDF",
            variable=self.show_pdf_var,
            command=self.update_section_display,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            display_frame,
            text="Show Grid",
            variable=self.show_grid_var,
            command=self.update_section_display,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            display_frame,
            text="Hide Empty",
            variable=self.hide_empty_var,
            command=self.filter_sections,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(display_frame, text="Sections:").pack(side=tk.LEFT, padx=5)
        section_count_spin = ttk.Spinbox(
            display_frame,
            from_=1,
            to=5,
            textvariable=self.section_count_var,
            width=5,
            command=self.on_section_count_changed,
        )
        section_count_spin.pack(side=tk.LEFT, padx=5)

        # Main content area with sections and controls
        content_paned = ttk.PanedWindow(section_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: Section display
        section_display_frame = ttk.Frame(content_paned)
        content_paned.add(section_display_frame, weight=3)

        # Create matplotlib figure for sections
        self.fig_sections = plt.figure(figsize=(14, 8))

        canvas_frame = ttk.Frame(section_display_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_sections = FigureCanvasTkAgg(self.fig_sections, master=canvas_frame)
        canvas_widget = self.canvas_sections.get_tk_widget()

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_sections = NavigationToolbar2Tk(self.canvas_sections, toolbar_frame)
        self.toolbar_sections.update()

        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        canvas_widget.bind("<MouseWheel>", self.on_mousewheel)
        canvas_widget.bind("<Button-4>", self.on_mousewheel)  # Linux
        canvas_widget.bind("<Button-5>", self.on_mousewheel)  # Linux
        self.canvas_sections.mpl_connect("pick_event", self.on_pick)
        self.canvas_sections.mpl_connect("motion_notify_event", self.on_section_hover)
        
        # Tooltip annotation (hidden initially)
        self.tooltip_annotation = None

        # Right: Unit assignment controls (with scrollable content)
        assignment_outer = ttk.Frame(content_paned)
        content_paned.add(assignment_outer, weight=1)

        # Compact actions bar at TOP (always visible)
        actions_frame = ttk.Frame(assignment_outer)
        actions_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # Export actions (left side)
        export_frame = ttk.LabelFrame(actions_frame, text="Export", padding=2)
        export_frame.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(export_frame, text="DXF", width=5,
                   command=self._export_all_sections_dxf).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_frame, text="GeoTIFF", width=7,
                   command=self.export_geotiff).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_frame, text="To PDF", width=6,
                   command=self.write_assignments_to_pdf).pack(side=tk.LEFT, padx=1)
        
        # Config actions
        config_frame = ttk.LabelFrame(actions_frame, text="Config", padding=2)
        config_frame.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(config_frame, text="Save", width=5,
                   command=self.save_strat_column).pack(side=tk.LEFT, padx=1)
        ttk.Button(config_frame, text="Load", width=5,
                   command=self.load_strat_column).pack(side=tk.LEFT, padx=1)
        
        # Auto-assign
        ttk.Button(actions_frame, text="Auto-Name", width=9,
                   command=self._auto_assign_from_pdf_names).pack(side=tk.LEFT, padx=3)

        # Mode indicator (right side)
        self.mode_label = ttk.Label(actions_frame, text="Mode: View Only", font=("Arial", 9, "bold"))
        self.mode_label.pack(side=tk.RIGHT, padx=5)

        # Scrollable canvas for the rest of the controls
        right_canvas = tk.Canvas(assignment_outer, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(assignment_outer, orient="vertical", command=right_canvas.yview)
        assignment_frame = ttk.Frame(right_canvas)
        
        assignment_frame.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        right_canvas.create_window((0, 0), window=assignment_frame, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Enable mousewheel scrolling
        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel, add='+')

        # Feature browser - show all extracted items (consolidated)
        browser_frame = ttk.LabelFrame(assignment_frame, text="Extracted Features", padding=5)
        browser_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Feature tree showing all polygons, faults, and contacts
        feature_tree_frame = ttk.Frame(browser_frame)
        feature_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.section_feature_tree = ttk.Treeview(
            feature_tree_frame,
            columns=("type", "formation", "assigned"),
            height=10,
            selectmode=tk.EXTENDED
        )
        self.section_feature_tree.heading("#0", text="Name")
        self.section_feature_tree.heading("type", text="Type")
        self.section_feature_tree.heading("formation", text="Original")
        self.section_feature_tree.heading("assigned", text="Assigned")
        self.section_feature_tree.column("#0", width=120)
        self.section_feature_tree.column("type", width=60)
        self.section_feature_tree.column("formation", width=80)
        self.section_feature_tree.column("assigned", width=80)
        
        feature_scrollbar = ttk.Scrollbar(feature_tree_frame, orient=tk.VERTICAL, 
                                          command=self.section_feature_tree.yview)
        self.section_feature_tree.configure(yscrollcommand=feature_scrollbar.set)
        self.section_feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.section_feature_tree.bind("<<TreeviewSelect>>", self.on_feature_tree_select)
        
        # Consolidated selection controls (grouped)
        selection_controls_frame = ttk.Frame(browser_frame)
        selection_controls_frame.pack(fill=tk.X, pady=3)
        
        # Row 1: List controls
        row1 = ttk.Frame(selection_controls_frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Button(row1, text="Refresh", width=8,
                   command=self.refresh_feature_browser).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="Highlight", width=8,
                   command=self.highlight_selected_features).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="Select Similar", width=10,
                   command=self.select_similar).pack(side=tk.LEFT, padx=1)
        
        # Row 2: Selection actions
        row2 = ttk.Frame(selection_controls_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Button(row2, text="Clear Selection", width=12,
                   command=self.clear_selection).pack(side=tk.LEFT, padx=1)
        ttk.Button(row2, text="Unassign Selected", width=14,
                   command=self.unassign_selected).pack(side=tk.LEFT, padx=1)

        # Classification mode selection
        mode_frame = ttk.LabelFrame(assignment_frame, text="Classification Mode", padding=5)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="none")
        mode_row = ttk.Frame(mode_frame)
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(
            mode_row, text="View", variable=self.mode_var,
            value="none", command=self.on_classification_mode_changed
        ).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(
            mode_row, text="Polygon", variable=self.mode_var,
            value="polygon", command=self.on_classification_mode_changed
        ).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(
            mode_row, text="Fault", variable=self.mode_var,
            value="fault", command=self.on_classification_mode_changed
        ).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(
            mode_row, text="Contact", variable=self.mode_var,
            value="contact", command=self.on_classification_mode_changed
        ).pack(side=tk.LEFT, padx=3)

        # Units frame with prospect grouping
        units_frame = ttk.LabelFrame(assignment_frame, text="Geological Units (by Prospect)", padding=5)
        units_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

        # Prospect management buttons
        prospect_btn_frame = ttk.Frame(units_frame)
        prospect_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(prospect_btn_frame, text="+ Add Prospect", command=self.add_new_prospect).pack(side=tk.LEFT, padx=2)
        ttk.Button(prospect_btn_frame, text="+ Add Unit", command=self.add_new_unit).pack(side=tk.LEFT, padx=2)
        ttk.Button(prospect_btn_frame, text="Expand All", command=self.expand_all_prospects).pack(side=tk.RIGHT, padx=2)
        ttk.Button(prospect_btn_frame, text="Collapse All", command=self.collapse_all_prospects).pack(side=tk.RIGHT, padx=2)

        # Scrollable canvas for units
        self.units_canvas = tk.Canvas(units_frame, height=200, highlightthickness=0)
        units_scrollbar = ttk.Scrollbar(units_frame, orient="vertical", command=self.units_canvas.yview)
        self.units_inner = ttk.Frame(self.units_canvas)

        self.units_inner.bind(
            "<Configure>", lambda e: self.units_canvas.configure(scrollregion=self.units_canvas.bbox("all"))
        )

        self.units_canvas.create_window((0, 0), window=self.units_inner, anchor="nw")
        self.units_canvas.configure(yscrollcommand=units_scrollbar.set)

        self.units_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        units_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mousewheel scrolling on units canvas
        def _on_units_mousewheel(event):
            self.units_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.units_canvas.bind("<MouseWheel>", _on_units_mousewheel)
        self.units_inner.bind("<MouseWheel>", _on_units_mousewheel)
        
        # Track expanded/collapsed state per prospect
        self.prospect_expanded = {}  # {prospect_name: bool}

        # Faults frame
        faults_frame = ttk.LabelFrame(assignment_frame, text="Faults", padding=5)
        faults_frame.pack(fill=tk.X, padx=5, pady=5)

        faults_canvas = tk.Canvas(faults_frame, height=100)
        faults_scrollbar = ttk.Scrollbar(faults_frame, orient="vertical", command=faults_canvas.yview)
        self.faults_inner = ttk.Frame(faults_canvas)

        self.faults_inner.bind(
            "<Configure>", lambda e: faults_canvas.configure(scrollregion=faults_canvas.bbox("all"))
        )

        faults_canvas.create_window((0, 0), window=self.faults_inner, anchor="nw")
        faults_canvas.configure(yscrollcommand=faults_scrollbar.set)

        faults_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        faults_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(faults_frame, text="+ Add Fault", command=self.add_new_fault).pack(fill=tk.X, pady=2)

        # Assignment info (simplified)
        info_frame = ttk.Frame(assignment_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=3)
        
        self.assignment_label = ttk.Label(info_frame, text="Click a unit/fault button, then click items to assign")
        self.assignment_label.pack(side=tk.LEFT)
        
        self.count_label = ttk.Label(info_frame, text="0 selected", font=("Arial", 9, "bold"))
        self.count_label.pack(side=tk.RIGHT)
        
        # Hidden listboxes for compatibility (not displayed but used by other methods)
        self.selected_listbox = tk.Listbox(assignment_frame, height=0)
        self.polyline_listbox = tk.Listbox(assignment_frame, height=0)

        # Initialize buttons from strat column
        self.refresh_unit_buttons()
        self.refresh_fault_buttons()

    def extract_contacts_grouped(self):
        """Extract contacts using new proximity-based method with grouping."""
        if not self.all_geological_units:
            messagebox.showwarning("Warning", "No geological units found. Process PDFs first.")
            return
        
        try:
            self.status_var.set("Extracting contacts (proximity-based)...")
            self.root.update()
            
            # Build unit assignments from current state
            unit_assignments = {}
            for unit_name, unit_data in self.all_geological_units.items():
                if unit_data.get("unit_assignment"):
                    unit_assignments[unit_name] = unit_data["unit_assignment"]
            
            # Use new contact extractor
            self.grouped_contacts = extract_contacts_grouped(
                all_sections_data=self.all_sections_data,
                unit_assignments=unit_assignments,
                sample_distance=2.0,
                proximity_threshold=10.0,
                min_contact_length=5.0,
                simplify_tolerance=1.0
            )
            
            # Also populate legacy all_contacts for backward compatibility
            self.all_contacts = []
            for group_name, group in self.grouped_contacts.items():
                for polyline in group.polylines:
                    contact = {
                        "name": f"{group.formation1}-{group.formation2}_contact",
                        "formation1": group.formation1,
                        "formation2": group.formation2,
                        "vertices": polyline.vertices,
                        "northing": polyline.northing,
                        "section_key": polyline.section_key,
                        "group": group_name
                    }
                    self.all_contacts.append(contact)
            
            total_polylines = sum(len(g.polylines) for g in self.grouped_contacts.values())
            
            self.status_var.set(f"Extracted {len(self.grouped_contacts)} contact groups ({total_polylines} polylines)")
            
            # Update display
            self.update_section_display()
            
            messagebox.showinfo(
                "Contacts Extracted",
                f"Extracted {len(self.grouped_contacts)} contact groups\n"
                f"Total polylines: {total_polylines}\n\n"
                f"Use 'Open Tie Line Editor' to draw correlation ties between sections."
            )
            
        except Exception as e:
            logger.error(f"Error extracting grouped contacts: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to extract contacts: {str(e)}")
            self.status_var.set("Contact extraction failed")

    def open_tie_line_editor(self):
        """Open the tie line editor window."""
        if not self.grouped_contacts:
            # Try to extract contacts first
            result = messagebox.askyesno(
                "No Contacts",
                "No grouped contacts found.\n\n"
                "Would you like to extract contacts now?"
            )
            if result:
                self.extract_contacts_grouped()
            
            if not self.grouped_contacts:
                return
        
        # Open the tie line editor
        editor = open_tie_line_editor(
            parent_app=self,
            all_sections_data=self.all_sections_data,
            grouped_contacts=self.grouped_contacts,
            strat_column=self.strat_column
        )
    
    def switch_to_tie_lines_tab(self):
        """Switch to the tie lines tab and refresh display."""
        if not self.grouped_contacts:
            result = messagebox.askyesno(
                "No Contacts",
                "No contacts extracted yet.\n\nExtract contacts now?"
            )
            if result:
                self.extract_contacts_grouped()
            if not self.grouped_contacts:
                return
        
        # Refresh tie line tab
        self.refresh_tie_group_list()
        self.update_tie_display()
        
        # Switch to Tie Lines tab (index 1)
        self.main_notebook.select(1)

    def receive_contacts_with_ties(self, grouped_contacts):
        """
        Receive grouped contacts with tie lines from the tie line editor.
        Called when the editor is closed.
        """
        self.grouped_contacts = grouped_contacts
        
        # Update legacy format too
        self.all_contacts = []
        for group_name, group in self.grouped_contacts.items():
            for polyline in group.polylines:
                contact = {
                    "name": f"{group.formation1}-{group.formation2}_contact",
                    "formation1": group.formation1,
                    "formation2": group.formation2,
                    "vertices": polyline.vertices,
                    "northing": polyline.northing,
                    "section_key": polyline.section_key,
                    "group": group_name
                }
                self.all_contacts.append(contact)
        
        total_ties = sum(len(g.tie_lines) for g in self.grouped_contacts.values())
        logger.info(f"Received contacts with {total_ties} tie lines")
        
        self.update_section_display()

    # Part 3b: Tie Lines Tab (Embedded Editor)
    def create_tie_lines_tab(self):
        """Create the embedded tie lines tab."""
        tie_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(tie_frame, text="Tie Lines")

        # Initialize tie line state
        self.tie_current_contact_group = None
        self.tie_line_start = None
        self.drawing_tie = False
        self.tie_sections_to_display = 3
        self.tie_current_center_section = 0
        self.tie_contact_lines = {}
        self.tie_line_artists = []
        self.temp_tie_line = None

        # Top controls
        control_frame = ttk.Frame(tie_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Navigation
        nav_frame = ttk.LabelFrame(control_frame, text="Navigation", padding=3)
        nav_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(nav_frame, text="Section:").pack(side=tk.LEFT, padx=2)
        self.tie_section_combo = ttk.Combobox(nav_frame, state="readonly", width=12)
        self.tie_section_combo.pack(side=tk.LEFT, padx=2)
        self.tie_section_combo.bind("<<ComboboxSelected>>", self.on_tie_section_changed)

        ttk.Button(nav_frame, text="<", command=self.tie_prev_section, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text=">", command=self.tie_next_section, width=3).pack(side=tk.LEFT, padx=1)

        ttk.Label(nav_frame, text="Show:").pack(side=tk.LEFT, padx=5)
        self.tie_section_count_var = tk.IntVar(value=3)
        ttk.Spinbox(nav_frame, from_=1, to=5, textvariable=self.tie_section_count_var,
                   width=3, command=self.on_tie_section_count_changed).pack(side=tk.LEFT, padx=2)

        # Actions
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding=3)
        action_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(action_frame, text="Auto-Suggest", command=self.tie_auto_suggest).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Clear Ties", command=self.tie_clear_for_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Accept All", command=self.tie_accept_suggestions).pack(side=tk.LEFT, padx=2)

        # Export
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding=3)
        export_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(export_frame, text="Contacts + Ties DXF", command=self.export_contacts_dxf).pack(side=tk.LEFT, padx=2)

        # Status label
        self.tie_status_label = ttk.Label(control_frame, text="Select a contact group", 
                                          font=("Arial", 9, "italic"))
        self.tie_status_label.pack(side=tk.RIGHT, padx=10)

        # Main content
        content_paned = ttk.PanedWindow(tie_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: Section display
        display_frame = ttk.Frame(content_paned)
        content_paned.add(display_frame, weight=3)

        self.fig_ties = plt.figure(figsize=(12, 8))
        self.ax_ties = self.fig_ties.add_subplot(111)

        self.canvas_ties = FigureCanvasTkAgg(self.fig_ties, master=display_frame)
        self.canvas_ties.draw()
        self.canvas_ties.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tie_toolbar_frame = ttk.Frame(display_frame)
        tie_toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas_ties, tie_toolbar_frame)

        # Connect events
        self.canvas_ties.mpl_connect('button_press_event', self.on_tie_click)
        self.canvas_ties.mpl_connect('motion_notify_event', self.on_tie_motion)
        self.canvas_ties.mpl_connect('key_press_event', self.on_tie_key)

        # Right: Control panel
        control_right = ttk.Frame(content_paned, width=300)
        content_paned.add(control_right, weight=1)

        # Contact groups listbox
        group_frame = ttk.LabelFrame(control_right, text="Contact Groups", padding=5)
        group_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(group_frame, text="Select contact to draw ties:").pack(anchor=tk.W)

        list_frame = ttk.Frame(group_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tie_group_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE, height=12, font=('Arial', 9)
        )
        self.tie_group_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tie_group_listbox.yview)
        self.tie_group_listbox.bind('<<ListboxSelect>>', self.on_tie_group_selected)

        # Info
        info_frame = ttk.LabelFrame(control_right, text="Current Contact", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.tie_info_label = ttk.Label(info_frame, text="No contact selected")
        self.tie_info_label.pack(anchor=tk.W)

        # Instructions
        instr_frame = ttk.LabelFrame(control_right, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, padx=5, pady=5)
        instructions = "1. Select a contact group\n2. Click contact to start tie\n3. Click adjacent section to complete\n4. Escape to cancel"
        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT, font=('Arial', 9)).pack(anchor=tk.W)

        # Summary
        summary_frame = ttk.LabelFrame(control_right, text="Summary", padding=5)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        self.tie_summary_label = ttk.Label(summary_frame, text="No contacts extracted")
        self.tie_summary_label.pack(anchor=tk.W)

    # =========================================================================
    # TIE LINE TAB METHODS (Embedded Editor)
    # =========================================================================
    def refresh_tie_group_list(self):
        """Refresh the tie line group listbox."""
        self.tie_group_listbox.delete(0, tk.END)

        for group_name, group in self.grouped_contacts.items():
            section_count = len(group.get_sections())
            tie_count = len(group.tie_lines)
            self.tie_group_listbox.insert(
                tk.END, f"{group_name} ({section_count}s, {tie_count}t)"
            )

        # Update section combo
        if self.northings:
            self.tie_section_combo['values'] = [f"N {int(n)}" for n in self.northings]
            if self.tie_section_combo.current() < 0:
                self.tie_section_combo.current(0)

        self.update_tie_summary()

    def update_tie_summary(self):
        """Update tie lines summary."""
        if not self.grouped_contacts:
            self.tie_summary_label.config(text="No contacts extracted")
            return

        total_ties = sum(len(g.tie_lines) for g in self.grouped_contacts.values())
        groups_with_ties = sum(1 for g in self.grouped_contacts.values() if g.tie_lines)

        text = f"Contact groups: {len(self.grouped_contacts)}\n"
        text += f"Groups with ties: {groups_with_ties}\n"
        text += f"Total tie lines: {total_ties}"
        self.tie_summary_label.config(text=text)

    def update_tie_display(self):
        """Update the tie lines display."""
        self.ax_ties.clear()
        self.tie_contact_lines = {}
        self.tie_line_artists = []

        if not self.northings:
            self.canvas_ties.draw()
            return

        # Get sections to display
        start_idx = self.tie_current_center_section
        end_idx = min(start_idx + self.tie_sections_to_display, len(self.northings))
        sections_to_show = self.northings[start_idx:end_idx]

        if not sections_to_show:
            self.canvas_ties.draw()
            return

        # Calculate ranges from section data
        easting_min = float('inf')
        easting_max = float('-inf')
        rl_min = float('inf')
        rl_max = float('-inf')

        for data in self.all_sections_data.values():
            for unit in data.get("units", {}).values():
                vertices = unit.get("vertices", [])
                for i in range(0, len(vertices), 2):
                    if i + 1 < len(vertices):
                        e, rl = vertices[i], vertices[i + 1]
                        if e > 10000:  # Real-world coordinates
                            easting_min = min(easting_min, e)
                            easting_max = max(easting_max, e)
                            rl_min = min(rl_min, rl)
                            rl_max = max(rl_max, rl)

        if easting_min == float('inf'):
            self.canvas_ties.draw()
            return

        rl_range = rl_max - rl_min
        section_spacing = rl_range * 1.3

        # Plot each section
        for idx, northing in enumerate(sections_to_show):
            y_offset = -idx * section_spacing

            # Find section data
            section_data = None
            for (pdf, page), data in self.all_sections_data.items():
                data_northing = data.get("northing")
                if data_northing is not None and abs(data_northing - northing) < 0.1:
                    section_data = data
                    break

            if section_data is None:
                continue

            # Plot units (faded, non-interactive)
            for unit in section_data.get("units", {}).values():
                vertices = unit.get("vertices", [])
                if len(vertices) >= 6:
                    coords = []
                    for j in range(0, len(vertices), 2):
                        if j + 1 < len(vertices):
                            coords.append((vertices[j], vertices[j + 1] + y_offset))

                    if coords:
                        color = unit.get("color", (0.5, 0.5, 0.5))
                        if isinstance(color, list):
                            color = tuple(color)
                        patch = MplPolygon(coords, facecolor=color, edgecolor='gray',
                                          linewidth=0.3, alpha=0.25)
                        self.ax_ties.add_patch(patch)

            # Plot contacts (interactive)
            self._plot_tie_contacts(northing, y_offset)

            # Section label
            self.ax_ties.text(
                easting_min - 50, rl_min + y_offset + rl_range / 2,
                f"N {int(northing)}", fontsize=10, fontweight='bold',
                ha='right', va='center'
            )

        # Plot existing tie lines
        self._plot_existing_ties(sections_to_show, section_spacing, rl_min)

        self.ax_ties.set_xlim(easting_min - 150, easting_max + 100)
        self.ax_ties.set_ylim(
            rl_min - (len(sections_to_show) - 1) * section_spacing - 100,
            rl_max + 100
        )
        self.ax_ties.set_xlabel("Easting (m)")
        self.ax_ties.set_ylabel("RL (m) - stacked by section")
        self.ax_ties.set_aspect('equal')

        self.canvas_ties.draw()

    def _plot_tie_contacts(self, northing, y_offset):
        """Plot contacts for tie line editing."""
        for group_name, group in self.grouped_contacts.items():
            is_active = group_name == self.tie_current_contact_group

            for polyline in group.polylines:
                if abs(polyline.northing - northing) < 0.1:
                    vertices = polyline.vertices
                    if len(vertices) >= 4:
                        x = [vertices[j] for j in range(0, len(vertices), 2)]
                        y = [vertices[j + 1] + y_offset for j in range(0, len(vertices), 2)]

                        color = '#FF00FF' if is_active else '#00AA00'
                        linewidth = 3 if is_active else 2

                        line, = self.ax_ties.plot(x, y, color=color, linewidth=linewidth,
                                                  alpha=0.9, picker=5)

                        self.tie_contact_lines[line] = {
                            'group': group_name,
                            'polyline': polyline,
                            'northing': northing,
                            'y_offset': y_offset
                        }

    def _plot_existing_ties(self, sections_to_show, section_spacing, rl_min):
        """Plot existing tie lines."""
        if not self.tie_current_contact_group:
            return

        group = self.grouped_contacts.get(self.tie_current_contact_group)
        if not group:
            return

        for tie in group.tie_lines:
            n1 = tie.get('northing1')
            n2 = tie.get('northing2')
            e1, rl1 = tie.get('easting1'), tie.get('rl1')
            e2, rl2 = tie.get('easting2'), tie.get('rl2')

            # Find y offsets
            y1, y2 = None, None
            for idx, northing in enumerate(sections_to_show):
                y_off = -idx * section_spacing
                if abs(northing - n1) < 0.1:
                    y1 = rl1 + y_off
                if abs(northing - n2) < 0.1:
                    y2 = rl2 + y_off

            if y1 is not None and y2 is not None:
                color = '#0088AA' if tie.get('auto_suggested') else '#00FFFF'
                line, = self.ax_ties.plot([e1, e2], [y1, y2], color=color,
                                         linewidth=1.5, linestyle='--', alpha=0.8)
                self.tie_line_artists.append(line)

    def on_tie_click(self, event):
        """Handle click in tie line editor."""
        if event.inaxes != self.ax_ties:
            return

        # Check if clicked on a contact
        for line, info in self.tie_contact_lines.items():
            contains, _ = line.contains(event)
            if contains:
                if not self.drawing_tie:
                    # Start new tie - snap to nearest vertex
                    raw_rl = event.ydata - info['y_offset']
                    snapped_e, snapped_rl = self._snap_to_nearest_vertex(
                        event.xdata, raw_rl, info['group'], info['northing']
                    )
                    
                    self.tie_current_contact_group = info['group']
                    self.tie_line_start = {
                        'northing': info['northing'],
                        'easting': snapped_e,
                        'rl': snapped_rl,
                        'y_offset': info['y_offset']
                    }
                    self.drawing_tie = True
                    self.tie_status_label.config(text=f"Drawing tie from N{int(info['northing'])}")

                    # Update listbox selection
                    group_names = list(self.grouped_contacts.keys())
                    try:
                        idx = group_names.index(info['group'])
                        self.tie_group_listbox.selection_clear(0, tk.END)
                        self.tie_group_listbox.selection_set(idx)
                    except ValueError:
                        pass

                    self.update_tie_display()
                else:
                    # Complete tie
                    if info['group'] == self.tie_current_contact_group:
                        start_northing = self.tie_line_start['northing']
                        end_northing = info['northing']

                        if abs(start_northing - end_northing) > 0.1:
                            # Snap end point to nearest vertex
                            raw_rl = event.ydata - info['y_offset']
                            snapped_e, snapped_rl = self._snap_to_nearest_vertex(
                                event.xdata, raw_rl, info['group'], end_northing
                            )
                            
                            # Add tie line
                            tie = {
                                'northing1': start_northing,
                                'easting1': self.tie_line_start['easting'],
                                'rl1': self.tie_line_start['rl'],
                                'northing2': end_northing,
                                'easting2': snapped_e,
                                'rl2': snapped_rl,
                                'auto_suggested': False
                            }

                            group = self.grouped_contacts[self.tie_current_contact_group]
                            group.tie_lines.append(tie)

                            self.tie_status_label.config(text="Added tie line")
                            self.update_tie_listbox_item()
                            self.update_tie_summary()

                    self.drawing_tie = False
                    self.tie_line_start = None
                    if self.temp_tie_line:
                        self.temp_tie_line.remove()
                        self.temp_tie_line = None
                    self.update_tie_display()
                return

    def on_tie_motion(self, event):
        """Handle mouse motion in tie editor."""
        if self.drawing_tie and self.tie_line_start and event.inaxes == self.ax_ties:
            # Draw temporary line
            if self.temp_tie_line:
                self.temp_tie_line.remove()

            y_start = self.tie_line_start['rl'] + self.tie_line_start['y_offset']

            self.temp_tie_line, = self.ax_ties.plot(
                [self.tie_line_start['easting'], event.xdata],
                [y_start, event.ydata],
                'c--', linewidth=1.5, alpha=0.7
            )
            self.canvas_ties.draw()

    def on_tie_key(self, event):
        """Handle key press in tie editor."""
        if event.key == 'escape':
            self.cancel_tie()

    def cancel_tie(self):
        """Cancel tie line in progress."""
        self.drawing_tie = False
        self.tie_line_start = None
        if self.temp_tie_line:
            self.temp_tie_line.remove()
            self.temp_tie_line = None
        self.tie_status_label.config(text="Tie cancelled")
        self.update_tie_display()

    def _snap_to_nearest_vertex(self, easting: float, rl: float, contact_group: str, northing: float) -> Tuple[float, float]:
        """
        Snap a click point to the nearest contact polyline vertex.
        
        Args:
            easting: Clicked easting coordinate
            rl: Clicked RL coordinate
            contact_group: Name of the contact group
            northing: Northing value of the section
            
        Returns:
            Tuple of (snapped_easting, snapped_rl)
        """
        if contact_group not in self.grouped_contacts:
            return (easting, rl)
        
        group = self.grouped_contacts[contact_group]
        
        # Get all polylines for this northing
        polylines = group.get_polylines_for_section(northing, tolerance=0.1)
        
        if not polylines:
            return (easting, rl)
        
        # Find nearest vertex across all polylines
        min_dist = float('inf')
        nearest_point = (easting, rl)
        
        for polyline in polylines:
            coords = polyline.get_coords()
            for coord_e, coord_rl in coords:
                dist = ((coord_e - easting)**2 + (coord_rl - rl)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = (coord_e, coord_rl)
        
        # Only snap if within reasonable distance (e.g., 50 meters)
        if min_dist < 50:
            return nearest_point
        else:
            return (easting, rl)
    
    def on_tie_group_selected(self, event):
        """Handle contact group selection."""
        selection = self.tie_group_listbox.curselection()
        if selection:
            idx = selection[0]
            group_names = list(self.grouped_contacts.keys())
            if idx < len(group_names):
                self.tie_current_contact_group = group_names[idx]
                self.cancel_tie()

                group = self.grouped_contacts[self.tie_current_contact_group]
                self.tie_info_label.config(
                    text=f"Group: {self.tie_current_contact_group}\n"
                         f"Sections: {len(group.get_sections())}\n"
                         f"Ties: {len(group.tie_lines)}"
                )
                self.update_tie_display()

    def on_tie_section_changed(self, event):
        """Handle section change in tie editor."""
        selection = self.tie_section_combo.current()
        if selection >= 0:
            self.tie_current_center_section = selection
            self.cancel_tie()
            self.update_tie_display()

    def on_tie_section_count_changed(self):
        """Handle section count change."""
        self.tie_sections_to_display = self.tie_section_count_var.get()
        self.cancel_tie()
        self.update_tie_display()

    def tie_prev_section(self):
        """Move to previous section in tie editor."""
        if self.tie_current_center_section > 0:
            self.tie_current_center_section -= 1
            self.tie_section_combo.current(self.tie_current_center_section)
            self.cancel_tie()
            self.update_tie_display()

    def tie_next_section(self):
        """Move to next section in tie editor."""
        if self.tie_current_center_section < len(self.northings) - 1:
            self.tie_current_center_section += 1
            self.tie_section_combo.current(self.tie_current_center_section)
            self.cancel_tie()
            self.update_tie_display()

    def tie_auto_suggest(self):
        """Auto-suggest tie lines for current contact group."""
        if not self.tie_current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return

        group = self.grouped_contacts.get(self.tie_current_contact_group)
        if not group:
            return

        # Simple auto-suggestion: connect endpoints on adjacent sections
        sections = sorted(group.get_sections(), reverse=True)

        suggestions = []
        for i in range(len(sections) - 1):
            n1 = sections[i]
            n2 = sections[i + 1]

            # Find polylines on each section
            pl1 = [p for p in group.polylines if abs(p.northing - n1) < 0.1]
            pl2 = [p for p in group.polylines if abs(p.northing - n2) < 0.1]

            if pl1 and pl2:
                v1 = pl1[0].vertices
                v2 = pl2[0].vertices

                if len(v1) >= 2 and len(v2) >= 2:
                    # Start points
                    suggestions.append({
                        'northing1': n1, 'easting1': v1[0], 'rl1': v1[1],
                        'northing2': n2, 'easting2': v2[0], 'rl2': v2[1],
                        'auto_suggested': True
                    })

                    # End points
                    suggestions.append({
                        'northing1': n1, 'easting1': v1[-2], 'rl1': v1[-1],
                        'northing2': n2, 'easting2': v2[-2], 'rl2': v2[-1],
                        'auto_suggested': True
                    })

        if suggestions:
            group.tie_lines.extend(suggestions)
            self.tie_status_label.config(text=f"Added {len(suggestions)} suggested ties")
            self.update_tie_display()
            self.update_tie_listbox_item()
            self.update_tie_summary()
        else:
            messagebox.showinfo("No Suggestions", "Could not auto-suggest ties.")

    def tie_accept_suggestions(self):
        """Accept all auto-suggested ties."""
        if not self.tie_current_contact_group:
            return

        group = self.grouped_contacts.get(self.tie_current_contact_group)
        if group:
            for tie in group.tie_lines:
                tie['auto_suggested'] = False

        self.update_tie_display()
        self.tie_status_label.config(text="All suggestions accepted")

    def tie_clear_for_group(self):
        """Clear ties for current group."""
        if not self.tie_current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return

        if not messagebox.askyesno("Confirm", f"Clear all ties for {self.tie_current_contact_group}?"):
            return

        group = self.grouped_contacts.get(self.tie_current_contact_group)
        if group:
            group.tie_lines = []

        self.update_tie_display()
        self.update_tie_listbox_item()
        self.update_tie_summary()
        self.tie_status_label.config(text="Ties cleared")

    def update_tie_listbox_item(self):
        """Update the listbox item for current group."""
        if not self.tie_current_contact_group:
            return

        group_names = list(self.grouped_contacts.keys())
        try:
            idx = group_names.index(self.tie_current_contact_group)
            group = self.grouped_contacts[self.tie_current_contact_group]
            section_count = len(group.get_sections())
            tie_count = len(group.tie_lines)

            self.tie_group_listbox.delete(idx)
            self.tie_group_listbox.insert(idx, f"{self.tie_current_contact_group} ({section_count}s, {tie_count}t)")
            self.tie_group_listbox.selection_set(idx)
        except ValueError:
            pass

    def export_contacts_dxf(self):
        """Export contacts AND tie lines to DXF (all groups with layers, ties included)."""
        if not self.grouped_contacts:
            messagebox.showwarning("No Contacts", "No contacts to export.")
            return

        total_ties = sum(len(g.tie_lines) for g in self.grouped_contacts.values())
        
        filepath = filedialog.asksaveasfilename(
            title="Export Contacts & Tie Lines DXF",
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            contacts_exported = 0
            ties_exported = 0
            
            with open(filepath, "w") as f:
                # DXF header
                f.write("0\nSECTION\n2\nENTITIES\n")
                
                for group_name, group in self.grouped_contacts.items():
                    # Sanitize layer name for contacts
                    contact_layer = group_name.replace("-", "_").replace(" ", "_").replace("/", "_")[:31]
                    tie_layer = f"{contact_layer}_ties"[:31]
                    
                    # Export contact POINTS (not connected polylines - avoids spike artifacts)
                    for polyline in group.polylines:
                        vertices = polyline.vertices
                        northing = polyline.northing
                        
                        if northing is None or len(vertices) < 4:
                            continue
                        
                        # Write each contact point as a separate POINT entity
                        for i in range(0, len(vertices), 2):
                            if i + 1 < len(vertices):
                                easting = vertices[i]
                                rl = vertices[i + 1]
                                if easting is not None and rl is not None:
                                    f.write("0\nPOINT\n")
                                    f.write(f"8\n{contact_layer}\n")
                                    f.write(f"10\n{float(easting):.2f}\n")  # X = Easting
                                    f.write(f"20\n{float(northing):.2f}\n")  # Y = Northing
                                    f.write(f"30\n{float(rl):.2f}\n")  # Z = RL
                                    contacts_exported += 1
                    
                    # Export tie lines for this group (snapped to contact vertices)
                    for tie in group.tie_lines:
                        from_point = tie.get('from_point')
                        to_point = tie.get('to_point')
                        n1 = tie.get('from_northing')
                        n2 = tie.get('to_northing')
                        
                        if from_point is None or to_point is None or n1 is None or n2 is None:
                            continue
                        if len(from_point) < 2 or len(to_point) < 2:
                            continue
                        
                        e1, rl1 = from_point[0], from_point[1]
                        e2, rl2 = to_point[0], to_point[1]
                        
                        # Write 3D LINE entity for tie
                        f.write("0\nLINE\n")
                        f.write(f"8\n{tie_layer}\n")
                        f.write(f"10\n{float(e1):.2f}\n")   # Start X (Easting)
                        f.write(f"20\n{float(n1):.2f}\n")   # Start Y (Northing)
                        f.write(f"30\n{float(rl1):.2f}\n")  # Start Z (RL)
                        f.write(f"11\n{float(e2):.2f}\n")   # End X (Easting)
                        f.write(f"21\n{float(n2):.2f}\n")   # End Y (Northing)
                        f.write(f"31\n{float(rl2):.2f}\n")  # End Z (RL)
                        
                        ties_exported += 1
                
                # DXF footer
                f.write("0\nENDSEC\n0\nEOF\n")
            
            self.status_var.set(f"Exported {contacts_exported} contacts, {ties_exported} ties to {filepath}")
            messagebox.showinfo(
                "Export Complete", 
                f"Exported to:\n{filepath}\n\n"
                f"Contact points: {contacts_exported}\n"
                f"Tie lines: {ties_exported}\n\n"
                f"Layers created per contact group:\n"
                f"  - [group_name] = contact points\n"
                f"  - [group_name]_ties = tie lines"
            )

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Export failed: {e}")

    def export_tie_lines_dxf(self):
        """Export tie lines to DXF (manual format, no external library needed)."""
        if not self.grouped_contacts:
            messagebox.showwarning("No Data", "No contacts extracted.")
            return

        total_ties = sum(len(g.tie_lines) for g in self.grouped_contacts.values())
        if total_ties == 0:
            messagebox.showwarning("No Ties", "No tie lines to export. Draw some ties first.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Tie Lines DXF",
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            ties_exported = 0
            
            with open(filepath, "w") as f:
                # DXF header
                f.write("0\nSECTION\n2\nENTITIES\n")
                
                for group_name, group in self.grouped_contacts.items():
                    # Sanitize layer name
                    layer_name = f"{group_name}_ties".replace("-", "_").replace(" ", "_").replace("/", "_")[:31]
                    
                    for tie in group.tie_lines:
                        # Extract coordinates from the actual data structure
                        # Tie lines store: from_point=(easting, rl), from_northing, to_point=(easting, rl), to_northing
                        from_point = tie.get('from_point')
                        to_point = tie.get('to_point')
                        n1 = tie.get('from_northing')
                        n2 = tie.get('to_northing')
                        
                        # Skip if any coordinate is missing
                        if from_point is None or to_point is None or n1 is None or n2 is None:
                            continue
                        if len(from_point) < 2 or len(to_point) < 2:
                            continue
                        
                        e1, rl1 = from_point[0], from_point[1]
                        e2, rl2 = to_point[0], to_point[1]
                        
                        # Write 3D LINE entity
                        f.write("0\nLINE\n")
                        f.write(f"8\n{layer_name}\n")
                        f.write(f"10\n{float(e1):.2f}\n")   # Start X (Easting)
                        f.write(f"20\n{float(n1):.2f}\n")   # Start Y (Northing)
                        f.write(f"30\n{float(rl1):.2f}\n")  # Start Z (RL)
                        f.write(f"11\n{float(e2):.2f}\n")   # End X (Easting)
                        f.write(f"21\n{float(n2):.2f}\n")   # End Y (Northing)
                        f.write(f"31\n{float(rl2):.2f}\n")  # End Z (RL)
                        
                        ties_exported += 1
                
                # DXF footer
                f.write("0\nENDSEC\n0\nEOF\n")
            
            self.status_var.set(f"Exported {ties_exported} tie lines to {filepath}")
            messagebox.showinfo("Export Complete", f"Exported {ties_exported} tie lines to:\n{filepath}")

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Export failed: {e}")

    # Part 4: 3D Visualization Tab
    def create_3d_tab(self):
        """Create the 3D visualization tab."""
        view3d_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(view3d_frame, text="3D View")

        # Control panel
        control_frame = ttk.Frame(view3d_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Display Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            control_frame,
            text="Wireframe",
            variable=self.display_mode_3d,
            value="wireframe",
            command=self.update_3d_display,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            control_frame,
            text="Solid",
            variable=self.display_mode_3d,
            value="solid",
            command=self.update_3d_display,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            control_frame,
            text="Transparent",
            variable=self.display_mode_3d,
            value="transparent",
            command=self.update_3d_display,
        ).pack(side=tk.LEFT)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Button(
            control_frame, text="Create Solids", command=self.create_solids_from_selected
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export 3D Model", command=self.export_3d_model).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Reset View", command=self.reset_3d_view).pack(
            side=tk.LEFT, padx=5
        )

        # 3D figure
        self.fig_3d = plt.figure(figsize=(12, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")

        canvas_frame = ttk.Frame(view3d_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, canvas_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, canvas_frame)
        toolbar_3d.update()

    def update_display(self):
        """Update the current display based on active tab."""
        current_tab = self.main_notebook.index(self.main_notebook.select())
        if current_tab == 0:  # Section viewer (was PDF viewer)
            self.update_section_display()
        elif current_tab == 1:  # Tie lines tab
            self.update_tie_display()
        elif current_tab == 2:  # 3D view
            self.update_3d_display()

    # Part 5: Core Processing Methods
    def open_pdf(self):
        """Open PDF file(s) and auto-process."""
        filenames = filedialog.askopenfilenames(
            title="Select PDF file(s)", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if filenames:
            for filename in filenames:
                pdf_path = Path(filename)
                if pdf_path not in self.pdf_list:
                    self.pdf_list.append(pdf_path)

            if filenames:
                first_new = Path(filenames[0])
                self.current_pdf_index = self.pdf_list.index(first_new)
                self.load_pdf_at_index(self.current_pdf_index)
                
                # Auto-process all loaded PDFs
                self.root.after(100, self._auto_process_after_load)

    def open_folder(self):
        """Open folder containing PDFs for batch processing."""
        folder_path = filedialog.askdirectory(title="Select folder containing PDF files")

        if folder_path:
            folder = Path(folder_path)
            pdf_files = list(folder.glob("*.pdf"))

            if not pdf_files:
                messagebox.showwarning("No PDFs", "No PDF files found in selected folder")
                return

            self.pdf_list = pdf_files
            if pdf_files:
                self.current_pdf_index = 0
                self.load_pdf_at_index(0)

            self.status_var.set(f"Loaded {len(pdf_files)} PDFs from folder")
            
            # Auto-process all loaded PDFs
            self.root.after(100, self._auto_process_after_load)
    
    def _auto_process_after_load(self):
        """Auto-process PDFs after loading (called with slight delay for UI responsiveness)."""
        if self.pdf_list:
            self.status_var.set("Auto-processing loaded PDFs...")
            self.root.update()
            self.process_all()
            
            # Auto-extract contacts if we have units
            if self.all_geological_units and not self.grouped_contacts:
                self.root.after(100, self._auto_extract_contacts_after_process)
    
    def _auto_extract_contacts_after_process(self):
        """Auto-extract contacts after processing (if assignments exist)."""
        # Only auto-extract if we have some assigned units
        has_assignments = any(
            unit.get("unit_assignment") 
            for unit in self.all_geological_units.values()
        )
        if has_assignments:
            self.status_var.set("Auto-extracting contacts...")
            self.root.update()
            self.extract_contacts_grouped()

    def load_pdf_at_index(self, index):
        """Load PDF at given index in the list."""
        if 0 <= index < len(self.pdf_list):
            try:
                self.pdf_path = self.pdf_list[index]
                if self.current_pdf:
                    self.current_pdf.close()

                self.current_pdf = fitz.open(str(self.pdf_path))

                if len(self.current_pdf) > 0:
                    self.current_page_num = 0
                    self.current_page = self.current_pdf[0]
                    # PDF viewer tab removed - skip UI updates
                    # self.update_page_label()
                    # self.update_pdf_display()
                    self.status_var.set(f"Loaded: {self.pdf_path.name}")
                    logger.info(f"Opened PDF: {self.pdf_path}")

                    # Auto-process coordinates and features
                    self.detect_coordinates(show_message=False)
                    self.extract_features(show_message=False)
                else:
                    messagebox.showwarning("Warning", "PDF has no pages")

            except Exception as e:
                logger.error(f"Failed to open PDF: {e}")
                messagebox.showerror("Error", f"Failed to open PDF: {str(e)}")

    def process_all(self):
        """Process all pages of all loaded PDFs."""
        if not self.pdf_list:
            messagebox.showwarning("Warning", "Please open PDF files first")
            return

        try:
            # Apply current extraction filter to feature extractor
            self.feature_extractor.set_extraction_filter(self.extraction_filter)
            
            self.all_sections_data = {}
            self.all_geological_units = {}
            self.all_contacts = []

            total_features = 0
            total_contacts = 0
            processed_pages = 0

            for pdf_path in self.pdf_list:
                self.status_var.set(f"Processing {pdf_path.name}...")
                self.root.update()

                doc = fitz.open(str(pdf_path))

                for page_num in range(len(doc)):
                    page = doc[page_num]

                    self.status_var.set(
                        f"Processing {pdf_path.name} - Page {page_num + 1}/{len(doc)}..."
                    )
                    self.root.update()

                    # Detect coordinates
                    coord_system = self.georeferencer.detect_coordinates(page, pdf_path)

                    # Extract features (uses the extraction_filter set above)
                    annotations = self.feature_extractor.extract_annotations(page)
                    self.feature_extractor.number_geological_units(annotations)

                    # Build transformation
                    transform = None
                    if coord_system:
                        self.georeferencer.coord_system = coord_system
                        transform = self.georeferencer.build_transformation()

                    # Process units
                    page_units = {}

                    # Section-level northing for all units/contacts on this page
                    section_northing = coord_system.get("northing") if coord_system else None

                    for unit_name, unit in self.feature_extractor.geological_units.items():
                        unique_name = f"{pdf_path.stem}_P{page_num+1}_{unit_name}"
                        unit_copy = unit.copy()

                        # Store original PDF vertices
                        unit_copy["pdf_vertices"] = unit_copy.get("vertices", []).copy()

                        # Transform to real-world coordinates
                        if transform and "vertices" in unit_copy:
                            pdf_vertices = unit_copy["vertices"]
                            rw_vertices = []

                            for i in range(0, len(pdf_vertices), 2):
                                if i + 1 < len(pdf_vertices):
                                    pdf_x = pdf_vertices[i]
                                    pdf_y = pdf_vertices[i + 1]
                                    easting, northing, rl = transform(pdf_x, pdf_y)
                                    rw_vertices.extend([easting, rl])

                            # Set vertices AFTER the loop completes (was incorrectly indented before)
                            unit_copy["vertices"] = rw_vertices
                            unit_copy["coordinate_system"] = "real_world"
                        else:
                            unit_copy["coordinate_system"] = "pdf"

                        unit_copy["source_pdf"] = str(pdf_path)
                        unit_copy["page_num"] = page_num
                        unit_copy["original_name"] = unit_name
                        unit_copy["unique_name"] = unique_name
                        
                        # Attach section northing so 3D solids know where this polygon sits in space
                        if coord_system and "northing" in coord_system:
                            unit_copy["northing"] = coord_system["northing"]
                        else:
                            unit_copy["northing"] = section_northing

                        page_units[unique_name] = unit_copy
                        self.all_geological_units[unique_name] = unit_copy


                    total_features += len(page_units)

                    # Extract polylines (PolyLines and Faults)
                    page_polylines = {}
                    fault_count = 0
                    polyline_count = 0
                    for annot_idx, annot in enumerate(annotations):
                        annot_type = annot.get("type", "")
                        
                        # Include both PolyLine types AND Fault types
                        if annot_type in ("PolyLine", "Fault"):
                            is_fault = (annot_type == "Fault")
                            
                            # Get the annotation name (author field)
                            annot_name = annot.get("name", "")
                            if not annot_name:
                                annot_name = annot.get("author", "")
                            
                            if is_fault:
                                fault_count += 1
                                fault_name = annot_name if annot_name else f"Fault{fault_count}"
                                polyline_name = f"F{fault_count}_{fault_name}"
                            else:
                                polyline_count += 1
                                # Use annotation name for non-fault polylines too
                                polyline_name = annot_name if annot_name else f"PL{polyline_count}"
                            
                            unique_polyline_name = f"{pdf_path.stem}_P{page_num+1}_{polyline_name}"

                            polyline_data = {
                                "name": unique_polyline_name,
                                "vertices": annot["vertices"].copy() if not transform else [],
                                "pdf_vertices": annot["vertices"].copy(),
                                "color": annot.get("color", (1, 0, 0) if is_fault else (0, 0, 0)),
                                "type": annot_type,
                                "is_fault": is_fault,
                                "fault_name": annot_name if annot_name else None,
                                "fault_assignment": annot_name if annot_name else None,
                                "author": annot.get("author", ""),  # Store author for auto-assignment
                            }

                            # Transform polyline vertices if transform available
                            if transform:
                                pdf_vertices = annot["vertices"]
                                rw_vertices = []
                                for i in range(0, len(pdf_vertices), 2):
                                    if i + 1 < len(pdf_vertices):
                                        pdf_x = pdf_vertices[i]
                                        pdf_y = pdf_vertices[i + 1]
                                        easting, northing, rl = transform(pdf_x, pdf_y)
                                        rw_vertices.extend([easting, rl])
                                polyline_data["vertices"] = rw_vertices

                            page_polylines[unique_polyline_name] = polyline_data
                    
                    if fault_count > 0:
                        logger.info(f"  Extracted {fault_count} faults, {polyline_count} other polylines")

                    # Extract contacts using new centerline method
                    page_contacts = []
                    if self.feature_extractor.geological_units:
                        # Build temporary section data for contact extraction
                        temp_section_data = {
                            (pdf_path, page_num): {
                                "units": dict(self.feature_extractor.geological_units),
                                "northing": section_northing,
                                "faults": list(page_polylines.values()),
                            }
                        }
                        
                        # Use new grouped contact extractor
                        grouped = extract_contacts_grouped(
                            temp_section_data,
                            sample_distance=2.0,
                            proximity_threshold=10.0,
                            min_contact_length=5.0,
                            simplify_tolerance=1.0
                        )
                        
                        # Flatten grouped contacts to list format
                        for group_name, group in grouped.items():
                            for polyline in group.polylines:
                                contact = {
                                    "name": group_name,
                                    "vertices": polyline.vertices,
                                    "formation1": group.formation1,
                                    "formation2": group.formation2,
                                    "source_pdf": str(pdf_path),
                                    "page_num": page_num,
                                    "unique_name": f"{pdf_path.stem}_P{page_num+1}_{group_name}",
                                    "northing": section_northing,
                                }
                                page_contacts.append(contact)
                                self.all_contacts.append(contact)

                        total_contacts += len(page_contacts)

                    # Build inverse transform for PDF coordinate conversion
                    inverse_transform = self.georeferencer.build_inverse_transformation()
                    
                    # Store section data
                    self.all_sections_data[(pdf_path, page_num)] = {
                        "coord_system": coord_system,
                        "units": page_units,
                        "contacts": page_contacts,
                        "polylines": page_polylines,
                        "northing": coord_system.get("northing") if coord_system else None,
                        "easting_min": coord_system.get("easting_min") if coord_system else None,
                        "easting_max": coord_system.get("easting_max") if coord_system else None,
                        "rl_min": coord_system.get("rl_min") if coord_system else None,
                        "rl_max": coord_system.get("rl_max") if coord_system else None,
                        "pdf_path": str(pdf_path),
                        "page_num": page_num,
                        "inverse_transform": inverse_transform,
                    }

                    processed_pages += 1

                    # Clear for next page
                    self.feature_extractor.geological_units = {}
                    self.feature_extractor.contacts = []

                doc.close()

            # Calculate global ranges
            self.calculate_global_ranges()
            
            # Auto-assign based on PDF annotation names
            self._auto_assign_from_pdf_names()

            # Update displays
            self.populate_feature_tree_all()
            self.filter_sections()
            self.update_section_display()

            self.status_var.set("Processing complete! Ready for analysis.")

            summary = f"Processing complete!\n\n"
            summary += f"Processed: {processed_pages} pages from {len(self.pdf_list)} PDFs\n"
            summary += f"Total geological units: {total_features}\n"
            summary += f"Total contacts: {total_contacts}\n\n"
            summary += "Switch to 'Section Viewer' tab to assign units"

            messagebox.showinfo("Processing Complete", summary)

        except Exception as e:
            logger.error(f"Error in process_all: {e}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")

    # Part 6: Section Viewer Methods
    def calculate_global_ranges(self):
        """Calculate the global min/max for eastings and RLs across all sections."""
        self.global_easting_min = float("inf")
        self.global_easting_max = float("-inf")
        self.global_rl_min = float("inf")
        self.global_rl_max = float("-inf")

        self.content_easting_min = float("inf")
        self.content_easting_max = float("-inf")
        self.content_rl_min = float("inf")
        self.content_rl_max = float("-inf")

        for (pdf, page), section_data in self.all_sections_data.items():
            easting_min = section_data.get("easting_min")
            easting_max = section_data.get("easting_max")
            rl_min = section_data.get("rl_min")
            rl_max = section_data.get("rl_max")

            if easting_min is not None and easting_max is not None:
                self.global_easting_min = min(self.global_easting_min, easting_min)
                self.global_easting_max = max(self.global_easting_max, easting_max)

            if rl_min is not None and rl_max is not None:
                self.global_rl_min = min(self.global_rl_min, rl_min)
                self.global_rl_max = max(self.global_rl_max, rl_max)

            # Check actual unit vertices
            for unit_name, unit in section_data.get("units", {}).items():
                vertices = unit.get("vertices", [])
                for j in range(0, len(vertices), 2):
                    if j + 1 < len(vertices):
                        easting = vertices[j]
                        rl = vertices[j + 1]
                        if easting > 10000:  # Real-world coordinates
                            self.content_easting_min = min(self.content_easting_min, easting)
                            self.content_easting_max = max(self.content_easting_max, easting)
                            self.content_rl_min = min(self.content_rl_min, rl)
                            self.content_rl_max = max(self.content_rl_max, rl)

        # Add margins
        if self.global_easting_min != float("inf"):
            self.global_easting_min -= 100
            self.global_easting_max += 100
            self.global_rl_min -= 50
            self.global_rl_max += 50

        # Update northings list
        self.northings = sorted(
            set(
                section_data.get("northing", 0)
                for section_data in self.all_sections_data.values()
                if section_data.get("northing") is not None
            ),
            reverse=True,
        )

        # Update section combo
        if self.northings:
            self.section_combo["values"] = [f"Section {int(n)}" for n in self.northings]
            if self.section_combo.current() < 0:
                self.section_combo.current(0)

    def filter_sections(self):
        """Filter sections based on display options."""
        all_northings = sorted(
            set(
                section_data.get("northing", 0)
                for section_data in self.all_sections_data.values()
                if section_data.get("northing") is not None
            ),
            reverse=True,
        )

        if self.hide_empty_var.get():
            self.northings = []
            for northing in all_northings:
                has_units = False
                for (pdf, page), data in self.all_sections_data.items():
                    if (
                        data.get("northing") is not None
                        and abs(data.get("northing") - northing) < 0.1
                    ):
                        if len(data.get("units", {})) > 0:
                            has_units = True
                            break

                if has_units:
                    self.northings.append(northing)
        else:
            self.northings = all_northings

        if self.northings:
            self.section_combo["values"] = [f"Section {int(n)}" for n in self.northings]
            if self.section_combo.current() >= len(self.northings):
                self.section_combo.current(0)
                self.current_center_section = 0

        self.update_section_display()

    def update_section_display(self):
        """Update the section display."""
        if self.view_mode.get() != "section":
            return

        self.fig_sections.clear()
        self.unit_patches.clear()

        if not self.northings:
            self.canvas_sections.draw()
            return

        # Determine sections to display
        start_idx = max(0, self.current_center_section - (self.section_count_var.get() - 1) // 2)
        end_idx = min(len(self.northings), start_idx + self.section_count_var.get())
        sections_to_show = self.northings[start_idx:end_idx]

        n_sections = len(sections_to_show)

        for i, northing in enumerate(sections_to_show):
            ax = self.fig_sections.add_subplot(n_sections, 1, i + 1)

            # Get section data
            section_data_list = [
                data
                for (pdf, page), data in self.all_sections_data.items()
                if data.get("northing") is not None and abs(data.get("northing") - northing) < 0.1
            ]

            if not section_data_list:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for section {int(northing)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            section_data = section_data_list[0]

            # Store polylines to draw after polygons (so they appear on top)
            polylines_to_draw = []
            for poly_name, polyline in section_data.get("polylines", {}).items():
                vertices = polyline.get("vertices", [])
                if len(vertices) >= 4:
                    polylines_to_draw.append((poly_name, polyline))

            # Display PDF background if enabled
            if self.show_pdf_var.get() and "pdf_path" in section_data:
                try:
                    pdf_path = section_data["pdf_path"]
                    page_num = section_data.get("page_num", 0)

                    doc = fitz.open(str(pdf_path))
                    page = doc[page_num]
                    mat = fitz.Matrix(1.0, 1.0)  # Full resolution for accuracy
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )

                    # Get coordinate system to properly position the PDF
                    easting_min = section_data.get("easting_min", self.global_easting_min)
                    easting_max = section_data.get("easting_max", self.global_easting_max)
                    rl_min = section_data.get("rl_min", self.global_rl_min)
                    rl_max = section_data.get("rl_max", self.global_rl_max)

                    # Display with correct extent
                    ax.imshow(
                        img,
                        extent=[easting_min, easting_max, rl_min, rl_max],
                        aspect="auto",
                        alpha=self.pdf_alpha_var.get(),
                        interpolation="bilinear",
                        origin="upper",
                    )
                    doc.close()
                except Exception as e:
                    logger.warning(f"Could not display PDF background: {e}")

            # Plot geological units
            for unit_name, unit in section_data.get("units", {}).items():
                vertices = unit.get("vertices", [])
                if len(vertices) >= 4:
                    xs, ys = [], []
                    for j in range(0, len(vertices), 2):
                        if j + 1 < len(vertices):
                            xs.append(vertices[j])
                            ys.append(vertices[j + 1])

                    if xs and xs[0] != xs[-1]:
                        xs.append(xs[0])
                        ys.append(ys[0])

                    is_selected = unit_name in self.selected_units
                    
                    # Get color from assignment if available
                    unit_assignment = unit.get("unit_assignment")
                    if unit_assignment and unit_assignment in self.defined_units:
                        # Use the assigned unit's color from strat column
                        assigned_color = self.defined_units[unit_assignment].get('color', (0.5, 0.5, 0.5))
                        if isinstance(assigned_color, str) and assigned_color.startswith('#'):
                            # Convert hex to RGB tuple
                            r = int(assigned_color[1:3], 16) / 255
                            g = int(assigned_color[3:5], 16) / 255
                            b = int(assigned_color[5:7], 16) / 255
                            color = (r, g, b)
                        else:
                            color = assigned_color
                        alpha = 0.6  # Higher alpha for assigned units
                        edgecolor = "darkblue"  # Distinct edge for assigned
                    else:
                        color = unit.get("color", (0.5, 0.5, 0.5))
                        alpha = 0.3
                        edgecolor = "black"

                    poly = MplPolygon(
                        list(zip(xs, ys)),
                        facecolor="red" if is_selected else color,
                        edgecolor="red" if is_selected else edgecolor,
                        alpha=0.7 if is_selected else alpha,
                        linewidth=2 if is_selected else 1,
                        picker=True,
                    )

                    # Store display name for tooltip
                    display_name = unit_assignment if unit_assignment else unit.get("formation", unit_name)
                    
                    self.unit_patches[poly] = {
                        "name": unit_name,
                        "unit": unit,
                        "northing": northing,
                        "section_data": section_data,
                        "display_name": display_name,
                        "assignment": unit_assignment,
                    }

                    ax.add_patch(poly)

            # Draw polylines on top of polygons for better visibility
            for poly_name, polyline in polylines_to_draw:
                vertices = polyline.get("vertices", [])
                if len(vertices) >= 4:
                    x = [vertices[i] for i in range(0, len(vertices), 2)]
                    z = [vertices[i + 1] for i in range(0, len(vertices), 2)]

                    # Check for fault assignment first, then fault_name
                    fault_assignment = polyline.get("fault_assignment") or polyline.get("fault_name")
                    is_fault = polyline.get("is_fault", False) or polyline.get("type") == "Fault"
                    
                    if fault_assignment or is_fault:
                        if fault_assignment and fault_assignment in self.defined_faults:
                            color = self.defined_faults[fault_assignment].get('color', 'red')
                        else:
                            color = polyline.get("color", "red") if is_fault else "red"
                        linewidth = 3.5  # Thicker for visibility
                        linestyle = "-"
                        zorder = 10  # Higher z-order to appear on top
                    else:
                        color = "black"  # Changed from gray for better visibility
                        linewidth = 2.5  # Thicker
                        linestyle = "--"
                        zorder = 9

                    line = ax.plot(
                        x,
                        z,
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        picker=8,  # Increased picker radius
                        zorder=zorder,  # Draw on top
                    )[0]

                    self.polyline_patches[line] = {
                        "name": poly_name,
                        "fault_name": fault_assignment,
                        "fault_assignment": fault_assignment,
                        "is_fault": is_fault,
                        "vertices": vertices,
                        "northing": northing,
                        "pdf": section_data.get("pdf_path"),
                        "page": section_data.get("page_num"),
                    }

            # Configure axes
            ax.set_title(f"Section {int(northing)}", fontsize=10)
            ax.set_xlabel("Easting (m)" if i == n_sections - 1 else "")
            ax.set_ylabel("RL (m)", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

            # Set limits
            if hasattr(self, "content_easting_min") and self.content_easting_min != float("inf"):
                easting_range = self.content_easting_max - self.content_easting_min
                rl_range = self.content_rl_max - self.content_rl_min
                margin_e = easting_range * 0.05 if easting_range > 0 else 100
                margin_rl = rl_range * 0.05 if rl_range > 0 else 50

                ax.set_xlim(
                    self.content_easting_min - margin_e, self.content_easting_max + margin_e
                )
                ax.set_ylim(self.content_rl_min - margin_rl, self.content_rl_max + margin_rl)

            ax.set_aspect("equal", adjustable="box")

            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)

        self.fig_sections.tight_layout()

        # Set up pick event handler
        if hasattr(self, "_pick_connection"):
            self.canvas_sections.mpl_disconnect(self._pick_connection)
        if self.unit_patches:
            self._pick_connection = self.canvas_sections.mpl_connect("pick_event", self.on_pick)

        self.canvas_sections.draw()

    #  Part 7: Additional Methods

    def update_pdf_display(self):
        """Display the current PDF page."""
        if not self.current_page:
            return

        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = self.current_page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            self.ax_pdf.clear()
            self.ax_pdf.imshow(img, aspect="auto")
            self.ax_pdf.set_title(f"PDF: {self.pdf_path.name if self.pdf_path else 'None'}")
            self.ax_pdf.set_xlim(0, img.shape[1])
            self.ax_pdf.set_ylim(img.shape[0], 0)
            self.ax_pdf.axis("off")

            if self.show_grid:
                self.ax_pdf.grid(True, alpha=0.3)

            self.canvas_pdf.draw()

        except Exception as e:
            logger.error(f"Error displaying page: {e}")

    def detect_coordinates(self, show_message=True):
        """Detect coordinates in the current PDF."""
        if not self.current_page:
            if show_message:
                messagebox.showwarning("Warning", "Please open a PDF first")
            return

        self.status_var.set("Detecting coordinates...")
        self.root.update()

        try:
            coord_system = self.georeferencer.detect_coordinates(self.current_page, self.pdf_path)

            if coord_system:
                self.georeferencer.coord_system = coord_system
                self.display_coordinate_info()
                self.status_var.set("Coordinates detected successfully")
                if show_message:
                    messagebox.showinfo("Success", "Coordinates detected successfully")
            else:
                self.status_var.set("Failed to detect coordinates")
                if show_message:
                    messagebox.showwarning("Warning", "Could not detect coordinate system")

        except Exception as e:
            logger.error(f"Error detecting coordinates: {e}")
            if show_message:
                messagebox.showerror("Error", f"Failed to detect coordinates: {str(e)}")

    def extract_features(self, show_message=True):
        """Extract features from current page."""
        if not self.current_page:
            if show_message:
                messagebox.showwarning("Warning", "Please open a PDF first")
            return

        self.status_var.set("Extracting features...")
        self.root.update()

        try:
            # Apply current extraction filter
            self.feature_extractor.set_extraction_filter(self.extraction_filter)
            
            self.annotations = self.feature_extractor.extract_annotations(self.current_page)
            self.feature_extractor.number_geological_units(self.annotations)
            self.populate_feature_tree()

            num_features = len(self.feature_extractor.geological_units)
            self.status_var.set(f"Extracted {num_features} geological units")

            if show_message:
                messagebox.showinfo("Success", f"Extracted {num_features} geological units")

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            if show_message:
                messagebox.showerror("Error", f"Failed to extract features: {str(e)}")

    def find_contacts(self, method="buffered", show_message=True):
        """Find contacts between features (legacy method - redirects to grouped extraction)."""
        # The old methods (extract_contacts_simplified, extract_contacts_advanced) have been 
        # removed. Redirect to the new grouped contact extraction which uses buffer-based
        # centerline detection.
        self.extract_contacts_grouped()


    def display_coordinate_info(self):
        """Display detected coordinate information."""
        if not self.georeferencer.coord_system:
            return

        coord_system = self.georeferencer.coord_system

        info = "=== VERTICAL CROSS-SECTION ===\n"
        info += f"Northing: {coord_system.get('northing', 'N/A'):.0f}\n"
        info += f"From: {coord_system.get('northing_text', 'N/A')}\n\n"

        if "easting_min" in coord_system:
            info += f"Easting range: {coord_system['easting_min']:.0f} - "
            info += f"{coord_system['easting_max']:.0f}\n"
        if "rl_min" in coord_system:
            info += f"RL range: {coord_system['rl_min']:.0f} - "
            info += f"{coord_system['rl_max']:.0f}\n"

        # PDF viewer tab removed - coord_text no longer exists
        if hasattr(self, 'coord_text'):
            self.coord_text.delete(1.0, tk.END)
            self.coord_text.insert(1.0, info)
        else:
            logger.info(f"Coordinate info: {info.replace(chr(10), ' | ')}")

    def populate_feature_tree(self):
        """Populate the feature tree with detected features."""
        # PDF viewer tab removed - feature_tree may not exist
        if not hasattr(self, 'feature_tree'):
            logger.info(f"Extracted {len(self.feature_extractor.geological_units)} units (no tree view)")
            return
            
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        for unit_name, unit in self.feature_extractor.geological_units.items():
            num_points = len(unit["vertices"]) // 2
            item = self.feature_tree.insert(
                "", "end", text="✓", values=(unit["type"], unit_name, num_points)
            )
            self.selected_items[item] = True

    def populate_feature_tree_all(self):
        """Populate the feature tree with ALL accumulated features."""
        # PDF viewer tab removed - feature_tree may not exist
        if not hasattr(self, 'feature_tree'):
            logger.info(f"Accumulated {len(self.all_geological_units)} units (no tree view)")
            return
            
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        pdf_groups = {}
        for unique_name, unit in self.all_geological_units.items():
            pdf_key = (unit["source_pdf"], unit["page_num"])
            if pdf_key not in pdf_groups:
                pdf_groups[pdf_key] = {"units": [], "contacts": []}
            pdf_groups[pdf_key]["units"].append((unique_name, unit))

        for contact in self.all_contacts:
            pdf_key = (contact["source_pdf"], contact["page_num"])
            if pdf_key in pdf_groups:
                pdf_groups[pdf_key]["contacts"].append(contact)

        for (pdf_path, page_num), group in sorted(pdf_groups.items()):
            pdf_name = Path(pdf_path).stem
            section_item = self.feature_tree.insert(
                "", "end", text="", values=("SECTION", f"{pdf_name} - Page {page_num+1}", "")
            )

            for unique_name, unit in group["units"]:
                num_points = len(unit["vertices"]) // 2
                item = self.feature_tree.insert(
                    section_item, "end", text="✓", values=(unit["type"], unique_name, num_points)
                )
                self.selected_items[item] = True

            if group["contacts"]:
                for contact in group["contacts"]:
                    num_points = len(contact["vertices"]) // 2
                    item = self.feature_tree.insert(
                        section_item,
                        "end",
                        text="✓",
                        values=("Contact", contact["unique_name"], num_points),
                    )
                    self.selected_items[item] = True

    # Part 8: 3d Visualisation Methods
    def update_3d_view(self):
        """Update the 3D view with selected units."""
        self.ax_3d.clear()

        if not self.selected_units:
            self.ax_3d.set_title("No units selected")
            self.canvas_3d.draw()
            return

        # Plot selected units in 3D
        for (pdf_path, page_num), section_data in self.all_sections_data.items():
            northing = section_data.get("northing", 0)

            for unit_name, unit in section_data.get("units", {}).items():
                if unit_name in self.selected_units and len(unit["vertices"]) >= 4:
                    eastings = []
                    rls = []
                    for i in range(0, len(unit["vertices"]), 2):
                        if i + 1 < len(unit["vertices"]):
                            eastings.append(unit["vertices"][i])
                            rls.append(unit["vertices"][i + 1])

                    if eastings and eastings[0] != eastings[-1]:
                        eastings.append(eastings[0])
                        rls.append(rls[0])

                    northings = [northing] * len(eastings)
                    color = unit.get("color", (0.5, 0.5, 0.5))

                    self.ax_3d.plot(eastings, northings, rls, color=color, linewidth=2, alpha=0.8)

        self.ax_3d.set_xlabel("Easting (m)")
        self.ax_3d.set_ylabel("Northing (m)")
        self.ax_3d.set_zlabel("RL (m)")
        self.ax_3d.set_title(f"{len(self.selected_units)} units selected")

        # Set equal aspect ratio for all axes
        self._set_axes_equal(self.ax_3d)

        self.ax_3d.view_init(elev=20, azim=45)

        self.canvas_3d.draw()

    def update_3d_display(self):
        """Update 3D display based on mode."""
        self.ax_3d.clear()

        if not self.selected_units:
            self.ax_3d.set_title("No units selected")
            self.canvas_3d.draw()
            return

        mode = self.display_mode_3d.get()

        if mode == "wireframe":
            self.draw_3d_wireframe()
        elif mode == "solid":
            self.draw_3d_solid()
        elif mode == "transparent":
            self.draw_3d_transparent()

        self.ax_3d.set_xlabel("Easting (m)")
        self.ax_3d.set_ylabel("Northing (m)")
        self.ax_3d.set_zlabel("RL (m)")
        self.ax_3d.set_title("3D Geological Model")
        self.ax_3d.invert_zaxis()

        # Set equal aspect ratio for all axes
        self._set_axes_equal(self.ax_3d)

        self.ax_3d.view_init(elev=20, azim=45)

        self.canvas_3d.draw()

    def _set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so geological features
        are displayed with proper proportions.

        Args:
            ax: matplotlib 3D axis
        """
        # Get the current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Calculate ranges
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]

        # Find the maximum range
        max_range = max(x_range, y_range, z_range)

        # Calculate centers
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2

        # Set new limits with equal ranges
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

        # Optionally set box aspect for newer matplotlib versions
        try:
            ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
        except AttributeError:
            # Older matplotlib versions don't have set_box_aspect
            pass

    def draw_3d_wireframe(self):
        """Draw sections as wireframes."""
        for (pdf_path, page_num), section_data in self.all_sections_data.items():
            northing = section_data.get("northing", 0)

            for unit_name, unit in section_data.get("units", {}).items():
                if unit_name in self.selected_units:
                    vertices = unit.get("vertices", [])
                    if len(vertices) >= 4:
                        x = [vertices[i] for i in range(0, len(vertices), 2)]
                        z = [vertices[i + 1] for i in range(0, len(vertices), 2)]

                        if x and x[0] != x[-1]:
                            x.append(x[0])
                            z.append(z[0])

                        y = [northing] * len(x)
                        color = unit.get("color", (0.5, 0.5, 0.5))

                        self.ax_3d.plot(x, y, z, color=color, alpha=0.8, linewidth=1.5)

    def draw_3d_solid(self):
        """Draw sections as solid meshes."""
        # Group units by northing first, then by formation
        sections_by_northing = {}

        for (pdf_path, page_num), section_data in self.all_sections_data.items():
            northing = section_data.get("northing", 0)

            if northing not in sections_by_northing:
                sections_by_northing[northing] = {}

            for unit_name, unit in section_data.get("units", {}).items():
                if unit_name in self.selected_units:
                    formation = unit.get("formation", "Unknown")
                    vertices = unit.get("vertices", [])

                    if len(vertices) >= 4:
                        coords = []
                        for i in range(0, len(vertices), 2):
                            if i + 1 < len(vertices):
                                coords.append((vertices[i], vertices[i + 1]))

                        if coords:
                            sections_by_northing[northing][formation] = {
                                "northing": northing,
                                "coords": np.array(coords),
                                "color": unit.get("color", (0.5, 0.5, 0.5)),
                                "unit_name": unit_name,
                            }

        # Get sorted list of northings
        sorted_northings = sorted(sections_by_northing.keys())

        # Only connect adjacent sections
        for i in range(len(sorted_northings) - 1):
            current_northing = sorted_northings[i]
            next_northing = sorted_northings[i + 1]

            current_formations = sections_by_northing[current_northing]
            next_formations = sections_by_northing[next_northing]

            # Find matching formations between adjacent sections
            for formation in current_formations:
                if formation in next_formations:
                    section1 = current_formations[formation]
                    section2 = next_formations[formation]

                    # Check if sections are reasonably close (not wildly different positions)
                    # This prevents connecting unrelated units that happen to have same formation name
                    coords1_mean = np.mean(section1["coords"], axis=0)
                    coords2_mean = np.mean(section2["coords"], axis=0)

                    # Calculate horizontal distance between unit centers
                    horiz_distance = abs(coords1_mean[0] - coords2_mean[0])

                    # Only connect if units are within reasonable distance (e.g., 500m)
                    # Adjust this threshold based on your data
                    if horiz_distance < 500:
                        self.create_mesh_between_sections(section1, section2, alpha=0.7)

    def draw_3d_transparent(self):
        """Draw sections as transparent meshes."""
        self.draw_3d_solid()  # Same as solid but alpha is already set

    def create_mesh_between_sections(self, section1, section2, alpha=0.7):
        """Create triangulated mesh between two sections.

        This version:
        - Resamples both outlines by arc length (not raw index) for better correspondence.
        - Builds a simple strip (no self-crossing Delaunay fan).
        - Records vertices and faces into self.mesh_vertices / self.mesh_faces
          so that export_3d_model can write the true 3D body.
        """
        try:
            coords1 = section1["coords"]
            coords2 = section2["coords"]
            northing1 = section1["northing"]
            northing2 = section2["northing"]
            color = section1["color"]

            # Need at least 3 vertices on each section to form a surface
            if len(coords1) < 3 or len(coords2) < 3:
                return

            # Helper: arc-length parameterization of a polyline
            def arc_length_param(coords):
                coords = np.asarray(coords)
                if len(coords) < 2:
                    t = np.linspace(0.0, 1.0, len(coords))
                    return coords[:, 0], coords[:, 1], t

                # Ensure closed polygon for length calculation
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])

                deltas = np.diff(coords, axis=0)
                seg_lengths = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
                cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
                total = cum_len[-1] if cum_len[-1] > 0 else 1.0
                t = cum_len / total
                return coords[:, 0], coords[:, 1], t

            x1, z1, t1 = arc_length_param(coords1)
            x2, z2, t2 = arc_length_param(coords2)

            # Use the larger of the two point counts for smoother interpolation
            n_points = max(len(x1), len(x2), 8)

            # Uniform parameter grid
            t_uniform = np.linspace(0.0, 1.0, n_points)

            # Interpolate x,z along arc length for each section
            x1_interp = np.interp(t_uniform, t1, x1)
            z1_interp = np.interp(t_uniform, t1, z1)

            x2_interp = np.interp(t_uniform, t2, x2)
            z2_interp = np.interp(t_uniform, t2, z2)

            # Build vertices for both sections
            # Keep track of global index offset for faces
            base_index = len(self.mesh_vertices)

            verts = []
            for i in range(n_points):
                verts.append([x1_interp[i], northing1, z1_interp[i]])
            for i in range(n_points):
                verts.append([x2_interp[i], northing2, z2_interp[i]])

            verts = np.asarray(verts)

            # Append to global mesh vertex list
            for v in verts:
                self.mesh_vertices.append(v.tolist())

            # Build faces as a simple strip between the two resampled outlines
            faces = []
            for i in range(n_points - 1):
                # Two triangles per quad: (i, i+1, i+n) and (i+1, i+n+1, i+n)
                i0 = base_index + i
                i1 = base_index + i + 1
                j0 = base_index + i + n_points
                j1 = base_index + i + n_points + 1

                faces.append([i0, i1, j0])
                faces.append([i1, j1, j0])

            # Append to global mesh face list
            self.mesh_faces.extend(faces)

            # Plot triangulated surface for visual feedback
            self.ax_3d.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=np.array([[f[0] - base_index, f[1] - base_index, f[2] - base_index] for f in faces]),
                color=color,
                alpha=alpha,
                edgecolor="none",
                shade=True,
            )

        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            # Fallback - use simple connection if something fails
            self.connect_sections_simple(section1, section2, alpha)


    def connect_sections_simple(self, section1, section2, alpha=0.7):
        """Simple connection between sections without triangulation."""
        coords1 = section1["coords"]
        coords2 = section2["coords"]
        northing1 = section1["northing"]
        northing2 = section2["northing"]
        color = section1["color"]

        n_connections = min(len(coords1), len(coords2))

        for i in range(0, n_connections, 2):  # Skip some for clarity
            x = [coords1[i, 0], coords2[i, 0]]
            y = [northing1, northing2]
            z = [coords1[i, 1], coords2[i, 1]]

            self.ax_3d.plot(x, y, z, color=color, alpha=alpha * 0.5, linewidth=0.5)

    def create_solids_from_selected(self):
        """Create solid meshes from selected units.

        Strategy:
        - For each formation, sort selected units by northing.
        - Split into blocks where the northing gap is reasonable so we do not
          bridge across big distances.
        - Within each block, only loft between adjacent sections if the unit
          outlines are similar enough (so pinched-out units are not forced).
        - Use create_mesh_between_sections for the actual lofting. That function
          already does dense, arc-length based resampling.
        """
        if not self.selected_units:
            messagebox.showwarning("No Selection", "Please select geological units first")
            return

        # Clear and redraw with only selected formations as solids
        self.ax_3d.clear()

        # Reset mesh buffers that export_3d_model / export_to_obj might use
        self.mesh_vertices = []
        self.mesh_faces = []

        # Group selected units by formation
        selected_formations = {}

        for unit_name in self.selected_units:
            if unit_name in self.all_geological_units:
                unit = self.all_geological_units[unit_name]
                vertices = unit.get("vertices", [])
                northing = unit.get("northing", None)

                # Need at least a polygon and a valid northing
                if northing is None or len(vertices) < 6:
                    continue

                formation = unit.get("formation", "Unknown")
                selected_formations.setdefault(formation, []).append(unit)

        # Parameters controlling connectivity
        max_gap_factor = 2.5       # how many times the median northing spacing we allow
        max_shape_mismatch = 2.0   # how many times the typical unit width we allow

        for formation, units in selected_formations.items():
            if len(units) < 2:
                continue

            # Sort units by northing
            units.sort(key=lambda u: u.get("northing", 0.0))

            # Split into blocks so we do not bridge huge gaps
            blocks = self._group_units_by_northing_gap(units, factor=max_gap_factor)

            for block in blocks:
                if len(block) < 2:
                    continue

                # Loft between adjacent units in this block if they are similar enough
                for i in range(len(block) - 1):
                    u1 = block[i]
                    u2 = block[i + 1]

                    if not self._units_are_compatible(u1, u2, max_shape_mismatch):
                        # Treat as pinch-out - skip this connection
                        continue

                    coords1 = np.array(
                        [
                            (u1["vertices"][j], u1["vertices"][j + 1])
                            for j in range(0, len(u1["vertices"]), 2)
                            if j + 1 < len(u1["vertices"])
                        ]
                    )
                    coords2 = np.array(
                        [
                            (u2["vertices"][j], u2["vertices"][j + 1])
                            for j in range(0, len(u2["vertices"]), 2)
                            if j + 1 < len(u2["vertices"])
                        ]
                    )

                    if len(coords1) < 3 or len(coords2) < 3:
                        continue

                    section1 = {
                        "coords": coords1,
                        "northing": u1.get("northing", 0.0),
                        "color": u1.get("color", (0.5, 0.5, 0.5)),
                    }
                    section2 = {
                        "coords": coords2,
                        "northing": u2.get("northing", 0.0),
                        "color": u2.get("color", (0.5, 0.5, 0.5)),
                    }

                    self.create_mesh_between_sections(section1, section2, alpha=0.9)

        self.ax_3d.set_xlabel("Easting (m)")
        self.ax_3d.set_ylabel("Northing (m)")
        self.ax_3d.set_zlabel("RL (m)")
        self.ax_3d.set_title("3D Solid Model - Selected Units")
        self.ax_3d.invert_zaxis()

        # Set equal aspect ratio for all axes
        self._set_axes_equal(self.ax_3d)

        self.ax_3d.view_init(elev=20, azim=45)
        self.canvas_3d.draw()

        messagebox.showinfo(
            "Solids Created", f"Created solid meshes for {len(selected_formations)} formations"
        )

    def _group_units_by_northing_gap(self, units, factor=2.5):
        """
        Split sorted units into blocks where northing gaps are not too large.

        units must already be sorted by unit["northing"].
        factor controls how many times the median spacing we allow
        before starting a new block.
        """
        northings = np.array([float(u.get("northing", 0.0)) for u in units], dtype=float)
        if len(northings) < 2:
            return [units]

        diffs = np.diff(northings)
        positive = diffs[diffs > 0]
        if positive.size == 0:
            max_gap = 1000.0
        else:
            median_gap = float(np.median(positive))
            max_gap = factor * median_gap if median_gap > 0 else 1000.0

        blocks = []
        current = [units[0]]

        for i in range(1, len(units)):
            if northings[i] - northings[i - 1] > max_gap:
                blocks.append(current)
                current = [units[i]]
            else:
                current.append(units[i])

        if current:
            blocks.append(current)

        return blocks

    def _units_are_compatible(self, u1, u2, max_shape_mismatch_factor=2.0):
        """
        Decide if two units should be lofted together.

        - If their centroids are far apart compared to their size, treat
          as pinch-out and do not connect.
        """
        c1, size1 = self._unit_centroid_and_size(u1)
        c2, size2 = self._unit_centroid_and_size(u2)

        if c1 is None or c2 is None:
            return False

        # Distance in easting - RL space
        dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        typical_size = max(size1, size2, 1.0)

        return dist <= max_shape_mismatch_factor * typical_size

    def _unit_centroid_and_size(self, unit):
        """
        Compute centroid and a characteristic size (max of width/height) for a unit
        in easting RL space.
        """
        verts = unit.get("vertices", [])
        if len(verts) < 6:
            return None, 0.0

        xs = []
        zs = []
        for i in range(0, len(verts), 2):
            if i + 1 < len(verts):
                xs.append(verts[i])
                zs.append(verts[i + 1])

        if not xs or not zs:
            return None, 0.0

        xs = np.array(xs, dtype=float)
        zs = np.array(zs, dtype=float)

        cx = float(xs.mean())
        cz = float(zs.mean())

        width = float(xs.max() - xs.min())
        height = float(zs.max() - zs.min())
        size = max(width, height)

        return (cx, cz), size


    def _sample_unit_outline_points(self, unit: Dict, northing: float, n_samples: int = 80) -> np.ndarray:
        """
        Sample points along a unit's outline in 3D.

        Returns an (N, 3) array of [easting, northing, RL] points.
        Uses arc-length parameterisation so sampling is uniform along the polygon.
        """
        verts = unit.get("vertices", [])
        if len(verts) < 6:
            return np.zeros((0, 3), dtype=float)

        # Build 2D (easting, RL) coordinates
        coords = []
        for i in range(0, len(verts), 2):
            if i + 1 < len(verts):
                coords.append((verts[i], verts[i + 1]))
        if len(coords) < 3:
            return np.zeros((0, 3), dtype=float)

        coords = np.asarray(coords, dtype=float)

        # Ensure closed polygon
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        # Arc-length parameterisation
        deltas = np.diff(coords, axis=0)
        seg_lengths = np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum_len[-1] if cum_len[-1] > 0 else 1.0
        t = cum_len / total

        # Target samples
        n = max(int(n_samples), 8)
        t_uniform = np.linspace(0.0, 1.0, n)

        x = np.interp(t_uniform, t, coords[:, 0])
        z = np.interp(t_uniform, t, coords[:, 1])
        y = np.full_like(x, float(northing))

        return np.vstack([x, y, z]).T

    def _build_convex_hull_mesh(self, points_3d: np.ndarray, ConvexHull):
        """
        Build a convex-hull mesh (shrink-wrap) from a 3D point cloud.

        Returns:
            (vertices, faces)
            vertices: (M, 3) array of unique vertex coordinates
            faces:    (K, 3) array of indices into vertices
        """
        if points_3d.shape[0] < 4:
            return None, None

        try:
            hull = ConvexHull(points_3d)
        except Exception:
            return None, None

        # Map original point indices used by hull to a compact 0..M-1 range
        used_indices = np.unique(hull.simplices.flatten())
        index_map = {int(old): i for i, old in enumerate(used_indices)}

        vertices = points_3d[used_indices]

        faces = []
        for tri in hull.simplices:
            i0 = index_map[int(tri[0])]
            i1 = index_map[int(tri[1])]
            i2 = index_map[int(tri[2])]
            faces.append([i0, i1, i2])

        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)

    def _group_units_by_northing_gap(self, units: List[Dict], factor: float = 2.5) -> List[List[Dict]]:
        """
        Split a sorted list of units (by northing) into blocks where the northing
        gaps are not excessively large.

        The idea is to avoid joining solids across big gaps between sections.
        """
        northings = np.array([float(u.get("northing", 0.0)) for u in units], dtype=float)
        if len(northings) < 2:
            return [units]

        diffs = np.diff(northings)
        median_gap = float(np.median(diffs)) if np.any(diffs > 0) else 0.0

        if median_gap <= 0:
            # Fallback fixed limit (in metres) if we cannot infer spacing
            max_gap = 1000.0
        else:
            max_gap = factor * median_gap

        blocks: List[List[Dict]] = []
        current_block: List[Dict] = [units[0]]

        for i in range(1, len(units)):
            if northings[i] - northings[i - 1] > max_gap:
                # Start a new block
                blocks.append(current_block)
                current_block = [units[i]]
            else:
                current_block.append(units[i])

        if current_block:
            blocks.append(current_block)

        return blocks

    def _mesh_block_pairwise(self, units_block: List[Dict]):
        """
        Fallback meshing: connect adjacent sections in a block using the existing
        strip approach (create_mesh_between_sections), but only within the block.

        units_block must already be sorted by northing.
        """
        if len(units_block) < 2:
            return

        for i in range(len(units_block) - 1):
            u1 = units_block[i]
            u2 = units_block[i + 1]

            coords1 = np.array(
                [(u1["vertices"][j], u1["vertices"][j + 1])
                 for j in range(0, len(u1["vertices"]), 2)
                 if j + 1 < len(u1["vertices"])]
            )
            coords2 = np.array(
                [(u2["vertices"][j], u2["vertices"][j + 1])
                 for j in range(0, len(u2["vertices"]), 2)
                 if j + 1 < len(u2["vertices"])]
            )

            if len(coords1) < 3 or len(coords2) < 3:
                continue

            section1 = {
                "coords": coords1,
                "northing": u1.get("northing", 0.0),
                "color": u1.get("color", (0.5, 0.5, 0.5)),
            }
            section2 = {
                "coords": coords2,
                "northing": u2.get("northing", 0.0),
                "color": u2.get("color", (0.5, 0.5, 0.5)),
            }

            self.create_mesh_between_sections(section1, section2, alpha=0.9)


    def reset_3d_view(self):
        """Reset the 3D view angle."""
        self.ax_3d.view_init(elev=20, azim=45)
        self.canvas_3d.draw()

    def export_3d_model(self):
        """Export 3D model to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("STL file", "*.stl"), ("PLY file", "*.ply")],
        )

        if filepath:
            ext = Path(filepath).suffix.lower()

            if ext == ".obj":
                self.export_to_obj(filepath)
            elif ext == ".stl":
                self.export_to_stl(filepath)
            elif ext == ".ply":
                self.export_to_ply(filepath)

            messagebox.showinfo("Exported", f"3D model exported to {filepath}")

    def export_to_obj(self, filepath):
        """Export to Wavefront OBJ format.

        If solid meshes have been created (mesh_vertices / mesh_faces),
        export those as a proper 3D body. Otherwise fall back to exporting
        flat section outlines for the selected units.
        """
        with open(filepath, "w") as f:
            f.write("# Geological 3D Model\n")
            f.write("# Generated by Geological Cross-Section Tool\n\n")

            # Prefer exporting the solid mesh if it exists
            if self.mesh_vertices and self.mesh_faces:
                # Write vertices
                for vx, vy, vz in self.mesh_vertices:
                    f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")

                f.write("\n")

                # Faces in OBJ are 1-based indices
                for i0, i1, i2 in self.mesh_faces:
                    f.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")

                return

            # Fallback: export each selected unit as a flat polygon at its northing
            vertex_count = 0

            for unit_name in self.selected_units:
                if unit_name in self.all_geological_units:
                    unit = self.all_geological_units[unit_name]
                    vertices = unit.get("vertices", [])
                    northing = unit.get("northing", 0)

                    if len(vertices) >= 4:
                        # Write vertices
                        for i in range(0, len(vertices), 2):
                            if i + 1 < len(vertices):
                                f.write(f"v {vertices[i]:.4f} {northing:.4f} {vertices[i+1]:.4f}\n")
                                vertex_count += 1

                        # Write a single face using the polygon vertex order
                        n_verts = len(vertices) // 2
                        f.write("f")
                        for i in range(n_verts):
                            f.write(f" {vertex_count - n_verts + i + 1}")
                        f.write("\n\n")



    def export_to_stl(self, filepath):
        """Export to STL format (simplified ASCII STL)."""
        try:
            with open(filepath, "w") as f:
                f.write("solid GeologicalModel\n")

                # Export each selected unit as triangulated mesh
                for unit_name in self.selected_units:
                    if unit_name in self.all_geological_units:
                        unit = self.all_geological_units[unit_name]
                        vertices = unit.get("vertices", [])
                        northing = unit.get("northing", 0)

                        # Create simple triangulation (fan from center)
                        if len(vertices) >= 6:  # At least 3 points
                            points = []
                            for i in range(0, len(vertices), 2):
                                if i + 1 < len(vertices):
                                    points.append([vertices[i], northing, vertices[i + 1]])

                            if len(points) >= 3:
                                # Fan triangulation from first point
                                for i in range(1, len(points) - 1):
                                    # Write triangle
                                    f.write("  facet normal 0 0 0\n")
                                    f.write("    outer loop\n")
                                    f.write(
                                        f"      vertex {points[0][0]} {points[0][1]} {points[0][2]}\n"
                                    )
                                    f.write(
                                        f"      vertex {points[i][0]} {points[i][1]} {points[i][2]}\n"
                                    )
                                    f.write(
                                        f"      vertex {points[i+1][0]} {points[i+1][1]} {points[i+1][2]}\n"
                                    )
                                    f.write("    endloop\n")
                                    f.write("  endfacet\n")

                f.write("endsolid GeologicalModel\n")

        except Exception as e:
            logger.error(f"Error exporting STL: {e}")
            messagebox.showerror("Export Error", f"Failed to export STL: {str(e)}")

    def export_to_ply(self, filepath):
        """Export to PLY format (simplified)."""
        messagebox.showinfo("PLY Export", "PLY export not yet implemented. Use OBJ format for now.")

    # Part 9: Strat Column Management Methods
    def add_strat_unit(self):
        """Add a new stratigraphic unit."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Stratigraphic Unit")
        dialog.geometry("400x250")

        # Name
        ttk.Label(dialog, text="Unit Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        # Age/Description
        ttk.Label(dialog, text="Age/Description:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        age_entry = ttk.Entry(dialog, width=30)
        age_entry.grid(row=1, column=1, padx=5, pady=5)

        # Color
        color_var = tk.StringVar(value="#808080")
        ttk.Label(dialog, text="Color:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        color_label = tk.Label(dialog, text="      ", bg=color_var.get())
        color_label.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        def choose_color():
            color = colorchooser.askcolor(initialcolor=color_var.get())
            if color[1]:
                color_var.set(color[1])
                color_label.config(bg=color[1])

        ttk.Button(dialog, text="Choose Color", command=choose_color).grid(
            row=2, column=2, padx=5, pady=5
        )

        # Unconformity checkbox
        is_unconformity_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="Is Unconformity", variable=is_unconformity_var).grid(
            row=3, column=1, padx=5, pady=5, sticky=tk.W
        )

        # Buttons
        def add():
            name = name_entry.get().strip()
            age = age_entry.get().strip()

            if name:
                hex_color = color_var.get()
                rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

                self.strat_column.add_unit(
                    name, rgb, age=age, is_unconformity=is_unconformity_var.get()
                )

                # Update strat buttons
                self.update_strat_buttons()
                dialog.destroy()

        ttk.Button(dialog, text="Add", command=add).grid(row=4, column=0, padx=5, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=4, column=1, padx=5, pady=10
        )

    def remove_strat_unit(self):
        """Remove selected stratigraphic unit."""
        # Create dialog with list of units
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Stratigraphic Unit")
        dialog.geometry("300x400")

        ttk.Label(dialog, text="Select unit to remove:").pack(pady=10)

        listbox = tk.Listbox(dialog, height=15)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for unit in self.strat_column.get_all_units_ordered():
            listbox.insert(tk.END, unit.name)

        def remove():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                units = self.strat_column.get_all_units_ordered()
                if idx < len(units):
                    unit_name = units[idx].name
                    self.strat_column.remove_unit(unit_name)
                    self.update_strat_buttons()
                    dialog.destroy()
                    messagebox.showinfo("Removed", f"Removed unit: {unit_name}")

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Remove", command=remove).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def update_strat_buttons(self):
        """Update stratigraphic unit assignment buttons after loading strat column."""
        # Clear existing defined units and reload from strat column
        self.defined_units.clear()
        self.defined_faults.clear()
        
        # Refresh both unit and fault buttons
        self.refresh_unit_buttons()
        self.refresh_fault_buttons()
        
        logger.info(f"Updated strat buttons: {len(self.strat_column.units)} units, {len(self.strat_column.faults)} faults")

    def save_strat_column(self):
        """Save stratigraphic column."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            self.strat_column.save(Path(filename))
            messagebox.showinfo("Success", "Stratigraphic column saved")

    def load_strat_column(self):
        """Load stratigraphic column."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            self.strat_column.load(Path(filename))
            # Update auto-labeler with new strat column
            self.auto_labeler = AutoLabeler(self.strat_column)
            self.update_strat_buttons()
            messagebox.showinfo("Success", "Stratigraphic column loaded")

    def detect_faults_from_contacts(self):
        """Detect faults based on non-sequential stratigraphic contacts."""
        if not self.feature_extractor.contacts:
            messagebox.showwarning("Warning", "No contacts found. Extract contacts first.")
            return

        if not self.strat_column.units:
            messagebox.showwarning("Warning", "No stratigraphic column defined.")
            return

        # Create strat order lookup (using v2 API)
        all_units = self.strat_column.get_all_units_ordered()
        strat_order = {unit.name: i for i, unit in enumerate(all_units)}

        faults_found = []

        for contact in self.feature_extractor.contacts:
            form1 = contact.get("formation1", "")
            form2 = contact.get("formation2", "")

            if form1 in strat_order and form2 in strat_order:
                order_diff = abs(strat_order[form1] - strat_order[form2])

                if order_diff > 1:  # Not adjacent = fault
                    contact["type"] = "Fault"
                    contact["fault_type"] = (
                        "Normal" if strat_order[form1] < strat_order[form2] else "Reverse"
                    )
                    faults_found.append(contact["name"])
                else:
                    contact["type"] = "Contact"

        self.update_display()

        if faults_found:
            messagebox.showinfo(
                "Faults Detected",
                f"Found {len(faults_found)} potential faults:\n"
                + "\n".join(faults_found[:10])
                + ("\n..." if len(faults_found) > 10 else ""),
            )
        else:
            messagebox.showinfo("No Faults", "No faults detected based on stratigraphy.")

    # Part 10: Export Methods

    def export_geotiff(self):
        """Export current page as GeoTIFF."""
        if not self.current_pdf or not self.pdf_path:
            messagebox.showwarning("Warning", "Please open a PDF first")
            return

        export_for_leapfrog = messagebox.askyesno(
            "Export Format",
            "Export for Leapfrog Geo?\n\nYes = Leapfrog format with corner points\nNo = Standard GeoTIFF",
        )

        output_file = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")],
            initialfile=f"{self.pdf_path.stem}.tif",
        )

        if output_file:
            success = self.georeferencer.export_geotiff(self.pdf_path, Path(output_file))

            if success:
                if export_for_leapfrog and self.georeferencer.coord_system:
                    self._create_leapfrog_corner_file(Path(output_file))
                messagebox.showinfo("Success", "GeoTIFF exported successfully")
            else:
                messagebox.showerror("Error", "Failed to export GeoTIFF")

    def _create_leapfrog_corner_file(self, tiff_path: Path):
        """Create a corner points file for Leapfrog import."""
        if not self.georeferencer.coord_system:
            return

        cs = self.georeferencer.coord_system
        northing = cs["northing"]

        corners_file = tiff_path.with_suffix(".corners")

        with open(corners_file, "w") as f:
            f.write("# Leapfrog Geo Corner Points File\n")
            f.write("# For vertical cross-section import\n")
            f.write("Easting,Northing,Elevation,ImageX,ImageY\n")

            # Corner points
            f.write(f"{cs['easting_min']:.2f},{northing:.2f},{cs['rl_max']:.2f},0,0\n")
            f.write(f"{cs['easting_max']:.2f},{northing:.2f},{cs['rl_max']:.2f},1,0\n")
            f.write(f"{cs['easting_min']:.2f},{northing:.2f},{cs['rl_min']:.2f},0,1\n")
            f.write(f"{cs['easting_max']:.2f},{northing:.2f},{cs['rl_min']:.2f},1,1\n")

        logger.info(f"Created Leapfrog corner points file: {corners_file}")

    def export_csv(self):
        """Export features as CSV."""
        if not self.feature_extractor.geological_units and not self.feature_extractor.contacts:
            messagebox.showwarning("Warning", "No features to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            self._export_csv_file(filename)

    def export_dxf(self):
        """Export features as DXF."""
        num_units = len(self.feature_extractor.geological_units)
        num_contacts = len(self.feature_extractor.contacts)
        
        logger.info(f"Export DXF: {num_units} units, {num_contacts} contacts")
        logger.info(f"Current PDF: {self.current_pdf}")
        logger.info(f"All sections data: {len(self.all_geological_units)} total units")
        
        if num_units == 0 and num_contacts == 0:
            # Check if we have data in all_geological_units (from section viewer)
            if len(self.all_geological_units) > 0:
                response = messagebox.askyesno(
                    "Export All Sections?",
                    f"No features loaded for current page, but {len(self.all_geological_units)} units "
                    f"found across all sections.\n\nExport all sections instead?"
                )
                if response:
                    self._export_all_sections_dxf()
                    return
            
            messagebox.showwarning("Warning", "No features to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".dxf", filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )

        if filename:
            self._export_dxf_file(filename)

    def _export_csv_file(self, filename):
        """Export data as CSV."""
        try:
            transform = None
            if self.georeferencer.coord_system:
                transform = self.georeferencer.build_transformation()

            if not transform:
                messagebox.showwarning(
                    "Warning", "No coordinate system detected. Export will use PDF coordinates."
                )

                def transform(x, y):
                    return (x, 0, y)

            simplify = messagebox.askyesno(
                "Export Options",
                "Simplify polylines?\n(Reduces point count while maintaining shape)",
            )

            with open(filename, "w") as f:
                # Export header
                f.write("Type,Name,Easting,Northing,RL,PolylineID,VertexNumber\n")

                polyline_id = 1
                northing = (
                    self.georeferencer.coord_system.get("northing", 0)
                    if self.georeferencer.coord_system
                    else 0
                )

                # Export units
                for unit_name, unit in self.feature_extractor.geological_units.items():
                    coords = []
                    for i in range(0, len(unit["vertices"]), 2):
                        if i + 1 < len(unit["vertices"]):
                            x, y = unit["vertices"][i], unit["vertices"][i + 1]
                            e, n, r = transform(x, y)
                            coords.append((e, r))

                    if simplify and len(coords) > 2:
                        coords = self.feature_extractor.simplify_line(coords)

                    for vertex_num, (e, r) in enumerate(coords):
                        f.write(
                            f"Unit,{unit_name},{e:.2f},{northing:.2f},{r:.2f},"
                            f"{polyline_id},{vertex_num+1}\n"
                        )

                    polyline_id += 1

                # Export contacts
                for contact in self.feature_extractor.contacts:
                    coords = []
                    for i in range(0, len(contact["vertices"]), 2):
                        if i + 1 < len(contact["vertices"]):
                            coords.append((contact["vertices"][i], contact["vertices"][i + 1]))

                    if simplify and len(coords) > 2:
                        coords = self.feature_extractor.simplify_line(coords)

                    for vertex_num, (e, r) in enumerate(coords):
                        f.write(
                            f"Contact,{contact['name']},{e:.2f},{northing:.2f},"
                            f"{r:.2f},{polyline_id},{vertex_num+1}\n"
                        )

                    polyline_id += 1

            logger.info(f"Exported to {filename}")
            messagebox.showinfo("Success", f"Exported to {filename}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def _export_dxf_file(self, filename):
        """Export data as DXF."""
        try:
            transform = (
                self.georeferencer.build_transformation()
                if self.georeferencer.coord_system
                else None
            )

            if not transform:
                messagebox.showwarning(
                    "Warning", "No coordinate system detected. Export will use PDF coordinates."
                )

                def transform(x, y):
                    return (x, 0, y)

            with open(filename, "w") as f:
                # DXF header
                f.write("0\nSECTION\n2\nENTITIES\n")

                northing = (
                    self.georeferencer.coord_system.get("northing", 0)
                    if self.georeferencer.coord_system
                    else 0
                )

                # Export units
                for unit_name, unit in self.feature_extractor.geological_units.items():
                    coords = []
                    for i in range(0, len(unit["vertices"]), 2):
                        if i + 1 < len(unit["vertices"]):
                            x, y = unit["vertices"][i], unit["vertices"][i + 1]
                            e, n, r = transform(x, y)
                            coords.append((e, northing, r))

                    if len(coords) >= 2:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])

                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\n{unit_name}\n")  # Layer
                        f.write("66\n1\n")  # Vertices follow
                        f.write("70\n8\n")  # 3D polyline

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\n{unit_name}\n")
                            f.write(f"10\n{x:.2f}\n")  # X = Easting
                            f.write(f"20\n{y:.2f}\n")  # Y = Northing
                            f.write(f"30\n{z:.2f}\n")  # Z = RL

                        f.write("0\nSEQEND\n")

                # Export contacts
                logger.info(f"Exporting {len(self.feature_extractor.contacts)} contacts to DXF")
                for contact in self.feature_extractor.contacts:
                    coords = []
                    for i in range(0, len(contact["vertices"]), 2):
                        if i + 1 < len(contact["vertices"]):
                            easting = contact["vertices"][i]
                            rl = contact["vertices"][i + 1]
                            coords.append((easting, northing, rl))

                    if len(coords) >= 2:
                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\nCONTACTS_{contact.get('formation1', 'UNIT1')}_{contact.get('formation2', 'UNIT2')}\n")  # Layer by formation pair
                        f.write("66\n1\n")
                        f.write("70\n8\n")  # 3D polyline

                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\nCONTACTS_{contact.get('formation1', 'UNIT1')}_{contact.get('formation2', 'UNIT2')}\n")
                            f.write(f"10\n{x:.2f}\n")  # X = Easting
                            f.write(f"20\n{y:.2f}\n")  # Y = Northing
                            f.write(f"30\n{z:.2f}\n")  # Z = RL

                        f.write("0\nSEQEND\n")
                        logger.debug(f"Exported contact {contact['name']} with {len(coords)} points")
                    else:
                        logger.warning(f"Contact {contact['name']} has insufficient points: {len(coords)}")

                f.write("0\nENDSEC\n0\nEOF\n")

            logger.info(f"Exported to {filename}")
            messagebox.showinfo("Success", f"Exported to {filename}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    
    def _export_all_sections_dxf(self):
        """Export all sections from batch processing to DXF.
        
        Exports:
        - Units (polygons) with their assigned formation names
        - Faults/polylines with fault_assignment
        - Contacts named as 'unit1-unit2_contact'
        """
        # Ask user what to export
        export_options = messagebox.askyesnocancel(
            "Export Options",
            "Export only ASSIGNED items?\n\n"
            "Yes = Only units/faults with assignments\n"
            "No = Export ALL items\n"
            "Cancel = Abort export"
        )
        
        if export_options is None:
            return
        
        only_assigned = export_options
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".dxf", 
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            units_exported = 0
            faults_exported = 0
            contacts_exported = 0
            skipped_no_northing = 0
            skipped_bad_coords = 0
            
            with open(filename, "w") as f:
                # DXF header
                f.write("0\nSECTION\n2\nENTITIES\n")
                
                # Export units from all_geological_units
                for unit_key, unit in self.all_geological_units.items():
                    formation = unit.get("formation", "UNKNOWN")
                    
                    # Skip unassigned if only_assigned is True
                    if only_assigned:
                        if formation == "UNKNOWN" or formation.startswith("UNIT") or formation.startswith("["):
                            continue
                    
                    # Get northing - handle None explicitly
                    northing = unit.get("northing")
                    if northing is None:
                        # Try to find northing from section data
                        source_pdf = unit.get("source_pdf", "")
                        page_num = unit.get("page_num", 0)
                        
                        # Look up in section data
                        for (pdf, pg), sec_data in self.all_sections_data.items():
                            if str(pdf) == source_pdf or (hasattr(pdf, 'name') and pdf.name in source_pdf):
                                if pg == page_num:
                                    northing = sec_data.get("northing")
                                    break
                        
                        if northing is None:
                            skipped_no_northing += 1
                            continue
                    
                    # Get vertices (already in real-world coords)
                    vertices = unit.get("vertices", [])
                    if len(vertices) < 6:  # Need at least 3 points
                        continue
                    
                    # Build coordinates, skipping any None values
                    coords = []
                    has_bad_coord = False
                    for i in range(0, len(vertices), 2):
                        if i + 1 < len(vertices):
                            easting = vertices[i]
                            rl = vertices[i + 1]
                            if easting is None or rl is None:
                                has_bad_coord = True
                                break
                            try:
                                coords.append((float(easting), float(northing), float(rl)))
                            except (TypeError, ValueError):
                                has_bad_coord = True
                                break
                    
                    if has_bad_coord or len(coords) < 3:
                        skipped_bad_coords += 1
                        continue
                    
                    # Close polygon
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    # Use formation as layer name (sanitize for DXF)
                    layer_name = formation.replace(" ", "_").replace("-", "_").replace("/", "_")
                    
                    f.write("0\nPOLYLINE\n")
                    f.write(f"8\n{layer_name}\n")
                    f.write("66\n1\n")
                    f.write("70\n8\n")  # 3D polyline
                    
                    for x, y, z in coords:
                        f.write("0\nVERTEX\n")
                        f.write(f"8\n{layer_name}\n")
                        f.write(f"10\n{x:.2f}\n")
                        f.write(f"20\n{y:.2f}\n")
                        f.write(f"30\n{z:.2f}\n")
                    
                    f.write("0\nSEQEND\n")
                    units_exported += 1
                
                # Export faults/polylines from section data
                for (pdf_path, page_num), section_data in self.all_sections_data.items():
                    northing = section_data.get("northing")
                    if northing is None:
                        continue  # Skip sections without valid northing
                    
                    for poly_name, polyline in section_data.get("polylines", {}).items():
                        fault_assignment = polyline.get("fault_assignment")
                        is_fault = polyline.get("is_fault", False) or polyline.get("type") == "Fault"
                        
                        # Skip unassigned if only_assigned is True
                        if only_assigned and not fault_assignment:
                            continue
                        
                        vertices = polyline.get("vertices", [])
                        if len(vertices) < 4:  # Need at least 2 points
                            continue
                        
                        # Build coordinates
                        coords = []
                        has_bad_coord = False
                        for i in range(0, len(vertices), 2):
                            if i + 1 < len(vertices):
                                e = vertices[i]
                                r = vertices[i + 1]
                                if e is None or r is None:
                                    has_bad_coord = True
                                    break
                                try:
                                    coords.append((float(e), float(northing), float(r)))
                                except (TypeError, ValueError):
                                    has_bad_coord = True
                                    break
                        
                        if has_bad_coord or len(coords) < 2:
                            continue
                        
                        # Layer name based on fault assignment
                        if fault_assignment:
                            layer_name = f"FAULT_{fault_assignment.replace(' ', '_').replace('-', '_')}"
                        else:
                            layer_name = "FAULTS_UNASSIGNED"
                        
                        f.write("0\nPOLYLINE\n")
                        f.write(f"8\n{layer_name}\n")
                        f.write("66\n1\n")
                        f.write("70\n8\n")
                        
                        for x, y, z in coords:
                            f.write("0\nVERTEX\n")
                            f.write(f"8\n{layer_name}\n")
                            f.write(f"10\n{x:.2f}\n")
                            f.write(f"20\n{y:.2f}\n")
                            f.write(f"30\n{z:.2f}\n")
                        
                        f.write("0\nSEQEND\n")
                        faults_exported += 1
                
                # Export contacts with proper naming
                for contact in self.all_contacts:
                    northing = contact.get("northing")
                    if northing is None:
                        continue  # Skip contacts without valid northing
                    
                    # Get formation names (prioritize formation1/formation2 over unit names)
                    formation1 = contact.get("formation1", contact.get("unit1", "UNIT1"))
                    formation2 = contact.get("formation2", contact.get("unit2", "UNIT2"))
                    
                    # Skip if both formations are unassigned and only_assigned is True
                    if only_assigned:
                        if (formation1 == "UNKNOWN" or formation1.startswith("UNIT") or formation1.startswith("[")) and \
                           (formation2 == "UNKNOWN" or formation2.startswith("UNIT") or formation2.startswith("[")):
                            continue
                    
                    vertices = contact.get("vertices", [])
                    
                    # Build coordinates - TRIM HOOKS BEFORE EXPORT
                    coords = []
                    has_bad_coord = False
                    for i in range(0, len(vertices), 2):
                        if i + 1 < len(vertices):
                            e = vertices[i]
                            r = vertices[i + 1]
                            if e is None or r is None:
                                has_bad_coord = True
                                break
                            try:
                                coords.append((float(e), float(northing), float(r)))
                            except (TypeError, ValueError):
                                has_bad_coord = True
                                break
                    
                    if has_bad_coord or len(coords) < 2:
                        continue
                    
                    # TRIM ENDPOINT HOOKS
                    coords = self._trim_hook_segments(coords)
                    if len(coords) < 2:
                        continue
                    
                    # Layer name: formation1-formation2_contact (clean names only)
                    layer_name = f"{formation1}-{formation2}_contact".replace(" ", "_").replace("/", "_")
                    
                    f.write("0\nPOLYLINE\n")
                    f.write(f"8\n{layer_name}\n")
                    f.write("66\n1\n")
                    f.write("70\n8\n")
                    
                    for x, y, z in coords:
                        f.write("0\nVERTEX\n")
                        f.write(f"8\n{layer_name}\n")
                        f.write(f"10\n{x:.2f}\n")
                        f.write(f"20\n{y:.2f}\n")
                        f.write(f"30\n{z:.2f}\n")
                    
                    f.write("0\nSEQEND\n")
                    contacts_exported += 1
                
                f.write("0\nENDSEC\n0\nEOF\n")
            
            summary = f"DXF Export Complete!\n\n"
            summary += f"Units exported: {units_exported}\n"
            summary += f"Faults exported: {faults_exported}\n"
            summary += f"Contacts exported: {contacts_exported}"
            if skipped_no_northing > 0:
                summary += f"\n\nSkipped (no northing): {skipped_no_northing}"
            if skipped_bad_coords > 0:
                summary += f"\nSkipped (bad coordinates): {skipped_bad_coords}"
            
            logger.info(f"Exported to {filename}: {units_exported} units, {faults_exported} faults, {contacts_exported} contacts")
            if skipped_no_northing > 0 or skipped_bad_coords > 0:
                logger.warning(f"Skipped: {skipped_no_northing} no northing, {skipped_bad_coords} bad coords")
            
            messagebox.showinfo("Success", summary)
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    # Part 11: Batch processing and Project Management methods:
    def batch_export(self):
        """Batch export PDFs to GeoTIFFs."""
        input_dir = filedialog.askdirectory(title="Select folder containing PDFs")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select output folder for GeoTIFFs")
        if not output_dir:
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            messagebox.showwarning("No PDFs", "No PDF files found in selected folder")
            return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Export Progress")
        progress_window.geometry("400x150")

        progress_label = ttk.Label(progress_window, text="Processing...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=350, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = len(pdf_files)

        current_file_label = ttk.Label(progress_window, text="")
        current_file_label.pack(pady=5)

        def update_progress(current, total, filename):
            progress_bar["value"] = current
            current_file_label["text"] = f"Processing: {filename}"
            progress_window.update()

        # Scan for missing northings
        self.status_var.set("Scanning PDFs for coordinate information...")
        northings = self.batch_processor.scan_for_missing_northings(pdf_files)

        missing = {
            path: None
            for path, page_dict in northings.items()
            for page_num, val in page_dict.items()
            if val is None
        }

        if missing:
            user_northings = self.get_missing_northings_dialog(missing)
            if not user_northings:
                progress_window.destroy()
                return

            self.batch_processor.northing_overrides = user_northings

        # Process files
        self.status_var.set(f"Batch processing {len(pdf_files)} files...")

        results = self.batch_processor.process_batch(
            pdf_files, output_path, progress_callback=update_progress
        )

        progress_window.destroy()

        # Show results
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful

        result_msg = f"Batch export complete!\n\nSuccessful: {successful}\nFailed: {failed}"

        if failed > 0:
            result_msg += "\n\nFailed files:\n"
            for filename, success in results.items():
                if not success:
                    result_msg += f"  - {filename}\n"

        messagebox.showinfo("Batch Export Results", result_msg)
        self.status_var.set("Batch export complete")

    def batch_export_dxf(self):
        """Batch export PDFs to DXF files."""
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("DXF Export Options")
        export_dialog.geometry("300x200")

        ttk.Label(
            export_dialog, text="What would you like to export?", font=("Arial", 10, "bold")
        ).pack(pady=10)

        export_units_var = tk.BooleanVar(value=True)
        export_contacts_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            export_dialog, text="Export Geological Units", variable=export_units_var
        ).pack(pady=5)
        ttk.Checkbutton(export_dialog, text="Export Contacts", variable=export_contacts_var).pack(
            pady=5
        )

        ttk.Label(
            export_dialog,
            text="Note: Units will be checked for overlaps",
            font=("Arial", 9, "italic"),
        ).pack(pady=10)

        result = {"export_units": True, "export_contacts": True}

        def proceed():
            result["export_units"] = export_units_var.get()
            result["export_contacts"] = export_contacts_var.get()
            export_dialog.destroy()

        def cancel():
            result["export_units"] = None
            export_dialog.destroy()

        button_frame = ttk.Frame(export_dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Proceed", command=proceed).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        export_dialog.wait_window()

        if result["export_units"] is None:
            return

        input_dir = filedialog.askdirectory(title="Select folder containing PDFs")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select output folder for DXF files")
        if not output_dir:
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            messagebox.showwarning("No PDFs", "No PDF files found in selected folder")
            return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch DXF Export Progress")
        progress_window.geometry("400x150")

        progress_label = ttk.Label(progress_window, text="Processing...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=350, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = len(pdf_files)

        current_file_label = ttk.Label(progress_window, text="")
        current_file_label.pack(pady=5)

        def update_progress(current, total, filename):
            progress_bar["value"] = current
            current_file_label["text"] = f"Processing: {filename}"
            progress_window.update()

        # Process files
        self.status_var.set(f"Batch processing {len(pdf_files)} files to DXF...")

        results = self.batch_processor.process_batch_to_dxf(
            pdf_files,
            output_path,
            progress_callback=update_progress,
            export_units=result["export_units"],
            export_contacts=result["export_contacts"],
        )

        progress_window.destroy()

        # Show results
        total_features = sum(len(features) for features in results.values())
        successful = sum(
            1 for pdf_results in results.values() for success in pdf_results.values() if success
        )

        result_msg = f"Batch DXF export complete!\n\n"
        result_msg += f"PDFs processed: {len(results)}\n"
        result_msg += f"Total features exported: {successful}/{total_features}\n"

        messagebox.showinfo("Batch DXF Export Results", result_msg)
        self.status_var.set("Batch DXF export complete")

    def _trim_hook_segments(self, coords: List[Tuple[float, float, float]], min_length: float = 5.0) -> List[Tuple[float, float, float]]:
        """
        Trim short segments or direction reversals from start/end of contact line.
        
        Args:
            coords: List of (easting, northing, rl) tuples
            min_length: Minimum segment length in meters
            
        Returns:
            Trimmed coordinate list
        """
        if len(coords) < 3:
            return coords
        
        import numpy as np
        
        # Trim from start
        start_idx = 0
        while start_idx < len(coords) - 2:
            dx = coords[start_idx + 1][0] - coords[start_idx][0]
            dz = coords[start_idx + 1][2] - coords[start_idx][2]
            seg_length = np.sqrt(dx**2 + dz**2)
            
            # Remove if too short
            if seg_length < min_length:
                start_idx += 1
                continue
            
            # Check for direction reversal
            if start_idx < len(coords) - 3:
                dx1 = coords[start_idx + 1][0] - coords[start_idx][0]
                dz1 = coords[start_idx + 1][2] - coords[start_idx][2]
                dx2 = coords[start_idx + 2][0] - coords[start_idx + 1][0]
                dz2 = coords[start_idx + 2][2] - coords[start_idx + 1][2]
                
                dot = dx1 * dx2 + dz1 * dz2
                if dot < 0:
                    start_idx += 1
                    continue
            
            break
        
        # Trim from end
        end_idx = len(coords)
        while end_idx > start_idx + 2:
            dx = coords[end_idx - 1][0] - coords[end_idx - 2][0]
            dz = coords[end_idx - 1][2] - coords[end_idx - 2][2]
            seg_length = np.sqrt(dx**2 + dz**2)
            
            # Remove if too short
            if seg_length < min_length:
                end_idx -= 1
                continue
            
            # Check for direction reversal
            if end_idx > start_idx + 3:
                dx1 = coords[end_idx - 2][0] - coords[end_idx - 3][0]
                dz1 = coords[end_idx - 2][2] - coords[end_idx - 3][2]
                dx2 = coords[end_idx - 1][0] - coords[end_idx - 2][0]
                dz2 = coords[end_idx - 1][2] - coords[end_idx - 2][2]
                
                dot = dx1 * dx2 + dz1 * dz2
                if dot < 0:
                    end_idx -= 1
                    continue
            
            break
        
        return coords[start_idx:end_idx]

    def get_missing_northings_dialog(self, missing_northings: Dict) -> Dict:
        """Create dialog to get northing values for PDFs that are missing them."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Missing Northing Values")
        dialog.geometry("600x400")

        ttk.Label(
            dialog,
            text="The following PDFs are missing northing values. Please enter them:",
            wraplength=550,
        ).pack(pady=10)

        # Create scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        entries = {}
        for i, (pdf_path, current_value) in enumerate(missing_northings.items()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(frame, text=f"{pdf_path.name}:", width=40).pack(side=tk.LEFT)

            entry = ttk.Entry(frame, width=20)
            if current_value:
                entry.insert(0, str(int(current_value)))
            else:
                # Try to parse from filename
                match = re.search(r"(\d{6}|\d{3}[,\s]\d{3})", pdf_path.stem)
                if match:
                    suggested = match.group(1).replace(",", "").replace(" ", "")
                    entry.insert(0, suggested)

            entry.pack(side=tk.LEFT, padx=5)
            entries[pdf_path] = entry

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        result = {}

        def apply():
            for pdf_path, entry in entries.items():
                try:
                    value = entry.get().strip()
                    if value:
                        value = value.replace(",", "").replace(" ", "")
                        northing = float(value)
                        result[pdf_path] = northing
                except ValueError:
                    messagebox.showerror(
                        "Invalid Input", f"Invalid northing value for {pdf_path.name}"
                    )
                    return
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Apply", command=apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

        return result

    def save_project(self):
        """Save current project."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                project_data = {
                    "pdf_paths": [str(p) for p in self.pdf_list],
                    "current_pdf_index": self.current_pdf_index,
                    "coord_system": self.georeferencer.coord_system,
                    "geological_units": {},
                    "contacts": [],
                    "strat_column": {
                        "version": 2,
                        "units": [u.to_dict() for u in self.strat_column.units.values()],
                        "prospects": [p.to_dict() for p in self.strat_column.prospects.values()],
                        "faults": [f.to_dict() for f in self.strat_column.faults.values()],
                    },
                    "all_sections_data": {},
                    "all_geological_units": self.all_geological_units,
                    "all_contacts": self.all_contacts,
                    "user_assigned_polygons": list(self.user_assigned_polygons),
                }

                # Save current units
                for name, unit in self.feature_extractor.geological_units.items():
                    project_data["geological_units"][name] = {
                        "type": unit["type"],
                        "vertices": unit["vertices"],
                        "color": (
                            list(unit["color"])
                            if isinstance(unit["color"], tuple)
                            else unit["color"]
                        ),
                        "formation": unit["formation"],
                        "unit_number": unit["unit_number"],
                    }

                # Save current contacts
                for contact in self.feature_extractor.contacts:
                    project_data["contacts"].append(
                        {
                            "name": contact["name"],
                            "unit1": contact["unit1"],
                            "unit2": contact["unit2"],
                            "vertices": contact["vertices"],
                        }
                    )

                # Save section data (simplified to avoid circular references)
                for key, section_data in self.all_sections_data.items():
                    pdf_str = str(key[0]) if isinstance(key[0], Path) else key[0]
                    page_num = key[1]
                    simple_key = f"{pdf_str}_{page_num}"

                    project_data["all_sections_data"][simple_key] = {
                        "northing": section_data.get("northing"),
                        "easting_min": section_data.get("easting_min"),
                        "easting_max": section_data.get("easting_max"),
                        "rl_min": section_data.get("rl_min"),
                        "rl_max": section_data.get("rl_max"),
                        "pdf_path": section_data.get("pdf_path"),
                        "page_num": section_data.get("page_num"),
                    }

                with open(filename, "w") as f:
                    json.dump(project_data, f, indent=2)

                messagebox.showinfo("Success", "Project saved successfully")

            except Exception as e:
                logger.error(f"Failed to save project: {e}")
                messagebox.showerror("Error", f"Failed to save project: {str(e)}")

    def write_assignments_to_pdf(self):
        """
        Write unit and fault assignments back to PDF files as new annotations.
        Uses PDFAnnotationWriter to create duplicate annotations with assignments
        in the Subject field, preserving original annotations.
        """
        if not self.all_sections_data:
            messagebox.showwarning("Warning", "No sections to save. Process PDFs first.")
            return
        
        # Count assignments
        unit_count = 0
        fault_count = 0
        for section_data in self.all_sections_data.values():
            for unit in section_data.get("units", {}).values():
                if unit.get("unit_assignment"):
                    unit_count += 1
            for polyline in section_data.get("polylines", {}).values():
                if polyline.get("fault_assignment"):
                    fault_count += 1
        
        if unit_count == 0 and fault_count == 0:
            messagebox.showwarning("Warning", "No assignments to write. Assign units/faults first.")
            return
        
        # Show options dialog
        dialog = WriteAssignmentsDialog(self.root, unit_count, fault_count, bool(self.grouped_contacts))
        options = dialog.result
        
        if not options:
            return  # User cancelled
        
        try:
            # Prepare contacts data if requested
            contacts_data = None
            if options.get("include_contacts") and self.grouped_contacts:
                contacts_data = self._prepare_contacts_for_export()
            
            # Use the batch writer
            results = write_assignments_batch(
                pdf_files=list(set(str(k[0]) for k in self.all_sections_data.keys())),
                all_sections_data=self.all_sections_data,
                contacts_data=contacts_data,
                create_backups=True,
                duplicate_mode=options.get("duplicate_mode", True)
            )
            
            # Show results
            msg = f"Modified {results['modified_files']} PDF file(s).\n"
            msg += f"Units written: {unit_count}\n"
            msg += f"Faults written: {fault_count}\n"
            if contacts_data:
                contact_count = sum(len(c) for c in contacts_data.values())
                msg += f"Contacts written: {contact_count}\n"
            
            if results['errors']:
                msg += f"\nErrors ({len(results['errors'])}):\n"
                msg += "\n".join(results['errors'][:5])
                if len(results['errors']) > 5:
                    msg += f"\n... and {len(results['errors']) - 5} more"
                messagebox.showwarning("Write Complete (with errors)", msg)
            else:
                messagebox.showinfo("Write Complete", msg)
            
        except Exception as e:
            logger.error(f"Error writing to PDFs: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to write to PDFs: {str(e)}")
    
    def _prepare_contacts_for_export(self) -> Dict[Tuple, List[Dict]]:
        """Prepare contacts data for PDF export."""
        contacts_data = {}
        
        for group_name, group in self.grouped_contacts.items():
            for polyline in group.polylines:
                section_key = polyline.section_key
                if section_key not in contacts_data:
                    contacts_data[section_key] = []
                
                # Convert ContactPolyline to dict format expected by writer
                contact_dict = {
                    "vertices": polyline.vertices,
                    "pdf_vertices": getattr(polyline, 'pdf_vertices', None) or polyline.vertices,
                    "unit1": group.formation1,
                    "unit2": group.formation2,
                    "color": getattr(group, 'color', (0, 0, 0)),
                }
                contacts_data[section_key].append(contact_dict)
        
        return contacts_data

    def load_project(self):
        """Load a project."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "r") as f:
                    project_data = json.load(f)

                # Load PDFs
                self.pdf_list = [Path(p) for p in project_data.get("pdf_paths", [])]
                self.current_pdf_index = project_data.get("current_pdf_index", 0)

                # Load first PDF if available
                if self.pdf_list and self.current_pdf_index < len(self.pdf_list):
                    self.load_pdf_at_index(self.current_pdf_index)

                # Load coordinate system
                self.georeferencer.coord_system = project_data.get("coord_system")
                self.display_coordinate_info()

                # Load geological units
                self.feature_extractor.geological_units = {}
                for name, unit_data in project_data.get("geological_units", {}).items():
                    unit = unit_data.copy()
                    if isinstance(unit.get("color"), list):
                        unit["color"] = tuple(unit["color"])
                    unit["name"] = name
                    unit["classification"] = "UNIT"
                    self.feature_extractor.geological_units[name] = unit

                # Load contacts
                self.feature_extractor.contacts = project_data.get("contacts", [])

                # Load strat column
                if "strat_column" in project_data:
                    strat_data = project_data["strat_column"]
                    version = strat_data.get("version", 1)
                    
                    if version >= 2:
                        # V2 format - load directly
                        self.strat_column.prospects.clear()
                        self.strat_column.units.clear()
                        self.strat_column.faults.clear()
                        
                        for p_data in strat_data.get("prospects", []):
                            prospect = Prospect.from_dict(p_data)
                            self.strat_column.prospects[prospect.name] = prospect
                        
                        for u_data in strat_data.get("units", []):
                            unit = StratUnit.from_dict(u_data)
                            self.strat_column.units[unit.name] = unit
                        
                        for f_data in strat_data.get("faults", []):
                            fault = Fault.from_dict(f_data)
                            self.strat_column.faults[fault.name] = fault
                    else:
                        # V1 format - use backward compatibility loader
                        self.strat_column._load_v1(strat_data)
                    
                    # Update auto-labeler
                    self.auto_labeler = AutoLabeler(self.strat_column)
                    self.update_strat_buttons()
                
                # Load user-assigned polygons
                self.user_assigned_polygons = set(project_data.get("user_assigned_polygons", []))

                # Load all data
                self.all_geological_units = project_data.get("all_geological_units", {})
                self.all_contacts = project_data.get("all_contacts", [])

                # Reconstruct section data
                self.all_sections_data = {}
                for simple_key, section_data in project_data.get("all_sections_data", {}).items():
                    # Try to reconstruct the key
                    parts = simple_key.rsplit("_", 1)
                    if len(parts) == 2:
                        pdf_path = Path(parts[0])
                        page_num = int(parts[1])
                        self.all_sections_data[(pdf_path, page_num)] = section_data

                # Update UI
                self.populate_feature_tree_all()
                self.calculate_global_ranges()
                self.filter_sections()
                self.update_display()

                messagebox.showinfo("Success", "Project loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load project: {e}")
                messagebox.showerror("Error", f"Failed to load project: {str(e)}")

    def correlate_sections(self):
        """Correlate stratigraphy between multiple sections."""
        input_dir = filedialog.askdirectory(title="Select folder containing cross-section PDFs")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select output folder for correlation results")
        if not output_dir:
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            messagebox.showwarning("No PDFs", "No PDF files found in selected folder")
            return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Section Correlation Progress")
        progress_window.geometry("500x250")

        progress_label = ttk.Label(progress_window, text="Processing sections...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=450, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = len(pdf_files)

        current_file_label = ttk.Label(progress_window, text="")
        current_file_label.pack(pady=5)

        results_text = tk.Text(progress_window, height=6, width=60)
        results_text.pack(pady=10)

        def update_progress(current, total, filename):
            progress_bar["value"] = current
            current_file_label["text"] = f"Processing: {filename}"
            progress_window.update()

        # Process with correlation
        self.status_var.set(f"Processing and correlating {len(pdf_files)} sections...")

        results = self.batch_processor.process_batch_with_correlation(
            pdf_files,
            output_path,
            correlate=True,
            export_ties=True,
            progress_callback=update_progress,
        )

        # Display results
        results_text.insert(1.0, f"Processing complete!\n\n")
        results_text.insert(tk.END, f"Sections processed: {len(results['sections'])}\n")
        results_text.insert(tk.END, f"Correlations found: {len(results['correlations'])}\n")
        results_text.insert(tk.END, f"Tie lines generated: {len(results['tie_lines'])}\n")
        results_text.insert(tk.END, f"\nExported files:\n")
        results_text.insert(tk.END, f"  - unified_sections.csv\n")
        results_text.insert(tk.END, f"  - all_sections.dxf\n")
        results_text.insert(tk.END, f"  - tie_lines.dxf\n")
        results_text.insert(tk.END, f"  - correlations.json\n")

        ttk.Button(progress_window, text="Close", command=progress_window.destroy).pack(pady=5)

        self.status_var.set("Section correlation complete")

    def set_northing(self):
        """Manually set northing value."""
        current = (
            self.georeferencer.coord_system.get("northing", 125000)
            if self.georeferencer.coord_system
            else 125000
        )

        is_default = current == 125000

        message = "No northing found in PDF. Please enter the Northing value:\n\n"
        message += "Expected format: 6-digit number (e.g., 114500 or 114,500)\n"
        message += "This should match the 'Location' box or section title on the page."

        if self.pdf_path:
            page_info = f"\n\nFile: {self.pdf_path.name}"
            if self.current_pdf:
                page_info += f"\nPage: {self.current_page_num + 1} of {len(self.current_pdf)}"
            message += page_info

        initial_value = "" if is_default else str(int(current))

        northing_str = simpledialog.askstring("Set Northing", message, initialvalue=initial_value)

        if northing_str:
            try:
                northing_str = northing_str.replace(",", "")
                northing = float(re.sub(r"[^0-9]", "", northing_str))

                if not self.georeferencer.coord_system:
                    self.georeferencer.coord_system = {
                        "northing": northing,
                        "northing_text": f"Manual entry: {northing_str}",
                        "easting_labels": [],
                        "rl_labels": [],
                        "page_rect": self.current_page.rect if self.current_page else None,
                    }
                else:
                    self.georeferencer.coord_system["northing"] = northing
                    self.georeferencer.coord_system["northing_text"] = (
                        f"Manual entry: {northing_str}"
                    )

                self.display_coordinate_info()
                self.status_var.set(f"Northing set to {northing}")
                messagebox.showinfo("Success", f"Northing set to: {northing}")

            except ValueError:
                messagebox.showerror("Error", "Invalid northing value")

    # Part 13: Tree Selection and Utility Methods
    def on_tree_click(self, event):
        """Handle click events on the tree view."""
        region = self.feature_tree.identify_region(event.x, event.y)
        if region == "tree":
            item = self.feature_tree.identify_row(event.y)
            if item:
                # Toggle selection
                self.selected_items[item] = not self.selected_items.get(item, True)
                # Update display
                if self.selected_items[item]:
                    self.feature_tree.item(item, text="✓")
                else:
                    self.feature_tree.item(item, text="")
                # Update visualization
                self.update_display()

    def select_all(self):
        """Select all items in feature tree."""
        for item in self.feature_tree.get_children():
            self.selected_items[item] = True
            self.feature_tree.item(item, text="✓")
            # Also select children if any
            for child in self.feature_tree.get_children(item):
                self.selected_items[child] = True
                self.feature_tree.item(child, text="✓")
        self.update_display()

    def select_none(self):
        """Deselect all items in feature tree."""
        for item in self.feature_tree.get_children():
            self.selected_items[item] = False
            self.feature_tree.item(item, text="")
            # Also deselect children if any
            for child in self.feature_tree.get_children(item):
                self.selected_items[child] = False
                self.feature_tree.item(child, text="")
        self.update_display()

    def invert_selection(self):
        """Invert selection in feature tree."""
        for item in self.feature_tree.get_children():
            self.selected_items[item] = not self.selected_items.get(item, True)
            if self.selected_items[item]:
                self.feature_tree.item(item, text="✓")
            else:
                self.feature_tree.item(item, text="")
            # Also invert children if any
            for child in self.feature_tree.get_children(item):
                self.selected_items[child] = not self.selected_items.get(child, True)
                if self.selected_items[child]:
                    self.feature_tree.item(child, text="✓")
                else:
                    self.feature_tree.item(child, text="")
        self.update_display()

    def on_pdf_click(self, event):
        """Handle mouse clicks on PDF canvas for feature selection."""
        if event.inaxes != self.ax_pdf or event.button != 1:  # Left click only
            return

        # Convert click to PDF coordinates
        x, y = event.xdata / 2, event.ydata / 2  # Divide by scale factor

        clicked_feature = None

        # Find clicked feature
        for unit_name, unit in self.feature_extractor.geological_units.items():
            coords = []
            for i in range(0, len(unit["vertices"]), 2):
                if i + 1 < len(unit["vertices"]):
                    coords.append((unit["vertices"][i], unit["vertices"][i + 1]))

            if len(coords) >= 3:
                from matplotlib.path import Path as MPath

                poly_path = MPath(coords)
                if poly_path.contains_point((x, y)):
                    clicked_feature = unit_name

                    # Toggle selection
                    if self.selected_feature_name == unit_name:
                        self.selected_feature_name = None
                    else:
                        self.selected_feature_name = unit_name

                        # Update tree selection
                        for item in self.feature_tree.get_children():
                            values = self.feature_tree.item(item, "values")
                            if len(values) > 1 and values[1] == unit_name:
                                self.feature_tree.selection_set(item)
                                break

                    self.status_var.set(
                        f"Selected: {unit_name}"
                        if self.selected_feature_name
                        else "Selection cleared"
                    )
                    break

        # Clear selection if clicked on empty space
        if not clicked_feature:
            self.selected_feature_name = None
            self.feature_tree.selection_set([])
            self.status_var.set("Selection cleared")

        self.update_pdf_display()

    # === PDF Calibration Tool Methods ===
    
    def open_calibration_tool(self):
        """Open the PDF calibration dialog to configure extraction filters."""
        # Determine which PDF to use for calibration
        pdf_path = None
        if self.pdf_list:
            pdf_path = self.pdf_list[0]
        elif self.pdf_path:
            pdf_path = self.pdf_path
        
        # Open calibration dialog
        dialog = PDFCalibrationDialog(self.root, pdf_path, self.extraction_filter)
        self.root.wait_window(dialog.dialog)
        
        # Get the configured filter
        self.extraction_filter = dialog.get_filter()
        
        # Apply to feature extractor
        self.feature_extractor.set_extraction_filter(self.extraction_filter)
        
        self.status_var.set("Extraction filter updated")
        logger.info("Extraction filter configured via calibration tool")
    
    def load_filter_config(self):
        """Load extraction filter configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Filter Configuration"
        )
        if filepath:
            try:
                self.extraction_filter.load(Path(filepath))
                self.feature_extractor.set_extraction_filter(self.extraction_filter)
                messagebox.showinfo("Success", f"Loaded filter config from {filepath}")
                self.status_var.set(f"Loaded filter: {Path(filepath).name}")
            except Exception as e:
                logger.error(f"Error loading filter config: {e}")
                messagebox.showerror("Error", f"Failed to load filter config: {str(e)}")
    
    def save_filter_config(self):
        """Save extraction filter configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Filter Configuration"
        )
        if filepath:
            try:
                self.extraction_filter.save(Path(filepath))
                messagebox.showinfo("Success", f"Saved filter config to {filepath}")
                self.status_var.set(f"Saved filter: {Path(filepath).name}")
            except Exception as e:
                logger.error(f"Error saving filter config: {e}")
                messagebox.showerror("Error", f"Failed to save filter config: {str(e)}")

    def show_debug_info(self):
        """Show debug information."""
        if not self.current_page:
            messagebox.showinfo("Debug Info", "No PDF loaded")
            return

        debug_info = "=== DEBUG INFORMATION ===\n\n"

        # Page info
        debug_info += f"PDF: {self.pdf_path.name if self.pdf_path else 'None'}\n"
        debug_info += f"Page size: {self.current_page.rect.width:.1f} x {self.current_page.rect.height:.1f}\n\n"

        # Coordinate system
        if self.georeferencer.coord_system:
            cs = self.georeferencer.coord_system
            debug_info += "Coordinate System:\n"
            debug_info += f"  Northing: {cs.get('northing', 'N/A')}\n"

            if "easting_min" in cs:
                debug_info += f"  Easting range: {cs['easting_min']} - {cs['easting_max']}\n"
                debug_info += f"  RL range: {cs['rl_min']} - {cs['rl_max']}\n"
                debug_info += f"  PDF X range: {cs['pdf_x_min']:.1f} - {cs['pdf_x_max']:.1f}\n"
                debug_info += f"  PDF Y range: {cs['pdf_y_min']:.1f} - {cs['pdf_y_max']:.1f}\n"

            if cs.get("scale_x"):
                debug_info += f"  Scale X: {cs['scale_x']:.2f} px/m\n"
                debug_info += f"  Scale Y: {cs['scale_y']:.2f} px/m\n"

        # Features
        debug_info += f"\nGeological Units: {len(self.feature_extractor.geological_units)}\n"
        debug_info += f"Contacts: {len(self.feature_extractor.contacts)}\n"

        # All sections data
        debug_info += f"\n=== ALL SECTIONS DATA ===\n"
        debug_info += f"Total sections: {len(self.all_sections_data)}\n"
        debug_info += f"Total units: {len(self.all_geological_units)}\n"
        debug_info += f"Total contacts: {len(self.all_contacts)}\n"
        debug_info += f"Northings: {self.northings}\n"

        # Show in window
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Information")
        debug_window.geometry("600x400")

        text_widget = scrolledtext.ScrolledText(debug_window, wrap=tk.WORD, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, debug_info)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(debug_window, text="Close", command=debug_window.destroy).pack(pady=5)

    # Final Part: Event Handlers and Utility Methods
    def on_pick(self, event):
        """Handle pick events on the sections."""
        # Get the mouse event to check for right-click
        mouse_event = event.mouseevent
        is_right_click = mouse_event.button == 3
        
        if event.artist in self.unit_patches:
            unit_data = self.unit_patches[event.artist]
            unit_name = unit_data["name"]
            
            # Right-click to clear assignment (works in any mode)
            if is_right_click:
                current_formation = self.all_geological_units.get(unit_name, {}).get("formation", "UNKNOWN")
                if current_formation != "UNKNOWN" and not current_formation.startswith("UNIT"):
                    if messagebox.askyesno("Clear Assignment", 
                                          f"Clear assignment '{current_formation}' from {unit_name}?"):
                        self.clear_polygon_assignment(unit_name)
                return

            # Check classification mode
            if self.classification_mode == "none":
                # View-only mode - just show info, no selection toggle
                self.status_var.set(f"Polygon: {unit_name} (View mode - switch to Polygon mode to interact)")
                return
            elif self.classification_mode == "contact":
                # Contact mode - ignore polygon clicks
                self.status_var.set(f"Contact mode - polygons not selectable")
                return
            elif self.classification_mode == "fault":
                # Fault mode - ignore polygon clicks
                self.status_var.set(f"Fault mode - use Polygon mode to interact with polygons")
                return
            elif self.classification_mode == "polygon":
                # Polygon mode - allow interaction
                if self.current_unit_assignment:
                    # Assign this polygon to the selected unit
                    self._assign_polygon_to_unit(unit_name, self.current_unit_assignment)
                    return
                else:
                    # Toggle selection
                    if unit_name in self.selected_units:
                        self.selected_units.remove(unit_name)
                        self.status_var.set(f"Deselected: {unit_name}")
                    else:
                        self.selected_units.add(unit_name)
                        self.status_var.set(f"Selected: {unit_name}")
                    
                    # Update displays
                    self.update_selection_display()
                    self.update_section_display()
                    self.update_3d_view()
        
        elif event.artist in self.polyline_patches:
            polyline_data = self.polyline_patches[event.artist]
            polyline_name = polyline_data.get("name", "Unknown")
            
            # Right-click to clear assignment (works in any mode)
            if is_right_click:
                # Find current assignment
                current_assignment = None
                for (pdf, page), section_data in self.all_sections_data.items():
                    if polyline_name in section_data.get('polylines', {}):
                        current_assignment = section_data['polylines'][polyline_name].get('fault_assignment')
                        break
                
                if current_assignment:
                    if messagebox.askyesno("Clear Assignment", 
                                          f"Clear assignment '{current_assignment}' from {polyline_name}?"):
                        self.clear_fault_assignment(polyline_name)
                return
            
            # Check classification mode
            if self.classification_mode == "none":
                # View-only mode - just show info
                self.status_var.set(f"Polyline: {polyline_name} (View mode - switch to Fault mode to interact)")
                return
            elif self.classification_mode == "contact":
                # Contact mode - ignore polyline clicks (for now)
                self.status_var.set(f"Contact mode - use Fault mode to interact with polylines")
                return
            elif self.classification_mode == "polygon":
                # Polygon mode - ignore polyline clicks
                self.status_var.set(f"Polygon mode - use Fault mode to interact with lines")
                return
            elif self.classification_mode == "fault":
                # Fault mode - allow interaction
                if self.current_fault_assignment:
                    self._assign_line_to_fault(polyline_name, self.current_fault_assignment)
                    return
                else:
                    # Toggle selection
                    if polyline_name in self.selected_polylines:
                        self.selected_polylines.discard(polyline_name)
                        self.status_var.set(f"Deselected: {polyline_name}")
                    else:
                        self.selected_polylines.add(polyline_name)
                        self.status_var.set(f"Selected: {polyline_name}")
                    
                    self.update_selection_display()
                    self.update_section_display()

    def on_section_hover(self, event):
        """Handle mouse hover over section display for tooltips."""
        if event.inaxes is None:
            # Mouse left the axes - hide tooltip
            if hasattr(self, 'tooltip_annotation') and self.tooltip_annotation:
                try:
                    self.tooltip_annotation.set_visible(False)
                    self.canvas_sections.draw_idle()
                except:
                    pass
            return
        
        # Check if hovering over any polygon
        found_feature = None
        try:
            for patch, data in self.unit_patches.items():
                if patch.contains(event)[0]:
                    found_feature = data
                    break
        except:
            pass
        
        # Check if hovering over any polyline
        if not found_feature:
            try:
                for line, data in self.polyline_patches.items():
                    if line.contains(event)[0]:
                        found_feature = data
                        break
            except:
                pass
        
        if found_feature:
            # Build tooltip text
            name = found_feature.get('name', 'Unknown')
            assignment = found_feature.get('assignment') or found_feature.get('fault_assignment')
            display = found_feature.get('display_name', name)
            
            if assignment:
                tooltip_text = f"{display}\n({name})"
            else:
                tooltip_text = f"{name}\n(Unassigned)"
            
            # Always recreate annotation on hover (avoids axes mismatch issues)
            # Hide old one first
            if hasattr(self, 'tooltip_annotation') and self.tooltip_annotation:
                try:
                    self.tooltip_annotation.set_visible(False)
                except:
                    pass
            
            # Create new annotation on current axes
            try:
                self.tooltip_annotation = event.inaxes.annotate(
                    tooltip_text,
                    xy=(event.xdata, event.ydata),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8,
                    color='white',
                    backgroundcolor='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', edgecolor='white', alpha=0.9),
                    zorder=100,
                )
                self.canvas_sections.draw_idle()
            except:
                pass
        else:
            # No feature under mouse - hide tooltip
            if hasattr(self, 'tooltip_annotation') and self.tooltip_annotation:
                try:
                    self.tooltip_annotation.set_visible(False)
                    self.canvas_sections.draw_idle()
                except:
                    pass

    def _auto_assign_from_pdf_names(self):
        """
        Auto-assign polygons and faults based on their PDF annotation subject field.
        
        Groups features by their 'formation' field (which comes from PDF subject)
        and ensures they have proper unit_assignment set.
        """
        logger.info("Starting auto-assign from PDF subject names...")
        
        # Group polygons by their formation (from PDF subject field)
        polygon_groups = defaultdict(list)
        for poly_name, poly_data in self.all_geological_units.items():
            formation = poly_data.get('formation', '')
            logger.debug(f"Polygon '{poly_name}': formation='{formation}', unit_assignment='{poly_data.get('unit_assignment')}'")
            
            # Skip generic/unknown formations
            if not formation or formation == 'UNKNOWN' or formation.startswith('UNIT'):
                continue
            
            # Clean the formation name
            clean_name = formation.strip().upper()
            if clean_name and not clean_name.startswith('['):
                polygon_groups[clean_name].append(poly_name)
        
        logger.info(f"Found {len(polygon_groups)} polygon groups from PDF subjects: {list(polygon_groups.keys())}")
        
        # Group faults by their fault_name (from PDF subject field)
        fault_groups = defaultdict(list)
        for (pdf, page), section_data in self.all_sections_data.items():
            for line_name, line_data in section_data.get('polylines', {}).items():
                # Use fault_name which comes from PDF subject
                fault_name = line_data.get('fault_name', '') or line_data.get('author', '')
                if fault_name and fault_name != 'UNKNOWN':
                    clean_name = fault_name.strip().upper()
                    if clean_name and not clean_name.startswith('['):
                        fault_groups[clean_name].append(line_name)
        
        logger.info(f"Found {len(fault_groups)} fault groups from PDF subjects: {list(fault_groups.keys())}")
        
        # Auto-assign polygons
        assigned_polygons = 0
        for formation_name, polygon_list in polygon_groups.items():
            # Check if this formation exists in defined_units
            if formation_name not in self.defined_units:
                logger.info(f"Auto-creating formation '{formation_name}' from PDF subject")
                color = self._generate_color_for_unit(formation_name)
                # Add to defined units
                self.defined_units[formation_name] = {
                    'name': formation_name,
                    'prospect': 'AUTO',
                    'color': color
                }
                # Also add to strat column
                self.strat_column.add_unit(formation_name, color, prospect='AUTO')
            
            # Assign all polygons with this formation name
            for poly_name in polygon_list:
                if poly_name in self.all_geological_units:
                    poly_data = self.all_geological_units[poly_name]
                    current_assignment = poly_data.get('unit_assignment')
                    
                    # Skip if user has manually assigned this polygon
                    if poly_name in self.user_assigned_polygons:
                        logger.debug(f"Skipping '{poly_name}' - user assigned")
                        continue
                    
                    # Only assign if not yet assigned
                    if current_assignment is None:
                        self._assign_polygon_to_unit(poly_name, formation_name)
                        assigned_polygons += 1
                        logger.debug(f"Assigned '{poly_name}' to '{formation_name}'")
        
        # Auto-assign faults
        assigned_faults = 0
        for fault_name, fault_list in fault_groups.items():
            # Check if this fault exists in defined faults
            if fault_name not in self.defined_faults:
                logger.info(f"Auto-creating fault '{fault_name}' from PDF annotations")
                self.defined_faults[fault_name] = {
                    'name': fault_name,
                    'color': '#FF0000',
                    'timing': len(self.defined_faults)
                }
                # Also add to strat column
                self.strat_column.add_fault(fault_name, color='#FF0000')
            
            # Assign all faults with this name
            for line_name in fault_list:
                for (pdf, page), section_data in self.all_sections_data.items():
                    if line_name in section_data.get('polylines', {}):
                        current_assignment = section_data['polylines'][line_name].get('fault_assignment')
                        # Only auto-assign if not already manually assigned
                        if not current_assignment:
                            self._assign_line_to_fault(line_name, fault_name)
                            assigned_faults += 1
                        break
        
        # Always log results and update display
        logger.info(f"Auto-assign complete: {assigned_polygons} polygons, {assigned_faults} faults")
        if assigned_polygons > 0 or assigned_faults > 0:
            self.status_var.set(f"Auto-assigned {assigned_polygons} polygons and {assigned_faults} faults from PDF names")
            self.refresh_unit_buttons()
            self.refresh_fault_buttons()
            self.update_section_display()
        else:
            self.status_var.set(f"No new assignments needed (found {len(polygon_groups)} polygon groups, {len(fault_groups)} fault groups)")
    
    def _generate_color_for_unit(self, unit_name: str) -> Tuple[float, float, float]:
        """Generate a consistent color for a unit based on its name."""
        # Simple hash-based color generation
        import hashlib
        hash_val = int(hashlib.md5(unit_name.encode()).hexdigest()[:6], 16)
        r = ((hash_val >> 16) & 0xFF) / 255.0
        g = ((hash_val >> 8) & 0xFF) / 255.0
        b = (hash_val & 0xFF) / 255.0
        return (r, g, b)
    
    def _assign_polygon_to_unit(self, polygon_name: str, target_unit: str):
        """Assign a polygon to a target unit and auto-suggest neighbors."""
        logger.info(f"Assigning polygon '{polygon_name}' to unit '{target_unit}'")
        
        # Mark as user-assigned (won't be overwritten by auto-labeling)
        self.user_assigned_polygons.add(polygon_name)
        
        if polygon_name in self.all_geological_units:
            self.all_geological_units[polygon_name]['unit_assignment'] = target_unit
            self.all_geological_units[polygon_name]['formation'] = target_unit
            if target_unit in self.defined_units:
                self.all_geological_units[polygon_name]['color'] = self.defined_units[target_unit]['color']
        
        # Find which section this polygon is in
        section_key = None
        section_data = None
        for (pdf, page), data in self.all_sections_data.items():
            if polygon_name in data.get('units', {}):
                section_key = (pdf, page)
                section_data = data
                data['units'][polygon_name]['unit_assignment'] = target_unit
                data['units'][polygon_name]['formation'] = target_unit
                if target_unit in self.defined_units:
                    data['units'][polygon_name]['color'] = self.defined_units[target_unit]['color']
                break
        
        # Auto-suggest labels for adjacent polygons
        if section_data:
            self._auto_label_neighbors(polygon_name, target_unit, section_key, section_data)
        
        self.update_section_display()
        self.status_var.set(f"Assigned {polygon_name} to {target_unit}")
    
    def _auto_label_neighbors(self, assigned_polygon: str, assigned_unit: str, 
                               section_key: tuple, section_data: dict):
        """Auto-label neighboring polygons based on stratigraphy."""
        # Get polygons and faults from this section
        section_polygons = section_data.get('units', {})
        section_faults = list(section_data.get('polylines', {}).values())
        
        # Build user assignments dict (polygon_name -> unit_name)
        user_assignments = {}
        for poly_name in self.user_assigned_polygons:
            if poly_name in section_polygons:
                unit_data = section_polygons[poly_name]
                assignment = unit_data.get('unit_assignment') or unit_data.get('formation')
                if assignment and assignment != 'UNKNOWN':
                    user_assignments[poly_name] = assignment
        
        # Get suggestions from auto-labeler
        try:
            suggestions = self.auto_labeler.suggest_labels(
                assigned_polygon_name=assigned_polygon,
                assigned_unit_name=assigned_unit,
                section_polygons=section_polygons,
                section_faults=section_faults,
                user_assignments=user_assignments,
            )
        except Exception as e:
            logger.warning(f"Auto-labeling error: {e}")
            return
        
        # Apply suggestions (don't overwrite user assignments)
        applied = 0
        for poly_name, suggested_unit in suggestions.items():
            if poly_name in self.user_assigned_polygons:
                continue  # Skip user assignments
            
            # Apply the suggestion
            if poly_name in section_polygons:
                section_polygons[poly_name]['unit_assignment'] = suggested_unit
                section_polygons[poly_name]['formation'] = suggested_unit
                if suggested_unit in self.defined_units:
                    section_polygons[poly_name]['color'] = self.defined_units[suggested_unit]['color']
                
                # Update global units too
                if poly_name in self.all_geological_units:
                    self.all_geological_units[poly_name]['unit_assignment'] = suggested_unit
                    self.all_geological_units[poly_name]['formation'] = suggested_unit
                    if suggested_unit in self.defined_units:
                        self.all_geological_units[poly_name]['color'] = self.defined_units[suggested_unit]['color']
                
                applied += 1
                logger.info(f"Auto-assigned '{poly_name}' to '{suggested_unit}'")
        
        if applied > 0:
            self.status_var.set(f"Assigned {assigned_polygon} to {assigned_unit} + {applied} auto-suggestions")

    def _assign_line_to_fault(self, line_name: str, target_fault: str):
        """Assign a line to a target fault."""
        logger.info(f"Assigning line '{line_name}' to fault '{target_fault}'")
        
        # Update in section data
        for (pdf, page), section_data in self.all_sections_data.items():
            if line_name in section_data.get('polylines', {}):
                section_data['polylines'][line_name]['fault_assignment'] = target_fault
                if target_fault in self.defined_faults:
                    section_data['polylines'][line_name]['color'] = self.defined_faults[target_fault]['color']
        
        self.update_section_display()
        self.status_var.set(f"Assigned {line_name} to {target_fault}")

    def update_selection_display(self):
        """Update the selected units listbox."""
        self.selected_listbox.delete(0, tk.END)

        for unit_name in sorted(self.selected_units):
            parts = unit_name.split("_")
            if len(parts) >= 3:
                display_name = f"{parts[0]} {parts[1]}: {'_'.join(parts[2:])}"
            else:
                display_name = unit_name
            self.selected_listbox.insert(tk.END, display_name)

        self.count_label.config(text=f"{len(self.selected_units)} units selected")

    def update_polyline_selection_display(self):
        """Update the selected polylines listbox."""
        if hasattr(self, "polyline_listbox"):
            self.polyline_listbox.delete(0, tk.END)
            for poly_name in sorted(self.selected_polylines):
                self.polyline_listbox.insert(tk.END, poly_name)

    def clear_selection(self):
        """Clear all selected units."""
        self.selected_units.clear()
        self.update_selection_display()
        self.update_section_display()
        self.update_3d_view()

    def select_similar(self):
        """Select units with similar properties."""
        if not self.selected_units:
            messagebox.showinfo("No Selection", "Please select at least one unit first")
            return

        selected_formations = set()
        for unit_name in self.selected_units:
            if unit_name in self.all_geological_units:
                unit = self.all_geological_units[unit_name]
                selected_formations.add(unit.get("formation", "UNIT"))

        new_selections = set()
        for unit_name, unit in self.all_geological_units.items():
            if unit.get("formation", "UNIT") in selected_formations:
                new_selections.add(unit_name)

        self.selected_units = new_selections
        self.update_selection_display()
        self.update_section_display()
        self.update_3d_view()

    def assign_selected_units(self, strat_unit):
        """Assign selected units to a stratigraphic unit."""
        if not self.selected_units:
            messagebox.showinfo("No Selection", "Please select units to assign")
            return

        count = len(self.selected_units)

        for unit_name in self.selected_units:
            if unit_name in self.all_geological_units:
                unit = self.all_geological_units[unit_name]
                unit["formation"] = strat_unit["name"]
                unit["color"] = strat_unit["color"]

                # Write back to PDF if possible
                if "source_pdf" in unit:
                    try:
                        self.write_assignment_to_pdf(unit, strat_unit)
                    except Exception as e:
                        logger.warning(f"Could not write assignment to PDF: {e}")

        self.clear_selection()
        messagebox.showinfo(
            "Assignment Complete", f"Assigned {count} units to {strat_unit['name']}"
        )

    def write_assignment_to_pdf(self, unit, strat_unit):
        """Write assignment back to the original PDF annotation."""
        pdf_path = unit.get("source_pdf")
        page_num = unit.get("page_num", 0)

        if not pdf_path:
            return

        doc = fitz.open(pdf_path)
        page = doc[page_num]

        for annot in page.annots():
            # Match annotation by vertices
            vertices = self._extract_vertices(annot)
            if vertices and len(vertices) == len(unit.get("vertices", [])):
                match = True
                # Check if vertices match (within tolerance)
                # Note: Need to handle coordinate system differences
                # This is simplified - you may need more robust matching

                if match:
                    info = annot.info
                    info["subject"] = strat_unit["name"]
                    info["content"] = f"Formation: {strat_unit['name']}"
                    annot.set_info(info)

                    if hasattr(annot, "set_colors"):
                        color_dict = {"stroke": strat_unit["color"], "fill": strat_unit["color"]}
                        annot.set_colors(color_dict)
                    break

        doc.save(pdf_path, incremental=True)
        doc.close()

    def _extract_vertices(self, annot) -> Optional[List[float]]:
        """Extract vertices from PDF annotation."""
        vertices = []

        if hasattr(annot, "vertices") and annot.vertices:
            for v in annot.vertices:
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    vertices.extend([float(v[0]), float(v[1])])
                else:
                    vertices.append(float(v))
        elif hasattr(annot, "rect"):
            rect = annot.rect
            if annot.type[0] == 3:  # Line
                vertices = [rect.x0, rect.y0, rect.x1, rect.y1]
            else:  # Rectangle
                vertices = [rect.x0, rect.y0, rect.x1, rect.y0, rect.x1, rect.y1, rect.x0, rect.y1]

        return vertices if vertices else None

    def get_annotation_vertices(self, annot) -> Optional[List[float]]:
        """Public wrapper for extracting vertices from annotations."""
        return self._extract_vertices(annot)

    def on_section_changed(self, event):
        """Handle section selection change."""
        selection = self.section_combo.current()
        if selection >= 0:
            self.current_center_section = selection
            self.update_section_display()
            self.refresh_feature_browser()

    def on_section_count_changed(self):
        """Handle section count change."""
        self.sections_to_display = self.section_count_var.get()
        self.update_section_display()

    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.num == 4 or event.delta > 0:
            self.prev_section()
        elif event.num == 5 or event.delta < 0:
            self.next_section()

    def prev_section(self):
        """Move to previous section."""
        if self.current_center_section > 0:
            self.current_center_section -= 1
            self.section_combo.current(self.current_center_section)
            self.update_section_display()
            self.refresh_feature_browser()

    def next_section(self):
        """Move to next section."""
        if self.current_center_section < len(self.northings) - 1:
            self.current_center_section += 1
            self.section_combo.current(self.current_center_section)
            self.update_section_display()
            self.refresh_feature_browser()

    def jump_to_section(self, index):
        """Jump to specific section index."""
        if 0 <= index < len(self.northings):
            self.current_center_section = index
            self.section_combo.current(index)
            self.update_section_display()
            self.refresh_feature_browser()

    def prev_page(self):
        """Navigate to previous page or PDF."""
        if not self.current_pdf:
            return

        if self.current_page_num > 0:
            self.current_page_num -= 1
            self.current_page = self.current_pdf[self.current_page_num]
            self.update_page_label()
            self.update_pdf_display()
        elif self.current_pdf_index > 0:
            self.current_pdf_index -= 1
            self.load_pdf_at_index(self.current_pdf_index)
            if self.current_pdf and len(self.current_pdf) > 0:
                self.current_page_num = len(self.current_pdf) - 1
                self.current_page = self.current_pdf[self.current_page_num]
                self.update_page_label()
                self.update_pdf_display()

    def next_page(self):
        """Navigate to next page or PDF."""
        if not self.current_pdf:
            return

        if self.current_page_num < len(self.current_pdf) - 1:
            self.current_page_num += 1
            self.current_page = self.current_pdf[self.current_page_num]
            self.update_page_label()
            self.update_pdf_display()
        elif self.current_pdf_index < len(self.pdf_list) - 1:
            self.current_pdf_index += 1
            self.load_pdf_at_index(self.current_pdf_index)

    def update_page_label(self):
        """Update page navigation label."""
        if self.current_pdf and self.pdf_list:
            pdf_info = f"PDF {self.current_pdf_index + 1}/{len(self.pdf_list)}"
            page_info = f"Page {self.current_page_num + 1}/{len(self.current_pdf)}"
            self.page_label.config(text=f"{pdf_info} | {page_info}")
        elif self.current_pdf:
            self.page_label.config(text=f"Page {self.current_page_num + 1}/{len(self.current_pdf)}")
        else:
            self.page_label.config(text="No PDF loaded")

    def on_classification_mode_changed(self):
        """Handle classification mode change."""
        mode = self.mode_var.get()
        self.classification_mode = mode
        
        if mode == "none":
            self.mode_label.config(text="Mode: View Only", foreground="black")
            self.assignment_label.config(text="Click a unit/fault button, then click items to assign")
        elif mode == "polygon":
            self.mode_label.config(text="Mode: POLYGON", foreground="blue")
            self.assignment_label.config(text="Click polygons to assign to selected unit")
        elif mode == "fault":
            self.mode_label.config(text="Mode: FAULT", foreground="red")
            self.assignment_label.config(text="Click lines to assign to selected fault")
        elif mode == "contact":
            self.mode_label.config(text="Mode: CONTACT", foreground="green")
            self.assignment_label.config(text="Click to select contacts (view/edit only)")
        
        # Clear assignments
        self.current_unit_assignment = None
        self.current_fault_assignment = None
        
        # Reset button visuals
        for btn in self.unit_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)
        for btn in self.fault_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)
        
        self.update_section_display()

    def refresh_feature_browser(self):
        """Refresh the feature browser tree with all extracted items from current section."""
        if not hasattr(self, 'section_feature_tree'):
            return
            
        # Clear existing items
        for item in self.section_feature_tree.get_children():
            self.section_feature_tree.delete(item)
        
        if not self.northings:
            return
        
        # Get current section
        current_northing = self.northings[self.current_center_section] if self.current_center_section < len(self.northings) else None
        if current_northing is None:
            return
        
        # Find section data for current northing
        for (pdf_path, page_num), section_data in self.all_sections_data.items():
            section_northing = section_data.get("northing")
            if section_northing is not None and abs(section_northing - current_northing) < 0.1:
                # Add polygons
                for unit_name, unit in section_data.get("units", {}).items():
                    assigned = unit.get("unit_assignment", "")
                    original = unit.get("formation", "UNIT")
                    self.section_feature_tree.insert(
                        "", "end", 
                        text=unit_name,
                        values=("Polygon", original, assigned),
                        tags=("polygon", unit_name)
                    )
                
                # Add faults/polylines
                for poly_name, polyline in section_data.get("polylines", {}).items():
                    is_fault = polyline.get("is_fault", False)
                    assigned = polyline.get("fault_assignment", "")
                    original = polyline.get("fault_name", "Line")
                    item_type = "Fault" if is_fault else "Line"
                    self.section_feature_tree.insert(
                        "", "end",
                        text=poly_name,
                        values=(item_type, original, assigned),
                        tags=("fault" if is_fault else "polyline", poly_name)
                    )
        
        # Add contacts from grouped_contacts for current section
        if self.grouped_contacts:
            for group_name, group in self.grouped_contacts.items():
                for polyline in group.polylines:
                    if abs(polyline.northing - current_northing) < 0.1:
                        contact_name = f"{group.formation1}-{group.formation2}"
                        self.section_feature_tree.insert(
                            "", "end",
                            text=contact_name,
                            values=("Contact", f"N={int(polyline.northing)}", group_name),
                            tags=("contact", group_name)
                        )
                        break  # Only show one entry per contact group

    def on_feature_tree_select(self, event):
        """Handle selection in the feature browser tree."""
        selection = self.section_feature_tree.selection()
        if not selection:
            return
        
        # Get selected item names
        selected_names = []
        for item in selection:
            tags = self.section_feature_tree.item(item, "tags")
            if len(tags) >= 2:
                selected_names.append(tags[1])  # The name is the second tag
        
        # Highlight in the plot
        if selected_names:
            self.highlight_features_by_name(selected_names)

    def highlight_selected_features(self):
        """Highlight features selected in the feature browser."""
        selection = self.section_feature_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select features in the list first")
            return
        
        selected_names = []
        for item in selection:
            item_text = self.section_feature_tree.item(item, "text")
            selected_names.append(item_text)
        
        self.highlight_features_by_name(selected_names)

    def highlight_features_by_name(self, names: List[str]):
        """Highlight specific features by name in the section display."""
        # Add to selected units/polylines
        for name in names:
            if name in self.all_geological_units:
                self.selected_units.add(name)
            # Also check polylines
            for (pdf_path, page_num), section_data in self.all_sections_data.items():
                if name in section_data.get("polylines", {}):
                    self.selected_polylines.add(name)
        
        # Update display
        self.update_section_display()
        self.update_selection_display()
        self.update_polyline_selection_display()
        self.count_label.config(text=f"{len(self.selected_units)} units, {len(self.selected_polylines)} polylines selected")

    def refresh_unit_buttons(self):
        """Refresh unit buttons organized by prospect groups with collapse/expand."""
        for widget in self.units_inner.winfo_children():
            widget.destroy()
        self.unit_buttons.clear()
        
        # Sync defined_units with strat_column
        self.defined_units.clear()
        for unit in self.strat_column.get_all_units_ordered():
            self.defined_units[unit.name] = {
                'name': unit.name,
                'color': unit.color,
                'thickness': unit.thickness,
                'prospect': unit.prospect,
                'order': unit.order
            }
        
        # Create UI for each prospect
        for prospect in self.strat_column.get_prospects_ordered():
            prospect_name = prospect.name
            
            # Initialize expanded state if not set
            if prospect_name not in self.prospect_expanded:
                self.prospect_expanded[prospect_name] = prospect.expanded
            
            is_expanded = self.prospect_expanded.get(prospect_name, True)
            
            # Prospect header frame
            header_frame = ttk.Frame(self.units_inner)
            header_frame.pack(fill=tk.X, pady=2, padx=2)
            
            # Expand/collapse button
            expand_text = "▼" if is_expanded else "▶"
            expand_btn = ttk.Button(
                header_frame, text=expand_text, width=2,
                command=lambda p=prospect_name: self.toggle_prospect(p)
            )
            expand_btn.pack(side=tk.LEFT, padx=1)
            
            # Prospect label with count
            units_in_prospect = self.strat_column.get_units_in_prospect(prospect_name)
            prospect_label = ttk.Label(
                header_frame, 
                text=f"📁 {prospect_name} ({len(units_in_prospect)} units)",
                font=("Arial", 9, "bold")
            )
            prospect_label.pack(side=tk.LEFT, padx=5)
            
            # Prospect edit/delete buttons
            ttk.Button(
                header_frame, text="✎", width=2,
                command=lambda p=prospect_name: self.edit_prospect(p)
            ).pack(side=tk.RIGHT, padx=1)
            
            if prospect_name != self.strat_column.DEFAULT_PROSPECT:
                ttk.Button(
                    header_frame, text="✕", width=2,
                    command=lambda p=prospect_name: self.delete_prospect(p)
                ).pack(side=tk.RIGHT, padx=1)
            
            # Units container (shown/hidden based on expanded state)
            if is_expanded:
                units_container = ttk.Frame(self.units_inner)
                units_container.pack(fill=tk.X, padx=(20, 2), pady=1)
                
                # Create rows for each unit in this prospect
                for unit in units_in_prospect:
                    self._create_unit_row(units_container, unit)
    
    def _create_unit_row(self, parent, unit: 'StratUnit'):
        """Create a single unit row with controls."""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=1)
        
        unit_name = unit.name
        color = unit.color
        thickness = unit.thickness
        
        # Convert color to hex
        hex_color, brightness = self._color_to_hex(color)
        
        # Move up/down buttons
        btn_up = ttk.Button(row_frame, text="▲", width=2, 
                           command=lambda u=unit_name: self.move_unit_up(u))
        btn_up.pack(side=tk.LEFT, padx=1)
        
        btn_down = ttk.Button(row_frame, text="▼", width=2,
                             command=lambda u=unit_name: self.move_unit_down(u))
        btn_down.pack(side=tk.LEFT, padx=1)
        
        # Main unit button
        thickness_text = f" [{thickness}m]" if thickness else ""
        btn = tk.Button(
            row_frame,
            text=f"{unit_name}{thickness_text}",
            bg=hex_color,
            fg="white" if brightness < 0.5 else "black",
            width=14,
            height=1,
            font=("Arial", 8, "bold"),
            command=lambda u=unit_name: self.select_unit_for_assignment(u),
            relief=tk.RAISED,
            bd=2
        )
        btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.unit_buttons[unit_name] = btn
        
        # Edit button
        btn_edit = ttk.Button(row_frame, text="✎", width=2,
                             command=lambda u=unit_name: self.edit_unit(u))
        btn_edit.pack(side=tk.LEFT, padx=1)
        
        # Delete button
        btn_del = ttk.Button(row_frame, text="✕", width=2,
                            command=lambda u=unit_name: self.delete_unit(u))
        btn_del.pack(side=tk.LEFT, padx=1)
    
    def _color_to_hex(self, color) -> tuple:
        """Convert color to hex string and calculate brightness."""
        if isinstance(color, (tuple, list)):
            hex_color = "#%02x%02x%02x" % tuple(int(c * 255) for c in color[:3])
            brightness = sum(color[:3]) / 3
        elif isinstance(color, str) and color.startswith('#'):
            hex_color = color
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255
            brightness = (r + g + b) / 3
        else:
            hex_color = "#808080"
            brightness = 0.5
        return hex_color, brightness
    
    def toggle_prospect(self, prospect_name: str):
        """Toggle prospect expand/collapse state."""
        self.prospect_expanded[prospect_name] = not self.prospect_expanded.get(prospect_name, True)
        self.refresh_unit_buttons()
    
    def expand_all_prospects(self):
        """Expand all prospect groups."""
        for prospect in self.strat_column.get_prospects_ordered():
            self.prospect_expanded[prospect.name] = True
        self.refresh_unit_buttons()
    
    def collapse_all_prospects(self):
        """Collapse all prospect groups."""
        for prospect in self.strat_column.get_prospects_ordered():
            self.prospect_expanded[prospect.name] = False
        self.refresh_unit_buttons()
    
    def add_new_prospect(self):
        """Add a new prospect group."""
        name = simpledialog.askstring("New Prospect", "Enter prospect name:", parent=self.root)
        if not name:
            return
        
        name = name.strip()
        if name in [p.name for p in self.strat_column.get_prospects_ordered()]:
            messagebox.showwarning("Duplicate", f"Prospect '{name}' already exists!")
            return
        
        self.strat_column.add_prospect(name)
        self.prospect_expanded[name] = True
        self.refresh_unit_buttons()
        logger.info(f"Added new prospect: {name}")
    
    def edit_prospect(self, prospect_name: str):
        """Edit a prospect's name."""
        new_name = simpledialog.askstring(
            "Edit Prospect", 
            f"Enter new name for '{prospect_name}':",
            initialvalue=prospect_name,
            parent=self.root
        )
        if not new_name or new_name.strip() == prospect_name:
            return
        
        new_name = new_name.strip()
        
        # Update prospect name in strat_column
        if prospect_name in self.strat_column.prospects:
            # Move all units to new prospect name
            for unit in self.strat_column.units.values():
                if unit.prospect == prospect_name:
                    unit.prospect = new_name
            
            # Update prospect itself
            prospect = self.strat_column.prospects.pop(prospect_name)
            prospect.name = new_name
            self.strat_column.prospects[new_name] = prospect
            
            # Update expanded state
            self.prospect_expanded[new_name] = self.prospect_expanded.pop(prospect_name, True)
        
        self.refresh_unit_buttons()
        logger.info(f"Renamed prospect '{prospect_name}' to '{new_name}'")
    
    def delete_prospect(self, prospect_name: str):
        """Delete a prospect (moves units to Default)."""
        if prospect_name == self.strat_column.DEFAULT_PROSPECT:
            messagebox.showwarning("Cannot Delete", "Cannot delete the Default prospect.")
            return
        
        units = self.strat_column.get_units_in_prospect(prospect_name)
        if units:
            if not messagebox.askyesno(
                "Confirm Delete",
                f"Prospect '{prospect_name}' has {len(units)} units.\n"
                f"They will be moved to 'Default'.\n\nContinue?"
            ):
                return
        
        self.strat_column.remove_prospect(prospect_name, move_units_to=self.strat_column.DEFAULT_PROSPECT)
        self.prospect_expanded.pop(prospect_name, None)
        self.refresh_unit_buttons()
        logger.info(f"Deleted prospect: {prospect_name}")

    def move_unit_up(self, unit_name: str):
        """Move a unit up in the stratigraphic order within its prospect."""
        unit = self.strat_column.get_unit(unit_name)
        if unit and unit.order > 0:
            # Find the unit above and swap
            prospect_units = self.strat_column.get_units_in_prospect(unit.prospect)
            for i, u in enumerate(prospect_units):
                if u.name == unit_name and i > 0:
                    # Swap orders
                    above_unit = prospect_units[i - 1]
                    unit.order, above_unit.order = above_unit.order, unit.order
                    break
        self.refresh_unit_buttons()

    def move_unit_down(self, unit_name: str):
        """Move a unit down in the stratigraphic order within its prospect."""
        unit = self.strat_column.get_unit(unit_name)
        if unit:
            prospect_units = self.strat_column.get_units_in_prospect(unit.prospect)
            for i, u in enumerate(prospect_units):
                if u.name == unit_name and i < len(prospect_units) - 1:
                    # Swap orders
                    below_unit = prospect_units[i + 1]
                    unit.order, below_unit.order = below_unit.order, unit.order
                    break
        self.refresh_unit_buttons()

    def edit_unit(self, unit_name: str):
        """Edit a unit's properties (color, thickness)."""
        # Create edit dialog
        edit_dialog = tk.Toplevel(self.root)
        edit_dialog.title(f"Edit Unit: {unit_name}")
        edit_dialog.geometry("300x200")
        edit_dialog.transient(self.root)
        edit_dialog.grab_set()
        
        # Get current values
        unit_data = self.defined_units.get(unit_name, {})
        current_color = unit_data.get('color', (0.5, 0.5, 0.5))
        current_thickness = unit_data.get('thickness')
        
        # Name (display only)
        ttk.Label(edit_dialog, text=f"Unit: {unit_name}", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Thickness
        thick_frame = ttk.Frame(edit_dialog)
        thick_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(thick_frame, text="Thickness (m):").pack(side=tk.LEFT)
        thickness_var = tk.StringVar(value=str(current_thickness) if current_thickness else "")
        thick_entry = ttk.Entry(thick_frame, textvariable=thickness_var, width=10)
        thick_entry.pack(side=tk.LEFT, padx=5)
        
        # Color button
        color_frame = ttk.Frame(edit_dialog)
        color_frame.pack(fill=tk.X, padx=10, pady=5)
        
        if isinstance(current_color, tuple):
            hex_color = "#%02x%02x%02x" % tuple(int(c * 255) for c in current_color[:3])
        else:
            hex_color = current_color
        
        color_var = tk.StringVar(value=hex_color)
        color_btn = tk.Button(color_frame, text="Change Color", bg=hex_color,
                             command=lambda: self._pick_color_for_edit(color_btn, color_var))
        color_btn.pack(side=tk.LEFT, padx=5)
        
        # Save/Cancel
        btn_frame = ttk.Frame(edit_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=20)
        
        def save_changes():
            # Update thickness
            try:
                new_thickness = float(thickness_var.get()) if thickness_var.get() else None
            except ValueError:
                new_thickness = None
            
            # Update color
            new_hex = color_var.get()
            r = int(new_hex[1:3], 16) / 255
            g = int(new_hex[3:5], 16) / 255
            b = int(new_hex[5:7], 16) / 255
            new_color = (r, g, b)
            
            # Update strat column (v2 uses dataclasses)
            unit = self.strat_column.get_unit(unit_name)
            if unit:
                unit.thickness = new_thickness
                unit.color = new_color
            
            # Update defined units
            if unit_name in self.defined_units:
                self.defined_units[unit_name]['thickness'] = new_thickness
                self.defined_units[unit_name]['color'] = new_color
            
            edit_dialog.destroy()
            self.refresh_unit_buttons()
        
        ttk.Button(btn_frame, text="Save", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=edit_dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _pick_color_for_edit(self, btn, color_var):
        """Pick a color and update the button."""
        color = colorchooser.askcolor(initialcolor=color_var.get(), parent=btn.winfo_toplevel())
        if color[1]:
            color_var.set(color[1])
            btn.config(bg=color[1])

    def delete_unit(self, unit_name: str):
        """Delete a unit from the stratigraphic column."""
        if messagebox.askyesno("Confirm Delete", f"Delete unit '{unit_name}'?"):
            self.strat_column.remove_unit(unit_name)
            self.defined_units.pop(unit_name, None)
            self.refresh_unit_buttons()
            logger.info(f"Deleted unit: {unit_name}")

    def refresh_fault_buttons(self):
        """Refresh fault buttons with edit/delete/move controls (matching units behavior)."""
        for widget in self.faults_inner.winfo_children():
            widget.destroy()
        self.fault_buttons.clear()
        
        # Sync with strat column faults (v2 uses Fault dataclass)
        for fault in self.strat_column.get_faults_ordered():
            if fault.name not in self.defined_faults:
                self.defined_faults[fault.name] = {
                    'name': fault.name,
                    'color': fault.color,
                    'type': fault.fault_type.value if hasattr(fault.fault_type, 'value') else fault.fault_type,
                    'timing': fault.timing  # Faults use 'timing' not 'order'
                }
        
        # Sort faults by timing
        sorted_faults = sorted(self.defined_faults.items(), 
                               key=lambda x: x[1].get('timing', 0))
        
        # Create rows for each fault
        for fault_name, fault_data in sorted_faults:
            row_frame = ttk.Frame(self.faults_inner)
            row_frame.pack(fill=tk.X, pady=1, padx=2)
            
            hex_color = fault_data.get('color', '#FF0000')
            
            try:
                r = int(hex_color[1:3], 16) / 255
                g = int(hex_color[3:5], 16) / 255
                b = int(hex_color[5:7], 16) / 255
                brightness = (r + g + b) / 3
            except:
                brightness = 0.5
            
            # Move up/down buttons
            btn_up = ttk.Button(row_frame, text="▲", width=2,
                               command=lambda f=fault_name: self.move_fault_up(f))
            btn_up.pack(side=tk.LEFT, padx=1)
            
            btn_down = ttk.Button(row_frame, text="▼", width=2,
                                 command=lambda f=fault_name: self.move_fault_down(f))
            btn_down.pack(side=tk.LEFT, padx=1)
            
            # Main fault button
            btn = tk.Button(
                row_frame,
                text=f"{fault_name}",
                bg=hex_color,
                fg="white" if brightness < 0.5 else "black",
                width=12,
                height=1,
                font=("Arial", 8, "bold"),
                command=lambda f=fault_name: self.select_fault_for_assignment(f),
                relief=tk.RAISED,
                bd=2
            )
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
            self.fault_buttons[fault_name] = btn
            
            # Edit button
            btn_edit = ttk.Button(row_frame, text="✎", width=2,
                                 command=lambda f=fault_name: self.edit_fault(f))
            btn_edit.pack(side=tk.LEFT, padx=1)
            
            # Delete button
            btn_del = ttk.Button(row_frame, text="✕", width=2,
                                command=lambda f=fault_name: self.delete_fault(f))
            btn_del.pack(side=tk.LEFT, padx=1)

    def edit_fault(self, fault_name: str):
        """Edit a fault's properties (color, type)."""
        fault_data = self.defined_faults.get(fault_name, {})
        current_color = fault_data.get('color', '#FF0000')
        current_type = fault_data.get('type', 'normal')
        
        # Pick new color
        color = colorchooser.askcolor(initialcolor=current_color, title=f"Choose color for {fault_name}")
        if color[1]:
            self.defined_faults[fault_name]['color'] = color[1]
            self.fault_colors[fault_name] = color[1]
            
            # Update in strat column (v2 uses Fault dataclass)
            fault = self.strat_column.faults.get(fault_name)
            if fault:
                fault.color = color[1]
            
            self.refresh_fault_buttons()
            self.update_section_display()

    def delete_fault(self, fault_name: str):
        """Delete a fault."""
        if messagebox.askyesno("Confirm Delete", f"Delete fault '{fault_name}'?"):
            self.strat_column.remove_fault(fault_name)
            self.defined_faults.pop(fault_name, None)
            self.fault_colors.pop(fault_name, None)
            self.refresh_fault_buttons()
            logger.info(f"Deleted fault: {fault_name}")

    def move_fault_up(self, fault_name: str):
        """Move a fault up in the display order."""
        fault = self.strat_column.faults.get(fault_name)
        if not fault:
            return
        
        # Get all faults sorted by timing
        faults = self.strat_column.get_faults_ordered()
        fault_list = list(faults)
        
        # Find current position
        current_idx = None
        for i, f in enumerate(fault_list):
            if f.name == fault_name:
                current_idx = i
                break
        
        if current_idx is None or current_idx == 0:
            return  # Can't move up
        
        # Swap timing with the fault above
        fault_above = fault_list[current_idx - 1]
        old_timing = fault.timing
        fault.timing = fault_above.timing
        fault_above.timing = old_timing
        
        # Update defined_faults
        self.defined_faults[fault_name]['timing'] = fault.timing
        self.defined_faults[fault_above.name]['timing'] = fault_above.timing
        
        self.refresh_fault_buttons()
        logger.info(f"Moved fault '{fault_name}' up")

    def move_fault_down(self, fault_name: str):
        """Move a fault down in the display order."""
        fault = self.strat_column.faults.get(fault_name)
        if not fault:
            return
        
        # Get all faults sorted by timing
        faults = self.strat_column.get_faults_ordered()
        fault_list = list(faults)
        
        # Find current position
        current_idx = None
        for i, f in enumerate(fault_list):
            if f.name == fault_name:
                current_idx = i
                break
        
        if current_idx is None or current_idx >= len(fault_list) - 1:
            return  # Can't move down
        
        # Swap timing with the fault below
        fault_below = fault_list[current_idx + 1]
        old_timing = fault.timing
        fault.timing = fault_below.timing
        fault_below.timing = old_timing
        
        # Update defined_faults
        self.defined_faults[fault_name]['timing'] = fault.timing
        self.defined_faults[fault_below.name]['timing'] = fault_below.timing
        
        self.refresh_fault_buttons()
        logger.info(f"Moved fault '{fault_name}' down")

    def select_unit_for_assignment(self, unit_name: str):
        """Select a unit for assignment. If features are selected, assign them all."""
        if self.classification_mode != "polygon":
            self.mode_var.set("polygon")
            self.on_classification_mode_changed()
        
        self.current_unit_assignment = unit_name
        self.current_fault_assignment = None
        
        for name, btn in self.unit_buttons.items():
            btn.config(relief=tk.SUNKEN if name == unit_name else tk.RAISED, bd=4 if name == unit_name else 2)
        for btn in self.fault_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)
        
        self.assignment_label.config(text=f"Assigning to: {unit_name}")
        logger.info(f"Selected unit for assignment: {unit_name}")
        
        # If features are selected (from "Select Similar"), assign them all
        if hasattr(self, 'selected_feature') and self.selected_feature:
            assigned_count = 0
            for feature_name in list(self.selected_feature):
                # Check if it's a polygon
                if feature_name in self.all_geological_units:
                    self._assign_polygon_to_unit(feature_name, unit_name)
                    assigned_count += 1
            
            if assigned_count > 0:
                self.status_var.set(f"Assigned {assigned_count} features to {unit_name}")
                logger.info(f"Bulk assigned {assigned_count} features to {unit_name}")
                self.selected_feature.clear()  # Clear selection after bulk assign
                self.update_section_display()

    def clear_polygon_assignment(self, polygon_name: str):
        """Clear the assignment from a polygon (reset to UNKNOWN)."""
        logger.info(f"Clearing assignment from polygon '{polygon_name}'")
        
        # Remove from user-assigned set so it can be auto-labeled again
        self.user_assigned_polygons.discard(polygon_name)
        
        if polygon_name in self.all_geological_units:
            self.all_geological_units[polygon_name]['formation'] = "UNKNOWN"
            self.all_geological_units[polygon_name]['unit_assignment'] = None
            self.all_geological_units[polygon_name]['color'] = (0.5, 0.5, 0.5)
        
        # Update in section data
        for (pdf, page), section_data in self.all_sections_data.items():
            if polygon_name in section_data.get('units', {}):
                section_data['units'][polygon_name]['formation'] = "UNKNOWN"
                section_data['units'][polygon_name]['unit_assignment'] = None
                section_data['units'][polygon_name]['color'] = (0.5, 0.5, 0.5)
        
        self.update_section_display()
        self.status_var.set(f"Cleared assignment from {polygon_name}")

    def clear_fault_assignment(self, line_name: str):
        """Clear the assignment from a fault line."""
        logger.info(f"Clearing assignment from line '{line_name}'")
        
        # Update in section data
        for (pdf, page), section_data in self.all_sections_data.items():
            if line_name in section_data.get('polylines', {}):
                section_data['polylines'][line_name]['fault_assignment'] = None
                section_data['polylines'][line_name]['color'] = (0, 0, 0)
        
        self.update_section_display()
        self.status_var.set(f"Cleared assignment from {line_name}")

    def unassign_selected(self):
        """Unassign all currently selected polygons and faults."""
        if not self.selected_feature:
            messagebox.showinfo("No Selection", "No features selected to unassign.")
            return
        
        unassigned_count = 0
        for feature_name in list(self.selected_feature):
            # Check if it's a polygon
            if feature_name in self.all_geological_units:
                self.clear_polygon_assignment(feature_name)
                unassigned_count += 1
            # Check if it's a fault/polyline
            else:
                for (pdf, page), section_data in self.all_sections_data.items():
                    if feature_name in section_data.get('polylines', {}):
                        self.clear_fault_assignment(feature_name)
                        unassigned_count += 1
                        break
        
        self.status_var.set(f"Unassigned {unassigned_count} features")
        logger.info(f"Unassigned {unassigned_count} features")
    
    def select_fault_for_assignment(self, fault_name: str):
        """Select a fault for assignment."""
        if self.classification_mode != "fault":
            self.mode_var.set("fault")
            self.on_classification_mode_changed()
        
        self.current_fault_assignment = fault_name
        self.current_unit_assignment = None
        
        for name, btn in self.fault_buttons.items():
            btn.config(relief=tk.SUNKEN if name == fault_name else tk.RAISED, bd=4 if name == fault_name else 2)
        for btn in self.unit_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)
        
        self.assignment_label.config(text=f"Assigning to: {fault_name}")
        logger.info(f"Selected fault for assignment: {fault_name}")

    def add_new_unit(self):
        """Add a new geological unit with prospect selection."""
        # Create dialog for unit details
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Unit")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Unit name
        ttk.Label(dialog, text="Unit Name:").pack(pady=5)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(pady=2)
        name_entry.focus()
        
        # Prospect selection
        ttk.Label(dialog, text="Prospect:").pack(pady=5)
        prospect_var = tk.StringVar(value=self.strat_column.DEFAULT_PROSPECT)
        prospect_names = [p.name for p in self.strat_column.get_prospects_ordered()]
        prospect_combo = ttk.Combobox(dialog, textvariable=prospect_var, values=prospect_names, width=27)
        prospect_combo.pack(pady=2)
        
        # Color
        color_var = tk.StringVar(value="#808080")
        color_frame = ttk.Frame(dialog)
        color_frame.pack(pady=10)
        ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT, padx=5)
        color_btn = tk.Button(color_frame, text="Choose...", bg="#808080", width=10,
                             command=lambda: self._pick_color_for_dialog(color_btn, color_var))
        color_btn.pack(side=tk.LEFT, padx=5)
        
        def save_unit():
            name = name_var.get().strip().upper()
            if not name:
                messagebox.showwarning("Error", "Please enter a unit name")
                return
            if name in self.strat_column.units:
                messagebox.showwarning("Duplicate", f"Unit '{name}' already exists!")
                return
            
            # Parse color
            hex_color = color_var.get()
            r = int(hex_color[1:3], 16) / 255
            g = int(hex_color[3:5], 16) / 255
            b = int(hex_color[5:7], 16) / 255
            rgb = (r, g, b)
            
            prospect = prospect_var.get()
            
            self.strat_column.add_unit(name, rgb, prospect=prospect)
            self.defined_units[name] = {'name': name, 'color': rgb, 'prospect': prospect}
            
            dialog.destroy()
            logger.info(f"Added new unit: {name} to prospect {prospect}")
            self.refresh_unit_buttons()
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Add Unit", command=save_unit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _pick_color_for_dialog(self, btn, color_var):
        """Pick a color for dialog."""
        color = colorchooser.askcolor(initialcolor=color_var.get(), parent=btn.winfo_toplevel())
        if color[1]:
            color_var.set(color[1])
            btn.config(bg=color[1])

    def add_new_fault(self):
        """Add a new fault."""
        name = simpledialog.askstring("New Fault", "Enter fault name (e.g., F1):", parent=self.root)
        if not name:
            return
        
        name = name.strip().upper()
        if name in self.strat_column.faults:
            messagebox.showwarning("Duplicate", f"Fault '{name}' already exists!")
            return
        
        color = colorchooser.askcolor(title=f"Choose color for {name}", parent=self.root)
        if color[1] is None:
            return
        
        hex_color = color[1]
        
        self.strat_column.add_fault(name, FaultType.NORMAL, hex_color)
        self.defined_faults[name] = {'name': name, 'color': hex_color, 'type': 'normal'}
        
        logger.info(f"Added new fault: {name}")
        self.refresh_fault_buttons()

    def on_closing(self):
        """Clean up when closing."""
        # Save config before closing
        self._save_config_on_exit()
        
        if self.current_pdf:
            self.current_pdf.close()
        self.root.destroy()

class WriteAssignmentsDialog:
    """Dialog for PDF annotation write options."""
    
    def __init__(self, parent, unit_count: int, fault_count: int, has_contacts: bool):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Write Assignments to PDFs")
        self.dialog.geometry("400x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.dialog.geometry(f"+{parent.winfo_x() + 200}+{parent.winfo_y() + 150}")
        
        # Info frame
        info_frame = ttk.LabelFrame(self.dialog, text="Summary", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text=f"Units with assignments: {unit_count}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Faults with assignments: {fault_count}").pack(anchor=tk.W)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.dialog, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.duplicate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Create new annotations (preserve originals)",
            variable=self.duplicate_var
        ).pack(anchor=tk.W)
        
        self.contacts_var = tk.BooleanVar(value=has_contacts)
        contacts_cb = ttk.Checkbutton(
            options_frame,
            text="Include extracted contacts as polylines",
            variable=self.contacts_var
        )
        contacts_cb.pack(anchor=tk.W)
        if not has_contacts:
            contacts_cb.configure(state=tk.DISABLED)
        
        # Info label
        ttk.Label(
            self.dialog,
            text="Backups will be created automatically.",
            font=("TkDefaultFont", 9, "italic")
        ).pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Write to PDFs", command=self._on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)
        
        # Wait for dialog
        self.dialog.wait_window()
    
    def _on_ok(self):
        self.result = {
            "duplicate_mode": self.duplicate_var.get(),
            "include_contacts": self.contacts_var.get()
        }
        self.dialog.destroy()
    
    def _on_cancel(self):
        self.result = None
        self.dialog.destroy()

def main():
    """Main entry point."""
    import re

    # Check for required packages
    required_packages = {
        "fitz": "PyMuPDF",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "shapely": "shapely",
        "rasterio": "rasterio",
    }

    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            logger.error(f"{package} not installed! Run: pip install {package}")
            print(f"ERROR: {package} not installed! Run: pip install {package}")
            sys.exit(1)

    root = tk.Tk()
    app = GeologicalCrossSectionGUI(root)
    root.mainloop()



if __name__ == "__main__":
    main()
