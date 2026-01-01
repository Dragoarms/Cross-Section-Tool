# geotools\section_viewer.py
"""
Interactive section viewer for geological unit assignment.
Provides stacked section view with classification modes for polygons and faults.
"""

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import fitz

# Import contact extraction
try:
    from .contact_extraction import extract_single_contact
except ImportError:
    from contact_extraction import extract_single_contact

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if not present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class ClassificationMode(Enum):
    """Classification mode enum for section viewer."""
    NONE = "none"
    POLYGON = "polygon"
    FAULT = "fault"


class SectionViewer:
    """Interactive viewer for stacked geological sections with assignment capabilities."""

    # Fault hit tolerance in data coordinates (makes faults easier to click)
    FAULT_HIT_TOLERANCE = 15.0  # pixels

    def __init__(self, parent_app, all_sections_data, all_geological_units, strat_column):
        """
        Initialize the section viewer.

        Args:
            parent_app: Reference to main GUI application
            all_sections_data: Dictionary of section data by (pdf_path, page_num)
            all_geological_units: Dictionary of all geological units
            strat_column: Reference to stratigraphic column
        """
        logger.info("=" * 60)
        logger.info("Initializing Section Viewer")
        logger.info("=" * 60)

        self.parent_app = parent_app
        self.all_sections_data = all_sections_data
        self.all_geological_units = all_geological_units
        self.strat_column = strat_column

        # Log data summary
        logger.info(f"Loaded {len(all_sections_data)} sections")
        logger.info(f"Loaded {len(all_geological_units)} geological units")

        # Classification mode state
        self.classification_mode = ClassificationMode.NONE
        self.current_unit_assignment = None  # The unit to assign when clicking
        self.current_fault_assignment = None  # The fault to assign when clicking

        # Defined units and faults (user-created)
        self.defined_units = {}  # {name: {'color': (r,g,b), 'name': name}}
        self.defined_faults = {}  # {name: {'color': hex_color, 'name': name, 'type': 'normal'|'reverse'|etc}}

        # Initialize with strat column units
        for unit in self.strat_column.strat_units:
            self.defined_units[unit['name']] = {
                'name': unit['name'],
                'color': unit.get('color', (0.5, 0.5, 0.5))
            }
        logger.debug(f"Initialized {len(self.defined_units)} units from strat column")

        # Initialize with strat column faults
        for fault in self.strat_column.faults:
            self.defined_faults[fault['name']] = {
                'name': fault['name'],
                'color': self._get_fault_color(fault['name']),
                'type': fault.get('fault_type', 'normal')
            }
        logger.debug(f"Initialized {len(self.defined_faults)} faults from strat column")

        # Selection tracking
        self.selected_units = set()  # Selected polygon unit names
        self.selected_faults = set()  # Selected fault line names

        # Matplotlib artist mappings
        self.unit_patches = {}  # {patch: unit_data}
        self.fault_lines = {}  # {line: fault_data}

        # Calculated contacts
        self.calculated_contacts = []  # List of contact dicts
        self.show_contacts = False
        self.contact_lines = {}  # {line: contact_data}

        # View parameters
        self.current_center_section = 0
        self.sections_to_display = 1
        self.vertical_spacing = 50

        # Get sorted northings
        self.northings = sorted(
            set(
                section_data.get("northing", 0)
                for section_data in all_sections_data.values()
                if section_data.get("northing") is not None
            ),
            reverse=True
        )
        logger.info(f"Found {len(self.northings)} unique northings")

        if not self.northings:
            logger.error("No valid northings found in section data!")
            messagebox.showerror("Error", "No valid northings found in section data!")
            return

        # Calculate global ranges
        self.calculate_global_ranges()

        # Storage for UI elements
        self.unit_buttons = {}
        self.fault_buttons = {}

        # Create the viewer window
        self.create_window()

    def _get_fault_color(self, fault_name: str) -> str:
        """Get a color for a fault, cycling through defaults."""
        default_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8800', '#8800FF', '#00AAAA']
        idx = len(self.defined_faults) % len(default_colors)
        return default_colors[idx]

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

            # Check unit vertices for content bounds
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

        if self.global_rl_min != float("inf"):
            self.global_rl_min -= 50
            self.global_rl_max += 50

        logger.info(f"Global ranges - E: [{self.global_easting_min:.1f}, {self.global_easting_max:.1f}]")
        logger.info(f"Global ranges - RL: [{self.global_rl_min:.1f}, {self.global_rl_max:.1f}]")

    def create_window(self):
        """Create the main viewer window."""
        self.window = tk.Toplevel(self.parent_app.root)
        self.window.title("Section Viewer - Classification Mode")
        self.window.geometry("1800x1000")

        # Main container
        main_paned = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Section viewer
        self.create_section_panel(main_paned)

        # Right panel - Classification controls
        self.create_classification_panel(main_paned)

        # Initial display
        self.update_section_display()

    def create_section_panel(self, parent):
        """Create the section viewing panel."""
        left_frame = ttk.Frame(parent)
        parent.add(left_frame, weight=3)

        # Navigation controls
        nav_frame = ttk.LabelFrame(left_frame, text="Navigation", padding=5)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(nav_frame, text="Current Section:").pack(side=tk.LEFT, padx=5)
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(
            nav_frame, textvariable=self.section_var, state="readonly", width=20
        )
        self.section_combo["values"] = [f"Section {int(n)}" for n in self.northings]
        if self.northings:
            self.section_combo.current(0)
        self.section_combo.pack(side=tk.LEFT, padx=5)
        self.section_combo.bind("<<ComboboxSelected>>", self.on_section_changed)

        ttk.Button(nav_frame, text="◄ Prev", command=self.prev_section).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ►", command=self.next_section).pack(side=tk.LEFT, padx=2)

        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(nav_frame, text="Sections:").pack(side=tk.LEFT, padx=5)
        self.section_count_var = tk.IntVar(value=1)
        section_count_spin = ttk.Spinbox(
            nav_frame, from_=1, to=5, textvariable=self.section_count_var,
            width=5, command=self.on_section_count_changed
        )
        section_count_spin.pack(side=tk.LEFT, padx=5)

        # Display options
        display_frame = ttk.LabelFrame(left_frame, text="Display Options", padding=5)
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            display_frame, text="Show Grid", variable=self.show_grid_var,
            command=self.update_section_display
        ).pack(side=tk.LEFT, padx=5)

        self.hide_empty_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            display_frame, text="Hide Empty", variable=self.hide_empty_var,
            command=self.filter_sections
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(display_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Contact controls
        self.show_contacts_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame, text="Show Contacts", variable=self.show_contacts_var,
            command=self.toggle_contacts
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            display_frame, text="Calculate Contacts", command=self.calculate_contacts
        ).pack(side=tk.LEFT, padx=5)

        # Mode indicator
        self.mode_label = ttk.Label(display_frame, text="Mode: None", font=('Arial', 10, 'bold'))
        self.mode_label.pack(side=tk.RIGHT, padx=10)

        # Create matplotlib figure
        self.fig_sections = plt.figure(figsize=(14, 8))

        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_sections = FigureCanvasTkAgg(self.fig_sections, master=canvas_frame)
        canvas_widget = self.canvas_sections.get_tk_widget()

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas_sections, toolbar_frame)
        self.toolbar.update()

        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        canvas_widget.bind("<MouseWheel>", self.on_mousewheel)
        canvas_widget.bind("<Button-4>", self.on_mousewheel)
        canvas_widget.bind("<Button-5>", self.on_mousewheel)

        # Connect pick event
        self.canvas_sections.mpl_connect("pick_event", self.on_pick)
        # Also connect button press for fault detection with larger tolerance
        self.canvas_sections.mpl_connect("button_press_event", self.on_click)

    def create_classification_panel(self, parent):
        """Create the classification control panel."""
        right_frame = ttk.Frame(parent)
        parent.add(right_frame, weight=1)

        # Mode selection frame
        mode_frame = ttk.LabelFrame(right_frame, text="Classification Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="none")

        ttk.Radiobutton(
            mode_frame, text="No Mode (View Only)", variable=self.mode_var,
            value="none", command=self.on_mode_changed
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            mode_frame, text="Polygon Classification", variable=self.mode_var,
            value="polygon", command=self.on_mode_changed
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            mode_frame, text="Fault Classification", variable=self.mode_var,
            value="fault", command=self.on_mode_changed
        ).pack(anchor=tk.W, pady=2)

        # Units frame (for polygon mode)
        self.units_frame = ttk.LabelFrame(right_frame, text="Geological Units", padding=10)
        self.units_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable unit buttons
        units_canvas = tk.Canvas(self.units_frame, height=250)
        units_scrollbar = ttk.Scrollbar(self.units_frame, orient="vertical", command=units_canvas.yview)
        self.units_inner = ttk.Frame(units_canvas)

        self.units_inner.bind(
            "<Configure>", lambda e: units_canvas.configure(scrollregion=units_canvas.bbox("all"))
        )

        units_canvas.create_window((0, 0), window=self.units_inner, anchor="nw")
        units_canvas.configure(yscrollcommand=units_scrollbar.set)

        # Populate unit buttons
        self.refresh_unit_buttons()

        units_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        units_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add unit button
        ttk.Button(
            self.units_frame, text="+ Add Unit", command=self.add_new_unit
        ).pack(fill=tk.X, pady=5)

        # Faults frame (for fault mode)
        self.faults_frame = ttk.LabelFrame(right_frame, text="Faults", padding=10)
        self.faults_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable fault buttons
        faults_canvas = tk.Canvas(self.faults_frame, height=150)
        faults_scrollbar = ttk.Scrollbar(self.faults_frame, orient="vertical", command=faults_canvas.yview)
        self.faults_inner = ttk.Frame(faults_canvas)

        self.faults_inner.bind(
            "<Configure>", lambda e: faults_canvas.configure(scrollregion=faults_canvas.bbox("all"))
        )

        faults_canvas.create_window((0, 0), window=self.faults_inner, anchor="nw")
        faults_canvas.configure(yscrollcommand=faults_scrollbar.set)

        # Populate fault buttons
        self.refresh_fault_buttons()

        faults_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        faults_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add fault button
        ttk.Button(
            self.faults_frame, text="+ Add Fault", command=self.add_new_fault
        ).pack(fill=tk.X, pady=5)

        # Selection info
        info_frame = ttk.LabelFrame(right_frame, text="Selection Info", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.selection_label = ttk.Label(info_frame, text="No selection")
        self.selection_label.pack(anchor=tk.W)

        self.assignment_label = ttk.Label(info_frame, text="Click a unit/fault button to select assignment")
        self.assignment_label.pack(anchor=tk.W)

        ttk.Button(info_frame, text="Clear Selection", command=self.clear_selection).pack(pady=5)

    def refresh_unit_buttons(self):
        """Refresh the unit buttons in the UI."""
        # Clear existing buttons
        for widget in self.units_inner.winfo_children():
            widget.destroy()
        self.unit_buttons.clear()

        # Create buttons for each defined unit
        for unit_name, unit_data in self.defined_units.items():
            color = unit_data.get('color', (0.5, 0.5, 0.5))
            if isinstance(color, tuple):
                hex_color = "#%02x%02x%02x" % tuple(int(c * 255) for c in color[:3])
            else:
                hex_color = color

            # Determine text color based on background brightness
            brightness = sum(color[:3]) / 3 if isinstance(color, tuple) else 0.5
            fg_color = "white" if brightness < 0.5 else "black"

            btn = tk.Button(
                self.units_inner,
                text=unit_name,
                bg=hex_color,
                fg=fg_color,
                width=18,
                height=2,
                font=("Arial", 9, "bold"),
                command=lambda u=unit_name: self.select_unit_for_assignment(u),
                relief=tk.RAISED,
                bd=2
            )
            btn.pack(pady=2, padx=5, fill=tk.X)
            self.unit_buttons[unit_name] = btn

        logger.debug(f"Refreshed {len(self.unit_buttons)} unit buttons")

    def refresh_fault_buttons(self):
        """Refresh the fault buttons in the UI."""
        # Clear existing buttons
        for widget in self.faults_inner.winfo_children():
            widget.destroy()
        self.fault_buttons.clear()

        # Create buttons for each defined fault
        for fault_name, fault_data in self.defined_faults.items():
            hex_color = fault_data.get('color', '#FF0000')

            # Determine text color
            try:
                r = int(hex_color[1:3], 16) / 255
                g = int(hex_color[3:5], 16) / 255
                b = int(hex_color[5:7], 16) / 255
                brightness = (r + g + b) / 3
                fg_color = "white" if brightness < 0.5 else "black"
            except:
                fg_color = "white"

            btn = tk.Button(
                self.faults_inner,
                text=f"{fault_name} ({fault_data.get('type', 'normal')})",
                bg=hex_color,
                fg=fg_color,
                width=18,
                height=2,
                font=("Arial", 9, "bold"),
                command=lambda f=fault_name: self.select_fault_for_assignment(f),
                relief=tk.RAISED,
                bd=2
            )
            btn.pack(pady=2, padx=5, fill=tk.X)
            self.fault_buttons[fault_name] = btn

        logger.debug(f"Refreshed {len(self.fault_buttons)} fault buttons")

    def add_new_unit(self):
        """Add a new geological unit."""
        name = simpledialog.askstring("New Unit", "Enter unit name:", parent=self.window)
        if not name:
            return

        name = name.strip().upper()
        if name in self.defined_units:
            messagebox.showwarning("Duplicate", f"Unit '{name}' already exists!")
            return

        # Pick color
        color = colorchooser.askcolor(title=f"Choose color for {name}", parent=self.window)
        if color[0] is None:
            return

        # Convert to 0-1 range
        rgb = tuple(c / 255.0 for c in color[0])

        self.defined_units[name] = {
            'name': name,
            'color': rgb
        }

        # Add to strat column
        self.strat_column.add_strat_unit(name, "Unknown", rgb)

        logger.info(f"Added new unit: {name} with color {rgb}")
        self.refresh_unit_buttons()

    def add_new_fault(self):
        """Add a new fault definition."""
        name = simpledialog.askstring("New Fault", "Enter fault name (e.g., F1):", parent=self.window)
        if not name:
            return

        name = name.strip().upper()
        if name in self.defined_faults:
            messagebox.showwarning("Duplicate", f"Fault '{name}' already exists!")
            return

        # Select fault type
        fault_types = ['normal', 'reverse', 'thrust', 'strike-slip']
        type_dialog = tk.Toplevel(self.window)
        type_dialog.title("Fault Type")
        type_dialog.geometry("200x150")
        type_dialog.transient(self.window)
        type_dialog.grab_set()

        selected_type = tk.StringVar(value='normal')
        ttk.Label(type_dialog, text="Select fault type:").pack(pady=5)
        for ft in fault_types:
            ttk.Radiobutton(type_dialog, text=ft.title(), variable=selected_type, value=ft).pack(anchor=tk.W, padx=20)

        def confirm():
            type_dialog.destroy()

        ttk.Button(type_dialog, text="OK", command=confirm).pack(pady=10)
        self.window.wait_window(type_dialog)

        fault_type = selected_type.get()

        # Pick color
        color = colorchooser.askcolor(title=f"Choose color for {name}", parent=self.window)
        if color[1] is None:
            return

        hex_color = color[1]

        self.defined_faults[name] = {
            'name': name,
            'color': hex_color,
            'type': fault_type
        }

        # Add to strat column
        self.strat_column.add_fault(name, fault_type)

        logger.info(f"Added new fault: {name} ({fault_type}) with color {hex_color}")
        self.refresh_fault_buttons()

    def select_unit_for_assignment(self, unit_name: str):
        """Select a unit for assignment when clicking polygons."""
        # Switch to polygon mode if not already
        if self.classification_mode != ClassificationMode.POLYGON:
            self.mode_var.set("polygon")
            self.on_mode_changed()

        self.current_unit_assignment = unit_name
        self.current_fault_assignment = None

        # Update button visuals
        for name, btn in self.unit_buttons.items():
            if name == unit_name:
                btn.config(relief=tk.SUNKEN, bd=4)
            else:
                btn.config(relief=tk.RAISED, bd=2)

        # Reset fault buttons
        for btn in self.fault_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)

        self.assignment_label.config(text=f"Assigning to: {unit_name}")
        logger.debug(f"Selected unit for assignment: {unit_name}")

    def select_fault_for_assignment(self, fault_name: str):
        """Select a fault for assignment when clicking lines."""
        # Switch to fault mode if not already
        if self.classification_mode != ClassificationMode.FAULT:
            self.mode_var.set("fault")
            self.on_mode_changed()

        self.current_fault_assignment = fault_name
        self.current_unit_assignment = None

        # Update button visuals
        for name, btn in self.fault_buttons.items():
            if name == fault_name:
                btn.config(relief=tk.SUNKEN, bd=4)
            else:
                btn.config(relief=tk.RAISED, bd=2)

        # Reset unit buttons
        for btn in self.unit_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)

        self.assignment_label.config(text=f"Assigning to: {fault_name}")
        logger.debug(f"Selected fault for assignment: {fault_name}")

    def on_mode_changed(self):
        """Handle classification mode change."""
        mode_str = self.mode_var.get()

        if mode_str == "none":
            self.classification_mode = ClassificationMode.NONE
            self.mode_label.config(text="Mode: View Only", foreground="black")
        elif mode_str == "polygon":
            self.classification_mode = ClassificationMode.POLYGON
            self.mode_label.config(text="Mode: POLYGON", foreground="blue")
        elif mode_str == "fault":
            self.classification_mode = ClassificationMode.FAULT
            self.mode_label.config(text="Mode: FAULT", foreground="red")

        # Clear current assignments when switching modes
        self.current_unit_assignment = None
        self.current_fault_assignment = None

        # Reset all button visuals
        for btn in self.unit_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)
        for btn in self.fault_buttons.values():
            btn.config(relief=tk.RAISED, bd=2)

        self.assignment_label.config(text="Click a unit/fault button to select assignment")
        logger.info(f"Classification mode changed to: {mode_str}")

        self.update_section_display()

    def on_pick(self, event):
        """Handle pick events on the sections."""
        if self.classification_mode == ClassificationMode.NONE:
            return

        artist = event.artist

        if self.classification_mode == ClassificationMode.POLYGON:
            if artist in self.unit_patches:
                self._handle_polygon_click(artist)

        elif self.classification_mode == ClassificationMode.FAULT:
            if artist in self.fault_lines:
                self._handle_fault_click(artist)

    def on_click(self, event):
        """Handle general click events for fault detection with larger tolerance."""
        if self.classification_mode != ClassificationMode.FAULT:
            return

        if event.inaxes is None:
            return

        # Check if we clicked near any fault line
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        # Check each fault line with expanded tolerance
        for line, fault_data in self.fault_lines.items():
            if self._is_near_line(line, click_x, click_y, self.FAULT_HIT_TOLERANCE):
                logger.debug(f"Click near fault line: {fault_data.get('name')}")
                self._handle_fault_click_by_data(fault_data)
                return

    def _is_near_line(self, line: Line2D, x: float, y: float, tolerance: float) -> bool:
        """Check if a point is near a line within tolerance (in data coordinates)."""
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        if len(xdata) < 2:
            return False

        # Check distance to each segment
        for i in range(len(xdata) - 1):
            x1, y1 = xdata[i], ydata[i]
            x2, y2 = xdata[i + 1], ydata[i + 1]

            # Calculate distance from point to line segment
            dist = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
            if dist <= tolerance:
                return True

        return False

    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _handle_polygon_click(self, artist):
        """Handle click on a polygon in polygon mode."""
        unit_data = self.unit_patches[artist]
        unit_name = unit_data.get('name', 'Unknown')

        if self.current_unit_assignment:
            # Assign the polygon to the selected unit
            self._assign_polygon_to_unit(unit_data, self.current_unit_assignment)
        else:
            # Just select/deselect
            if unit_name in self.selected_units:
                self.selected_units.discard(unit_name)
            else:
                self.selected_units.add(unit_name)

        self.update_selection_display()
        self.update_section_display()

    def _handle_fault_click(self, artist):
        """Handle click on a fault line."""
        fault_data = self.fault_lines[artist]
        self._handle_fault_click_by_data(fault_data)

    def _handle_fault_click_by_data(self, fault_data):
        """Handle fault click by data dict."""
        fault_name = fault_data.get('name', 'Unknown')

        if self.current_fault_assignment:
            # Assign the line to the selected fault
            self._assign_line_to_fault(fault_data, self.current_fault_assignment)
        else:
            # Just select/deselect
            if fault_name in self.selected_faults:
                self.selected_faults.discard(fault_name)
            else:
                self.selected_faults.add(fault_name)

        self.update_selection_display()
        self.update_section_display()

    def _assign_polygon_to_unit(self, unit_data: Dict, target_unit: str):
        """Assign a polygon to a target unit."""
        unit_name = unit_data.get('name')
        if not unit_name:
            return

        logger.info(f"Assigning polygon '{unit_name}' to unit '{target_unit}'")

        # Update in all_geological_units
        if unit_name in self.all_geological_units:
            self.all_geological_units[unit_name]['formation'] = target_unit
            if target_unit in self.defined_units:
                self.all_geological_units[unit_name]['color'] = self.defined_units[target_unit]['color']

        # Update in section data
        for (pdf, page), section_data in self.all_sections_data.items():
            if unit_name in section_data.get('units', {}):
                section_data['units'][unit_name]['formation'] = target_unit
                if target_unit in self.defined_units:
                    section_data['units'][unit_name]['color'] = self.defined_units[target_unit]['color']

        self.update_section_display()

    def _assign_line_to_fault(self, fault_data: Dict, target_fault: str):
        """Assign a line to a target fault."""
        line_name = fault_data.get('name')
        if not line_name:
            return

        logger.info(f"Assigning line '{line_name}' to fault '{target_fault}'")

        # Update the fault assignment in the data
        fault_data['fault_assignment'] = target_fault
        if target_fault in self.defined_faults:
            fault_data['color'] = self.defined_faults[target_fault]['color']

        # Update in section data
        for (pdf, page), section_data in self.all_sections_data.items():
            if 'faults' in section_data:
                for fault in section_data['faults']:
                    if fault.get('name') == line_name:
                        fault['fault_assignment'] = target_fault
                        if target_fault in self.defined_faults:
                            fault['color'] = self.defined_faults[target_fault]['color']

        self.update_section_display()

    def update_selection_display(self):
        """Update the selection info display."""
        total = len(self.selected_units) + len(self.selected_faults)
        if total == 0:
            self.selection_label.config(text="No selection")
        else:
            parts = []
            if self.selected_units:
                parts.append(f"{len(self.selected_units)} polygons")
            if self.selected_faults:
                parts.append(f"{len(self.selected_faults)} faults")
            self.selection_label.config(text=f"Selected: {', '.join(parts)}")

    def clear_selection(self):
        """Clear all selections."""
        self.selected_units.clear()
        self.selected_faults.clear()
        self.update_selection_display()
        self.update_section_display()

    def calculate_contacts(self):
        """Calculate contacts between adjacent polygons."""
        logger.info("Calculating contacts...")

        try:
            from shapely.geometry import Polygon, LineString
            from shapely.ops import unary_union
        except ImportError:
            messagebox.showerror("Error", "Shapely library required for contact calculation")
            return

        self.calculated_contacts = []
        contacts_found = 0

        for (pdf, page), section_data in self.all_sections_data.items():
            northing = section_data.get('northing')
            units = section_data.get('units', {})

            if len(units) < 2:
                continue

            logger.debug(f"Processing section at northing {northing} with {len(units)} units")

            # Build polygons
            polygons = {}
            for unit_name, unit in units.items():
                vertices = unit.get('vertices', [])
                if len(vertices) < 6:
                    continue

                coords = []
                for i in range(0, len(vertices), 2):
                    if i + 1 < len(vertices):
                        coords.append((vertices[i], vertices[i + 1]))

                if len(coords) >= 3:
                    try:
                        # Close polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        poly = Polygon(coords)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty:
                            polygons[unit_name] = {
                                'polygon': poly,
                                'formation': unit.get('formation', 'Unknown'),
                                'unit': unit
                            }
                    except Exception as e:
                        logger.warning(f"Could not create polygon for {unit_name}: {e}")

            # Find contacts between adjacent polygons
            unit_names = list(polygons.keys())
            for i, name1 in enumerate(unit_names):
                for name2 in unit_names[i + 1:]:
                    poly1 = polygons[name1]['polygon']
                    poly2 = polygons[name2]['polygon']

        logger.info(f"Calculated {contacts_found} contacts")
        self.show_contacts_var.set(True)
        self.show_contacts = True
        self.update_section_display()

        messagebox.showinfo("Contacts Calculated", f"Found {contacts_found} contacts")

    def toggle_contacts(self):
        """Toggle contact display."""
        self.show_contacts = self.show_contacts_var.get()
        self.update_section_display()

    def filter_sections(self):
        """Filter sections based on display options."""
        all_northings = sorted(
            set(
                section_data.get("northing", 0)
                for section_data in self.all_sections_data.values()
                if section_data.get("northing") is not None
            ),
            reverse=True
        )

        if self.hide_empty_var.get():
            self.northings = []
            for northing in all_northings:
                for (pdf, page), data in self.all_sections_data.items():
                    if data.get("northing") is not None and abs(data.get("northing") - northing) < 0.1:
                        if len(data.get("units", {})) > 0 or len(data.get("faults", [])) > 0:
                            self.northings.append(northing)
                            break
            logger.info(f"Filtered to {len(self.northings)} non-empty sections")
        else:
            self.northings = all_northings
            logger.info(f"Showing all {len(self.northings)} sections")

        # Update section selector
        self.section_combo["values"] = [f"Section {int(n)}" for n in self.northings]
        if self.northings and self.section_combo.current() >= len(self.northings):
            self.section_combo.current(0)
            self.current_center_section = 0

        self.update_section_display()

    def update_section_display(self):
        """Update the section display."""
        logger.debug("Updating section display")

        self.fig_sections.clear()
        self.unit_patches.clear()
        self.fault_lines.clear()
        self.contact_lines.clear()

        # Determine sections to display
        start_idx = max(0, self.current_center_section)
        end_idx = min(len(self.northings), start_idx + self.sections_to_display)
        sections_to_show = self.northings[start_idx:end_idx]

        if not sections_to_show:
            logger.warning("No sections to show")
            self.canvas_sections.draw()
            return

        n_sections = len(sections_to_show)

        for i, northing in enumerate(sections_to_show):
            ax = self.fig_sections.add_subplot(n_sections, 1, i + 1)

            # Get section data
            section_data_list = [
                data for (pdf, page), data in self.all_sections_data.items()
                if data.get("northing") is not None and abs(data.get("northing") - northing) < 0.1
            ]

            if not section_data_list:
                ax.text(0.5, 0.5, f"No data for section {int(northing)}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            section_data = section_data_list[0]

            # Plot polygons (units)
            self._plot_polygons(ax, section_data, northing)

            # Plot faults
            self._plot_faults(ax, section_data, northing)

            # Plot contacts if enabled
            if self.show_contacts:
                self._plot_contacts(ax, northing)

            # Configure axes
            ax.set_title(f"Section {int(northing)}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Easting (m)" if i == n_sections - 1 else "", fontsize=9)
            ax.set_ylabel("RL (m)", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Set axis limits
            if self.content_easting_min != float("inf"):
                margin_e = (self.content_easting_max - self.content_easting_min) * 0.05
                margin_rl = (self.content_rl_max - self.content_rl_min) * 0.05
                ax.set_xlim(self.content_easting_min - margin_e, self.content_easting_max + margin_e)
                ax.set_ylim(self.content_rl_min - margin_rl, self.content_rl_max + margin_rl)

            ax.set_aspect("equal", adjustable="box")

            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)
                ax.minorticks_on()
                ax.grid(True, alpha=0.1, which="minor", linestyle=":")

        self.fig_sections.subplots_adjust(hspace=0.2, left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.canvas_sections.draw()

    def _plot_polygons(self, ax, section_data, northing):
        """Plot polygon units on the axes."""
        units = section_data.get("units", {})
        polygons_plotted = 0

        for unit_name, unit in units.items():
            vertices = unit.get("vertices", [])
            if len(vertices) < 6:
                continue

            xs = [vertices[j] for j in range(0, len(vertices), 2)]
            ys = [vertices[j + 1] for j in range(0, len(vertices), 2) if j + 1 < len(vertices)]

            if not xs or not ys:
                continue

            # Close polygon
            if xs[0] != xs[-1]:
                xs.append(xs[0])
                ys.append(ys[0])

            # Determine color and style based on selection and mode
            is_selected = unit_name in self.selected_units
            color = unit.get("color", (0.5, 0.5, 0.5))

            # In polygon mode, make polygons more prominent
            if self.classification_mode == ClassificationMode.POLYGON:
                alpha = 0.7 if is_selected else 0.5
                linewidth = 3 if is_selected else 1.5
                edgecolor = "yellow" if is_selected else "black"
            else:
                alpha = 0.3
                linewidth = 1
                edgecolor = "gray"

            # Only pickable in polygon mode
            picker = (self.classification_mode == ClassificationMode.POLYGON)

            try:
                poly = MplPolygon(
                    list(zip(xs, ys)),
                    facecolor=color,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    linewidth=linewidth,
                    picker=picker
                )

                self.unit_patches[poly] = {
                    "name": unit_name,
                    "unit": unit,
                    "northing": northing,
                    "section_data": section_data
                }

                ax.add_patch(poly)
                polygons_plotted += 1

            except Exception as e:
                logger.warning(f"Failed to plot polygon {unit_name}: {e}")

        logger.debug(f"Plotted {polygons_plotted} polygons for section {int(northing)}")

    def _plot_faults(self, ax, section_data, northing):
        """Plot fault lines on the axes."""
        faults = section_data.get("faults", [])
        faults_plotted = 0

        for fault in faults:
            vertices = fault.get("vertices", [])
            if len(vertices) < 4:
                continue

            xs = [vertices[j] for j in range(0, len(vertices), 2)]
            ys = [vertices[j + 1] for j in range(0, len(vertices), 2) if j + 1 < len(vertices)]

            if not xs or not ys:
                continue

            fault_name = fault.get("name", f"Fault_{faults_plotted}")
            is_selected = fault_name in self.selected_faults

            # Get color from assignment or default
            if 'fault_assignment' in fault and fault['fault_assignment'] in self.defined_faults:
                color = self.defined_faults[fault['fault_assignment']]['color']
            elif 'color' in fault:
                color = fault['color']
            else:
                color = 'red'

            # In fault mode, make faults more prominent
            if self.classification_mode == ClassificationMode.FAULT:
                linewidth = 4 if is_selected else 2.5
                alpha = 1.0
                linestyle = '-'
            else:
                linewidth = 1.5
                alpha = 0.5
                linestyle = '--'

            # Pickable in fault mode (but we also use click detection)
            picker = (self.classification_mode == ClassificationMode.FAULT)

            try:
                line, = ax.plot(
                    xs, ys,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                    picker=picker,
                    pickradius=10  # Larger pick radius for faults
                )

                self.fault_lines[line] = {
                    "name": fault_name,
                    "fault": fault,
                    "northing": northing,
                    "section_data": section_data
                }

                faults_plotted += 1

            except Exception as e:
                logger.warning(f"Failed to plot fault {fault_name}: {e}")

        logger.debug(f"Plotted {faults_plotted} faults for section {int(northing)}")

    def _plot_contacts(self, ax, northing):
        """Plot calculated contacts on the axes."""
        contacts_plotted = 0

        for contact in self.calculated_contacts:
            if abs(contact.get('northing', 0) - northing) > 0.1:
                continue

            vertices = contact.get('vertices', [])
            if len(vertices) < 4:
                continue

            xs = [vertices[j] for j in range(0, len(vertices), 2)]
            ys = [vertices[j + 1] for j in range(0, len(vertices), 2) if j + 1 < len(vertices)]

            if not xs or not ys:
                continue

            try:
                line, = ax.plot(
                    xs, ys,
                    color='purple',
                    linewidth=2,
                    alpha=0.8,
                    linestyle='-',
                    marker='.',
                    markersize=3
                )

                self.contact_lines[line] = contact
                contacts_plotted += 1

                # Add label at midpoint
                mid_idx = len(xs) // 2
                ax.annotate(
                    contact['name'],
                    (xs[mid_idx], ys[mid_idx]),
                    fontsize=6,
                    ha='center',
                    va='bottom',
                    color='purple',
                    alpha=0.7
                )

            except Exception as e:
                logger.warning(f"Failed to plot contact {contact.get('name')}: {e}")

        logger.debug(f"Plotted {contacts_plotted} contacts for section {int(northing)}")

    def on_section_changed(self, event):
        """Handle section selection change."""
        selection = self.section_combo.current()
        if selection >= 0:
            self.current_center_section = selection
            self.update_section_display()

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

    def next_section(self):
        """Move to next section."""
        if self.current_center_section < len(self.northings) - 1:
            self.current_center_section += 1
            self.section_combo.current(self.current_center_section)
            self.update_section_display()

    def jump_to_section(self, index):
        """Jump to specific section index."""
        if 0 <= index < len(self.northings):
            self.current_center_section = index
            self.section_combo.current(index)
            self.update_section_display()
