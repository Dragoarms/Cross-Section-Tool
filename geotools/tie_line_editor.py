# geotools\tie_line_editor.py
"""
Tie Line Editor for geological cross-section correlation.

Provides an interactive interface for drawing tie lines between contacts
on adjacent sections. Tie lines indicate which parts of a contact on one
section correspond to parts on the next section, enabling 3D wireframing.

Key features:
- Stacked section view (similar to SectionViewer)
- Units displayed but not interactive
- Contacts are interactive and selectable
- Tie lines drawn between adjacent sections only
- Formation pair validation (can't tie A-B to C-D)
- Auto-suggestion of tie lines based on geometry
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging

# Import contact extraction types
try:
    from .contact_extraction import GroupedContact, ContactPolyline, ContactExtractor
except ImportError:
    from contact_extraction import GroupedContact, ContactPolyline, ContactExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class TieLinePoint:
    """A point selected for tie line drawing."""
    northing: float
    easting: float
    rl: float
    contact_group: str  # The contact group this point belongs to
    polyline_idx: int   # Index of the polyline within the group
    vertex_idx: int = -1  # Index of the vertex within the polyline (-1 if not snapped)


class TieLineEditor:
    """
    Interactive editor for drawing tie lines between geological contacts.
    
    Tie lines connect corresponding points on contacts across adjacent sections,
    enabling proper 3D wireframe construction.
    """
    
    # Visual styling
    CONTACT_COLOR_UNSELECTED = '#00AA00'  # Green
    CONTACT_COLOR_SELECTED = '#FFFF00'    # Yellow
    CONTACT_COLOR_ACTIVE = '#FF00FF'      # Magenta (for the active group)
    TIE_LINE_COLOR = '#00FFFF'            # Cyan
    TIE_LINE_SUGGESTED_COLOR = '#0088AA'  # Darker cyan for suggestions
    UNIT_ALPHA = 0.25                     # Units are faded
    CONTACT_LINE_WIDTH = 2.5
    TIE_LINE_WIDTH = 1.5

    # Snapping configuration
    SNAP_RADIUS = 50  # Only snap if within this distance (in coordinate units)
    SNAP_HIGHLIGHT_COLOR = '#FF0000'  # Red highlight for snap target
    SNAP_HIGHLIGHT_SIZE = 80  # Size of snap highlight marker
    
    def __init__(
        self,
        parent_app,
        all_sections_data: Dict[Tuple, Dict],
        grouped_contacts: Dict[str, GroupedContact],
        strat_column
    ):
        """
        Initialize the tie line editor.
        
        Args:
            parent_app: Reference to main GUI application
            all_sections_data: Section data dictionary
            grouped_contacts: Dictionary of grouped contacts
            strat_column: Stratigraphic column for unit colors
        """
        logger.info("=" * 60)
        logger.info("Initializing Tie Line Editor")
        logger.info("=" * 60)
        
        self.parent_app = parent_app
        self.all_sections_data = all_sections_data
        self.grouped_contacts = grouped_contacts
        self.strat_column = strat_column
        
        # State
        self.current_contact_group: Optional[str] = None  # Selected contact group name
        self.tie_line_start: Optional[TieLinePoint] = None  # First point of tie being drawn
        self.drawing_tie = False
        
        # View state
        self.current_center_section = 0
        self.sections_to_display = 3  # Show 3 sections at a time for tie drawing
        
        # Get sorted northings
        self.northings = sorted(
            set(
                section_data.get("northing", 0)
                for section_data in all_sections_data.values()
                if section_data.get("northing") is not None
            ),
            reverse=True
        )
        
        if not self.northings:
            logger.error("No valid northings found!")
            messagebox.showerror("Error", "No valid northings found in section data!")
            return
        
        logger.info(f"Found {len(self.northings)} sections")
        logger.info(f"Found {len(grouped_contacts)} contact groups")
        
        # Calculate display ranges
        self.calculate_ranges()
        
        # Matplotlib artist tracking
        self.contact_lines = {}  # {line_artist: {'group': str, 'polyline': ContactPolyline}}
        self.tie_line_artists = []  # List of tie line artists
        self.temp_tie_line = None  # Temporary line while drawing
        self.snap_highlight = None  # Visual indicator for snap target

        # Editing state
        self.edit_mode = False
        self.selected_vertex = None  # (polyline, vertex_idx, northing)
        self.dragging_vertex = False
        
        # Create window
        self.create_window()
    
    def calculate_ranges(self):
        """Calculate global coordinate ranges for display."""
        self.easting_min = float('inf')
        self.easting_max = float('-inf')
        self.rl_min = float('inf')
        self.rl_max = float('-inf')
        
        for section_data in self.all_sections_data.values():
            for unit in section_data.get("units", {}).values():
                vertices = unit.get("vertices", [])
                for i in range(0, len(vertices), 2):
                    if i + 1 < len(vertices):
                        e, rl = vertices[i], vertices[i + 1]
                        if e > 10000:  # Real-world coordinates
                            self.easting_min = min(self.easting_min, e)
                            self.easting_max = max(self.easting_max, e)
                            self.rl_min = min(self.rl_min, rl)
                            self.rl_max = max(self.rl_max, rl)
        
        # Add margins
        if self.easting_min != float('inf'):
            margin_e = (self.easting_max - self.easting_min) * 0.05
            margin_rl = (self.rl_max - self.rl_min) * 0.05
            self.easting_min -= margin_e
            self.easting_max += margin_e
            self.rl_min -= margin_rl
            self.rl_max += margin_rl
    
    def create_window(self):
        """Create the main editor window."""
        self.window = tk.Toplevel(self.parent_app.root)
        self.window.title("Tie Line Editor - Contact Correlation")
        self.window.geometry("1600x900")
        
        # Main horizontal layout
        main_paned = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Section display
        self.create_section_panel(main_paned)
        
        # Right: Controls
        self.create_control_panel(main_paned)
        
        # Status bar
        self.status_var = tk.StringVar(value="Select a contact group to begin")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Initial display
        self.update_display()
    
    def create_section_panel(self, parent):
        """Create the section display panel."""
        section_frame = ttk.Frame(parent, width=1100)
        parent.add(section_frame, weight=3)
        
        # Navigation controls
        nav_frame = ttk.Frame(section_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(nav_frame, text="Section:").pack(side=tk.LEFT, padx=5)
        
        self.section_combo = ttk.Combobox(
            nav_frame,
            values=[f"N {int(n)}" for n in self.northings],
            state="readonly",
            width=15
        )
        self.section_combo.pack(side=tk.LEFT, padx=5)
        self.section_combo.current(0)
        self.section_combo.bind("<<ComboboxSelected>>", self.on_section_changed)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_section).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_section).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(nav_frame, text="Show sections:").pack(side=tk.LEFT, padx=5)
        self.section_count_var = tk.IntVar(value=3)
        section_spin = ttk.Spinbox(
            nav_frame,
            from_=1, to=5,
            textvariable=self.section_count_var,
            width=5,
            command=self.on_section_count_changed
        )
        section_spin.pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=section_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(section_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key)
    
    def create_control_panel(self, parent):
        """Create the control panel."""
        control_frame = ttk.Frame(parent, width=450)
        parent.add(control_frame, weight=1)
        
        # Contact group selection
        group_frame = ttk.LabelFrame(control_frame, text="Contact Groups", padding=5)
        group_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(group_frame, text="Select contact to draw ties:").pack(anchor=tk.W)
        
        # Scrollable list of contact groups
        list_frame = ttk.Frame(group_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.group_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            height=10,
            font=('Arial', 10)
        )
        self.group_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.group_listbox.yview)
        
        # Populate list
        for group_name, group in self.grouped_contacts.items():
            section_count = len(group.get_sections())
            tie_count = len(group.tie_lines)
            self.group_listbox.insert(tk.END, f"{group_name} ({section_count} sections, {tie_count} ties)")
        
        self.group_listbox.bind('<<ListboxSelect>>', self.on_group_selected)
        
        # Tie line controls
        tie_frame = ttk.LabelFrame(control_frame, text="Tie Lines", padding=5)
        tie_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tie_info_label = ttk.Label(tie_frame, text="No contact selected")
        self.tie_info_label.pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(tie_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            btn_frame, text="Auto-Suggest Ties",
            command=self.auto_suggest_ties
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame, text="Clear Ties",
            command=self.clear_ties_for_group
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame, text="Accept Suggestions",
            command=self.accept_suggestions
        ).pack(side=tk.LEFT, padx=2)

        # Contact Editing controls
        edit_frame = ttk.LabelFrame(control_frame, text="Contact Editing", padding=5)
        edit_frame.pack(fill=tk.X, padx=5, pady=5)

        # Edit mode toggle
        self.edit_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            edit_frame, text="Enable vertex editing",
            variable=self.edit_mode_var,
            command=self.toggle_edit_mode
        ).pack(anchor=tk.W)

        edit_btn_frame = ttk.Frame(edit_frame)
        edit_btn_frame.pack(fill=tk.X, pady=3)

        ttk.Button(
            edit_btn_frame, text="Clean Hooks",
            command=self.clean_contact_hooks
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            edit_btn_frame, text="Smooth",
            command=self.smooth_contact
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            edit_btn_frame, text="Delete Vertex",
            command=self.delete_selected_vertex
        ).pack(side=tk.LEFT, padx=2)

        self.edit_info_label = ttk.Label(
            edit_frame,
            text="Select a contact group, then\nenable editing to modify vertices",
            font=('Arial', 8)
        )
        self.edit_info_label.pack(anchor=tk.W, pady=2)

        # Instructions
        instr_frame = ttk.LabelFrame(control_frame, text="Instructions", padding=5)
        instr_frame.pack(fill=tk.X, padx=5, pady=5)

        instructions = """
1. Select a contact group from the list
2. Click on a contact line to start a tie
3. Click on the corresponding contact in
   an ADJACENT section to complete the tie
4. Press Escape to cancel a tie in progress
5. Ties can only connect adjacent sections
6. Use Auto-Suggest for automatic ties

Editing Mode:
- Enable editing to modify vertices
- Right-click a vertex to delete it
- Drag vertices to move them
- Use Clean Hooks to remove artifacts
- Start/end/inflection points are best
        """
        ttk.Label(instr_frame, text=instructions.strip(), justify=tk.LEFT).pack(anchor=tk.W)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(control_frame, text="Summary", padding=5)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_label = ttk.Label(summary_frame, text="")
        self.summary_label.pack(anchor=tk.W)
        self.update_summary()
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(
            action_frame, text="Export Contacts + Ties",
            command=self.export_contacts
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            action_frame, text="Close",
            command=self.window.destroy
        ).pack(fill=tk.X, pady=2)
    
    def update_summary(self):
        """Update the summary label."""
        total_ties = sum(len(g.tie_lines) for g in self.grouped_contacts.values())
        groups_with_ties = sum(1 for g in self.grouped_contacts.values() if g.tie_lines)
        
        text = f"Total contact groups: {len(self.grouped_contacts)}\n"
        text += f"Groups with ties: {groups_with_ties}\n"
        text += f"Total tie lines: {total_ties}"
        
        self.summary_label.config(text=text)
    
    def update_display(self):
        """Update the section display."""
        self.ax.clear()
        self.contact_lines = {}
        self.tie_line_artists = []
        
        # Get sections to display
        start_idx = self.current_center_section
        end_idx = min(start_idx + self.sections_to_display, len(self.northings))
        
        sections_to_show = self.northings[start_idx:end_idx]
        
        if not sections_to_show:
            self.canvas.draw()
            return
        
        # Calculate vertical spacing between sections
        rl_range = self.rl_max - self.rl_min
        section_spacing = rl_range * 1.2
        
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
            self._plot_units(section_data, y_offset, northing)
            
            # Plot contacts (interactive)
            self._plot_contacts(northing, y_offset)
            
            # Section label
            self.ax.text(
                self.easting_min + 50, y_offset + self.rl_max - 20,
                f"N {int(northing)}",
                fontsize=12, fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Section boundary line
            self.ax.axhline(
                y=y_offset + self.rl_min,
                color='gray', linestyle='--', alpha=0.5, linewidth=0.5
            )
        
        # Plot existing tie lines
        self._plot_tie_lines(sections_to_show, section_spacing)
        
        # Set axis limits
        self.ax.set_xlim(self.easting_min, self.easting_max)
        
        total_height = len(sections_to_show) * section_spacing
        self.ax.set_ylim(
            -total_height + self.rl_min,
            self.rl_max + 20
        )
        
        self.ax.set_xlabel("Easting")
        self.ax.set_ylabel("RL (with section offset)")
        self.ax.set_aspect('equal')
        
        # Title
        if self.current_contact_group:
            self.ax.set_title(f"Tie Line Editor - {self.current_contact_group}")
        else:
            self.ax.set_title("Tie Line Editor - Select a contact group")
        
        self.canvas.draw()
    
    def _plot_units(self, section_data: Dict, y_offset: float, northing: float):
        """Plot geological units (faded, non-interactive)."""
        units = section_data.get("units", {})
        
        for unit_name, unit in units.items():
            vertices = unit.get("vertices", [])
            if len(vertices) < 6:
                continue
            
            xs = [vertices[i] for i in range(0, len(vertices), 2)]
            ys = [vertices[i + 1] + y_offset for i in range(0, len(vertices), 2) if i + 1 < len(vertices)]
            
            if not xs or not ys:
                continue
            
            # Close polygon
            if xs[0] != xs[-1]:
                xs.append(xs[0])
                ys.append(ys[0])
            
            color = unit.get("color", (0.5, 0.5, 0.5))
            
            try:
                poly = MplPolygon(
                    list(zip(xs, ys)),
                    facecolor=color,
                    edgecolor='gray',
                    alpha=self.UNIT_ALPHA,
                    linewidth=0.5,
                    picker=False  # Not interactive
                )
                self.ax.add_patch(poly)
            except Exception as e:
                logger.warning(f"Error plotting unit {unit_name}: {e}")
    
    def _plot_contacts(self, northing: float, y_offset: float):
        """Plot contacts for a section."""
        for group_name, group in self.grouped_contacts.items():
            polylines = group.get_polylines_for_section(northing)
            
            is_active_group = (group_name == self.current_contact_group)
            
            for pl_idx, polyline in enumerate(polylines):
                coords = polyline.get_coords()
                if len(coords) < 2:
                    continue
                
                xs = [c[0] for c in coords]
                ys = [c[1] + y_offset for c in coords]
                
                # Determine color
                if is_active_group:
                    color = self.CONTACT_COLOR_ACTIVE
                    linewidth = self.CONTACT_LINE_WIDTH * 1.5
                    zorder = 10
                else:
                    color = self.CONTACT_COLOR_UNSELECTED
                    linewidth = self.CONTACT_LINE_WIDTH
                    zorder = 5
                
                try:
                    line, = self.ax.plot(
                        xs, ys,
                        color=color,
                        linewidth=linewidth,
                        alpha=0.9,
                        picker=True,
                        pickradius=8,
                        zorder=zorder
                    )
                    
                    # Show vertex nodes for active contact group
                    if is_active_group:
                        self.ax.scatter(
                            xs, ys,
                            s=30,
                            c='white',
                            edgecolors=color,
                            linewidths=1.5,
                            zorder=zorder + 1,
                            alpha=0.8
                        )
                    
                    self.contact_lines[line] = {
                        'group': group_name,
                        'polyline': polyline,
                        'polyline_idx': pl_idx,
                        'northing': northing,
                        'y_offset': y_offset
                    }
                except Exception as e:
                    logger.warning(f"Error plotting contact: {e}")
    
    def _plot_tie_lines(self, sections_to_show: List[float], section_spacing: float):
        """Plot existing tie lines."""
        if not self.current_contact_group:
            return
        
        group = self.grouped_contacts.get(self.current_contact_group)
        if not group:
            return
        
        # Build northing to y_offset mapping
        northing_offsets = {}
        for idx, northing in enumerate(sections_to_show):
            northing_offsets[northing] = -idx * section_spacing
        
        for tie in group.tie_lines:
            from_n = tie['from_northing']
            to_n = tie['to_northing']
            
            # Check if both sections are visible
            if from_n not in northing_offsets or to_n not in northing_offsets:
                continue
            
            from_offset = northing_offsets[from_n]
            to_offset = northing_offsets[to_n]
            
            from_pt = tie['from_point']
            to_pt = tie['to_point']
            
            xs = [from_pt[0], to_pt[0]]
            ys = [from_pt[1] + from_offset, to_pt[1] + to_offset]
            
            # Use different color for suggested vs manual ties
            color = self.TIE_LINE_SUGGESTED_COLOR if tie.get('auto_suggested') else self.TIE_LINE_COLOR
            
            line, = self.ax.plot(
                xs, ys,
                color=color,
                linewidth=self.TIE_LINE_WIDTH,
                linestyle='--',
                alpha=0.8,
                marker='o',
                markersize=4,
                zorder=15
            )
            self.tie_line_artists.append(line)
    
    def _find_snap_target(self, click_x: float, click_y: float) -> Optional[Tuple[Dict, int, float, Tuple[float, float]]]:
        """
        Find the nearest snap target (vertex node) within SNAP_RADIUS.

        Args:
            click_x: Click x coordinate
            click_y: Click y coordinate (with y_offset already applied in display)

        Returns:
            Tuple of (contact_data, vertex_idx, distance, (easting, rl)) or None if no snap target
        """
        best_snap = None
        min_dist = float('inf')

        for line, data in self.contact_lines.items():
            # Only consider the active contact group
            if self.current_contact_group and data['group'] != self.current_contact_group:
                continue

            polyline = data['polyline']
            y_offset = data['y_offset']
            coords = polyline.get_coords()

            for i, (e, rl) in enumerate(coords):
                # Calculate distance to this vertex (accounting for y_offset)
                dist = np.sqrt((e - click_x)**2 + (rl + y_offset - click_y)**2)
                if dist < min_dist and dist <= self.SNAP_RADIUS:
                    min_dist = dist
                    best_snap = (data, i, dist, (e, rl))

        return best_snap

    def on_click(self, event):
        """Handle mouse click on the plot."""
        if event.inaxes != self.ax:
            return

        # Handle edit mode clicks (left or right click)
        if self.edit_mode and (event.button == 1 or event.button == 3):
            if self.on_edit_click(event):
                return

        if event.button != 1:  # Left click only for tie line mode
            return

        click_x, click_y = event.xdata, event.ydata

        # If no contact group selected, check if clicking on any contact to select it
        if self.current_contact_group is None:
            for line, data in self.contact_lines.items():
                if line.contains(event)[0]:
                    self.select_contact_group(data['group'])
                    return
            return

        # Find snap target within SNAP_RADIUS
        snap_target = self._find_snap_target(click_x, click_y)

        if snap_target is None:
            self.status_var.set(f"Click closer to a node (within {self.SNAP_RADIUS} units)")
            return

        data, nearest_idx, snap_dist, nearest_point = snap_target
        group_name = data['group']
        northing = data['northing']
        y_offset = data['y_offset']
        polyline = data['polyline']

        # If clicking a different contact group, warn user
        if group_name != self.current_contact_group:
            messagebox.showwarning(
                "Wrong Contact",
                f"You're drawing ties for '{self.current_contact_group}'.\n"
                f"Clicked contact is '{group_name}'.\n\n"
                "You can only tie contacts of the same type."
            )
            return

        if not self.drawing_tie:
            # Start a new tie line
            self.tie_line_start = TieLinePoint(
                northing=northing,
                easting=nearest_point[0],
                rl=nearest_point[1],
                contact_group=group_name,
                polyline_idx=data['polyline_idx'],
                vertex_idx=nearest_idx
            )
            self.drawing_tie = True
            self.status_var.set(
                f"Tie started at N={int(northing)}, E={nearest_point[0]:.1f} "
                f"(snapped to node {nearest_idx}). Click adjacent section to complete."
            )

            # Draw start point marker
            self.ax.plot(
                nearest_point[0], nearest_point[1] + y_offset,
                'o', color='red', markersize=12, zorder=20
            )
            self.canvas.draw()

        else:
            # Complete the tie line
            start = self.tie_line_start

            # Validate: must be adjacent section
            start_idx = self.northings.index(start.northing) if start.northing in self.northings else -1
            end_idx = self.northings.index(northing) if northing in self.northings else -1

            if abs(start_idx - end_idx) != 1:
                messagebox.showwarning(
                    "Invalid Tie",
                    "Tie lines can only connect ADJACENT sections.\n"
                    "You cannot skip sections."
                )
                return

            # Add the tie line to the group
            group = self.grouped_contacts[self.current_contact_group]

            # Ensure from_northing > to_northing (north to south)
            if start.northing > northing:
                group.tie_lines.append({
                    "from_northing": start.northing,
                    "from_point": (start.easting, start.rl),
                    "from_vertex_idx": start.vertex_idx,
                    "from_polyline_idx": start.polyline_idx,
                    "to_northing": northing,
                    "to_point": nearest_point,
                    "to_vertex_idx": nearest_idx,
                    "to_polyline_idx": data['polyline_idx']
                })
            else:
                group.tie_lines.append({
                    "from_northing": northing,
                    "from_point": nearest_point,
                    "from_vertex_idx": nearest_idx,
                    "from_polyline_idx": data['polyline_idx'],
                    "to_northing": start.northing,
                    "to_point": (start.easting, start.rl),
                    "to_vertex_idx": start.vertex_idx,
                    "to_polyline_idx": start.polyline_idx
                })

            self.drawing_tie = False
            self.tie_line_start = None

            self.status_var.set(f"Tie line added to {self.current_contact_group} (snapped to nodes)")
            self.update_display()
            self.update_listbox_item()
            self.update_summary()
    
    def on_motion(self, event):
        """Handle mouse motion for tie line preview and snap feedback."""
        if event.inaxes != self.ax:
            # Remove snap highlight if moving outside plot
            if self.snap_highlight is not None:
                self.snap_highlight.remove()
                self.snap_highlight = None
                self.canvas.draw_idle()
            return

        # Remove previous snap highlight
        if self.snap_highlight is not None:
            self.snap_highlight.remove()
            self.snap_highlight = None

        # Show snap highlight when near a node (for active contact group)
        if self.current_contact_group:
            snap_target = self._find_snap_target(event.xdata, event.ydata)
            if snap_target:
                data, vertex_idx, snap_dist, (e, rl) = snap_target
                y_offset = data['y_offset']
                # Draw snap highlight
                self.snap_highlight = self.ax.scatter(
                    [e], [rl + y_offset],
                    s=self.SNAP_HIGHLIGHT_SIZE,
                    c=self.SNAP_HIGHLIGHT_COLOR,
                    marker='o',
                    alpha=0.6,
                    zorder=25,
                    edgecolors='white',
                    linewidths=2
                )

        # Handle tie line preview (only when drawing)
        if not self.drawing_tie or self.tie_line_start is None:
            self.canvas.draw_idle()
            return

        # Remove previous temp line
        if self.temp_tie_line is not None:
            self.temp_tie_line.remove()
            self.temp_tie_line = None

        # Find the y_offset for the start section
        start_idx = self.current_center_section
        end_idx = min(start_idx + self.sections_to_display, len(self.northings))
        sections_to_show = self.northings[start_idx:end_idx]

        rl_range = self.rl_max - self.rl_min
        section_spacing = rl_range * 1.2

        start_y_offset = 0
        for idx, n in enumerate(sections_to_show):
            if abs(n - self.tie_line_start.northing) < 0.1:
                start_y_offset = -idx * section_spacing
                break

        # Determine end point - snap to node if within radius
        end_x, end_y = event.xdata, event.ydata
        line_color = 'gray'  # Gray when not snapped

        snap_target = self._find_snap_target(event.xdata, event.ydata)
        if snap_target:
            data, vertex_idx, snap_dist, (e, rl) = snap_target
            y_offset = data['y_offset']
            end_x = e
            end_y = rl + y_offset
            line_color = 'green'  # Green when snapped to valid node

        # Draw temp line
        self.temp_tie_line, = self.ax.plot(
            [self.tie_line_start.easting, end_x],
            [self.tie_line_start.rl + start_y_offset, end_y],
            color=line_color,
            linewidth=2.0,
            linestyle=':',
            alpha=0.8,
            zorder=20
        )

        self.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle key presses."""
        if event.key == 'escape':
            self.cancel_tie()
    
    def cancel_tie(self):
        """Cancel the current tie line being drawn."""
        if self.drawing_tie:
            self.drawing_tie = False
            self.tie_line_start = None
            if self.temp_tie_line is not None:
                self.temp_tie_line.remove()
                self.temp_tie_line = None
            if self.snap_highlight is not None:
                self.snap_highlight.remove()
                self.snap_highlight = None
            self.status_var.set("Tie line cancelled")
            self.update_display()
    
    def on_group_selected(self, event):
        """Handle contact group selection from listbox."""
        selection = self.group_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        group_names = list(self.grouped_contacts.keys())
        if idx < len(group_names):
            self.select_contact_group(group_names[idx])
    
    def select_contact_group(self, group_name: str):
        """Select a contact group for tie line editing."""
        self.current_contact_group = group_name
        self.cancel_tie()  # Cancel any in-progress tie
        
        group = self.grouped_contacts.get(group_name)
        if group:
            sections = group.get_sections()
            tie_count = len(group.tie_lines)
            self.tie_info_label.config(
                text=f"Group: {group_name}\n"
                     f"Sections: {len(sections)}\n"
                     f"Existing ties: {tie_count}"
            )
        
        self.status_var.set(f"Selected contact group: {group_name}")
        self.update_display()
    
    def auto_suggest_ties(self):
        """Auto-suggest tie lines for the current contact group."""
        if not self.current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return
        
        group = self.grouped_contacts.get(self.current_contact_group)
        if not group:
            return
        
        # Use ContactExtractor to suggest ties
        extractor = ContactExtractor()
        suggestions = extractor.suggest_tie_lines(group)
        
        if not suggestions:
            messagebox.showinfo("No Suggestions", "Could not auto-suggest tie lines for this contact.")
            return
        
        # Add suggestions (marked as auto-suggested)
        for suggestion in suggestions:
            suggestion['auto_suggested'] = True
            group.tie_lines.append(suggestion)
        
        self.status_var.set(f"Added {len(suggestions)} suggested tie lines")
        self.update_display()
        self.update_listbox_item()
        self.update_summary()
    
    def accept_suggestions(self):
        """Convert auto-suggested ties to manual (accepted) ties."""
        if not self.current_contact_group:
            return
        
        group = self.grouped_contacts.get(self.current_contact_group)
        if not group:
            return
        
        for tie in group.tie_lines:
            tie['auto_suggested'] = False
        
        self.update_display()
        self.status_var.set("All suggestions accepted")
    
    def clear_ties_for_group(self):
        """Clear all tie lines for the current contact group."""
        if not self.current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return
        
        if not messagebox.askyesno("Confirm", f"Clear all tie lines for {self.current_contact_group}?"):
            return
        
        group = self.grouped_contacts.get(self.current_contact_group)
        if group:
            group.tie_lines = []
        
        self.status_var.set(f"Cleared tie lines for {self.current_contact_group}")
        self.update_display()
        self.update_listbox_item()
        self.update_summary()
    
    def update_listbox_item(self):
        """Update the listbox item for the current contact group."""
        if not self.current_contact_group:
            return
        
        group_names = list(self.grouped_contacts.keys())
        try:
            idx = group_names.index(self.current_contact_group)
            group = self.grouped_contacts[self.current_contact_group]
            section_count = len(group.get_sections())
            tie_count = len(group.tie_lines)
            
            self.group_listbox.delete(idx)
            self.group_listbox.insert(idx, f"{self.current_contact_group} ({section_count} sections, {tie_count} ties)")
            self.group_listbox.selection_set(idx)
        except ValueError:
            pass
    
    def on_section_changed(self, event):
        """Handle section selection change."""
        selection = self.section_combo.current()
        if selection >= 0:
            self.current_center_section = selection
            self.cancel_tie()
            self.update_display()
    
    def on_section_count_changed(self):
        """Handle section count change."""
        self.sections_to_display = self.section_count_var.get()
        self.cancel_tie()
        self.update_display()
    
    def prev_section(self):
        """Move to previous section."""
        if self.current_center_section > 0:
            self.current_center_section -= 1
            self.section_combo.current(self.current_center_section)
            self.cancel_tie()
            self.update_display()
    
    def next_section(self):
        """Move to next section."""
        if self.current_center_section < len(self.northings) - 1:
            self.current_center_section += 1
            self.section_combo.current(self.current_center_section)
            self.cancel_tie()
            self.update_display()
    
    # === Contact Editing Methods ===

    def toggle_edit_mode(self):
        """Toggle vertex editing mode on/off."""
        self.edit_mode = self.edit_mode_var.get()
        self.selected_vertex = None
        self.dragging_vertex = False

        if self.edit_mode:
            self.edit_info_label.config(text="EDIT MODE: Click vertices to select\nRight-click to delete")
            self.status_var.set("Edit mode enabled - click vertices to select")
        else:
            self.edit_info_label.config(text="Select a contact group, then\nenable editing to modify vertices")
            self.status_var.set("Edit mode disabled")

        self.update_display()

    def clean_contact_hooks(self):
        """Clean hooks from all contacts in the current group."""
        if not self.current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return

        group = self.grouped_contacts.get(self.current_contact_group)
        if not group:
            return

        from geotools.contact_postprocess import remove_hooks_iterative

        cleaned_count = 0
        for polyline in group.polylines:
            vertices = polyline.vertices
            if len(vertices) < 8:
                continue

            # Convert to coordinate list
            coords = []
            for i in range(0, len(vertices), 2):
                if i + 1 < len(vertices):
                    coords.append((vertices[i], vertices[i + 1]))

            original_len = len(coords)
            coords = remove_hooks_iterative(coords, max_iterations=3)

            if len(coords) < original_len:
                new_vertices = []
                for x, y in coords:
                    new_vertices.extend([x, y])
                polyline.vertices = new_vertices
                cleaned_count += 1

        self.status_var.set(f"Cleaned hooks from {cleaned_count} contacts")
        self.update_display()

    def smooth_contact(self):
        """Apply smoothing to all contacts in the current group."""
        if not self.current_contact_group:
            messagebox.showwarning("No Selection", "Please select a contact group first.")
            return

        group = self.grouped_contacts.get(self.current_contact_group)
        if not group:
            return

        from geotools.contact_postprocess import smooth_contact as smooth_func

        smoothed_count = 0
        for polyline in group.polylines:
            vertices = polyline.vertices
            if len(vertices) < 6:
                continue

            # Convert to coordinate list
            coords = []
            for i in range(0, len(vertices), 2):
                if i + 1 < len(vertices):
                    coords.append((vertices[i], vertices[i + 1]))

            coords = smooth_func(coords, window_size=3, preserve_endpoints=True)

            new_vertices = []
            for x, y in coords:
                new_vertices.extend([x, y])
            polyline.vertices = new_vertices
            smoothed_count += 1

        self.status_var.set(f"Smoothed {smoothed_count} contacts")
        self.update_display()

    def delete_selected_vertex(self):
        """Delete the currently selected vertex."""
        if not self.selected_vertex:
            messagebox.showinfo("No Selection", "Click on a vertex to select it first.")
            return

        polyline, vertex_idx, northing = self.selected_vertex
        num_vertices = len(polyline.vertices) // 2

        if num_vertices <= 2:
            messagebox.showwarning("Cannot Delete", "Contact must have at least 2 vertices.")
            return

        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Delete vertex {vertex_idx} from this contact?"):
            return

        # Delete the vertex
        start_idx = vertex_idx * 2
        del polyline.vertices[start_idx:start_idx + 2]

        self.selected_vertex = None
        self.status_var.set(f"Deleted vertex {vertex_idx}")
        self.update_display()

    def on_edit_click(self, event):
        """Handle click in edit mode - select vertex."""
        if not self.edit_mode or event.inaxes != self.ax:
            return False

        click_x, click_y = event.xdata, event.ydata

        # Find nearest vertex across all displayed contacts
        best_match = None
        min_dist = self.SNAP_RADIUS

        for line, data in self.contact_lines.items():
            if self.current_contact_group and data['group'] != self.current_contact_group:
                continue

            polyline = data['polyline']
            y_offset = data['y_offset']
            coords = polyline.get_coords()

            for i, (e, rl) in enumerate(coords):
                dist = np.sqrt((e - click_x)**2 + (rl + y_offset - click_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = (polyline, i, data['northing'], y_offset, e, rl)

        if best_match:
            polyline, idx, northing, y_offset, e, rl = best_match
            self.selected_vertex = (polyline, idx, northing)

            # Right-click to delete
            if event.button == 3:
                self.delete_selected_vertex()
                return True

            self.status_var.set(f"Selected vertex {idx} at E={e:.1f}, RL={rl:.1f}")
            self.update_display()

            # Highlight selected vertex
            self.ax.scatter(
                [e], [rl + y_offset],
                s=150, c='yellow', marker='s',
                edgecolors='red', linewidths=2, zorder=30
            )
            self.canvas.draw()
            return True

        return False

    def export_contacts(self):
        """Export contacts with tie lines."""
        # This will integrate with the main app's export functionality
        # For now, just pass the data back
        if hasattr(self.parent_app, 'receive_contacts_with_ties'):
            self.parent_app.receive_contacts_with_ties(self.grouped_contacts)
            messagebox.showinfo("Export", "Contacts and tie lines passed to main application.\nUse Export menu to save to DXF.")
        else:
            messagebox.showinfo(
                "Export Ready",
                f"Contacts: {len(self.grouped_contacts)} groups\n"
                f"Tie lines: {sum(len(g.tie_lines) for g in self.grouped_contacts.values())}\n\n"
                "Close this window and use Export > Contacts to DXF"
            )

        self.window.destroy()


def open_tie_line_editor(parent_app, all_sections_data, grouped_contacts, strat_column):
    """
    Open the tie line editor window.
    
    Args:
        parent_app: Main GUI application
        all_sections_data: Section data dictionary
        grouped_contacts: Dictionary of GroupedContact objects
        strat_column: Stratigraphic column
    
    Returns:
        TieLineEditor instance
    """
    return TieLineEditor(parent_app, all_sections_data, grouped_contacts, strat_column)
