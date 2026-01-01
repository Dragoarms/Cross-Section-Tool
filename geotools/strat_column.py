# geotools\strat_column.py
"""
Enhanced stratigraphic column module for managing geological units with proper relationships.
Includes support for faults with colors and types.
"""

from typing import List, Dict, Optional, Tuple, Set
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StratColumn:
    """Manage stratigraphic column with units, intrusives, and structural features."""

    # Default fault colors for cycling
    DEFAULT_FAULT_COLORS = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00AA00',  # Green
        '#FF8800',  # Orange
        '#8800FF',  # Purple
        '#00AAAA',  # Cyan
        '#FF00FF',  # Magenta
        '#888800',  # Olive
    ]

    def __init__(self):
        # Stratigraphic units (ordered from youngest to oldest)
        self.strat_units = []  # List of unit dicts
        self.unconformities = set()  # Set of unit names that are unconformities

        # Intrusive units (ordered by relative age, youngest to oldest)
        self.intrusive_units = []

        # Structural features
        self.faults = []  # List of fault dicts with timing and color
        self.fold_hinges = []  # List of major fold hinges

        # Visual properties
        self.unit_colors = {}
        self.unit_patterns = {}
        self.fault_colors = {}  # {fault_name: hex_color}

        # Relationship tracking
        self.allowed_contacts = set()  # Set of valid contact tuples
        self.lateral_continuity = {}  # Unit name -> bool (can pinch out)

    def add_strat_unit(
        self,
        name: str,
        age: str,
        color: tuple,
        pattern: Optional[str] = None,
        position: Optional[int] = None,
        is_unconformity: bool = False,
        can_pinch_out: bool = True,
        thickness: Optional[float] = None,
    ):
        """Add a stratigraphic unit to the column."""
        # Check for duplicate
        if any(u['name'] == name for u in self.strat_units):
            logger.warning(f"Unit '{name}' already exists, skipping")
            return

        unit = {
            "name": name,
            "age": age,
            "color": color,
            "pattern": pattern,
            "type": "stratigraphic",
            "is_unconformity": is_unconformity,
            "can_pinch_out": can_pinch_out,
            "thickness": thickness,
        }

        if position is not None:
            self.strat_units.insert(position, unit)
        else:
            self.strat_units.append(unit)

        if is_unconformity:
            self.unconformities.add(name)

        self.unit_colors[name] = color
        if pattern:
            self.unit_patterns[name] = pattern

        self.lateral_continuity[name] = can_pinch_out

        # Rebuild allowed contacts
        self._rebuild_allowed_contacts()

        logger.info(f"Added strat unit: {name} {'(unconformity)' if is_unconformity else ''}")

    def remove_strat_unit(self, name: str) -> bool:
        """Remove a stratigraphic unit by name."""
        for i, unit in enumerate(self.strat_units):
            if unit['name'] == name:
                self.strat_units.pop(i)
                self.unconformities.discard(name)
                self.unit_colors.pop(name, None)
                self.unit_patterns.pop(name, None)
                self.lateral_continuity.pop(name, None)
                self._rebuild_allowed_contacts()
                logger.info(f"Removed strat unit: {name}")
                return True
        return False

    def add_intrusive(
        self,
        name: str,
        age: str,
        color: tuple,
        pattern: Optional[str] = None,
        position: Optional[int] = None,
    ):
        """Add an intrusive unit."""
        # Check for duplicate
        if any(u['name'] == name for u in self.intrusive_units):
            logger.warning(f"Intrusive '{name}' already exists, skipping")
            return

        unit = {
            "name": name,
            "age": age,
            "color": color,
            "pattern": pattern,
            "type": "intrusive",
        }

        if position is not None:
            self.intrusive_units.insert(position, unit)
        else:
            self.intrusive_units.append(unit)

        self.unit_colors[name] = color
        if pattern:
            self.unit_patterns[name] = pattern

        logger.info(f"Added intrusive unit: {name}")

    def add_fault(
        self,
        name: str,
        fault_type: str = 'normal',  # 'normal', 'reverse', 'thrust', 'strike-slip'
        timing: Optional[int] = None,  # Relative timing (1 = oldest)
        color: Optional[str] = None,  # Hex color
    ):
        """
        Add a fault to the structural features.
        
        Args:
            name: Fault name (e.g., 'F1', 'Main Fault')
            fault_type: Type of fault movement
            timing: Relative timing (lower = older)
            color: Hex color string (auto-assigned if None)
        """
        # Check for duplicate
        if any(f['name'] == name for f in self.faults):
            logger.warning(f"Fault '{name}' already exists, skipping")
            return

        # Auto-assign color if not provided
        if color is None:
            color_idx = len(self.faults) % len(self.DEFAULT_FAULT_COLORS)
            color = self.DEFAULT_FAULT_COLORS[color_idx]

        fault = {
            "name": name,
            "type": "fault",
            "fault_type": fault_type,
            "timing": timing or len(self.faults) + 1,
            "color": color,
        }
        self.faults.append(fault)
        self.fault_colors[name] = color

        # Sort faults by timing
        self.faults.sort(key=lambda f: f.get("timing", 999))

        logger.info(f"Added {fault_type} fault: {name} (color: {color})")

    def remove_fault(self, name: str) -> bool:
        """Remove a fault by name."""
        for i, fault in enumerate(self.faults):
            if fault['name'] == name:
                self.faults.pop(i)
                self.fault_colors.pop(name, None)
                logger.info(f"Removed fault: {name}")
                return True
        return False

    def update_fault_color(self, name: str, color: str):
        """Update a fault's color."""
        for fault in self.faults:
            if fault['name'] == name:
                fault['color'] = color
                self.fault_colors[name] = color
                logger.debug(f"Updated fault {name} color to {color}")
                return True
        return False

    def get_fault(self, name: str) -> Optional[Dict]:
        """Get fault by name."""
        for fault in self.faults:
            if fault['name'] == name:
                return fault
        return None

    def get_fault_color(self, name: str) -> str:
        """Get fault color by name, with fallback."""
        return self.fault_colors.get(name, '#FF0000')

    def add_fold_hinge(self, name: str, fold_type: str):  # 'antiform' or 'synform'
        """Add a major fold hinge for correlation."""
        fold = {"name": name, "type": "fold_hinge", "fold_type": fold_type}
        self.fold_hinges.append(fold)
        logger.info(f"Added {fold_type} hinge: {name}")

    def _rebuild_allowed_contacts(self):
        """Rebuild the set of allowed contacts based on stratigraphy."""
        self.allowed_contacts = set()

        # Add conformable contacts (adjacent units in stratigraphy)
        for i in range(len(self.strat_units) - 1):
            younger = self.strat_units[i]["name"]
            older = self.strat_units[i + 1]["name"]

            # Contact is named older-younger (following geological convention)
            self.allowed_contacts.add((older, younger))

            # If the younger unit is an unconformity, it can contact any older unit
            if younger in self.unconformities:
                for j in range(i + 1, len(self.strat_units)):
                    much_older = self.strat_units[j]["name"]
                    self.allowed_contacts.add((much_older, younger))

            # If units can pinch out, allow skip contacts
            if self.lateral_continuity.get(younger, False):
                for j in range(i + 2, len(self.strat_units)):
                    much_older = self.strat_units[j]["name"]
                    next_younger = self.strat_units[i - 1]["name"] if i > 0 else None
                    if next_younger:
                        self.allowed_contacts.add((much_older, next_younger))

        # Add intrusive contacts
        for intrusive in self.intrusive_units:
            for strat in self.strat_units:
                self.allowed_contacts.add((strat["name"], intrusive["name"]))

            for other_intrusive in self.intrusive_units:
                if intrusive["name"] != other_intrusive["name"]:
                    self.allowed_contacts.add((other_intrusive["name"], intrusive["name"]))

    def validate_contact(
        self, unit1: str, unit2: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate if a contact between two units is geologically valid.

        Returns:
            (is_valid, contact_type, error_message)
            contact_type can be: 'conformable', 'unconformable', 'intrusive', 'fault_required'
        """
        contact = (unit1, unit2)
        reverse_contact = (unit2, unit1)

        if contact in self.allowed_contacts:
            if unit2 in self.unconformities:
                return (True, "unconformable", None)
            elif any(u["name"] == unit2 for u in self.intrusive_units):
                return (True, "intrusive", None)
            else:
                return (True, "conformable", None)

        elif reverse_contact in self.allowed_contacts:
            return (
                False,
                "reversed",
                f"Contact should be {unit2}-{unit1} (older-younger)",
            )

        else:
            unit1_idx = self._get_strat_index(unit1)
            unit2_idx = self._get_strat_index(unit2)

            if unit1_idx is not None and unit2_idx is not None:
                if abs(unit1_idx - unit2_idx) > 1:
                    return (
                        False,
                        "fault_required",
                        f"Contact between {unit1} and {unit2} requires a fault (non-adjacent units)",
                    )

            return (
                False,
                "invalid",
                f"Contact between {unit1} and {unit2} is not geologically valid",
            )

    def _get_strat_index(self, unit_name: str) -> Optional[int]:
        """Get the index of a unit in the stratigraphic column."""
        for i, unit in enumerate(self.strat_units):
            if unit["name"] == unit_name:
                return i
        return None

    def get_expected_contacts(self) -> List[str]:
        """Get a list of expected contacts based on stratigraphy."""
        contacts = []

        for i in range(len(self.strat_units) - 1):
            younger = self.strat_units[i]["name"]
            older = self.strat_units[i + 1]["name"]

            if younger in self.unconformities:
                contacts.append(f"{older}-{younger} (unconformable)")
            else:
                contacts.append(f"{older}-{younger}")

        return contacts

    def detect_missing_units(
        self, observed_contacts: List[Tuple[str, str]]
    ) -> List[str]:
        """Detect units that might be missing based on observed contacts."""
        missing = []

        for unit1, unit2 in observed_contacts:
            is_valid, contact_type, error = self.validate_contact(unit1, unit2)

            if contact_type == "fault_required":
                idx1 = self._get_strat_index(unit1)
                idx2 = self._get_strat_index(unit2)

                if idx1 is not None and idx2 is not None:
                    min_idx = min(idx1, idx2)
                    max_idx = max(idx1, idx2)

                    for i in range(min_idx + 1, max_idx):
                        missing_unit = self.strat_units[i]["name"]
                        if self.lateral_continuity.get(missing_unit, False):
                            missing.append(f"{missing_unit} (pinched out?)")
                        else:
                            missing.append(f"{missing_unit} (faulted out?)")

        return missing

    def get_unit_by_name(self, name: str) -> Optional[Dict]:
        """Get a unit by name from strat or intrusive units."""
        for unit in self.strat_units:
            if unit['name'] == name:
                return unit
        for unit in self.intrusive_units:
            if unit['name'] == name:
                return unit
        return None

    def get_all_unit_names(self) -> List[str]:
        """Get all unit names (strat + intrusive)."""
        names = [u['name'] for u in self.strat_units]
        names.extend([u['name'] for u in self.intrusive_units])
        return names

    def get_all_fault_names(self) -> List[str]:
        """Get all fault names."""
        return [f['name'] for f in self.faults]

    def save_column(self, filepath: Path):
        """Save complete stratigraphic system to JSON."""
        data = {
            "strat_units": self.strat_units,
            "intrusive_units": self.intrusive_units,
            "faults": self.faults,
            "fold_hinges": self.fold_hinges,
            "unconformities": list(self.unconformities),
            "colors": {k: list(v) if isinstance(v, tuple) else v 
                      for k, v in self.unit_colors.items()},
            "fault_colors": self.fault_colors,
            "patterns": self.unit_patterns,
            "lateral_continuity": self.lateral_continuity,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved stratigraphic column to {filepath}")

    def load_column(self, filepath: Path):
        """Load complete stratigraphic system from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load strat units and convert colors from lists to tuples
        self.strat_units = data.get("strat_units", [])
        for unit in self.strat_units:
            if 'color' in unit and isinstance(unit['color'], list):
                unit['color'] = tuple(unit['color'])
        
        # Load intrusive units and convert colors
        self.intrusive_units = data.get("intrusive_units", [])
        for unit in self.intrusive_units:
            if 'color' in unit and isinstance(unit['color'], list):
                unit['color'] = tuple(unit['color'])
        
        self.faults = data.get("faults", [])
        self.fold_hinges = data.get("fold_hinges", [])
        self.unconformities = set(data.get("unconformities", []))
        
        # Load colors
        self.unit_colors = {}
        for k, v in data.get("colors", {}).items():
            if isinstance(v, list):
                self.unit_colors[k] = tuple(v)
            else:
                self.unit_colors[k] = v

        self.fault_colors = data.get("fault_colors", {})
        
        # Ensure all faults have colors
        for fault in self.faults:
            if fault['name'] not in self.fault_colors:
                self.fault_colors[fault['name']] = fault.get('color', '#FF0000')

        self.unit_patterns = data.get("patterns", {})
        self.lateral_continuity = data.get("lateral_continuity", {})

        # Rebuild allowed contacts
        self._rebuild_allowed_contacts()

        logger.info(f"Loaded stratigraphic column from {filepath}")
        logger.info(f"  - {len(self.strat_units)} strat units")
        logger.info(f"  - {len(self.intrusive_units)} intrusive units")
        logger.info(f"  - {len(self.faults)} faults")

    def get_summary(self) -> Dict:
        """Get a summary of the stratigraphic column."""
        return {
            "strat_units": len(self.strat_units),
            "intrusive_units": len(self.intrusive_units),
            "faults": len(self.faults),
            "fold_hinges": len(self.fold_hinges),
            "unconformities": len(self.unconformities),
            "allowed_contacts": len(self.allowed_contacts),
        }
