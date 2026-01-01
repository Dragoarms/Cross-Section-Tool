# geotools\strat_column_v2.py
"""
Enhanced stratigraphic column module v2.

Key improvements over v1:
1. Prospect-based grouping (group units by prospect/area)
2. Drag-and-drop reordering support
3. Auto-labeling logic based on stratigraphic position
4. Fault-aware contact validation
5. Dataclass-based for better type safety
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FaultType(Enum):
    """Types of geological faults."""
    NORMAL = "normal"
    REVERSE = "reverse"
    THRUST = "thrust"
    STRIKE_SLIP = "strike-slip"
    UNKNOWN = "unknown"


@dataclass
class StratUnit:
    """A single stratigraphic unit."""
    name: str
    color: Tuple[float, float, float]  # RGB 0-1
    prospect: Optional[str] = None  # Prospect/group name
    age: str = "Unknown"
    order: int = 0  # Position within prospect (0 = youngest/top)
    is_unconformity: bool = False
    can_pinch_out: bool = True
    thickness: Optional[float] = None
    pattern: Optional[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "color": list(self.color),
            "prospect": self.prospect,
            "age": self.age,
            "order": self.order,
            "is_unconformity": self.is_unconformity,
            "can_pinch_out": self.can_pinch_out,
            "thickness": self.thickness,
            "pattern": self.pattern,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StratUnit":
        """Create from dictionary."""
        color = data.get("color", [0.5, 0.5, 0.5])
        if isinstance(color, list):
            color = tuple(color)
        return cls(
            name=data["name"],
            color=color,
            prospect=data.get("prospect"),
            age=data.get("age", "Unknown"),
            order=data.get("order", 0),
            is_unconformity=data.get("is_unconformity", False),
            can_pinch_out=data.get("can_pinch_out", True),
            thickness=data.get("thickness"),
            pattern=data.get("pattern"),
            description=data.get("description", ""),
        )


@dataclass
class Prospect:
    """A prospect/area grouping for stratigraphic units."""
    name: str
    order: int = 0  # Display order (0 = top)
    color: Optional[str] = None  # Header color (hex)
    expanded: bool = True  # UI collapse state
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "order": self.order,
            "color": self.color,
            "expanded": self.expanded,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Prospect":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            order=data.get("order", 0),
            color=data.get("color"),
            expanded=data.get("expanded", True),
            description=data.get("description", ""),
        )


@dataclass
class Fault:
    """A geological fault."""
    name: str
    fault_type: FaultType = FaultType.NORMAL
    color: str = "#FF0000"  # Hex color
    timing: int = 0  # Relative timing (lower = older)
    prospect: Optional[str] = None  # Optional prospect association
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "fault_type": self.fault_type.value,
            "color": self.color,
            "timing": self.timing,
            "prospect": self.prospect,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Fault":
        """Create from dictionary."""
        fault_type = data.get("fault_type", "normal")
        if isinstance(fault_type, str):
            try:
                fault_type = FaultType(fault_type)
            except ValueError:
                fault_type = FaultType.UNKNOWN
        return cls(
            name=data["name"],
            fault_type=fault_type,
            color=data.get("color", "#FF0000"),
            timing=data.get("timing", 0),
            prospect=data.get("prospect"),
            description=data.get("description", ""),
        )


class StratColumnV2:
    """
    Enhanced stratigraphic column manager with prospect grouping.
    
    Units are organized by prospect, with each prospect having its own
    ordered sequence from youngest (top) to oldest (bottom).
    """
    
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
    
    # Default prospect for units without explicit assignment
    DEFAULT_PROSPECT = "Default"
    
    def __init__(self):
        self.units: Dict[str, StratUnit] = {}  # {name: StratUnit}
        self.prospects: Dict[str, Prospect] = {}  # {name: Prospect}
        self.faults: Dict[str, Fault] = {}  # {name: Fault}
        
        # Create default prospect
        self.prospects[self.DEFAULT_PROSPECT] = Prospect(
            name=self.DEFAULT_PROSPECT, 
            order=0,
            description="Default grouping for units"
        )
    
    # === Unit Management ===
    
    def add_unit(
        self,
        name: str,
        color: Tuple[float, float, float],
        prospect: Optional[str] = None,
        age: str = "Unknown",
        position: Optional[int] = None,
        is_unconformity: bool = False,
        can_pinch_out: bool = True,
    ) -> bool:
        """
        Add a stratigraphic unit.
        
        Args:
            name: Unit name (must be unique)
            color: RGB color tuple (0-1 range)
            prospect: Prospect to assign to (creates if doesn't exist)
            age: Age description
            position: Position within prospect (None = append at bottom)
            is_unconformity: Whether this is an unconformable contact
            can_pinch_out: Whether unit can pinch out laterally
            
        Returns:
            True if added, False if name already exists
        """
        if name in self.units:
            logger.warning(f"Unit '{name}' already exists")
            return False
        
        # Use default prospect if none specified
        if prospect is None:
            prospect = self.DEFAULT_PROSPECT
        
        # Create prospect if it doesn't exist
        if prospect not in self.prospects:
            self.add_prospect(prospect)
        
        # Determine order within prospect
        if position is None:
            # Find max order in this prospect and add at end
            prospect_units = self.get_units_in_prospect(prospect)
            if prospect_units:
                position = max(u.order for u in prospect_units) + 1
            else:
                position = 0
        else:
            # Shift existing units down to make room
            self._shift_units_in_prospect(prospect, position, shift=1)
        
        unit = StratUnit(
            name=name,
            color=color,
            prospect=prospect,
            age=age,
            order=position,
            is_unconformity=is_unconformity,
            can_pinch_out=can_pinch_out,
        )
        self.units[name] = unit
        
        logger.info(f"Added unit '{name}' to prospect '{prospect}' at position {position}")
        return True
    
    def remove_unit(self, name: str) -> bool:
        """Remove a unit by name."""
        if name not in self.units:
            return False
        
        unit = self.units[name]
        prospect = unit.prospect
        order = unit.order
        
        del self.units[name]
        
        # Shift remaining units up to fill gap
        self._shift_units_in_prospect(prospect, order + 1, shift=-1)
        
        logger.info(f"Removed unit '{name}'")
        return True
    
    def move_unit(self, name: str, new_prospect: str, new_position: int) -> bool:
        """
        Move a unit to a new prospect and/or position.
        
        Args:
            name: Unit name to move
            new_prospect: Target prospect
            new_position: Position within target prospect
            
        Returns:
            True if moved successfully
        """
        if name not in self.units:
            return False
        
        unit = self.units[name]
        old_prospect = unit.prospect
        old_position = unit.order
        
        # Create new prospect if needed
        if new_prospect not in self.prospects:
            self.add_prospect(new_prospect)
        
        # Remove from old position
        if old_prospect == new_prospect:
            # Moving within same prospect
            if new_position == old_position:
                return True  # No change needed
            
            # Shift units between old and new positions
            if new_position > old_position:
                # Moving down - shift units up
                for u in self.units.values():
                    if u.prospect == old_prospect and old_position < u.order <= new_position:
                        u.order -= 1
            else:
                # Moving up - shift units down
                for u in self.units.values():
                    if u.prospect == old_prospect and new_position <= u.order < old_position:
                        u.order += 1
        else:
            # Moving to different prospect
            # Shift units in old prospect up
            self._shift_units_in_prospect(old_prospect, old_position + 1, shift=-1)
            # Shift units in new prospect down
            self._shift_units_in_prospect(new_prospect, new_position, shift=1)
        
        # Update unit
        unit.prospect = new_prospect
        unit.order = new_position
        
        logger.info(f"Moved unit '{name}' to prospect '{new_prospect}' position {new_position}")
        return True
    
    def _shift_units_in_prospect(self, prospect: str, from_position: int, shift: int):
        """Shift all units in a prospect from a position by shift amount."""
        for unit in self.units.values():
            if unit.prospect == prospect and unit.order >= from_position:
                unit.order += shift
    
    def get_unit(self, name: str) -> Optional[StratUnit]:
        """Get a unit by name."""
        return self.units.get(name)
    
    def get_units_in_prospect(self, prospect: str) -> List[StratUnit]:
        """Get all units in a prospect, ordered by position (young to old)."""
        units = [u for u in self.units.values() if u.prospect == prospect]
        return sorted(units, key=lambda u: u.order)
    
    def get_all_units_ordered(self) -> List[StratUnit]:
        """Get all units ordered by prospect then position."""
        result = []
        for prospect in self.get_prospects_ordered():
            result.extend(self.get_units_in_prospect(prospect.name))
        return result
    
    # === Prospect Management ===
    
    def add_prospect(self, name: str, position: Optional[int] = None, color: Optional[str] = None) -> bool:
        """Add a new prospect."""
        if name in self.prospects:
            logger.warning(f"Prospect '{name}' already exists")
            return False
        
        if position is None:
            position = len(self.prospects)
        else:
            # Shift existing prospects
            for p in self.prospects.values():
                if p.order >= position:
                    p.order += 1
        
        self.prospects[name] = Prospect(name=name, order=position, color=color)
        logger.info(f"Added prospect '{name}' at position {position}")
        return True
    
    def remove_prospect(self, name: str, move_units_to: str = None) -> bool:
        """
        Remove a prospect.
        
        Args:
            name: Prospect to remove
            move_units_to: Where to move units (None = also delete units)
        """
        if name not in self.prospects:
            return False
        
        if name == self.DEFAULT_PROSPECT:
            logger.warning("Cannot remove default prospect")
            return False
        
        # Handle units in this prospect
        units_to_move = [u.name for u in self.units.values() if u.prospect == name]
        
        if move_units_to:
            if move_units_to not in self.prospects:
                self.add_prospect(move_units_to)
            for unit_name in units_to_move:
                self.move_unit(unit_name, move_units_to, 
                              len(self.get_units_in_prospect(move_units_to)))
        else:
            for unit_name in units_to_move:
                self.remove_unit(unit_name)
        
        order = self.prospects[name].order
        del self.prospects[name]
        
        # Shift remaining prospects
        for p in self.prospects.values():
            if p.order > order:
                p.order -= 1
        
        logger.info(f"Removed prospect '{name}'")
        return True
    
    def move_prospect(self, name: str, new_position: int) -> bool:
        """Move a prospect to a new position."""
        if name not in self.prospects:
            return False
        
        prospect = self.prospects[name]
        old_position = prospect.order
        
        if new_position == old_position:
            return True
        
        # Shift other prospects
        if new_position > old_position:
            for p in self.prospects.values():
                if old_position < p.order <= new_position:
                    p.order -= 1
        else:
            for p in self.prospects.values():
                if new_position <= p.order < old_position:
                    p.order += 1
        
        prospect.order = new_position
        return True
    
    def get_prospects_ordered(self) -> List[Prospect]:
        """Get all prospects ordered by position."""
        return sorted(self.prospects.values(), key=lambda p: p.order)
    
    # === Fault Management ===
    
    def add_fault(
        self,
        name: str,
        fault_type: FaultType = FaultType.NORMAL,
        color: Optional[str] = None,
        timing: Optional[int] = None,
        prospect: Optional[str] = None,
    ) -> bool:
        """Add a fault."""
        if name in self.faults:
            logger.warning(f"Fault '{name}' already exists")
            return False
        
        if color is None:
            color_idx = len(self.faults) % len(self.DEFAULT_FAULT_COLORS)
            color = self.DEFAULT_FAULT_COLORS[color_idx]
        
        if timing is None:
            timing = len(self.faults)
        
        self.faults[name] = Fault(
            name=name,
            fault_type=fault_type,
            color=color,
            timing=timing,
            prospect=prospect,
        )
        
        logger.info(f"Added fault '{name}' ({fault_type.value})")
        return True
    
    def remove_fault(self, name: str) -> bool:
        """Remove a fault."""
        if name not in self.faults:
            return False
        del self.faults[name]
        logger.info(f"Removed fault '{name}'")
        return True
    
    def get_faults_ordered(self) -> List[Fault]:
        """Get faults ordered by timing (oldest first)."""
        return sorted(self.faults.values(), key=lambda f: f.timing)
    
    # === Auto-Labeling Logic ===
    
    def suggest_adjacent_assignments(
        self,
        assigned_unit_name: str,
        adjacent_polygon_positions: Dict[str, str],  # {polygon_name: "above" or "below"}
        already_assigned: Set[str],  # Polygon names that already have user assignments
        fault_separated: Set[str],  # Polygon names separated by fault from assigned_unit
    ) -> Dict[str, str]:
        """
        Suggest unit assignments for polygons adjacent to an assigned polygon.
        
        Args:
            assigned_unit_name: Name of the unit just assigned
            adjacent_polygon_positions: Dict mapping polygon names to their relative 
                                        position ("above" or "below") relative to assigned
            already_assigned: Polygon names that already have user assignments (skip these)
            fault_separated: Polygon names separated by a fault (skip these)
            
        Returns:
            Dict mapping polygon names to suggested unit names
        """
        suggestions = {}
        
        unit = self.get_unit(assigned_unit_name)
        if not unit:
            logger.warning(f"Unit '{assigned_unit_name}' not found")
            return suggestions
        
        prospect_units = self.get_units_in_prospect(unit.prospect)
        unit_idx = None
        for i, u in enumerate(prospect_units):
            if u.name == assigned_unit_name:
                unit_idx = i
                break
        
        if unit_idx is None:
            return suggestions
        
        for polygon_name, position in adjacent_polygon_positions.items():
            # Skip if already assigned by user
            if polygon_name in already_assigned:
                continue
            
            # Skip if separated by fault
            if polygon_name in fault_separated:
                continue
            
            if position == "above":
                # Above = younger = lower index in order
                if unit_idx > 0:
                    suggested_unit = prospect_units[unit_idx - 1]
                    suggestions[polygon_name] = suggested_unit.name
                    logger.debug(f"Suggesting '{suggested_unit.name}' for polygon '{polygon_name}' (above)")
            
            elif position == "below":
                # Below = older = higher index in order
                if unit_idx < len(prospect_units) - 1:
                    suggested_unit = prospect_units[unit_idx + 1]
                    suggestions[polygon_name] = suggested_unit.name
                    logger.debug(f"Suggesting '{suggested_unit.name}' for polygon '{polygon_name}' (below)")
        
        return suggestions
    
    def get_expected_contact_name(self, unit1: str, unit2: str) -> Optional[str]:
        """
        Get the expected contact name for two units.
        
        Returns canonical name (alphabetically sorted) or None if units
        are from different prospects or not adjacent.
        """
        u1 = self.get_unit(unit1)
        u2 = self.get_unit(unit2)
        
        if not u1 or not u2:
            return None
        
        if u1.prospect != u2.prospect:
            # Different prospects - might be fault contact
            return None
        
        # Check if adjacent
        if abs(u1.order - u2.order) != 1:
            # Not adjacent - might be fault contact or missing unit
            return None
        
        # Return canonical name
        names = sorted([unit1, unit2])
        return f"{names[0]}-{names[1]}"
    
    def validate_contact(
        self, unit1: str, unit2: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate if a contact between two units is geologically valid.
        
        Returns:
            (is_valid, contact_type, error_message)
            contact_type: 'conformable', 'unconformable', 'fault_contact', 'invalid'
        """
        u1 = self.get_unit(unit1)
        u2 = self.get_unit(unit2)
        
        if not u1 or not u2:
            return (False, "invalid", f"Unknown unit(s): {unit1 if not u1 else ''} {unit2 if not u2 else ''}")
        
        if u1.prospect != u2.prospect:
            return (False, "fault_contact", f"Units from different prospects ({u1.prospect} vs {u2.prospect})")
        
        order_diff = abs(u1.order - u2.order)
        
        if order_diff == 1:
            # Adjacent units
            younger = u1 if u1.order < u2.order else u2
            if younger.is_unconformity:
                return (True, "unconformable", None)
            return (True, "conformable", None)
        
        elif order_diff == 0:
            return (False, "invalid", "Same unit cannot contact itself")
        
        else:
            # Non-adjacent
            return (False, "fault_contact", f"Non-adjacent units (gap of {order_diff - 1} units)")
    
    # === Persistence ===
    
    def save(self, filepath: Path):
        """Save to JSON file."""
        data = {
            "version": 2,
            "prospects": [p.to_dict() for p in self.prospects.values()],
            "units": [u.to_dict() for u in self.units.values()],
            "faults": [f.to_dict() for f in self.faults.values()],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved stratigraphic column to {filepath}")
    
    def load(self, filepath: Path):
        """Load from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        version = data.get("version", 1)
        
        if version == 1:
            self._load_v1(data)
        else:
            self._load_v2(data)
        
        logger.info(f"Loaded stratigraphic column from {filepath}")
        logger.info(f"  - {len(self.prospects)} prospects")
        logger.info(f"  - {len(self.units)} units")
        logger.info(f"  - {len(self.faults)} faults")
    
    def _load_v2(self, data: Dict):
        """Load v2 format."""
        self.prospects.clear()
        self.units.clear()
        self.faults.clear()
        
        for p_data in data.get("prospects", []):
            prospect = Prospect.from_dict(p_data)
            self.prospects[prospect.name] = prospect
        
        # Ensure default prospect exists
        if self.DEFAULT_PROSPECT not in self.prospects:
            self.prospects[self.DEFAULT_PROSPECT] = Prospect(
                name=self.DEFAULT_PROSPECT, order=0
            )
        
        for u_data in data.get("units", []):
            unit = StratUnit.from_dict(u_data)
            self.units[unit.name] = unit
        
        for f_data in data.get("faults", []):
            fault = Fault.from_dict(f_data)
            self.faults[fault.name] = fault
    
    def _load_v1(self, data: Dict):
        """Load v1 format (backward compatibility)."""
        self.prospects.clear()
        self.units.clear()
        self.faults.clear()
        
        # Create default prospect
        self.prospects[self.DEFAULT_PROSPECT] = Prospect(
            name=self.DEFAULT_PROSPECT, order=0
        )
        
        # Load strat units
        for i, u_data in enumerate(data.get("strat_units", [])):
            color = u_data.get("color", [0.5, 0.5, 0.5])
            if isinstance(color, list):
                color = tuple(color)
            
            unit = StratUnit(
                name=u_data["name"],
                color=color,
                prospect=self.DEFAULT_PROSPECT,
                age=u_data.get("age", "Unknown"),
                order=i,
                is_unconformity=u_data.get("is_unconformity", False),
                can_pinch_out=u_data.get("can_pinch_out", True),
                thickness=u_data.get("thickness"),
                pattern=u_data.get("pattern"),
            )
            self.units[unit.name] = unit
        
        # Load faults
        for f_data in data.get("faults", []):
            fault_type_str = f_data.get("fault_type", "normal")
            try:
                fault_type = FaultType(fault_type_str)
            except ValueError:
                fault_type = FaultType.UNKNOWN
            
            fault = Fault(
                name=f_data["name"],
                fault_type=fault_type,
                color=f_data.get("color", "#FF0000"),
                timing=f_data.get("timing", 0),
            )
            self.faults[fault.name] = fault
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "prospects": len(self.prospects),
            "units": len(self.units),
            "faults": len(self.faults),
            "units_per_prospect": {
                p.name: len(self.get_units_in_prospect(p.name))
                for p in self.prospects.values()
            },
        }
    
    # === Backward Compatibility Properties ===
    
    @property
    def strat_units(self) -> List[Dict]:
        """For backward compatibility with v1 code."""
        return [u.to_dict() for u in self.get_all_units_ordered()]
    
    @property
    def unit_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """For backward compatibility."""
        return {u.name: u.color for u in self.units.values()}
    
    @property
    def fault_colors(self) -> Dict[str, str]:
        """For backward compatibility."""
        return {f.name: f.color for f in self.faults.values()}
    
    def get_all_unit_names(self) -> List[str]:
        """Get all unit names."""
        return list(self.units.keys())
    
    def get_all_fault_names(self) -> List[str]:
        """Get all fault names."""
        return list(self.faults.keys())
