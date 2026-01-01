# geotools\section_correlation.py
"""
Cross-section correlation module for matching geological units between sections.
Handles stratigraphy matching, tie line generation, and volume interpolation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import json

logger = logging.getLogger(__name__)


class SectionCorrelator:
    """Correlate geological units between multiple cross-sections."""

    def __init__(self):
        self.sections = {}  # {northing: section_data}
        self.correlations = []  # List of matched units between sections
        self.tie_lines = []  # 3D tie lines between correlated units

    def add_section(
        self, northing: float, units: Dict, contacts: List, pdf_name: str = None
    ):
        """
        Add a cross-section to the correlation dataset.

        Args:
            northing: The northing value (Y coordinate) of the section
            units: Dictionary of geological units from feature_extractor
            contacts: List of contacts from feature_extractor
            pdf_name: Source PDF name for reference
        """
        self.sections[northing] = {
            "units": units,
            "contacts": contacts,
            "pdf_name": pdf_name,
            "northing": northing,
        }
        logger.info(f"Added section at northing {northing} with {len(units)} units")

    def find_correlations(
        self,
        max_rl_difference: float = 50.0,
        max_easting_offset: float = 100.0,
        min_overlap_ratio: float = 0.3,
    ) -> List[Dict]:
        """
        Find correlations between units in adjacent sections.

        Args:
            max_rl_difference: Maximum RL difference to consider units matching
            max_easting_offset: Maximum easting offset for matching
            min_overlap_ratio: Minimum overlap ratio in easting range

        Returns:
            List of correlation dictionaries
        """
        self.correlations = []

        # Sort sections by northing
        sorted_northings = sorted(self.sections.keys())

        for i in range(len(sorted_northings) - 1):
            north1 = sorted_northings[i]
            north2 = sorted_northings[i + 1]

            section1 = self.sections[north1]
            section2 = self.sections[north2]

            # Compare each unit in section1 with units in section2
            for unit1_name, unit1 in section1["units"].items():
                # Get unit1 bounds
                bounds1 = self._get_unit_bounds(unit1)
                if not bounds1:
                    continue

                formation1 = unit1.get("formation", "UNKNOWN")

                # Find potential matches in section2
                for unit2_name, unit2 in section2["units"].items():
                    formation2 = unit2.get("formation", "UNKNOWN")

                    # Must be same formation type
                    if formation1 != formation2:
                        continue

                    bounds2 = self._get_unit_bounds(unit2)
                    if not bounds2:
                        continue

                    # Check RL overlap
                    rl_overlap = self._calculate_overlap(
                        (bounds1["rl_min"], bounds1["rl_max"]),
                        (bounds2["rl_min"], bounds2["rl_max"]),
                    )

                    if (
                        abs(bounds1["rl_center"] - bounds2["rl_center"])
                        > max_rl_difference
                    ):
                        continue

                    # Check easting overlap
                    easting_overlap = self._calculate_overlap(
                        (bounds1["e_min"], bounds1["e_max"]),
                        (bounds2["e_min"], bounds2["e_max"]),
                    )

                    if easting_overlap < min_overlap_ratio:
                        continue

                    # Calculate correlation score
                    score = self._calculate_correlation_score(
                        bounds1, bounds2, rl_overlap, easting_overlap
                    )

                    if score > 0.5:  # Threshold for accepting correlation
                        correlation = {
                            "north1": north1,
                            "north2": north2,
                            "unit1": unit1_name,
                            "unit2": unit2_name,
                            "formation": formation1,
                            "score": score,
                            "bounds1": bounds1,
                            "bounds2": bounds2,
                        }
                        self.correlations.append(correlation)

                        logger.info(
                            f"Correlated {unit1_name}@{north1} with "
                            f"{unit2_name}@{north2} (score: {score:.2f})"
                        )

        return self.correlations

    def generate_tie_lines(self) -> List[Dict]:
        """
        Generate 3D tie lines between correlated units.

        Returns:
            List of tie line dictionaries with 3D coordinates
        """
        self.tie_lines = []

        for corr in self.correlations:
            # Get unit boundaries
            unit1 = self.sections[corr["north1"]]["units"][corr["unit1"]]
            unit2 = self.sections[corr["north2"]]["units"][corr["unit2"]]

            # Generate tie points at corners and midpoints
            tie_points1 = self._get_tie_points(unit1, corr["north1"])
            tie_points2 = self._get_tie_points(unit2, corr["north2"])

            # Match tie points based on proximity
            for pt1 in tie_points1:
                # Find closest point in section2
                min_dist = float("inf")
                best_pt2 = None

                for pt2 in tie_points2:
                    # Distance in easting-RL space (ignore northing)
                    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_pt2 = pt2

                if best_pt2:
                    tie_line = {
                        "name": f"{corr['unit1']}-{corr['unit2']}_tie",
                        "formation": corr["formation"],
                        "start": pt1,  # (easting, northing, rl)
                        "end": best_pt2,
                        "correlation": corr,
                    }
                    self.tie_lines.append(tie_line)

        logger.info(f"Generated {len(self.tie_lines)} tie lines")
        return self.tie_lines

    def export_tie_lines_dxf(self, output_path: Path):
        """Export tie lines as 3D DXF."""
        with open(output_path, "w") as f:
            # DXF header
            f.write("0\nSECTION\n2\nENTITIES\n")

            for tie in self.tie_lines:
                # 3D polyline
                f.write("0\nPOLYLINE\n")
                f.write(f"8\n{tie['formation']}_ties\n")  # Layer
                f.write("66\n1\n")  # Vertices follow
                f.write("70\n8\n")  # 3D polyline

                # Start point
                f.write("0\nVERTEX\n")
                f.write(f"8\n{tie['formation']}_ties\n")
                f.write(f"10\n{tie['start'][0]:.2f}\n")  # X = Easting
                f.write(f"20\n{tie['start'][1]:.2f}\n")  # Y = Northing
                f.write(f"30\n{tie['start'][2]:.2f}\n")  # Z = RL

                # End point
                f.write("0\nVERTEX\n")
                f.write(f"8\n{tie['formation']}_ties\n")
                f.write(f"10\n{tie['end'][0]:.2f}\n")
                f.write(f"20\n{tie['end'][1]:.2f}\n")
                f.write(f"30\n{tie['end'][2]:.2f}\n")

                f.write("0\nSEQEND\n")

            f.write("0\nENDSEC\n0\nEOF\n")

        logger.info(f"Exported {len(self.tie_lines)} tie lines to {output_path}")

    def export_correlations(self, output_path: Path):
        """Export correlations to JSON."""
        export_data = {"sections": {}, "correlations": [], "tie_lines": []}

        # Convert sections data
        for northing, section in self.sections.items():
            export_data["sections"][str(northing)] = {
                "pdf_name": section["pdf_name"],
                "northing": northing,
                "num_units": len(section["units"]),
                "num_contacts": len(section["contacts"]),
            }

        # Export correlations
        for corr in self.correlations:
            export_data["correlations"].append(
                {
                    "north1": corr["north1"],
                    "north2": corr["north2"],
                    "unit1": corr["unit1"],
                    "unit2": corr["unit2"],
                    "formation": corr["formation"],
                    "score": corr["score"],
                }
            )

        # Export tie lines
        for tie in self.tie_lines:
            export_data["tie_lines"].append(
                {
                    "name": tie["name"],
                    "formation": tie["formation"],
                    "start": tie["start"],
                    "end": tie["end"],
                }
            )

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported correlations to {output_path}")

    def _get_unit_bounds(self, unit: Dict) -> Optional[Dict]:
        """Get bounding box of a unit in real-world coordinates."""
        if len(unit["vertices"]) < 4:
            return None

        eastings = [unit["vertices"][i] for i in range(0, len(unit["vertices"]), 2)]
        rls = [unit["vertices"][i + 1] for i in range(0, len(unit["vertices"]), 2)]

        if not eastings or not rls:
            return None

        return {
            "e_min": min(eastings),
            "e_max": max(eastings),
            "e_center": (min(eastings) + max(eastings)) / 2,
            "rl_min": min(rls),
            "rl_max": max(rls),
            "rl_center": (min(rls) + max(rls)) / 2,
            "width": max(eastings) - min(eastings),
            "height": max(rls) - min(rls),
        }

    def _calculate_overlap(self, range1: Tuple, range2: Tuple) -> float:
        """Calculate overlap ratio between two ranges."""
        min1, max1 = range1
        min2, max2 = range2

        overlap = min(max1, max2) - max(min1, min2)
        if overlap <= 0:
            return 0.0

        size1 = max1 - min1
        size2 = max2 - min2

        if size1 == 0 or size2 == 0:
            return 0.0

        return overlap / min(size1, size2)

    def _calculate_correlation_score(
        self, bounds1: Dict, bounds2: Dict, rl_overlap: float, easting_overlap: float
    ) -> float:
        """Calculate correlation score between two units."""
        # Elevation similarity (most important)
        rl_diff = abs(bounds1["rl_center"] - bounds2["rl_center"])
        rl_score = max(0, 1 - rl_diff / 100)  # Normalize to 0-1

        # Size similarity
        size_ratio = (
            min(
                bounds1["width"] / bounds2["width"], bounds2["width"] / bounds1["width"]
            )
            if bounds2["width"] > 0
            else 0
        )

        # Weighted score
        score = (
            rl_score * 0.5  # Elevation match is most important
            + easting_overlap * 0.3  # Lateral position overlap
            + size_ratio * 0.2  # Similar size
        )

        return score

    def _get_tie_points(self, unit: Dict, northing: float) -> List[Tuple]:
        """Get tie points for a unit (corners and center)."""
        bounds = self._get_unit_bounds(unit)
        if not bounds:
            return []

        # Generate tie points at key locations
        points = [
            # Corners
            (bounds["e_min"], northing, bounds["rl_min"]),
            (bounds["e_min"], northing, bounds["rl_max"]),
            (bounds["e_max"], northing, bounds["rl_min"]),
            (bounds["e_max"], northing, bounds["rl_max"]),
            # Center
            (bounds["e_center"], northing, bounds["rl_center"]),
        ]

        return points
