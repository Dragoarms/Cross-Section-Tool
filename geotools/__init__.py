#!/usr/bin/env python3
"""
GeoTools - Geological Cross-Section Tool Suite

A comprehensive toolkit for processing geological cross-sections from PDFs,
including coordinate detection, feature extraction, contact identification,
and DXF export for use with Leapfrog and other geological modeling software.
"""

from .georeferencing import GeoReferencer
from .feature_extraction import FeatureExtractor
from .strat_column import StratColumn
from .batch_processor import BatchProcessor
from .section_correlation import SectionCorrelator

# GUI components are optional (require tkinter)
try:
    from .pdf_calibration import ExtractionFilter, PDFCalibrationDialog, open_calibration_dialog
    _GUI_AVAILABLE = True
except ImportError:
    ExtractionFilter = None
    PDFCalibrationDialog = None
    open_calibration_dialog = None
    _GUI_AVAILABLE = False

__version__ = "1.1.0"
__author__ = "GeoTools Team"

__all__ = [
    "GeoReferencer",
    "FeatureExtractor",
    "StratColumn",
    "BatchProcessor",
    "SectionCorrelator",
    "ExtractionFilter",
    "PDFCalibrationDialog",
    "open_calibration_dialog",
]
