# Geological Cross-Section Tool Suite

A comprehensive Python toolkit for extracting, georeferencing, and analyzing geological features from PDF cross-sections. This tool automates the digitization of geological cross-sections, enabling 3D visualization, stratigraphic correlation, and export to GIS formats.

## 🌟 Features

### Core Capabilities
- **PDF Feature Extraction**: Automatically extract colored polygons and polylines representing geological units
- **Intelligent Georeferencing**: Detect coordinate systems (Easting, Northing, RL) from PDF labels
- **Formation Identification**: Identify geological formations by color or annotation metadata
- **Contact Detection**: Automatically detect and validate contacts between geological units
- **Stratigraphic Management**: Define and validate stratigraphic sequences with unconformities and intrusives

### Advanced Features
- **Batch Processing**: Process multiple PDFs with multiple pages simultaneously
- **3D Visualization**: Interactive 3D view of geological units across sections
- **Section Correlation**: Correlate units between multiple cross-sections with tie line generation
- **Multiple Export Formats**: 
  - GeoTIFF with full georeferencing (compatible with GIS software)
  - DXF for CAD applications (AutoCAD, Leapfrog Geo)
  - CSV for data analysis
- **Smart Assignment**: Visual stratigraphic assignment interface with write-back to source PDFs

## 📋 Requirements

- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- RAM: 4GB minimum, 8GB recommended for large datasets

## 🚀 Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/geological-cross-section-tools.git
cd geological-cross-section-tools

Create a virtual environment (recommended):

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt

Install in development mode:

bashpip install -e .
Using pip (when published)
bashpip install geological-cross-section-tools
🎯 Quick Start
1. Launch the GUI
bashpython -m geotools.main_gui
2. Basic Workflow
pythonfrom geotools import GeoReferencer, FeatureExtractor, BatchProcessor

# Initialize modules
georef = GeoReferencer()
extractor = FeatureExtractor()

# Process a single PDF
import fitz
doc = fitz.open("cross_section.pdf")
page = doc[0]

# Detect coordinates
coord_system = georef.detect_coordinates(page)
print(f"Detected northing: {coord_system['northing']}")

# Extract geological units
annotations = extractor.extract_annotations(page)
extractor.number_geological_units(annotations)
print(f"Found {len(extractor.geological_units)} units")
📖 Detailed Usage
GUI Mode
The GUI provides an intuitive interface for all operations:

Open PDF(s): File → Open PDF or Open Folder for batch mode
Process: Click "Process All" to extract features from all pages
Review: Check detected coordinates and extracted units
Assign Stratigraphy: Use "3D View" for visual assignment
Export: Choose format (GeoTIFF, DXF, CSV) and export

Batch Processing
Process multiple PDFs programmatically:
pythonfrom geotools import BatchProcessor
from pathlib import Path

processor = BatchProcessor()

# Set up paths
pdf_files = list(Path("input_folder").glob("*.pdf"))
output_dir = Path("output_folder")

# Handle missing northings
missing_northings = processor.scan_for_missing_northings(pdf_files)
if missing_northings:
    # Provide northings manually
    processor.northing_overrides = {
        Path("section1.pdf"): 122800,
        Path("section2.pdf"): 123000
    }

# Process to DXF
results = processor.process_batch_to_dxf(
    pdf_files, 
    output_dir,
    export_units=True,
    export_contacts=True
)
Coordinate System Detection
The tool automatically detects coordinate labels in PDFs:
python# Coordinate detection with fallback options
coord_system = georef.detect_coordinates(page, pdf_path)

if not coord_system:
    # Try filename parsing
    # Supports formats like "KM_122800_section.pdf"
    pass

if not coord_system:
    # Manual entry
    georef.coord_system = {
        'northing': 122800,
        'easting_labels': [...],
        'rl_labels': [...]
    }
Stratigraphic Column Management
Define your stratigraphic sequence:
pythonfrom geotools import StratColumn

strat = StratColumn()

# Add stratigraphic units (youngest to oldest)
strat.add_strat_unit("SCH2", "Schist 2", color=(0.2, 0.7, 0.2))
strat.add_strat_unit("BIF2", "Banded Iron Fm 2", color=(0.2, 0.2, 0.8))
strat.add_strat_unit("UNCONFORMITY", "Major Unconformity", 
                     color=(0.8, 0.8, 0.8), is_unconformity=True)
strat.add_strat_unit("PHY", "Phyllite", color=(0.6, 0.4, 0.2))

# Validate contacts
is_valid, contact_type, error = strat.validate_contact("BIF2", "SCH2")
Section Correlation
Correlate units between multiple sections:
pythonfrom geotools import SectionCorrelator

correlator = SectionCorrelator()

# Add sections
correlator.add_section(122800, units_dict_1, contacts_1, "section1.pdf")
correlator.add_section(123000, units_dict_2, contacts_2, "section2.pdf")

# Find correlations
correlations = correlator.find_correlations(
    max_rl_difference=50.0,
    min_overlap_ratio=0.3
)

# Generate 3D tie lines
tie_lines = correlator.generate_tie_lines()

# Export
correlator.export_tie_lines_dxf(Path("tie_lines.dxf"))
🏗️ Architecture
┌─────────────────┐
│    main_gui     │  ← User Interface
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┬──────────┐
    │         │          │          │          │
┌───▼──┐ ┌───▼───┐ ┌────▼────┐ ┌──▼──┐ ┌────▼─────┐
│georef│ │feature│ │  batch  │ │strat│ │  section │
│      │ │extract│ │processor│ │column│ │correlation│
└──────┘ └───────┘ └─────────┘ └─────┘ └──────────┘
    │         │           │         │          │
    └─────────┴───────────┴─────────┴──────────┘
                         │
                  ┌──────▼──────┐
                  │ debug_utils │
                  └─────────────┘
Module Responsibilities

main_gui.py: GUI application and user interaction
georeferencing.py: Coordinate system detection and transformation
feature_extraction.py: Extract geological features from PDFs
batch_processor.py: Handle multiple PDFs efficiently
strat_column.py: Stratigraphic relationships and validation
section_correlation.py: Cross-section correlation and tie lines
section_viewer.py: 3D visualization and assignment interface
utils/debug_utils.py: Logging and debugging infrastructure

🎨 Color Conventions
The tool recognizes geological formations by color:
ColorFormationRGB RangeBlueBIF (Banded Iron Formation)B > R and B > GGreenSCH (Schist) / AMP (Amphibolite)G > R and G > BBrownPHY (Phyllite)R≈G, B < 0.3GrayFaults/StructuresR≈G≈B
📊 Output Formats
GeoTIFF

Full georeferencing with corner points or GCPs
Compatible with QGIS, ArcGIS, Leapfrog Geo
Includes metadata tags for section orientation

DXF

3D polylines with proper coordinates
Layer organization by formation
X=Easting, Y=Northing, Z=RL/Elevation
Compatible with AutoCAD, Leapfrog, MineSight

CSV

Structured data export
Columns: Type, Name, Easting, Northing, RL, VertexNumber
Suitable for further analysis in Excel or Python

🧪 Testing
Run the test suite:
bash# All tests
pytest tests/

# Specific module
pytest tests/test_georeferencing.py

# With coverage
pytest --cov=geotools tests/

# Debug mode
python test_runner.py --verbosity DEBUG
🐛 Debugging
The toolkit includes comprehensive debugging features:
pythonfrom geotools.utils import debug_trace, debug_value

@debug_trace(log_args=True, measure_time=True)
def process_section(section_data):
    debug_value(logger, "section_northing", section_data['northing'])
    # Processing code...
Enable debug logging:
pythonfrom geotools.utils import DebugLogger

debug_logger = DebugLogger()
debug_logger.set_global_level("DEBUG")