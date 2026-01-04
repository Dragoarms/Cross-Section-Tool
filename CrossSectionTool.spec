# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Geological Cross-Section Tool Suite.

To build on Windows:
    1. Install Python 3.8+ with tkinter
    2. pip install pyinstaller PyMuPDF numpy matplotlib shapely scipy rasterio pillow
    3. pyinstaller CrossSectionTool.spec --clean

To build on Linux/Mac:
    1. Ensure tkinter is installed (sudo apt install python3-tk on Ubuntu)
    2. Run the same commands as above

This creates a standalone executable that includes all dependencies.
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

# Determine platform
is_windows = sys.platform == 'win32'

a = Analysis(
    ['run_app.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include config files
        ('configs.json', '.'),
        # Include the geotools package
        ('geotools', 'geotools'),
    ],
    hiddenimports=[
        # Matplotlib backends
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        'matplotlib.figure',
        'matplotlib.patches',
        'matplotlib.path',
        'mpl_toolkits.mplot3d',
        # Core dependencies
        'numpy',
        'numpy.lib.format',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
        'scipy',
        'scipy.interpolate',
        'scipy.spatial',
        'fitz',  # PyMuPDF
        'PIL',
        'PIL.Image',
        # Rasterio and GDAL
        'rasterio',
        'rasterio.transform',
        'rasterio.control',
        'affine',
        # Geotools modules
        'geotools',
        'geotools.main_gui',
        'geotools.georeferencing',
        'geotools.feature_extraction',
        'geotools.strat_column_v2',
        'geotools.strat_column',
        'geotools.auto_labeler',
        'geotools.batch_processor',
        'geotools.section_correlation',
        'geotools.pdf_calibration',
        'geotools.contact_extraction',
        'geotools.tie_line_editor',
        'geotools.pdf_annotation_writer',
        'geotools.model_boundary',
        'geotools.section_viewer',
        'geotools.contact_postprocess',
        'geotools.utils',
        'geotools.utils.debug_utils',
        # JSON and other stdlib
        'json',
        'logging',
        'pathlib',
        'collections',
        'dataclasses',
        're',
        'typing',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'pytest',
        'pytest_cov',
        'black',
        'flake8',
        'mypy',
        'IPython',
        'notebook',
        'sphinx',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CrossSectionTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add 'icon.ico' path if you have one
)
