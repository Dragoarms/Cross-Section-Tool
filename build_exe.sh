#!/bin/bash
# ============================================
# Build script for Cross-Section Tool (Linux/Mac)
# ============================================
#
# Prerequisites:
#   1. Python 3.8+ with tkinter
#   2. Run from project root directory
#
# Usage:
#   chmod +x build_exe.sh
#   ./build_exe.sh
#
# ============================================

echo "============================================"
echo "  Cross-Section Tool - Executable Builder"
echo "============================================"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ ! -d "build_venv" ]; then
    python3 -m venv build_venv
fi

echo "[2/4] Installing dependencies..."
source build_venv/bin/activate

pip install --upgrade pip
pip install pyinstaller
pip install PyMuPDF numpy matplotlib shapely scipy rasterio pillow

echo "[3/4] Building executable..."
pyinstaller CrossSectionTool.spec --clean --noconfirm

echo "[4/4] Build complete!"
echo ""

if [ -f "dist/CrossSectionTool" ]; then
    echo "SUCCESS! Executable created at:"
    echo "  dist/CrossSectionTool"
    echo ""
    echo "Make it executable with: chmod +x dist/CrossSectionTool"
else
    echo "WARNING: Executable may not have been created."
    echo "Check the build output for errors."
fi
