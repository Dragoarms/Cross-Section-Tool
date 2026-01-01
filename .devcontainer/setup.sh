#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    python3-tk \
    libgdal-dev \
    gdal-bin \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    x11-utils

echo "=== Creating Python virtual environment ==="
python -m venv ~/.venv
source ~/.venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing package in development mode ==="
pip install -e .

echo "=== Setup complete ==="
echo ""
echo "To run the GUI application:"
echo "  1. Open the Desktop via port 6080 (noVNC) in your browser"
echo "  2. Run: source ~/.venv/bin/activate && python -m geotools"
echo ""
echo "To run tests:"
echo "  pytest tests/"
