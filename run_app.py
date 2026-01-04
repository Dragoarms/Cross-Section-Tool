#!/usr/bin/env python3
"""
Entry point for Geological Cross-Section Tool Suite.
This script is used by PyInstaller to create a standalone executable.
"""

import sys
import os

# Ensure the package can be found
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = os.path.dirname(sys.executable)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

# Add the application path to sys.path
if application_path not in sys.path:
    sys.path.insert(0, application_path)

def main():
    """Launch the main GUI application."""
    import tkinter as tk
    from geotools.main_gui import GeologicalCrossSectionGUI

    root = tk.Tk()
    app = GeologicalCrossSectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
