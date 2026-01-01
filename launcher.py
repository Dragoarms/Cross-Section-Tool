# launcher.py
"""
Cross-platform launcher for Geological Cross-Section Tool Suite
"""

import sys
import subprocess
import os
from pathlib import Path
import venv


def setup_venv():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path(".venv")

    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created!")

    return venv_path


def get_python_executable():
    """Get the path to the Python executable in the venv."""
    venv_path = Path(".venv")

    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    return python_exe


def install_package(python_exe):
    """Install the package in editable mode."""
    try:
        # Check if package is installed
        result = subprocess.run(
            [str(python_exe), "-c", "import geotools"], capture_output=True, text=True
        )

        if result.returncode != 0:
            print("Installing package...")
            subprocess.run([str(python_exe), "-m", "pip", "install", "-e", "."])
            print("Package installed!")
    except Exception as e:
        print(f"Error checking/installing package: {e}")


def main():
    """Main launcher function."""
    print("=" * 50)
    print("  Geological Cross-Section Tool Suite Launcher")
    print("=" * 50)
    print()

    # Setup venv
    venv_path = setup_venv()
    python_exe = get_python_executable()

    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        input("Press Enter to exit...")
        return 1

    # Install/update package
    install_package(python_exe)

    # Launch GUI
    print("\nLaunching GUI...")
    print("-" * 50)

    try:
        # Auto-fix Tcl/Tk path for Tkinter inside venv (PEP-8)
        tcl_root = Path(sys.base_prefix) / "tcl"
        tcl_dir = next((p for p in tcl_root.glob("tcl8.*") if p.is_dir()), None)
        tk_dir = next((p for p in tcl_root.glob("tk8.*") if p.is_dir()), None)

        env = os.environ.copy()
        if tcl_dir and tk_dir:
            env["TCL_LIBRARY"] = str(tcl_dir)
            env["TK_LIBRARY"] = str(tk_dir)
            print(f"Detected Tcl/Tk → {env['TCL_LIBRARY']} | {env['TK_LIBRARY']}")
        else:
            print("Warning: Could not auto-detect Tcl/Tk; proceeding without overrides.")

        result = subprocess.run([str(python_exe), "-m", "geotools.main_gui"], check=False, env=env)

        if result.returncode != 0:
            print("\n" + "=" * 50)
            print("An error occurred while running the GUI.")
            input("Press Enter to exit...")
            return result.returncode

    except KeyboardInterrupt:
        print("\n\nGUI closed by user.")
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        input("Press Enter to exit...")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
