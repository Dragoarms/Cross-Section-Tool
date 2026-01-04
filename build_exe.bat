@echo off
REM ============================================
REM Build script for Cross-Section Tool Windows EXE
REM ============================================
REM
REM Prerequisites:
REM   1. Python 3.8+ installed (with tkinter)
REM   2. Run this from the project root directory
REM
REM Usage:
REM   build_exe.bat
REM
REM ============================================

echo ============================================
echo   Cross-Section Tool - Windows EXE Builder
echo ============================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist "build_venv" (
    python -m venv build_venv
)

echo [2/4] Activating environment and installing dependencies...
call build_venv\Scripts\activate.bat

pip install --upgrade pip
pip install pyinstaller
pip install PyMuPDF numpy matplotlib shapely scipy rasterio pillow

echo [3/4] Building executable...
pyinstaller CrossSectionTool.spec --clean --noconfirm

echo [4/4] Build complete!
echo.

if exist "dist\CrossSectionTool.exe" (
    echo SUCCESS! Executable created at:
    echo   dist\CrossSectionTool.exe
    echo.
    echo You can distribute this single file to users.
) else (
    echo WARNING: Executable may not have been created.
    echo Check the build output for errors.
)

echo.
pause
