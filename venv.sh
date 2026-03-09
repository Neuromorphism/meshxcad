#!/usr/bin/env bash
# venv.sh — Create a Python virtual environment for MeshXCAD
# Works on Linux and macOS.
#
# Usage:
#   chmod +x venv.sh
#   ./venv.sh
#   source .venv/bin/activate

set -euo pipefail

VENV_DIR=".venv"
PYTHON=""

# ---------- locate python ----------
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    echo "Error: Python 3.8+ required (found $PY_VERSION)."
    exit 1
fi

echo "Using $PYTHON ($PY_VERSION)"

# ---------- create venv ----------
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR — removing and recreating."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment in $VENV_DIR ..."
"$PYTHON" -m venv "$VENV_DIR"

# ---------- activate ----------
source "$VENV_DIR/bin/activate"

# ---------- upgrade pip ----------
echo "Upgrading pip ..."
pip install --upgrade pip

# ---------- core dependencies ----------
echo "Installing core dependencies ..."
pip install numpy scipy

# ---------- visualization ----------
echo "Installing visualization dependencies ..."
pip install matplotlib

# ---------- CAD kernel (OCP / cadquery) ----------
echo "Installing OpenCASCADE Python bindings ..."
# cadquery bundles OCP and is the easiest cross-platform way to get it
pip install cadquery

# ---------- FreeCAD ----------
# FreeCAD cannot be installed via pip in most cases.
# On Linux it is available through system packages; on macOS via Homebrew.
echo ""
echo "--- FreeCAD (optional) ---"
if "$PYTHON" -c "import FreeCAD" 2>/dev/null; then
    echo "FreeCAD is already importable — skipping."
else
    OS="$(uname -s)"
    echo "FreeCAD is NOT installed in this environment."
    echo "Some features (parametric CAD generation, .FCStd I/O) require FreeCAD."
    echo ""
    if [ "$OS" = "Linux" ]; then
        echo "Install FreeCAD on Linux:"
        echo "  sudo apt install freecad          # Debian / Ubuntu"
        echo "  sudo dnf install freecad           # Fedora"
        echo "  conda install -c conda-forge freecad   # via conda"
    elif [ "$OS" = "Darwin" ]; then
        echo "Install FreeCAD on macOS:"
        echo "  brew install --cask freecad        # via Homebrew"
        echo "  conda install -c conda-forge freecad   # via conda"
    fi
    echo ""
    echo "After installing, you may need to add FreeCAD's lib path to PYTHONPATH."
    echo "Example (adjust for your install):"
    if [ "$OS" = "Linux" ]; then
        echo '  export PYTHONPATH="/usr/lib/freecad-python3/lib:$PYTHONPATH"'
    elif [ "$OS" = "Darwin" ]; then
        echo '  export PYTHONPATH="/Applications/FreeCAD.app/Contents/Resources/lib:$PYTHONPATH"'
    fi
fi

# ---------- testing ----------
echo ""
echo "Installing test dependencies ..."
pip install pytest

# ---------- install meshxcad in editable mode ----------
echo "Installing meshxcad in editable mode ..."
pip install -e .

# ---------- summary ----------
echo ""
echo "============================================"
echo " MeshXCAD virtual environment ready!"
echo "============================================"
echo ""
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run tests:"
echo "  pytest tests/"
echo ""
echo "CLI usage:"
echo "  python -m meshxcad.cli transfer --plain model.step --detail scan.stl --output result.stl"
echo ""
