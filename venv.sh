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
# FreeCAD cannot be installed via pip. Its Python modules are compiled against a
# specific Python version (often 3.11), so "import FreeCAD" only works when the
# interpreter version matches. We detect FreeCAD by looking for the binary and
# its lib directory instead.
echo ""
echo "--- FreeCAD (optional) ---"
OS="$(uname -s)"
FREECAD_FOUND=false
FREECAD_LIB=""

# Check common install locations
if [ "$OS" = "Darwin" ]; then
    # macOS: Homebrew cask or standalone .app
    for d in \
        "/Applications/FreeCAD.app/Contents/Resources/lib" \
        "/Applications/FreeCAD.app/Contents/lib" \
        "$HOME/Applications/FreeCAD.app/Contents/Resources/lib" \
        "$HOME/Applications/FreeCAD.app/Contents/lib"; do
        if [ -d "$d" ]; then
            FREECAD_FOUND=true
            FREECAD_LIB="$d"
            break
        fi
    done
    # Also check if the binary is on PATH
    if ! $FREECAD_FOUND && command -v freecad &>/dev/null; then
        FREECAD_FOUND=true
    fi
else
    # Linux: system package or Snap/Flatpak
    for d in \
        "/usr/lib/freecad-python3/lib" \
        "/usr/lib/freecad/lib" \
        "/usr/lib64/freecad/lib" \
        "/usr/share/freecad/lib" \
        "/snap/freecad/current/usr/lib/freecad-python3/lib" \
        "/snap/freecad/current/usr/lib/freecad/lib"; do
        if [ -d "$d" ]; then
            FREECAD_FOUND=true
            FREECAD_LIB="$d"
            break
        fi
    done
    if ! $FREECAD_FOUND && command -v freecad &>/dev/null; then
        FREECAD_FOUND=true
    fi
fi

# Also try the import as a last resort — it works when the venv Python version
# happens to match FreeCAD's compiled version.
if ! $FREECAD_FOUND; then
    if python -c "import FreeCAD" 2>/dev/null; then
        FREECAD_FOUND=true
    fi
fi

if $FREECAD_FOUND; then
    echo "FreeCAD detected."
    if [ -n "$FREECAD_LIB" ]; then
        echo "  Library path: $FREECAD_LIB"
        echo ""
        echo "  To make FreeCAD importable from this venv, add to your shell profile"
        echo "  or run before activating:"
        echo "    export PYTHONPATH=\"$FREECAD_LIB:\$PYTHONPATH\""
        echo ""
        echo "  NOTE: FreeCAD's Python modules are compiled against a specific Python"
        echo "  version (check with: ls $FREECAD_LIB/FreeCAD.so or .pyd)."
        echo "  If your venv Python ($PY_VERSION) does not match, you will get import"
        echo "  errors. In that case, install a matching Python version or use conda:"
        echo "    conda install -c conda-forge freecad python=$PY_VERSION"
    fi
else
    echo "FreeCAD is NOT installed."
    echo "Some features (parametric CAD generation, .FCStd I/O) require FreeCAD."
    echo ""
    if [ "$OS" = "Linux" ]; then
        echo "Install FreeCAD on Linux:"
        echo "  sudo apt install freecad          # Debian / Ubuntu"
        echo "  sudo dnf install freecad           # Fedora"
        echo "  conda install -c conda-forge freecad   # via conda"
        echo ""
        echo "Then add its lib path to PYTHONPATH (adjust for your install):"
        echo '  export PYTHONPATH="/usr/lib/freecad-python3/lib:$PYTHONPATH"'
    elif [ "$OS" = "Darwin" ]; then
        echo "Install FreeCAD on macOS:"
        echo "  brew install --cask freecad        # via Homebrew"
        echo "  conda install -c conda-forge freecad   # via conda"
        echo ""
        echo "Then add its lib path to PYTHONPATH (adjust for your install):"
        echo '  export PYTHONPATH="/Applications/FreeCAD.app/Contents/Resources/lib:$PYTHONPATH"'
    fi
    echo ""
    echo "NOTE: FreeCAD's Python modules are compiled against a specific Python"
    echo "version. Your venv uses Python $PY_VERSION. If FreeCAD was built against"
    echo "a different version, 'import FreeCAD' will fail even with the correct"
    echo "PYTHONPATH. Use conda to get a matching set:"
    echo "  conda install -c conda-forge freecad python=$PY_VERSION"
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
