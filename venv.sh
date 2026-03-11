#!/usr/bin/env bash
# venv.sh — Create a Python virtual environment for MeshXCAD
# Works on Linux and macOS.
#
# This script detects FreeCAD's required Python version and creates the venv
# with a matching interpreter so that all features work out of the box.
#
# Usage:
#   chmod +x venv.sh
#   ./venv.sh
#   source .venv/bin/activate

set -euo pipefail

VENV_DIR=".venv"
OS="$(uname -s)"

# ============================================================
# 1. Find FreeCAD and determine its required Python version
# ============================================================
echo "--- Detecting FreeCAD ---"

FREECAD_LIB=""
FREECAD_PYVER=""

# Collect candidate FreeCAD lib directories
FC_LIB_CANDIDATES=()
if [ "$OS" = "Darwin" ]; then
    FC_LIB_CANDIDATES+=(
        "/Applications/FreeCAD.app/Contents/Resources/lib"
        "/Applications/FreeCAD.app/Contents/lib"
        "$HOME/Applications/FreeCAD.app/Contents/Resources/lib"
        "$HOME/Applications/FreeCAD.app/Contents/lib"
    )
    # Homebrew may also symlink into a versioned prefix
    if command -v brew &>/dev/null; then
        brew_prefix="$(brew --prefix 2>/dev/null || true)"
        if [ -n "$brew_prefix" ]; then
            FC_LIB_CANDIDATES+=("$brew_prefix/lib/freecad/lib")
        fi
    fi
else
    FC_LIB_CANDIDATES+=(
        "/usr/lib/freecad-python3/lib"
        "/usr/lib/freecad/lib"
        "/usr/lib64/freecad/lib"
        "/usr/share/freecad/lib"
        "/snap/freecad/current/usr/lib/freecad-python3/lib"
        "/snap/freecad/current/usr/lib/freecad/lib"
    )
fi

for d in "${FC_LIB_CANDIDATES[@]}"; do
    if [ -d "$d" ]; then
        FREECAD_LIB="$d"
        break
    fi
done

# If we didn't find a lib dir, try to find it via the freecad binary
if [ -z "$FREECAD_LIB" ] && command -v freecad &>/dev/null; then
    fc_bin="$(command -v freecad)"
    # Follow symlinks to find the real location
    if command -v readlink &>/dev/null; then
        fc_real="$(readlink -f "$fc_bin" 2>/dev/null || echo "$fc_bin")"
        fc_dir="$(dirname "$fc_real")"
        for suffix in "../lib" "../lib/freecad/lib" "../lib/freecad-python3/lib"; do
            candidate="$fc_dir/$suffix"
            if [ -d "$candidate" ]; then
                FREECAD_LIB="$(cd "$candidate" && pwd)"
                break
            fi
        done
    fi
fi

# Probe FreeCAD's compiled Python version from its shared library
detect_freecad_pyver() {
    local libdir="$1"
    local pyver=""

    # Method 1: check for FreeCAD.so and use ldd/otool to find linked libpython
    local fc_so=""
    for f in "$libdir/FreeCAD.so" "$libdir/FreeCAD.dylib" "$libdir/_FreeCAD.so"; do
        if [ -f "$f" ]; then
            fc_so="$f"
            break
        fi
    done

    if [ -n "$fc_so" ]; then
        if [ "$OS" = "Darwin" ]; then
            pyver=$(otool -L "$fc_so" 2>/dev/null | grep -oE 'libpython3\.[0-9]+' | head -1 | sed 's/libpython//')
        else
            pyver=$(ldd "$fc_so" 2>/dev/null | grep -oE 'libpython3\.[0-9]+' | head -1 | sed 's/libpython//')
        fi
    fi

    # Method 2: look for python3.X directories or .so files in the lib tree
    if [ -z "$pyver" ]; then
        pyver=$(find "$libdir" -maxdepth 2 -name "python3.*" -type d 2>/dev/null \
            | grep -oE 'python3\.[0-9]+' | head -1 | sed 's/python//')
    fi

    # Method 3: check parent directories for python version hints
    if [ -z "$pyver" ]; then
        pyver=$(echo "$libdir" | grep -oE 'python3\.[0-9]+' | head -1 | sed 's/python//')
    fi

    echo "$pyver"
}

if [ -n "$FREECAD_LIB" ]; then
    echo "FreeCAD library found: $FREECAD_LIB"
    FREECAD_PYVER=$(detect_freecad_pyver "$FREECAD_LIB")
fi

# If we still don't have a version, try running freecad to ask it
if [ -z "$FREECAD_PYVER" ] && command -v freecad &>/dev/null; then
    FREECAD_PYVER=$(freecad --console -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
fi

if [ -n "$FREECAD_PYVER" ]; then
    echo "FreeCAD requires Python $FREECAD_PYVER"
else
    if [ -n "$FREECAD_LIB" ]; then
        echo "WARNING: Found FreeCAD but could not determine its Python version."
        echo "         Defaulting to system Python. FreeCAD import may fail."
    else
        echo "FreeCAD not found. Will install with system Python."
        echo "FreeCAD features (.FCStd I/O, parametric CAD) will be unavailable."
    fi
fi

# ============================================================
# 2. Find or install the required Python version
# ============================================================
PYTHON=""
REQUIRED="$FREECAD_PYVER"

find_python_version() {
    # Try exact version binaries first
    local ver="$1"
    for candidate in "python${ver}" "python$(echo "$ver" | tr -d '.')"; do
        if command -v "$candidate" &>/dev/null; then
            local actual
            actual=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
            if [ "$actual" = "$ver" ]; then
                echo "$candidate"
                return
            fi
        fi
    done
    # Try pyenv shims
    if command -v pyenv &>/dev/null; then
        local installed
        installed=$(pyenv versions --bare 2>/dev/null | grep "^${ver}" | tail -1)
        if [ -n "$installed" ]; then
            local pyenv_root
            pyenv_root="$(pyenv root)"
            local bin="$pyenv_root/versions/$installed/bin/python3"
            if [ -x "$bin" ]; then
                echo "$bin"
                return
            fi
        fi
    fi
}

if [ -n "$REQUIRED" ]; then
    echo ""
    echo "--- Locating Python $REQUIRED ---"
    PYTHON=$(find_python_version "$REQUIRED")

    if [ -z "$PYTHON" ]; then
        echo "Python $REQUIRED not found. Attempting to install it..."
        echo ""

        if [ "$OS" = "Darwin" ]; then
            # macOS: try Homebrew first, then pyenv
            if command -v brew &>/dev/null; then
                BREW_PKG="python@${REQUIRED}"
                echo "Running: brew install $BREW_PKG"
                if brew install "$BREW_PKG" 2>&1; then
                    brew_prefix="$(brew --prefix "$BREW_PKG" 2>/dev/null || true)"
                    if [ -x "$brew_prefix/bin/python${REQUIRED}" ]; then
                        PYTHON="$brew_prefix/bin/python${REQUIRED}"
                    fi
                fi
            fi
            if [ -z "$PYTHON" ] && command -v pyenv &>/dev/null; then
                echo "Trying pyenv..."
                latest=$(pyenv install --list 2>/dev/null | tr -d ' ' | grep "^${REQUIRED}\." | grep -v '[a-zA-Z]' | tail -1)
                if [ -n "$latest" ]; then
                    echo "Running: pyenv install $latest"
                    pyenv install -s "$latest"
                    PYTHON=$(find_python_version "$REQUIRED")
                fi
            fi
        else
            # Linux: try deadsnakes PPA (Debian/Ubuntu), then dnf, then pyenv
            if command -v apt-get &>/dev/null; then
                echo "Adding deadsnakes PPA and installing Python $REQUIRED..."
                if command -v add-apt-repository &>/dev/null; then
                    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                fi
                sudo apt-get update -qq 2>/dev/null || true
                sudo apt-get install -y "python${REQUIRED}" "python${REQUIRED}-venv" "python${REQUIRED}-dev" 2>&1 || true
                PYTHON=$(find_python_version "$REQUIRED")
            elif command -v dnf &>/dev/null; then
                echo "Running: dnf install python${REQUIRED}"
                sudo dnf install -y "python${REQUIRED}" 2>&1 || true
                PYTHON=$(find_python_version "$REQUIRED")
            fi
            if [ -z "$PYTHON" ] && command -v pyenv &>/dev/null; then
                echo "Trying pyenv..."
                latest=$(pyenv install --list 2>/dev/null | tr -d ' ' | grep "^${REQUIRED}\." | grep -v '[a-zA-Z]' | tail -1)
                if [ -n "$latest" ]; then
                    echo "Running: pyenv install $latest"
                    pyenv install -s "$latest"
                    PYTHON=$(find_python_version "$REQUIRED")
                fi
            fi
        fi

        if [ -z "$PYTHON" ]; then
            echo ""
            echo "ERROR: Could not install Python $REQUIRED automatically."
            echo ""
            echo "Please install Python $REQUIRED manually:"
            if [ "$OS" = "Darwin" ]; then
                echo "  brew install python@${REQUIRED}"
                echo "  # or"
                echo "  pyenv install ${REQUIRED}"
            else
                echo "  # Debian/Ubuntu:"
                echo "  sudo apt install python${REQUIRED} python${REQUIRED}-venv"
                echo "  # Fedora:"
                echo "  sudo dnf install python${REQUIRED}"
                echo "  # Or use pyenv:"
                echo "  pyenv install ${REQUIRED}"
            fi
            echo ""
            echo "Then re-run this script."
            exit 1
        fi
    fi

    echo "Found: $PYTHON"
else
    # No FreeCAD version requirement — use system python
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "Error: No suitable Python 3 interpreter found."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using $PYTHON (Python $PY_VERSION)"

# Verify version match if we have a requirement
if [ -n "$REQUIRED" ] && [ "$PY_VERSION" != "$REQUIRED" ]; then
    echo "ERROR: Interpreter reports $PY_VERSION but FreeCAD needs $REQUIRED."
    exit 1
fi

# ============================================================
# 3. Create the virtual environment
# ============================================================
echo ""
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing $VENV_DIR ..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment in $VENV_DIR (Python $PY_VERSION) ..."
"$PYTHON" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

# ============================================================
# 4. Configure pip (SSL cert & mirror/trusted-host)
# ============================================================
PIP_EXTRA_ARGS=()

# SSL certificate — use SSL_CERT_FILE env var for corporate/custom CAs
if [ -n "${SSL_CERT_FILE:-}" ]; then
    if [ -f "$SSL_CERT_FILE" ]; then
        echo "Using SSL certificate: $SSL_CERT_FILE"
        PIP_EXTRA_ARGS+=("--cert" "$SSL_CERT_FILE")

        # Write pip.conf so the cert persists inside the venv
        cat > "$VENV_DIR/pip.conf" <<PIPEOF
[global]
cert = $SSL_CERT_FILE
PIPEOF

        # Export so pip's requests library and other tools respect the CA
        export PIP_CERT="$SSL_CERT_FILE"
        export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
        export CURL_CA_BUNDLE="$SSL_CERT_FILE"
        export SSL_CERT_FILE="$SSL_CERT_FILE"
    else
        echo "WARNING: SSL_CERT_FILE is set to '$SSL_CERT_FILE' but file does not exist."
        echo "         Continuing without custom certificate."
    fi
fi

# Detect pip mirror (Nexus, Artifactory, devpi, etc.) and add --trusted-host
# so that self-signed or corporate-CA HTTPS mirrors work out of the box.
detect_pip_mirror_host() {
    local index_url=""

    # 1. Check PIP_INDEX_URL environment variable
    if [ -n "${PIP_INDEX_URL:-}" ]; then
        index_url="$PIP_INDEX_URL"
    fi

    # 2. Check PIP_EXTRA_INDEX_URL environment variable
    local extra_index_url="${PIP_EXTRA_INDEX_URL:-}"

    # 3. Check pip config files for index-url / extra-index-url
    if [ -z "$index_url" ]; then
        for cfg in "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf" "/etc/pip.conf"; do
            if [ -f "$cfg" ]; then
                local url
                url=$(grep -E '^\s*index-url\s*=' "$cfg" 2>/dev/null | head -1 | sed 's/^[^=]*=\s*//' | tr -d '[:space:]')
                if [ -n "$url" ]; then
                    index_url="$url"
                fi
                if [ -z "$extra_index_url" ]; then
                    extra_index_url=$(grep -E '^\s*extra-index-url\s*=' "$cfg" 2>/dev/null | head -1 | sed 's/^[^=]*=\s*//' | tr -d '[:space:]')
                fi
                break
            fi
        done
    fi

    # Extract hostnames from URLs that are NOT pypi.org
    local hosts=()
    for url in $index_url $extra_index_url; do
        local host
        host=$(echo "$url" | sed -E 's|^https?://([^/:]+).*|\1|')
        if [ -n "$host" ] && [[ "$host" != *pypi.org ]] && [[ "$host" != *pythonhosted.org ]]; then
            hosts+=("$host")
        fi
    done

    # Deduplicate
    printf '%s\n' "${hosts[@]}" | sort -u
}

MIRROR_HOSTS=$(detect_pip_mirror_host)
if [ -n "$MIRROR_HOSTS" ]; then
    echo "Detected pip mirror host(s) — adding --trusted-host:"
    while IFS= read -r host; do
        echo "  $host"
        PIP_EXTRA_ARGS+=("--trusted-host" "$host")
    done <<< "$MIRROR_HOSTS"
fi

# ============================================================
# 5. Install dependencies
# ============================================================
echo ""
echo "Upgrading pip ..."
pip install --upgrade pip "${PIP_EXTRA_ARGS[@]}"

echo "Installing core dependencies ..."
pip install numpy scipy "${PIP_EXTRA_ARGS[@]}"

echo "Installing visualization dependencies ..."
pip install matplotlib "${PIP_EXTRA_ARGS[@]}"

echo "Installing OpenCASCADE Python bindings ..."
pip install cadquery "${PIP_EXTRA_ARGS[@]}"

echo "Installing test dependencies ..."
pip install pytest "${PIP_EXTRA_ARGS[@]}"

echo "Installing meshxcad in editable mode ..."
pip install -e . "${PIP_EXTRA_ARGS[@]}"

# ============================================================
# 6. Configure FreeCAD PYTHONPATH
# ============================================================
if [ -n "$FREECAD_LIB" ]; then
    # Write an activation hook so FreeCAD is importable automatically
    SITECUSTOMIZE="$VENV_DIR/lib/python${PY_VERSION}/site-packages/freecad_path.pth"
    echo "$FREECAD_LIB" > "$SITECUSTOMIZE"
    echo ""
    echo "Added $FREECAD_LIB to venv site-packages (.pth file)"

    # Verify FreeCAD actually imports
    echo "Verifying FreeCAD import ..."
    if python -c "import FreeCAD; print(f'  FreeCAD {FreeCAD.Version()[0]}.{FreeCAD.Version()[1]} OK')" 2>/dev/null; then
        echo "  FreeCAD is working."
    else
        echo "  WARNING: FreeCAD import failed. The library may have additional"
        echo "  dependencies. Check: python -c 'import FreeCAD'"
    fi
fi

# ============================================================
# 7. Summary
# ============================================================
echo ""
echo "============================================"
echo " MeshXCAD virtual environment ready!"
echo " Python $PY_VERSION | $([ -n "$FREECAD_LIB" ] && echo "FreeCAD: yes" || echo "FreeCAD: no")"
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
