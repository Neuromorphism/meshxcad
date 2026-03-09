# venv.ps1 — Create a Python virtual environment for MeshXCAD on Windows
#
# Usage (from PowerShell):
#   .\venv.ps1
#   .\.venv\Scripts\Activate.ps1
#
# If scripts are blocked, run first:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"

# ---------- locate python ----------
$Python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3") {
            $Python = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $Python) {
    Write-Host "Error: Python 3 not found. Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$PyVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$PyMajor = & $Python -c "import sys; print(sys.version_info.major)"
$PyMinor = & $Python -c "import sys; print(sys.version_info.minor)"

if ([int]$PyMajor -lt 3 -or ([int]$PyMajor -eq 3 -and [int]$PyMinor -lt 8)) {
    Write-Host "Error: Python 3.8+ required (found $PyVersion)." -ForegroundColor Red
    exit 1
}

Write-Host "Using $Python ($PyVersion)" -ForegroundColor Cyan

# ---------- create venv ----------
if (Test-Path $VenvDir) {
    Write-Host "Virtual environment already exists at $VenvDir — removing and recreating."
    Remove-Item -Recurse -Force $VenvDir
}

Write-Host "Creating virtual environment in $VenvDir ..."
& $Python -m venv $VenvDir

# ---------- activate ----------
& "$VenvDir\Scripts\Activate.ps1"

# ---------- upgrade pip ----------
Write-Host "Upgrading pip ..."
pip install --upgrade pip

# ---------- core dependencies ----------
Write-Host "Installing core dependencies ..."
pip install numpy scipy

# ---------- visualization ----------
Write-Host "Installing visualization dependencies ..."
pip install matplotlib

# ---------- CAD kernel (OCP / cadquery) ----------
Write-Host "Installing OpenCASCADE Python bindings ..."
# cadquery bundles OCP and is the easiest cross-platform way to get it
pip install cadquery

# ---------- FreeCAD ----------
Write-Host ""
Write-Host "--- FreeCAD (optional) ---" -ForegroundColor Yellow
$FreecadAvailable = $false
try {
    & $Python -c "import FreeCAD" 2>$null
    $FreecadAvailable = $true
} catch {}

if ($FreecadAvailable) {
    Write-Host "FreeCAD is already importable — skipping."
} else {
    Write-Host "FreeCAD is NOT installed in this environment."
    Write-Host "Some features (parametric CAD generation, .FCStd I/O) require FreeCAD."
    Write-Host ""
    Write-Host "Install FreeCAD on Windows:"
    Write-Host "  1. Download from https://www.freecad.org/downloads.php"
    Write-Host "  2. Install, then add FreeCAD's lib and bin to PYTHONPATH:"
    Write-Host '     $env:PYTHONPATH = "C:\Program Files\FreeCAD 0.21\bin;C:\Program Files\FreeCAD 0.21\lib;$env:PYTHONPATH"'
    Write-Host "  Adjust the path to match your FreeCAD install location."
    Write-Host ""
    Write-Host "Alternatively, install via conda:"
    Write-Host "  conda install -c conda-forge freecad"
}

# ---------- testing ----------
Write-Host ""
Write-Host "Installing test dependencies ..."
pip install pytest

# ---------- install meshxcad in editable mode ----------
Write-Host "Installing meshxcad in editable mode ..."
pip install -e .

# ---------- summary ----------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " MeshXCAD virtual environment ready!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run tests:"
Write-Host "  pytest tests\"
Write-Host ""
Write-Host "CLI usage:"
Write-Host "  python -m meshxcad.cli transfer --plain model.step --detail scan.stl --output result.stl"
Write-Host ""
