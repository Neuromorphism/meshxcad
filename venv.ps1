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
# FreeCAD cannot be installed via pip. Its Python modules are compiled against a
# specific Python version (often 3.11), so "import FreeCAD" only works when the
# interpreter version matches. We detect FreeCAD by looking for common install
# directories instead.
Write-Host ""
Write-Host "--- FreeCAD (optional) ---" -ForegroundColor Yellow
$FreecadFound = $false
$FreecadLib = ""

# Check common Windows install locations
$SearchDirs = @(
    "${env:ProgramFiles}\FreeCAD *\bin",
    "${env:ProgramFiles}\FreeCAD *\lib",
    "${env:LOCALAPPDATA}\FreeCAD *\bin",
    "${env:LOCALAPPDATA}\FreeCAD *\lib",
    "${env:ProgramFiles(x86)}\FreeCAD *\bin"
)
foreach ($pattern in $SearchDirs) {
    $matches = Resolve-Path -Path $pattern -ErrorAction SilentlyContinue
    if ($matches) {
        $FreecadFound = $true
        # Prefer the bin directory (contains FreeCAD.pyd on Windows)
        foreach ($m in $matches) {
            if ($m.Path -match "\\bin$") {
                $FreecadLib = $m.Path
                break
            }
        }
        if (-not $FreecadLib) {
            $FreecadLib = $matches[0].Path
        }
        break
    }
}

# Also check if freecad is on PATH
if (-not $FreecadFound) {
    try {
        $fc = Get-Command freecad -ErrorAction SilentlyContinue
        if ($fc) { $FreecadFound = $true }
    } catch {}
}

# Last resort: try import (works only if venv Python version matches FreeCAD's)
if (-not $FreecadFound) {
    try {
        & python -c "import FreeCAD" 2>$null
        if ($LASTEXITCODE -eq 0) { $FreecadFound = $true }
    } catch {}
}

if ($FreecadFound) {
    Write-Host "FreeCAD detected."
    if ($FreecadLib) {
        Write-Host "  Library path: $FreecadLib"
        Write-Host ""
        Write-Host "  To make FreeCAD importable from this venv, set PYTHONPATH before activating:"
        Write-Host "    `$env:PYTHONPATH = `"$FreecadLib;`$env:PYTHONPATH`""
        Write-Host ""
        Write-Host "  NOTE: FreeCAD's Python modules are compiled against a specific Python"
        Write-Host "  version. If your venv Python ($PyVersion) does not match, you will get"
        Write-Host "  import errors. In that case, install a matching Python version or use conda:"
        Write-Host "    conda install -c conda-forge freecad python=$PyVersion"
    }
} else {
    Write-Host "FreeCAD is NOT installed."
    Write-Host "Some features (parametric CAD generation, .FCStd I/O) require FreeCAD."
    Write-Host ""
    Write-Host "Install FreeCAD on Windows:"
    Write-Host "  1. Download from https://www.freecad.org/downloads.php"
    Write-Host "  2. Install, then add FreeCAD's bin dir to PYTHONPATH:"
    Write-Host '     $env:PYTHONPATH = "C:\Program Files\FreeCAD 1.0\bin;$env:PYTHONPATH"'
    Write-Host "     (Adjust the path to match your FreeCAD install location.)"
    Write-Host ""
    Write-Host "Alternatively, install via conda:"
    Write-Host "  conda install -c conda-forge freecad"
    Write-Host ""
    Write-Host "NOTE: FreeCAD's Python modules are compiled against a specific Python"
    Write-Host "version. Your venv uses Python $PyVersion. If FreeCAD was built against"
    Write-Host "a different version, 'import FreeCAD' will fail even with the correct"
    Write-Host "PYTHONPATH. Use conda to get a matching set:"
    Write-Host "  conda install -c conda-forge freecad python=$PyVersion"
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
